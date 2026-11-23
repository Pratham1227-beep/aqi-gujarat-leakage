"""Evaluation harness for running experiments across cities, setups, models, and horizons.

Saves predictions to results/preds/ and logs metrics to results/runs.csv.
Computes Block Bootstrap 95% CI and Diebold-Mariano test vs Persistence.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aqi.eval.bootstrap import block_bootstrap_mae
from aqi.eval.dm_test import diebold_mariano_test
from aqi.eval.leakage_checks import (
    check_feature_target_correlation,
    check_temporal_separation,
)
from aqi.eval.metrics import (
    compute_band_accuracy,
    compute_directional_accuracy,
    compute_mae,
    compute_r2,
    compute_rmse,
    compute_skill,
    compute_spike_recall,
)
from aqi.features.build import build_task_f, build_task_n
from aqi.models.base import Forecaster
from aqi.models.registry import MODEL_REGISTRY, get_model
from aqi.splits.chrono_split import chrono_split
from aqi.splits.random_split import random_split
from aqi.splits.rolling_origin import rolling_origin_splits

log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
PREDS_DIR = RESULTS_DIR / "preds"
RUNS_CSV = RESULTS_DIR / "runs.csv"

# Standard models for the benchmark
BENCHMARK_MODELS = [
    "persistence",
    "seasonal_naive",
    "moving_average",
    "ridge",
    "random_forest",
    "xgboost",
]

# Setups A & B require pollutants (only Ahmedabad and Chennai qualify)
POLLUTANT_CITIES = ["ahmedabad", "chennai"]
ALL_QUALIFIED_CITIES = ["ahmedabad", "chennai", "gandhinagar", "ankleshwar", "vapi", "surat"]


def _init_results_dir() -> None:
    PREDS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "logs").mkdir(parents=True, exist_ok=True)


def load_city_data(city: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{city.lower()}_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    return pd.read_parquet(path)


def run_experiment(
    model: Forecaster,
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    setup: str,
    dates: pd.Series | None = None,
    persistence_mae: float | None = None,
) -> dict[str, Any]:
    """Execute a single experimental run and return metrics."""
    check_feature_target_correlation(X, y)

    if dates is not None and setup in ("B", "C"):
        check_temporal_separation(dates.iloc[train_idx], dates.iloc[test_idx], setup)

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    y_test_arr = y_test.to_numpy()
    mae = compute_mae(y_test_arr, y_pred)
    rmse = compute_rmse(y_test_arr, y_pred)
    r2 = compute_r2(y_test_arr, y_pred)
    skill = (
        compute_skill(mae, persistence_mae)
        if persistence_mae is not None
        else float("nan")
    )
    band_acc = compute_band_accuracy(y_test_arr, y_pred)
    spike_rec = compute_spike_recall(y_test_arr, y_pred)
    dir_acc = compute_directional_accuracy(y_test_arr, y_pred)

    # Block bootstrap 95% CI on MAE
    ci_low, ci_high = block_bootstrap_mae(y_test_arr, y_pred, block_length=7, n_resamples=1000)

    return {
        "model": model.name,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "mae": mae,
        "mae_ci_low": ci_low,
        "mae_ci_high": ci_high,
        "rmse": rmse,
        "r2": r2,
        "skill": skill,
        "band_acc": band_acc,
        "spike_recall": spike_rec,
        "directional_acc": dir_acc,
        "y_pred": y_pred,
        "y_true": y_test_arr,
        "dates_test": dates.iloc[test_idx].to_numpy() if dates is not None else None,
    }


RUNS_COLUMNS = [
    "city",
    "setup",
    "model",
    "horizon",
    "seed",
    "n_train",
    "n_test",
    "mae",
    "mae_ci_low",
    "mae_ci_high",
    "rmse",
    "r2",
    "skill",
    "band_acc",
    "spike_recall",
    "directional_acc",
    "dm_stat",
    "dm_p",
]


def append_to_runs_csv(row: dict[str, Any]) -> None:
    _init_results_dir()
    # Align to standard columns
    row_aligned = {col: row.get(col, np.nan) for col in RUNS_COLUMNS}
    df_row = pd.DataFrame([row_aligned])
    if not RUNS_CSV.exists():
        df_row.to_csv(RUNS_CSV, index=False)
    else:
        df_row.to_csv(RUNS_CSV, mode="a", header=False, index=False)


def save_predictions(city: str, setup: str, model_name: str, horizon: int, seed: int | None, y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray | None) -> None:
    _init_results_dir()
    seed_str = f"_s{seed}" if seed is not None else ""
    filename = f"{city}_{setup}_{model_name}_h{horizon}{seed_str}.csv"
    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })
    if dates is not None:
        df["date"] = dates
    df.to_csv(PREDS_DIR / filename, index=False)


def run_setup_a(city: str, models: list[str] | None = None, seeds: list[int] | None = None) -> list[dict[str, Any]]:
    """Execute Setup A: Random 80/20 split on Task N (same-day pollutants) across multiple seeds."""
    if city.lower() not in POLLUTANT_CITIES:
        log.warning("Skipping Setup A for %s: requires pollutant concentrations", city)
        return []

    models = models or BENCHMARK_MODELS
    seeds = seeds or [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    df = load_city_data(city)

    # Impute missing features in Task N using median across whole city dataset (reproducing literature baseline)
    X, y, meta = build_task_n(df)
    valid_mask = y.notna() & X["pm25"].notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    results = []
    for seed in seeds:
        train_idx, test_idx = random_split(X, test_fraction=0.2, seed=seed)
        
        # In Setup A, persistence predicts mean of training target (random splits break sequential order)
        y_train_mean = float(y.iloc[train_idx].mean())
        persistence_mae = float(np.abs(y.iloc[test_idx] - y_train_mean).mean())

        for m_name in models:
            if m_name in ("persistence", "seasonal_naive", "moving_average"):
                continue  # time-series lag baselines not well-defined under shuffled random split
            model = get_model(m_name, seed=seed)
            res = run_experiment(model, X, y, train_idx, test_idx, setup="A", persistence_mae=persistence_mae)
            res.update({
                "city": city.lower(),
                "setup": "A",
                "horizon": 0,
                "seed": seed,
            })
            append_to_runs_csv(res)
            save_predictions(city.lower(), "A", m_name, 0, seed, res["y_true"], res["y_pred"], None)
            results.append(res)
            log.info("Setup A | %s | %s | Seed %d | MAE: %.2f | R2: %.4f", city, m_name, seed, res["mae"], res["r2"])

    return results


def run_setup_b(city: str, models: list[str] | None = None) -> list[dict[str, Any]]:
    """Execute Setup B: Chronological 80/20 split on Task N (same-day pollutants)."""
    if city.lower() not in POLLUTANT_CITIES:
        log.warning("Skipping Setup B for %s: requires pollutant concentrations", city)
        return []

    models = models or BENCHMARK_MODELS
    df = load_city_data(city)
    dates = pd.to_datetime(df["date"])

    X, y, meta = build_task_n(df)
    valid_mask = y.notna() & X["pm25"].notna()
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    dates = dates[valid_mask].reset_index(drop=True)

    train_idx, test_idx = chrono_split(X, test_fraction=0.2)
    
    # Chronological persistence: last observation in train or sequential lag
    y_test_arr = y.iloc[test_idx].to_numpy()
    persistence_preds = y.iloc[test_idx].shift(1).fillna(y.iloc[train_idx[-1]]).to_numpy()
    persistence_mae = float(np.abs(y_test_arr - persistence_preds).mean())

    results = []
    for m_name in models:
        if m_name in ("persistence", "seasonal_naive", "moving_average"):
            continue
        model = get_model(m_name, seed=42)
        res = run_experiment(model, X, y, train_idx, test_idx, setup="B", dates=dates, persistence_mae=persistence_mae)
        
        # Diebold-Mariano test vs persistence
        dm_stat, dm_p = diebold_mariano_test(
            y_test_arr - res["y_pred"],
            y_test_arr - persistence_preds,
            h=1,
        )
        res.update({
            "city": city.lower(),
            "setup": "B",
            "horizon": 0,
            "seed": 42,
            "dm_stat": dm_stat,
            "dm_p": dm_p,
        })
        append_to_runs_csv(res)
        save_predictions(city.lower(), "B", m_name, 0, 42, res["y_true"], res["y_pred"], res["dates_test"])
        results.append(res)
        log.info("Setup B | %s | %s | MAE: %.2f | R2: %.4f | DM p: %.4f", city, m_name, res["mae"], res["r2"], dm_p)

    return results


def run_setup_c(city: str, models: list[str] | None = None, horizons: list[int] | None = None, refit_cadence: int = 30) -> list[dict[str, Any]]:
    """Execute Setup C: Rolling-origin leak-free evaluation on Task F for horizons h=1 and h=3."""
    models = models or BENCHMARK_MODELS
    horizons = horizons or [1, 3]
    df = load_city_data(city)

    results = []
    for h in horizons:
        X, y, meta = build_task_f(df, horizon=h, shift_all=1)
        dates = pd.to_datetime(df.loc[y.index, "date"]).reset_index(drop=True)
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        # Baseline: persistence forecast
        # For horizon h at origin t, persistence predicts y_t (which is aqi_official_lag_1 when y is t+h and shift=1)
        pers_model = get_model("persistence")
        pers_preds_all: list[float] = []
        pers_actuals_all: list[float] = []
        test_dates_all: list[Any] = []

        # Iterate over rolling origin splits
        # Final 20% of data is test window
        splits = list(rolling_origin_splits(X, test_fraction=0.2, refit_cadence=refit_cadence))
        
        # First compute persistence predictions across all test points
        for train_idx, test_idx, _refit in splits:
            pers_model.fit(X.iloc[train_idx], y.iloc[train_idx])
            p_val = pers_model.predict(X.iloc[test_idx])
            pers_preds_all.extend(p_val)
            pers_actuals_all.extend(y.iloc[test_idx].to_numpy())
            test_dates_all.extend(dates.iloc[test_idx].to_numpy())

        pers_preds_arr = np.array(pers_preds_all)
        pers_actuals_arr = np.array(pers_actuals_all)
        pers_mae = compute_mae(pers_actuals_arr, pers_preds_arr)

        # Log persistence baseline run
        pers_res = {
            "city": city.lower(),
            "setup": "C",
            "model": "persistence",
            "horizon": h,
            "seed": 42,
            "n_train": len(splits[0][0]),
            "n_test": len(pers_actuals_arr),
            "mae": pers_mae,
            "rmse": compute_rmse(pers_actuals_arr, pers_preds_arr),
            "r2": compute_r2(pers_actuals_arr, pers_preds_arr),
            "skill": 0.0,
            "band_acc": compute_band_accuracy(pers_actuals_arr, pers_preds_arr),
            "directional_acc": compute_directional_accuracy(pers_actuals_arr, pers_preds_arr),
            "dm_stat": 0.0,
            "dm_p": 1.0,
        }
        append_to_runs_csv(pers_res)
        save_predictions(city.lower(), "C", "persistence", h, 42, pers_actuals_arr, pers_preds_arr, np.array(test_dates_all))
        results.append(pers_res)
        log.info("Setup C | %s | Persistence | h=%d | MAE: %.2f | R2: %.4f", city, h, pers_mae, pers_res["r2"])

        # Now evaluate all other registered models
        for m_name in models:
            if m_name == "persistence":
                continue

            preds_all: list[float] = []
            actuals_all: list[float] = []
            model = get_model(m_name, seed=42) if "seed" in MODEL_REGISTRY[m_name].__init__.__code__.co_varnames else get_model(m_name)

            for train_idx, test_idx, refit in splits:
                if refit:
                    model.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = model.predict(X.iloc[test_idx])
                preds_all.extend(preds)
                actuals_all.extend(y.iloc[test_idx].to_numpy())

            preds_arr = np.array(preds_all)
            actuals_arr = np.array(actuals_all)

            m_mae = compute_mae(actuals_arr, preds_arr)
            m_rmse = compute_rmse(actuals_arr, preds_arr)
            m_r2 = compute_r2(actuals_arr, preds_arr)
            m_skill = compute_skill(m_mae, pers_mae)
            m_band = compute_band_accuracy(actuals_arr, preds_arr)
            m_dir = compute_directional_accuracy(actuals_arr, preds_arr)

            # Diebold-Mariano test vs persistence
            dm_stat, dm_p = diebold_mariano_test(
                actuals_arr - preds_arr,
                actuals_arr - pers_preds_arr,
                h=h,
            )

            res = {
                "city": city.lower(),
                "setup": "C",
                "model": m_name,
                "horizon": h,
                "seed": 42,
                "n_train": len(splits[0][0]),
                "n_test": len(actuals_arr),
                "mae": m_mae,
                "rmse": m_rmse,
                "r2": m_r2,
                "skill": m_skill,
                "band_acc": m_band,
                "directional_acc": m_dir,
                "dm_stat": dm_stat,
                "dm_p": dm_p,
            }
            append_to_runs_csv(res)
            save_predictions(city.lower(), "C", m_name, h, 42, actuals_arr, preds_arr, np.array(test_dates_all))
            results.append(res)
            log.info("Setup C | %s | %s | h=%d | MAE: %.2f | R2: %.4f | Skill: %.3f | DM p: %.4f",
                     city, m_name, h, m_mae, m_r2, m_skill, dm_p)

    return results


def run_all_experiments(cities: list[str] | None = None, setups: list[str] | None = None, horizons: list[int] | None = None) -> list[dict[str, Any]]:
    """Execute the full experimental suite across all cities and setups."""
    cities = cities or ALL_QUALIFIED_CITIES
    setups = setups or ["A", "B", "C"]
    horizons = horizons or [1, 3]

    all_results = []
    for city in cities:
        log.info("=== Starting experiments for city: %s ===", city)
        if "A" in setups and city in POLLUTANT_CITIES:
            all_results.extend(run_setup_a(city))
        if "B" in setups and city in POLLUTANT_CITIES:
            all_results.extend(run_setup_b(city))
        if "C" in setups:
            all_results.extend(run_setup_c(city, horizons=horizons))
    return all_results
