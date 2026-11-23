"""Evaluation harness for running experiments across cities, setups, models, and horizons."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from aqi.eval.leakage_checks import (
    check_feature_target_correlation,
    check_temporal_separation,
)
from aqi.eval.metrics import (
    compute_band_accuracy,
    compute_mae,
    compute_r2,
    compute_rmse,
    compute_skill,
    compute_spike_recall,
)
from aqi.models.base import Forecaster


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
    # Leakage check: feature correlation with future target
    check_feature_target_correlation(X, y)

    # Temporal separation check
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

    return {
        "model": model.name,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "skill": skill,
        "band_acc": band_acc,
        "spike_recall": spike_rec,
        "y_pred": y_pred,
    }
