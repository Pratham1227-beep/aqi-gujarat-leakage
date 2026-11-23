"""Leakage test 2: Shuffled-target test."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aqi.eval.metrics import compute_mae, compute_skill
from aqi.models.baselines import PersistenceForecaster
from aqi.models.sklearn_models import RidgeForecaster


def test_shuffled_target_destroys_skill() -> None:
    rng = np.random.default_rng(42)
    n = 200
    # Synthetic autocorrelated signal
    x_val = np.linspace(0, 100, n)
    y = pd.Series(x_val + rng.normal(0, 1, n))
    X = pd.DataFrame({"aqi_official_lag_1": y.shift(1).fillna(0)})

    train_idx = np.arange(150)
    test_idx = np.arange(150, n)

    # Persistence baseline
    pers = PersistenceForecaster(target_lag_col="aqi_official_lag_1")
    pers.fit(X.iloc[train_idx], y.iloc[train_idx])
    y_pred_pers = pers.predict(X.iloc[test_idx])
    mae_pers = compute_mae(y.iloc[test_idx].to_numpy(), y_pred_pers)

    # Fit Ridge on SHUFFLED training target
    y_shuffled = y.iloc[train_idx].sample(frac=1.0, random_state=42).reset_index(drop=True)
    ridge = RidgeForecaster()
    ridge.fit(X.iloc[train_idx].reset_index(drop=True), y_shuffled)
    y_pred_ridge = ridge.predict(X.iloc[test_idx])
    mae_ridge = compute_mae(y.iloc[test_idx].to_numpy(), y_pred_ridge)

    skill = compute_skill(mae_ridge, mae_pers)
    # Shuffled target model skill must fall to or below 0 (persistence)
    assert skill <= 0.05
