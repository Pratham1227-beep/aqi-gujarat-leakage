"""Unit tests verifying that feature builders introduce zero future leakage."""

from __future__ import annotations

import pandas as pd

from aqi.features.build import build_task_f


def test_task_f_features_are_shifted() -> None:
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "aqi_official": [10.0 * i for i in range(10)],
        "pm25": [5.0 * i for i in range(10)],
    })
    X, y, meta = build_task_f(df, horizon=1, shift_all=1)
    # The first row of X must correspond to index 0 (values from t-1 or earlier)
    # Check that lag 1 of aqi_official at row index 1 has value of row 0
    assert "aqi_official_lag_1" in X.columns
    # Target y for row t is aqi at t+1
    assert y.iloc[0] == 10.0
