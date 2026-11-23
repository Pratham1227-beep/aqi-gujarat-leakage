"""Unit tests verifying that feature builders introduce zero future leakage."""

from __future__ import annotations

import pandas as pd
import pytest

from aqi.features.build import build_task_f, build_task_n


def test_task_f_features_are_shifted() -> None:
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "aqi_official": [10.0 * i for i in range(10)],
        "pm25": [5.0 * i for i in range(10)],
    })
    X, y, meta = build_task_f(df, horizon=1, shift_all=1)
    assert "aqi_official_lag_1" in X.columns
    # Target y for row t is aqi at t+1
    assert y.iloc[0] == 10.0
    # Crucial: Unlagged target and unlagged pollutants must not be in X
    assert "aqi_official" not in X.columns
    assert "pm25" not in X.columns


@pytest.mark.parametrize("h", [1, 3])
def test_task_f_horizons(h: int) -> None:
    dates = pd.date_range("2023-01-01", periods=20, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "aqi_official": [10.0 * i for i in range(20)],
        "pm25": [2.0 * i for i in range(20)],
    })
    X, y, meta = build_task_f(df, horizon=h, shift_all=1)
    assert meta["horizon"] == h
    assert len(X) == 20 - h
    # Target y at row 0 is AQI at 0 + h
    assert y.iloc[0] == 10.0 * h
    # Every feature must be lagged, rolling, calendar, or weather
    for col in X.columns:
        assert (
            "_lag_" in col
            or "_roll_" in col
            or col.startswith("calendar_")
            or col.startswith("weather_")
        ), f"Unexpected unlagged/unfiltered column in Task F: {col}"


def test_task_n_matrix() -> None:
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "aqi_official": [10.0 * i for i in range(10)],
        "pm25": [5.0 * i for i in range(10)],
        "no2": [3.0 * i for i in range(10)],
    })
    X, y, meta = build_task_n(df)
    assert meta["task"] == "N"
    assert "pm25" in X.columns
    assert "no2" in X.columns
    assert "aqi_official" not in X.columns
    assert y.iloc[0] == 0.0
