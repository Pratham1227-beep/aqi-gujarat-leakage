"""Task N (nowcast) vs Task F (forecast) matrix builders."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from aqi.features.calendar_feats import add_calendar, add_calendar_for_forecast
from aqi.features.lags import add_pollutant_lags, add_target_lags
from aqi.features.rolling import add_rolling_features


def build_task_n(
    df: pd.DataFrame,
    target_col: str = "aqi_official",
    pollutant_cols: list[str] | None = None,
    use_calendar: bool = True,
    use_weather: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Build Task N (nowcast) feature matrix and target.

    X holds day-t pollutants, calendar features, optional weather at day t.
    y is AQI at t. Used for setups A and B only.
    """
    if pollutant_cols is None:
        pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]

    X = df.copy()
    y = X.pop(target_col)

    if use_calendar:
        X = add_calendar(X)

    feature_cols = [
        c
        for c in X.columns
        if c in pollutant_cols
        or any(c.startswith(f"{p}_") for p in pollutant_cols)
        or c.startswith("calendar_")
        or c.startswith("weather_")
    ]
    X = X[feature_cols]

    meta = {
        "task": "N",
        "target_col": target_col,
        "features": feature_cols,
        "note": "Same-day pollutants used. Used for setups A and B only.",
    }
    return X, y, meta


def build_task_f(
    df: pd.DataFrame,
    target_col: str = "aqi_official",
    horizon: int = 1,
    target_lags: Sequence[int] = (1, 2, 3, 7, 14),
    pollutant_cols: list[str] | None = None,
    pollutant_lags: Sequence[int] = (1, 2, 3, 7),
    rolling_windows: Sequence[int] = (3, 7, 14, 30),
    rolling_stats: Sequence[str] = ("mean", "std"),
    shift_all: int = 1,
    use_calendar: bool = True,
    use_weather: bool = False,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Build Task F (forecast) feature matrix and target.

    X holds information up to day t only, y is AQI at t+horizon.
    All features are shifted by *shift_all* (default 1) to prevent leakage.
    """
    if pollutant_cols is None:
        pollutant_cols = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]

    X = df.copy()
    y = X[target_col].shift(-horizon)

    # Target lags
    X = add_target_lags(X, target_col, target_lags)

    # Pollutant lags
    X = add_pollutant_lags(X, pollutant_cols, pollutant_lags)

    # Rolling stats of AQI (shifted by shift_all)
    X = add_rolling_features(
        X, target_col, rolling_windows, rolling_stats, shift=shift_all
    )

    # Calendar features for day t+horizon
    if use_calendar:
        X = add_calendar_for_forecast(X, horizon)

    # Drop rows where y is NaN (end of series)
    mask = y.notna()
    X = X[mask]
    y = y[mask]

    feature_cols = [
        c
        for c in X.columns
        if "_lag_" in c
        or "_roll_" in c
        or c.startswith("calendar_")
        or c.startswith("weather_")
    ]
    X = X[feature_cols]

    meta = {
        "task": "F",
        "target_col": target_col,
        "horizon": horizon,
        "features": feature_cols,
        "note": "All features shifted by shift_all. Data up to day t only.",
    }
    return X, y, meta
