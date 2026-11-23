"""Unit tests for CPCB AQI reconstruction."""

from __future__ import annotations

import pandas as pd

from aqi.data.reconstruct import compute_aqi, sub_index


def test_sub_index_breakpoints() -> None:
    assert sub_index("pm25", 15.0) == 25.0
    assert sub_index("pm25", 30.0) == 50.0
    assert sub_index("pm25", 60.0) == 100.0


def test_compute_aqi_minimum_pollutants() -> None:
    # Need >= 3 pollutants and at least one of PM2.5 or PM10
    df_valid = pd.DataFrame([{
        "pm25": 30.0,
        "no2": 40.0,
        "so2": 40.0,
    }])
    res = compute_aqi(df_valid)
    assert res.loc[0, "aqi_recomputed"] == 50.0

    # Without PM: invalid
    df_no_pm = pd.DataFrame([{
        "no2": 40.0,
        "so2": 40.0,
        "nh3": 50.0,
    }])
    res_no_pm = compute_aqi(df_no_pm)
    assert pd.isna(res_no_pm.loc[0, "aqi_recomputed"])
