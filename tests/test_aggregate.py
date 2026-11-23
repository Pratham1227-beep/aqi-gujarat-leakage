"""Unit tests for hourly to daily aggregation and completeness rules."""

from __future__ import annotations

import pandas as pd

from aqi.data.aggregate import daily_means


def test_daily_means_completeness() -> None:
    # 24 hours for day 1 (valid), 10 hours for day 2 (< 16, invalid)
    dates_d1 = pd.date_range("2023-01-01 00:00", "2023-01-01 23:00", freq="h")
    dates_d2 = pd.date_range("2023-01-02 00:00", "2023-01-02 09:00", freq="h")
    all_dates = dates_d1.union(dates_d2)

    hourly = pd.DataFrame({
        "date": all_dates,
        "pm25": [50.0] * len(all_dates),
    })
    daily = daily_means(hourly, value_cols=["pm25"])
    assert len(daily) == 2
    assert daily.loc[0, "n_hours_pm25"] == 24
    assert daily.loc[0, "pm25"] == 50.0
    assert daily.loc[1, "n_hours_pm25"] == 10
    assert pd.isna(daily.loc[1, "pm25"])
