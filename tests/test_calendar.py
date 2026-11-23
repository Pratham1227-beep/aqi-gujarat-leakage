"""Unit tests for complete calendar reindexing."""

from __future__ import annotations

import pandas as pd

from aqi.data.calendar import build_daily_index, reindex_to_complete


def test_build_daily_index() -> None:
    idx = build_daily_index("2023-01-01", "2023-01-05")
    assert len(idx) == 5
    assert idx[0] == pd.Timestamp("2023-01-01")
    assert idx[-1] == pd.Timestamp("2023-01-05")


def test_reindex_to_complete_inserts_missing_dates() -> None:
    # Gap on Jan 2 and Jan 4
    df = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-03", "2023-01-05"],
        "pm25": [25.0, 30.0, 45.0],
    })
    reindexed = reindex_to_complete(df)
    assert len(reindexed) == 5
    assert list(reindexed["date"]) == list(pd.date_range("2023-01-01", "2023-01-05"))
    assert pd.isna(reindexed.loc[1, "pm25"])
    assert pd.isna(reindexed.loc[3, "pm25"])
