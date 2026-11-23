"""Complete daily calendar index per city.

Builds a full daily date range for each city so no date is missing before
features or splits are constructed.
"""

from __future__ import annotations

import pandas as pd


def build_daily_index(start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Return a complete daily DatetimeIndex from *start* to *end* inclusive."""
    return pd.date_range(start=start, end=end, freq="D", name="date")


def reindex_to_complete(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Reindex *df* to a complete daily calendar between its min and max dates.

    Missing dates are inserted as NaN rows so downstream code can see gaps
    explicitly instead of silently skipping them.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    full = build_daily_index(df[date_col].min(), df[date_col].max())
    df = df.set_index(date_col).reindex(full)
    df.index.name = date_col
    return df.reset_index()
