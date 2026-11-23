"""Hourly-to-daily aggregation with CPCB completeness rules.

Rules:
- Daily mean of hourly values.
- NaN unless at least 16 of 24 hours exist (CO and O3: 8-hour rolling max per
  CPCB, then daily max).
Keep ``n_hours_*`` counts.
"""

from __future__ import annotations

import pandas as pd


def daily_means(
    hourly_df: pd.DataFrame,
    hours_col: str = "hour",
    value_cols: list[str] | None = None,
    min_hours: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with daily aggregated means.

    Parameters
    ----------
    hourly_df:
        Hourly DataFrame indexed or with a ``date`` column (or a datetime index).
    hours_col:
        Column name if ``hourly_df`` is not indexed.
    value_cols:
        List of pollutant column names to aggregate. If ``None`` all columns
        that look like pollutant values are used.
    min_hours:
        Minimum number of valid hours per pollutant to keep the daily value.
        Defaults to 16 for PM2.5/PM10 and 8 for CO/O3 per CPCB.

    Returns
    -------
    daily_df
        Daily means with ``n_hours_<pollutant>`` columns added.
    """
    df = hourly_df.copy()
    if value_cols is None:
        # Heuristic: all float/int columns that aren't dates/ids
        exclude = {"date", "city", "source", "year", "month", "day"}
        value_cols = [
            c for c in df.columns if c not in exclude and df[c].dtype in [float, int]
        ]

    if min_hours is None:
        min_hours = {
            "pm25": 16,
            "pm10": 16,
            "no2": 16,
            "so2": 16,
            "co": 8,
            "o3": 8,
            "nh3": 16,
        }

    # If there's a datetime index, derive date
    if df.index.name == "date" or (isinstance(df.index, pd.DatetimeIndex)):
        df = df.reset_index().rename(columns={"date": "date"})
    elif "date" not in df.columns and "hour" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).date if hasattr(pd.to_datetime(df["hour"]).values[0], "date") else pd.to_datetime(df["hour"]).date

    daily_rows = []
    for date_, group in df.groupby(df["date"].dt.date if "date" in df.columns else df.index.get_level_values(0).date):
        row = {"date": pd.Timestamp(date_)}
        for col in value_cols:
            vals = pd.to_numeric(group[col], errors="coerce").dropna()
            n = len(vals)
            row[f"n_hours_{col}"] = n
            if n >= min_hours.get(col, 16):
                if col in ("co", "o3"):
                    # CPCB: 8-hour rolling max first, then daily max
                    rolling_max = vals.rolling(window=8, min_periods=1).max()
                    daily_val = rolling_max.max()
                else:
                    daily_val = vals.mean()
                row[col] = daily_val
            else:
                row[col] = float("nan")
        daily_rows.append(row)

    daily_df = pd.DataFrame(daily_rows)
    return daily_df