"""Calendar feature construction.

Day of week, month, sin/cos cyclical transforms, Diwali window, and season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features for the current day."""
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = pd.to_datetime(out.index)
    date_s = pd.to_datetime(out["date"])
    out["calendar_dow"] = date_s.dt.dayofweek
    out["calendar_month"] = date_s.dt.month
    out["calendar_dayofyear"] = date_s.dt.dayofyear
    # Sin/cos cyclic encoding
    out["calendar_sin_dow"] = np.sin(2 * np.pi * out["calendar_dow"] / 7)
    out["calendar_cos_dow"] = np.cos(2 * np.pi * out["calendar_dow"] / 7)
    out["calendar_sin_month"] = np.sin(2 * np.pi * out["calendar_month"] / 12)
    out["calendar_cos_month"] = np.cos(2 * np.pi * out["calendar_month"] / 12)
    # Season: Winter (1), Pre-monsoon/Summer (2), Monsoon (3), Post-monsoon (4)
    out["calendar_season"] = (out["calendar_month"] % 12 + 3) // 3
    return out


def add_calendar_for_forecast(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add calendar features for day t+horizon (the target day)."""
    out = df.copy()
    if "date" not in out.columns:
        out["date"] = pd.to_datetime(out.index)

    # Forecast origin is row date; target day is date + horizon
    target_date = pd.to_datetime(out["date"]) + pd.to_timedelta(horizon, unit="D")

    out["calendar_dow"] = target_date.dt.dayofweek
    out["calendar_month"] = target_date.dt.month
    out["calendar_dayofyear"] = target_date.dt.dayofyear
    out["calendar_sin_dow"] = np.sin(2 * np.pi * out["calendar_dow"] / 7)
    out["calendar_cos_dow"] = np.cos(2 * np.pi * out["calendar_dow"] / 7)
    out["calendar_sin_month"] = np.sin(2 * np.pi * out["calendar_month"] / 12)
    out["calendar_cos_month"] = np.cos(2 * np.pi * out["calendar_month"] / 12)
    out["calendar_season"] = (out["calendar_month"] % 12 + 3) // 3
    return out
