"""Lag feature construction.

Pure functions: take a daily DataFrame and lag specifications.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_target_lags(
    df: pd.DataFrame,
    target_col: str,
    lags: Sequence[int],
) -> pd.DataFrame:
    """Add target lag columns (t-1, t-2, ...). All shifted by lag before splitting."""
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_pollutant_lags(
    df: pd.DataFrame,
    pollutant_cols: Sequence[str],
    lags: Sequence[int],
) -> pd.DataFrame:
    """Add pollutant lag columns."""
    out = df.copy()
    for col in pollutant_cols:
        if col not in out.columns:
            continue
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
    return out
