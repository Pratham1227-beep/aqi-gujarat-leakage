"""Rolling feature construction.

Rolling statistics shifted by at least 1 day before splitting to prevent leakage.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: Sequence[int],
    stats: Sequence[str] = ("mean", "std"),
    shift: int = 1,
) -> pd.DataFrame:
    """Add rolling statistics shifted by *shift* (default 1 day before split)."""
    out = df.copy()
    for window in windows:
        for stat in stats:
            if stat == "mean":
                rolled = out[col].rolling(window=window, min_periods=1).mean()
            elif stat == "std":
                rolled = out[col].rolling(window=window, min_periods=1).std()
            else:
                continue
            out[f"{col}_roll_{stat}_{window}"] = rolled.shift(shift)
    return out
