"""Physical range and flag validation for hourly/daily pollutant values.

Drops physically impossible values and honours instrument flags if present.
Logging counts removed per pollutant happens in the build pipeline.
"""

from __future__ import annotations

import pandas as pd

# Physical plausibility bounds (units as in CPCB data).
VALIDITY_RULES: dict[str, tuple[float, float]] = {
    "pm25": (0.0, 1000.0),  # ug/m3
    "pm10": (0.0, 2000.0),
    "no2": (0.0, 500.0),  # ppb
    "so2": (0.0, 200.0),  # ppb
    "co": (0.0, 100.0),  # mg/m3
    "o3": (0.0, 300.0),  # ppb
    "nh3": (0.0, 500.0),  # ug/m3
}


def flag_invalid_values(
    df: pd.DataFrame,
    rules: dict[str, tuple[float, float]] | None = None,
    flag_cols: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return *df* with invalid pollutant values set to NaN, and a count of removals.

    Parameters
    ----------
    df:
        DataFrame with one column per pollutant.
    rules:
        Mapping pollutant -> (low, high) inclusive bounds. Defaults to
        :data:`VALIDITY_RULES`.
    flag_cols:
        Optional set of boolean columns; rows where the flag is ``False`` are
        treated as invalid for the matching pollutant column (e.g. ``pm25_flag``
        maps to ``pm25``).

    Returns
    -------
    (cleaned_df, removal_counts)
    """
    rules = rules or VALIDITY_RULES
    flag_cols = flag_cols or set()
    out = df.copy()
    removed: dict[str, int] = {}

    for pollutant, (low, high) in rules.items():
        if pollutant not in out.columns:
            continue
        mask = (out[pollutant] < low) | (out[pollutant] > high)
        n_removed = int(mask.sum())
        out.loc[mask, pollutant] = pd.NA
        removed[pollutant] = n_removed

        # Honour instrument flag if a matching *_flag column exists.
        flag_col = f"{pollutant}_flag"
        if flag_col in out.columns:
            bad = out[flag_col].eq(False) & out[pollutant].notna()
            out.loc[bad, pollutant] = pd.NA
            removed[pollutant] += int(bad.sum())

    return out, removed