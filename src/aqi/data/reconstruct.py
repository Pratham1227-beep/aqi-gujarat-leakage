"""CPCB sub-index and AQI reconstruction from pollutant concentrations.

Implements CPCB sub-indices by piecewise linear interpolation over the CPCB
breakpoint table (see docs/cpcb_breakpoints.md). AQI is the maximum sub-index,
valid only if at least 3 pollutants are present and one of them is PM2.5 or PM10.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Breakpoints: (low_conc, high_conc, low_index, high_index)
BREAKPOINTS: dict[str, list[tuple[float, float, float, float]]] = {
    "pm25": [
        (0.0, 30.0, 0.0, 50.0),
        (31.0, 60.0, 51.0, 100.0),
        (61.0, 90.0, 101.0, 200.0),
        (91.0, 120.0, 201.0, 300.0),
        (121.0, 250.0, 301.0, 400.0),
        (251.0, 380.0, 401.0, 500.0),
    ],
    "pm10": [
        (0.0, 50.0, 0.0, 50.0),
        (51.0, 100.0, 51.0, 100.0),
        (101.0, 250.0, 101.0, 200.0),
        (251.0, 350.0, 201.0, 300.0),
        (351.0, 430.0, 301.0, 400.0),
        (431.0, 500.0, 401.0, 500.0),
    ],
    "no2": [
        (0.0, 40.0, 0.0, 50.0),
        (41.0, 80.0, 51.0, 100.0),
        (81.0, 180.0, 101.0, 200.0),
        (181.0, 280.0, 201.0, 300.0),
        (281.0, 400.0, 301.0, 400.0),
        (401.0, 500.0, 401.0, 500.0),
    ],
    "so2": [
        (0.0, 40.0, 0.0, 50.0),
        (41.0, 80.0, 51.0, 100.0),
        (81.0, 380.0, 101.0, 200.0),
        (381.0, 800.0, 201.0, 300.0),
        (801.0, 1600.0, 301.0, 400.0),
        (1601.0, 2000.0, 401.0, 500.0),
    ],
    "co": [
        (0.0, 1.0, 0.0, 50.0),
        (1.1, 2.0, 51.0, 100.0),
        (2.1, 10.0, 101.0, 200.0),
        (10.1, 17.0, 201.0, 300.0),
        (17.1, 34.0, 301.0, 400.0),
        (34.1, 50.0, 401.0, 500.0),
    ],
    "o3": [
        (0.0, 50.0, 0.0, 50.0),
        (51.0, 100.0, 51.0, 100.0),
        (101.0, 168.0, 101.0, 200.0),
        (169.0, 208.0, 201.0, 300.0),
        (209.0, 748.0, 301.0, 400.0),
        (749.0, 1000.0, 401.0, 500.0),
    ],
    "nh3": [
        (0.0, 200.0, 0.0, 50.0),
        (201.0, 400.0, 51.0, 100.0),
        (401.0, 800.0, 101.0, 200.0),
        (801.0, 1200.0, 201.0, 300.0),
        (1201.0, 1800.0, 301.0, 400.0),
        (1801.0, 2400.0, 401.0, 500.0),
    ],
}

# Index ranges for each category
INDEX_RANGES: list[tuple[int, int]] = [
    (0, 50),
    (51, 100),
    (101, 200),
    (201, 300),
    (301, 400),
    (401, 500),
]


def sub_index(pollutant: str, concentration: float | None) -> float | None:
    """Compute the CPCB sub-index for a single pollutant concentration.

    Returns ``None`` if concentration is missing or outside the table.
    """
    if concentration is None or pd.isna(concentration) or concentration < 0:
        return None
    bps = BREAKPOINTS.get(pollutant)
    if bps is None:
        return None

    for bp_lo, bp_hi, i_lo, i_hi in bps:
        if bp_lo <= concentration <= bp_hi:
            # Linear interpolation
            return float(i_lo + (i_hi - i_lo) / (bp_hi - bp_lo) * (concentration - bp_lo))

    # Outside the table
    if concentration > bps[-1][1]:
        return float(bps[-1][3])
    return float(bps[0][2])


def compute_aqi(df: pd.DataFrame, pollutant_cols: list[str] | None = None) -> pd.DataFrame:
    """Compute the AQI from pollutant concentrations.

    Returns a copy of *df* with additional columns:
    - ``sub_index_<pollutant>`` for each pollutant
    - ``aqi_recomputed`` (float, NaN if invalid)
    - ``aqi_bucket_recomputed`` (int, 0-5 if valid, -1 if invalid)

    AQI is valid only if at least 3 pollutants have valid sub-indices and at
    least one of PM2.5 or PM10 is present.
    """
    if pollutant_cols is None:
        pollutant_cols = [c for c in df.columns if c in BREAKPOINTS]

    out = df.copy()

    for pollutant in pollutant_cols:
        col = f"sub_index_{pollutant}"
        out[col] = df[pollutant].apply(lambda x, p=pollutant: sub_index(p, x) if pd.notna(x) else None)

    def _row_aqi(row: pd.Series) -> tuple[float | None, int]:
        sub_indices = []
        for p in pollutant_cols:
            val = row.get(f"sub_index_{p}")
            if val is not None and pd.notna(val):
                sub_indices.append(float(val))
        # Need at least 3 valid pollutants AND at least one of PM2.5 or PM10
        has_pm = any(
            row.get(f"sub_index_{p}") is not None and pd.notna(row.get(f"sub_index_{p}"))
            for p in ("pm25", "pm10")
        )
        if len(sub_indices) >= 3 and has_pm:
            return max(sub_indices), _bucket_of_aqi(max(sub_indices))
        return None, -1

    aqi_results = out.apply(_row_aqi, axis=1, result_type="expand")
    out["aqi_recomputed"] = aqi_results[0]
    out["aqi_bucket_recomputed"] = aqi_results[1]
    return out


def _bucket_of_aqi(aqi: float) -> int:
    """Return the 0-5 bucket index for a given AQI value."""
    for i, (lo, hi) in enumerate(INDEX_RANGES):
        if lo <= aqi <= hi:
            return i
    return -1


def validate_against_official(
    df: pd.DataFrame,
    col_recomputed: str = "aqi_recomputed",
    col_official: str = "aqi_official",
) -> dict:
    """Return validation metrics: MAE and R2 between recomputed and official AQI."""
    valid = df[df[col_official].notna() & df[col_recomputed].notna()]
    if len(valid) == 0:
        return {"n": 0, "mae": None, "r2": None}
    mae = float(np.abs(valid[col_recomputed] - valid[col_official]).mean())
    if valid[col_recomputed].std() > 0 and valid[col_official].std() > 0:
        r2 = float(valid[col_recomputed].corr(valid[col_official]) ** 2)
    else:
        r2 = None
    return {"n": len(valid), "mae": mae, "r2": r2}
