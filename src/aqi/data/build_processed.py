"""Build processed daily parquets from interim files.

Steps per city
--------------
1. Load interim parquet (aqi_official from bulletin + pollutants from Kaggle)
2. Reindex to a complete daily date range (no gaps)
3. Flag invalid pollutant values per CPCB validity table
4. Write processed parquet to data/processed/<city>_daily.parquet

The processed parquet is what all feature-engineering and modelling code reads.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from aqi.data.missingness import report_missingness
from aqi.data.reconstruct import compute_aqi, validate_against_official
from aqi.data.validity import flag_invalid_values

log = logging.getLogger(__name__)

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")

# Validity ranges for pollutant columns (CPCB guidelines)
# Values outside these are set to NaN during flagging.
POLLUTANT_COLS = ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]


def reindex_complete(df: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """Reindex to a complete date range, inserting NaN rows for missing dates."""
    df = df.set_index("date").sort_index()
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq=freq, name="date")
    df = df.reindex(full_idx)
    # Forward-fill city name for the inserted rows
    df["city"] = df["city"].ffill().bfill()
    return df.reset_index()


def build_processed(city_stem: str, interim_dir: Path | None = None, out_dir: Path | None = None) -> pd.DataFrame:
    """Build the processed daily parquet for one city."""
    interim_dir = interim_dir or INTERIM
    out_dir = out_dir or PROCESSED
    out_dir.mkdir(parents=True, exist_ok=True)

    path = interim_dir / f"{city_stem}_interim.parquet"
    df = pd.read_parquet(path)

    # 1. Complete daily reindex
    df = reindex_complete(df)

    # 2. Flag invalid pollutant values → set to NaN
    df, removed = flag_invalid_values(df)
    for pollutant, n in removed.items():
        if n:
            log.info("  %s: removed %d invalid %s values", city_stem, n, pollutant)

    # 3. Clip aqi_official to [0, 500] (should already be but be safe)
    if "aqi_official" in df.columns:
        df["aqi_official"] = pd.to_numeric(df["aqi_official"], errors="coerce").clip(0, 500)

    # 4. Compute sub-indices and recomputed AQI where pollutants exist
    df = compute_aqi(df)
    if "aqi_recomputed" in df.columns and "aqi_official" in df.columns:
        val = validate_against_official(df)
        if val["n"] > 0:
            log.info("  %s: recomputed AQI validation against official: n=%d, MAE=%.2f, R2=%s",
                     city_stem, val["n"], val["mae"], f"{val['r2']:.4f}" if val['r2'] else "None")

    # 5. Add a boolean missing-aqi column (useful for missingness analysis)
    df["aqi_missing"] = df["aqi_official"].isna()

    # 6. Generate and save missingness report
    pollutant_cols = [c for c in df.columns if c in POLLUTANT_COLS]
    # Also include aqi_official in missingness assessment
    cols_to_report = ["aqi_official"] + pollutant_cols
    missingness_df = report_missingness(df, city_stem, cols_to_report)
    missingness_path = out_dir / f"{city_stem}_missingness.csv"
    missingness_df.to_csv(missingness_path, index=False)

    out_path = out_dir / f"{city_stem}_daily.parquet"
    df.to_parquet(out_path, index=False)
    log.info("Wrote %s — %d rows, aqi_missing=%.1f%%",
             out_path, len(df), 100 * df["aqi_missing"].mean())
    return df


def build_all_processed(interim_dir: Path | None = None, out_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Build processed parquets for all cities with interim files."""
    interim_dir = interim_dir or INTERIM
    out_dir = out_dir or PROCESSED
    stems = [p.stem.replace("_interim", "") for p in sorted(interim_dir.glob("*_interim.parquet"))]
    results = {}
    for stem in stems:
        results[stem] = build_processed(stem, interim_dir, out_dir)
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_all_processed()
