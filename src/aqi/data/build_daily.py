"""Build the final ``<city>_daily.parquet`` from interim data.

Steps:
1. Calendar (complete daily index)
2. Validity (physical ranges, flags)
3. Aggregation (hourly -> daily, completeness rules)
4. Reconstruction (CPCB sub-indices, AQI recompute, validate against official)
5. Missingness report
6. Write to data/processed/<city>_daily.parquet
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aqi.data.calendar import reindex_to_complete
from aqi.data.missingness import report_missingness
from aqi.data.reconstruct import compute_aqi, validate_against_official
from aqi.data.validity import flag_invalid_values


def build_daily(
    interim_path: str | Path,
    output_path: str | Path,
    city: str = "",
    source: str = "",
    min_dates: tuple[str | None, str | None] = (None, None),
) -> tuple[pd.DataFrame, dict]:
    """Build a daily parquet from an interim hourly/individual-pollutant DataFrame.

    Parameters
    ----------
    interim_path:
        Path to interim CSV or parquet with hourly or daily data.
    output_path:
        Path to write the daily parquet.
    city:
        City name for the output.
    source:
        Source name for the output.
    min_dates:
        Optional (start, end) to restrict the date range.

    Returns
    -------
    (final_df, stats_dict)
    """
    raw = _load_interim(interim_path)
    if min_dates[0] is not None and "date" in raw.columns:
        raw = raw[raw["date"] >= min_dates[0]]
    if min_dates[1] is not None and "date" in raw.columns:
        raw = raw[raw["date"] <= min_dates[1]]

    # Reindex to complete daily calendar
    if "date" not in raw.columns:
        raw["date"] = pd.to_datetime(raw.index)
    daily = reindex_to_complete(raw)

    # Validity check
    daily, removed_counts = flag_invalid_values(daily)

    # Compute sub-indices and AQI
    daily = compute_aqi(daily)

    # Stats
    stats = {
        "city": city,
        "source": source,
        "n_days": len(daily),
        "removed_counts": removed_counts,
        "aqi_stats": {},
    }
    if "aqi_official" in daily.columns and "aqi_recomputed" in daily.columns:
        stats["aqi_stats"] = validate_against_official(daily)

    # Missingness report
    pollutant_cols = [
        c for c in daily.columns if c in ["pm25", "pm10", "no2", "so2", "co", "o3", "nh3"]
    ]
    missingness_df = report_missingness(daily, city, pollutant_cols)
    output_p = Path(output_path)
    missingness_path = output_p.parent / f"{city}_missingness.csv"
    missingness_df.to_csv(missingness_path, index=False)

    # Write final
    output_p.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_p, index=False)

    return daily, stats


def _load_interim(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported interim file format: {path}")
