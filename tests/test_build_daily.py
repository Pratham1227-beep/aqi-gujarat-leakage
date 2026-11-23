"""Unit tests for built daily datasets."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROCESSED = Path("data/processed")
CITIES = ["ahmedabad", "chennai", "gandhinagar", "ankleshwar", "vapi", "surat"]


@pytest.mark.parametrize("city", CITIES)
def test_processed_parquet_exists_and_schema(city: str) -> None:
    path = PROCESSED / f"{city}_daily.parquet"
    assert path.exists(), f"Missing processed parquet: {path}"
    df = pd.read_parquet(path)
    assert len(df) > 0
    # Core columns must exist
    for col in ["date", "city", "aqi_official", "aqi_missing"]:
        assert col in df.columns, f"Missing column {col} in {city}"
    # Date must be monotonic increasing
    df["date"] = pd.to_datetime(df["date"])
    assert df["date"].is_monotonic_increasing
    # Dates must have complete daily frequency
    diffs = df["date"].diff().dropna()
    assert (diffs == pd.Timedelta(days=1)).all(), f"Gaps detected in date range for {city}"


@pytest.mark.parametrize("city", CITIES)
def test_missingness_csv_exists(city: str) -> None:
    path = PROCESSED / f"{city}_missingness.csv"
    assert path.exists(), f"Missing missingness CSV: {path}"
    df = pd.read_csv(path)
    assert "pollutant" in df.columns
    assert "share_missing" in df.columns
    assert "longest_gap_days" in df.columns


def test_ahmedabad_pollutant_validity() -> None:
    path = PROCESSED / "ahmedabad_daily.parquet"
    df = pd.read_parquet(path)
    # CO must not exceed 100 mg/m3
    assert (df["co"].dropna() <= 100.0).all()
    # AQI official must not exceed 500
    assert (df["aqi_official"].dropna() <= 500.0).all()
    assert (df["aqi_official"].dropna() >= 0.0).all()
