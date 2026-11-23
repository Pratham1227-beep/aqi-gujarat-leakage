"""Build interim daily parquet per city.

Merge strategy
--------------
- aqi_official  ← India_AQI_Bulletins_Master.csv  (Index Value, integer 0-500)
- pm25..nh3     ← city_day.csv  (Kaggle, available for Ahmedabad + Chennai only)
- prominent_pollutant ← bulletin (categorical)

For bulletin-only cities (Gandhinagar, Ankleshwar, Vapi, Surat) the pollutant
columns are all NaN; Setup C (next-day AQI forecast from lag-only features) is
still valid.

Output columns
--------------
date, city, aqi_official, air_quality_category, prominent_pollutant,
pm25, pm10, no2, so2, co, o3, nh3
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

RAW = Path("data/raw")
INTERIM = Path("data/interim")

# Kaggle column → internal name
KAGGLE_COL_MAP: dict[str, str] = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
    "O3": "o3",
    "NH3": "nh3",
}

# Cities present in Kaggle city_day.csv (with pollutant concentrations)
KAGGLE_CITIES: set[str] = {"Ahmedabad", "Chennai"}

# All cities to build parquets for (bulletin key → parquet stem)
TARGET_CITIES: dict[str, str] = {
    "Ahmedabad": "ahmedabad",
    "Chennai": "chennai",
    "Surat": "surat",
    "GandhiNagar": "gandhinagar",
    "Ankleshwar": "ankleshwar",
    "Vapi": "vapi",
}

AQI_CATEGORY_MAP: dict[str, str] = {
    "Good": "good",
    "Satisfactory": "satisfactory",
    "Moderate": "moderate",
    "Poor": "poor",
    "Very poor": "very_poor",
    "Severe": "severe",
}


def _load_bulletin() -> pd.DataFrame:
    """Load the CPCB bulletin CSV and return cleaned DataFrame."""
    path = RAW / "India_AQI_Bulletins_Master.csv"
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    df["aqi_official"] = pd.to_numeric(df["Index Value"], errors="coerce").clip(0, 500)
    df["air_quality_category"] = df["Air Quality"].map(AQI_CATEGORY_MAP)
    df["prominent_pollutant"] = df["Prominent Pollutant"].str.strip().str.lower()
    df = df.rename(columns={"City": "city"})
    return df[["date", "city", "aqi_official", "air_quality_category", "prominent_pollutant"]]


def _load_kaggle_pollutants(city: str) -> pd.DataFrame:
    """Load Kaggle city_day.csv pollutant columns for a single city."""
    path = RAW / "city_day.csv"
    df = pd.read_csv(path, low_memory=False)
    sub = df[df["City"] == city].copy()
    sub["date"] = pd.to_datetime(sub["Date"])
    sub = sub.rename(columns=KAGGLE_COL_MAP)
    keep = ["date"] + [v for v in KAGGLE_COL_MAP.values() if v in sub.columns]
    return sub[keep].drop_duplicates(subset="date")


def build_city(city_bulletin: str, city_stem: str) -> pd.DataFrame:
    """Build the interim daily DataFrame for one city.

    Parameters
    ----------
    city_bulletin : str
        City name as it appears in the bulletin CSV.
    city_stem : str
        Output file stem (lowercase, no spaces).
    """
    bulletin = _load_bulletin()
    city_df = bulletin[bulletin["city"] == city_bulletin].copy()
    city_df = city_df.drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)

    if city_bulletin in KAGGLE_CITIES:
        kaggle = _load_kaggle_pollutants(city_bulletin)
        city_df = city_df.merge(kaggle, on="date", how="left")
        log.info("%s: merged %d bulletin rows with Kaggle pollutants", city_bulletin, len(city_df))
    else:
        for col in KAGGLE_COL_MAP.values():
            city_df[col] = float("nan")
        log.info("%s: bulletin-only city (%d rows)", city_bulletin, len(city_df))

    city_df["city"] = city_stem  # normalise to lowercase stem
    return city_df


def build_all(output_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Build interim parquets for all target cities.

    Returns a dict mapping city stem → DataFrame.
    """
    out_dir = output_dir or INTERIM
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}
    for bulletin_name, stem in TARGET_CITIES.items():
        df = build_city(bulletin_name, stem)
        path = out_dir / f"{stem}_interim.parquet"
        df.to_parquet(path, index=False)
        results[stem] = df
        log.info("Wrote %s (%d rows, %d cols)", path, len(df), df.shape[1])
        # Quick summary
        n_aqi = df["aqi_official"].notna().sum()
        n_pm25 = df["pm25"].notna().sum() if "pm25" in df.columns else 0
        log.info(
            "  aqi_official valid=%d / %d  |  pm25 valid=%d",
            n_aqi,
            len(df),
            n_pm25,
        )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_all()
