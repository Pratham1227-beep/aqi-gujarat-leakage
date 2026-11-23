"""Read the Kaggle city_day file (2015-2020) from a local path.

Used only to reproduce earlier-paper numbers. The user downloads it manually;
we do not scrape Kaggle.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from aqi.data.manifest import Manifest


def ingest_kaggle(
    local_path: str | Path,
    output_dir: str | Path = "data/raw",
) -> Path | None:
    """Copy the Kaggle city_day CSV into raw/ and log it in the manifest.

    The file is expected to have columns:
    Date, City, PM2.5, PM10, NO2, SO2, CO, O3, NH3, AQI, AQI_Bucket
    """
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(raw_dir)

    src = Path(local_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Kaggle file not found at {local_path}. "
            "Download it manually from Kaggle (city_day.csv) and place it there."
        )

    target = raw_dir / "city_day.csv"
    df = pd.read_csv(src)
    df.to_csv(target, index=False)

    manifest.add(
        source="kaggle",
        url="https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india",
        filename="city_day.csv",
        licence="CC0",
        rows=len(df),
    )
    return target