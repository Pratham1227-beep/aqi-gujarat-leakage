"""Download CPCB daily city AQI bulletins from Zenodo.

Fetches India_AQI_Bulletins_Master.csv (CPCB daily city AQI, 2015-2023).
"""

from __future__ import annotations

from pathlib import Path

import requests

from aqi.data.manifest import Manifest

ZENODO_API = "https://zenodo.org/api/records"
ZENODO_RECORD = "4264575"  # placeholder record ID


def ingest_zenodo(
    output_dir: str | Path = "data/raw",
    record_id: str = ZENODO_RECORD,
    force: bool = False,
) -> Path | None:
    """Download the Zenodo AQI bulletin dataset.

    Returns the path to the downloaded file, or None if it already exists
    and *force* is False.
    """
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(raw_dir)

    url = f"{ZENODO_API}/{record_id}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    record = response.json()

    for entry in record.get("files", []):
        file_url = entry.get("links", {}).get("self", "")
        filename = str(entry.get("key", ""))
        if "AQI" not in filename.upper() and "bulletin" not in filename.lower():
            continue

        target: Path = raw_dir / filename
        if target.exists() and not force:
            continue

        r = requests.get(file_url, stream=True, timeout=300)
        r.raise_for_status()
        with target.open("wb") as fh:
            for chunk in r.iter_content(chunk_size=65536):
                fh.write(chunk)

        manifest.add(source="zenodo", url=file_url, filename=filename, licence="CC-BY", rows=None)
        return target

    return None
