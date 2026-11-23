"""Read Kaggle air quality files (2015-2020) from a local path.

Used to reproduce earlier-paper numbers. The user downloads them manually.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from aqi.data.manifest import Manifest


def ingest_kaggle(
    local_path: str | Path,
    output_dir: str | Path = "data/raw",
) -> list[Path]:
    """Copy Kaggle air quality CSV(s) into raw/ and log in the manifest."""
    raw_dir = Path(output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(raw_dir)

    src = Path(local_path)
    if not src.exists():
        raise FileNotFoundError(
            f"Kaggle path not found at {local_path}. "
            "Please check the path."
        )

    files_to_copy: list[Path] = []
    if src.is_dir():
        files_to_copy = sorted(src.glob("*.csv"))
    elif src.is_file():
        files_to_copy = [src]

    copied_paths: list[Path] = []
    for f in files_to_copy:
        target = raw_dir / f.name
        shutil.copy2(f, target)
        # Count rows
        with target.open("r", encoding="utf-8", errors="ignore") as fh:
            rows = sum(1 for _ in fh) - 1

        manifest.add(
            source="kaggle",
            url="https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india",
            filename=f.name,
            licence="CC0",
            rows=rows,
        )
        copied_paths.append(target)

    return copied_paths
