"""Manifest tracking for raw data sources.

Each source that enters ``data/raw/`` gets one row in ``MANIFEST.csv``:
source, URL, retrieved date, SHA256, licence, rows.
"""

from __future__ import annotations

import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path


def sha256_of_file(path: Path) -> str:
    """Return the SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    """Append-only ledger of downloaded raw files."""

    def __init__(self, raw_dir: str | Path):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.raw_dir / "MANIFEST.csv"
        self._fields = [
            "source",
            "url",
            "retrieved_at",
            "filename",
            "sha256",
            "licence",
            "rows",
        ]
        if not self.path.exists():
            self._write_header()

    def _write_header(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fields)
            writer.writeheader()

    def add(
        self,
        source: str,
        url: str,
        filename: str,
        licence: str,
        rows: int | None = None,
    ) -> None:
        """Append a row for *filename* located under ``raw_dir``."""
        path = self.raw_dir / filename
        entry = {
            "source": source,
            "url": url,
            "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "filename": filename,
            "sha256": sha256_of_file(path) if path.exists() else "",
            "licence": licence,
            "rows": rows if rows is not None else "",
        }
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self._fields)
            writer.writerow(entry)

    def entries(self) -> list[dict[str, str]]:
        """Return all manifest rows as a list of dicts."""
        if not self.path.exists():
            return []
        with self.path.open("r", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))


def load_manifest(raw_dir: str | Path) -> Manifest:
    return Manifest(raw_dir)
