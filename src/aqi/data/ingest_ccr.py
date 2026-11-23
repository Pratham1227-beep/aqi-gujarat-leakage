"""Pull station data from the CPCB CCR dashboard.

If the dashboard blocks automation, this module stops and the user downloads
files by hand into data/raw/ccr/.
"""

from __future__ import annotations


def ingest_ccr(output_dir: str = "data/raw/ccr") -> list[str]:
    """Attempt automated CCR ingestion; returns list of file paths.

    The CPCB CCR dashboard is often hard to scrape. If an automated fetch is
    not possible in this environment, this raises a friendly error so the user
    can fall back to manual downloads.
    """
    raise RuntimeError(
        "Automated CPCB CCR ingestion is not supported yet. "
        "Please download CCR station files by hand into data/raw/ccr/."
    )
