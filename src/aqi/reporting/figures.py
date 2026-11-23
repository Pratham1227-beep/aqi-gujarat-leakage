"""Generate analysis figures from results/runs.csv."""

from __future__ import annotations

from pathlib import Path


def generate_figures(
    runs_csv: str | Path = "results/runs.csv",
    output_dir: str | Path = "results/figures",
) -> None:
    """Generate publication figures from experimental results."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
