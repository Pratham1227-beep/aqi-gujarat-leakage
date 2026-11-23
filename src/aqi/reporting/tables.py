"""Generate LaTeX and CSV tables from results/runs.csv."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_tables(
    runs_csv: str | Path = "results/runs.csv",
    output_dir: str | Path = "results/tables",
) -> None:
    """Generate summary tables from experimental runs."""
    path = Path(runs_csv)
    if not path.exists():
        return
    df = pd.read_csv(path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = df.groupby(["city", "setup", "model"]).agg({
        "mae": "mean",
        "rmse": "mean",
        "r2": "mean",
    })
    summary.to_csv(out_dir / "summary.csv")
