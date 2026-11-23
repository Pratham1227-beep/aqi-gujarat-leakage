"""Command-line interface for the AQI study."""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for aqi commands."""
    parser = argparse.ArgumentParser(description="AQI Gujarat Leakage Study CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Ingest
    subparsers.add_parser("ingest", help="Ingest raw data sources")

    # Build daily
    subparsers.add_parser("build-daily", help="Build processed daily parquets")

    # Tune
    subparsers.add_parser("tune", help="Tune model hyperparameters")

    # Run
    run_parser = subparsers.add_parser("run", help="Run model experiment")
    run_parser.add_argument("--city", default="Ahmedabad")
    run_parser.add_argument("--setup", default="A")
    run_parser.add_argument("--model", default="all")
    run_parser.add_argument("--horizon", type=int, default=1)

    # Run all
    subparsers.add_parser("run-all", help="Run full experiment grid")

    # Results & figures
    subparsers.add_parser("tables", help="Generate tables from runs.csv")
    subparsers.add_parser("figures", help="Generate figures")

    args = parser.parse_args(argv or sys.argv[1:])
    if not args.command:
        parser.print_help()
        return 0

    print(f"Executing command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
