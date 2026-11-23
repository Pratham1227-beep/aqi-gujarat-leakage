"""Command-line interface for the AQI study."""

from __future__ import annotations

import argparse
import logging
import sys

from aqi.harness import (
    run_all_experiments,
    run_setup_a,
    run_setup_b,
    run_setup_c,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


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
    run_parser.add_argument("--city", default="ahmedabad", help="City name (e.g. ahmedabad, chennai, gandhinagar)")
    run_parser.add_argument("--setup", default="C", choices=["A", "B", "C"], help="Experiment setup (A, B, or C)")
    run_parser.add_argument("--horizon", type=int, default=1, choices=[1, 3], help="Forecast horizon (1 or 3, Setup C only)")
    run_parser.add_argument("--model", default=None, help="Specific model to run (default: all benchmark models)")

    # Run all
    run_all_parser = subparsers.add_parser("run-all", help="Run full experiment grid")
    run_all_parser.add_argument("--city", default=None, help="Optional single city filter")
    run_all_parser.add_argument("--setup", default=None, help="Optional single setup filter")

    # Results & figures
    subparsers.add_parser("tables", help="Generate tables from runs.csv")
    subparsers.add_parser("figures", help="Generate figures")

    args = parser.parse_args(argv or sys.argv[1:])
    if not args.command:
        parser.print_help()
        return 0

    if args.command == "run":
        models = [args.model] if args.model else None
        city = args.city.lower()
        if args.setup == "A":
            run_setup_a(city, models=models)
        elif args.setup == "B":
            run_setup_b(city, models=models)
        elif args.setup == "C":
            run_setup_c(city, models=models, horizons=[args.horizon])
        return 0

    if args.command == "run-all":
        cities = [args.city.lower()] if args.city else None
        setups = [args.setup.upper()] if args.setup else None
        run_all_experiments(cities=cities, setups=setups)
        return 0

    print(f"Executing command: {args.command}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
