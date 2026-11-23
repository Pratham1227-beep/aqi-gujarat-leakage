"""Hyperparameter tuning on earliest rolling window with time-ordered inner folds."""

from __future__ import annotations

from typing import Any


def tune_model(
    model_name: str,
    param_grid: list[dict[str, Any]],
) -> dict[str, Any]:
    """Stub for one-time tuning on earliest window using time-ordered inner folds."""
    if not param_grid:
        return {}
    return param_grid[0]
