"""Centralised, deterministic seed management."""

from __future__ import annotations

import random

import numpy as np
import torch

_numpy_seed: int = 0
_python_seed: int = 0


def _reset_numpy() -> None:
    np.random.seed(_numpy_seed)


def _reset_random() -> None:
    random.seed(_python_seed)


def _reset_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_all_seeds(numpy_seed: int, python_seed: int, torch_seed: int) -> None:
    """Set deterministic seeds for numpy, python random and torch."""
    global _numpy_seed, _python_seed
    _numpy_seed = numpy_seed
    _python_seed = python_seed
    _reset_numpy()
    _reset_random()
    _reset_torch(torch_seed)
