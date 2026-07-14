"""Evaluation metric helpers."""

from __future__ import annotations

import numpy as np


def policy_uniform_deviation(policy: list[float] | np.ndarray) -> float:
    """L1 deviation from the uniform Nash policy."""

    probs = np.asarray(policy, dtype=np.float64)
    return float(np.abs(probs - (1.0 / 3.0)).sum())
