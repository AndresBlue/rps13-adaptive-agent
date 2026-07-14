"""Base interfaces for opponent-move prediction experts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from rps13.constants import NUM_ACTIONS
from rps13.game.state import GameObservation


@dataclass(slots=True)
class ExpertPrediction:
    """Single expert output for the next human move."""

    name: str
    probs: np.ndarray
    pattern_stability: float = 0.0
    meta_level: str = "P.0"
    debug: dict = field(default_factory=dict)


class BaseExpert(ABC):
    """Predicts P(human next move | history)."""

    name: str = "base"

    def reset(self) -> None:
        """Reset per-match state."""

    @abstractmethod
    def predict(self, observation: GameObservation) -> ExpertPrediction:
        """Return a probability vector over human moves."""

    @staticmethod
    def normalize(probs: np.ndarray) -> np.ndarray:
        arr = np.asarray(probs, dtype=np.float64)
        arr = np.where(np.isfinite(arr), arr, 0.0)
        arr = np.maximum(arr, 0.0)
        total = arr.sum()
        if total <= 0:
            return np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
        return arr / total
