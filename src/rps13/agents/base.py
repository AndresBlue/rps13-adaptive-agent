"""Base interfaces for agents and scripted opponents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from rps13.constants import NUM_ACTIONS
from rps13.game.state import GameObservation, RoundRecord


@dataclass(slots=True)
class AgentDecision:
    """Action plus optional policy/debug information."""

    action: int
    policy: list[float]
    debug: dict[str, Any]


class BaseAgent(ABC):
    """Base class for AI-side agents."""

    def reset(self) -> None:
        """Reset any per-match state."""

    @abstractmethod
    def policy(self, observation: GameObservation) -> np.ndarray:
        """Return a probability distribution over ROCK, PAPER, SCISSORS."""

    def select_action(self, observation: GameObservation) -> AgentDecision:
        """Sample an action from the current policy."""

        probs = self.policy(observation)
        action = int(np.random.choice(NUM_ACTIONS, p=probs))
        return AgentDecision(action=action, policy=probs.tolist(), debug={})


class BaseOpponent(ABC):
    """Base class for human-side scripted opponents used in training/eval."""

    name = "base"

    def __init__(self, seed: int | None = None, noise: float = 0.0) -> None:
        self.seed = seed
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def reset(self) -> None:
        """Reset per-match state."""

    @abstractmethod
    def choose_action(self, observation: GameObservation) -> int:
        """Choose the next human-side action."""

    def observe_round(self, record: RoundRecord) -> None:
        """Receive a completed round."""

    def maybe_noise(self, action: int) -> int:
        """With probability ``noise``, replace action by a random action."""

        if self.noise > 0 and self.rng.random() < self.noise:
            return int(self.rng.integers(0, NUM_ACTIONS))
        return int(action)

    def random_action(self) -> int:
        """Sample a uniform valid action."""

        return int(self.rng.integers(0, NUM_ACTIONS))
