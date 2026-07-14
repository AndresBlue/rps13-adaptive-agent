"""Uniform random AI agent."""

from __future__ import annotations

import numpy as np

from rps13.agents.base import BaseAgent
from rps13.constants import NUM_ACTIONS
from rps13.game.state import GameObservation


class RandomAgent(BaseAgent):
    """AI agent that plays the Nash-equilibrium uniform policy."""

    def policy(self, observation: GameObservation) -> np.ndarray:
        return np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
