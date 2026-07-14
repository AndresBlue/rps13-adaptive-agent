"""Neural actor-critic inference wrapper."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from rps13.agents.base import BaseAgent
from rps13.constants import NUM_ACTIONS
from rps13.data.feature_engineering import build_sequence_arrays
from rps13.game.state import GameObservation
from rps13.models.actor_critic import ActorCriticGRU


class ActorCriticAgent(BaseAgent):
    """Inference wrapper for ``ActorCriticGRU`` checkpoints."""

    def __init__(self, checkpoint_path: str | Path, sequence_length: int = 20) -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("model_config", {})
        self.model = ActorCriticGRU(**config)
        self.model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        self.model.eval()
        self.sequence_length = sequence_length

    @torch.no_grad()
    def policy(self, observation: GameObservation) -> np.ndarray:
        arrays = build_sequence_arrays(observation.history, self.sequence_length, observation.target_score)
        logits, _value, _hidden = self.model(
            torch.as_tensor(arrays["human_moves"]).long().unsqueeze(0),
            torch.as_tensor(arrays["ai_moves"]).long().unsqueeze(0),
            torch.as_tensor(arrays["results"]).long().unsqueeze(0),
            torch.as_tensor(arrays["numeric"]).float().unsqueeze(0),
            torch.as_tensor([max(int(arrays["length"]), 1)]).long(),
        )
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        probs = np.maximum(probs, 1e-4)
        return (probs / probs.sum()).astype(np.float32)


def uniform_policy() -> np.ndarray:
    return np.ones(NUM_ACTIONS, dtype=np.float32) / NUM_ACTIONS
