"""GRU opponent move predictor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from rps13.constants import NUM_ACTIONS, PAD_ACTION, PAD_RESULT_INDEX
from rps13.data.feature_engineering import NUMERIC_FEATURE_KEYS, build_sequence_arrays
from rps13.game.state import RoundRecord


class OpponentPredictorGRU(nn.Module):
    """Predicts the next human move from previous rounds."""

    def __init__(
        self,
        numeric_dim: int = len(NUMERIC_FEATURE_KEYS),
        action_embedding_dim: int = 8,
        result_embedding_dim: int = 4,
        hidden_dim: int = 96,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.numeric_dim = numeric_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.action_embedding = nn.Embedding(NUM_ACTIONS + 1, action_embedding_dim, padding_idx=PAD_ACTION)
        self.ai_action_embedding = nn.Embedding(NUM_ACTIONS + 1, action_embedding_dim, padding_idx=PAD_ACTION)
        self.result_embedding = nn.Embedding(4, result_embedding_dim, padding_idx=PAD_RESULT_INDEX)
        input_dim = action_embedding_dim * 2 + result_embedding_dim + numeric_dim
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, NUM_ACTIONS),
        )

    def forward(
        self,
        human_moves: torch.Tensor,
        ai_moves: torch.Tensor,
        results: torch.Tensor,
        numeric_features: torch.Tensor,
        lengths: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits ``[batch, 3]`` and final hidden state."""

        human_emb = self.action_embedding(human_moves.clamp(min=0, max=PAD_ACTION))
        ai_emb = self.ai_action_embedding(ai_moves.clamp(min=0, max=PAD_ACTION))
        result_emb = self.result_embedding(results.clamp(min=0, max=PAD_RESULT_INDEX))
        x = torch.cat([human_emb, ai_emb, result_emb, numeric_features.float()], dim=-1)
        out, hidden_out = self.gru(x, hidden)
        if lengths is None:
            last = out[:, -1, :]
        else:
            idx = torch.clamp(lengths.long(), min=1, max=out.shape[1]) - 1
            last = out[torch.arange(out.shape[0], device=out.device), idx]
        logits = self.head(last)
        return logits, hidden_out

    @torch.no_grad()
    def predict_proba(
        self,
        history: list[RoundRecord] | list[dict[str, Any]],
        sequence_length: int = 20,
        target_score: int = 13,
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        """Predict next human move probabilities for a single history."""

        device = device or next(self.parameters()).device
        arrays = build_sequence_arrays(history, sequence_length=sequence_length, target_score=target_score)
        tensors = {
            "human_moves": torch.as_tensor(arrays["human_moves"], dtype=torch.long, device=device).unsqueeze(0),
            "ai_moves": torch.as_tensor(arrays["ai_moves"], dtype=torch.long, device=device).unsqueeze(0),
            "results": torch.as_tensor(arrays["results"], dtype=torch.long, device=device).unsqueeze(0),
            "numeric_features": torch.as_tensor(arrays["numeric"], dtype=torch.float32, device=device).unsqueeze(0),
            "lengths": torch.as_tensor([max(int(arrays["length"]), 1)], dtype=torch.long, device=device),
        }
        was_training = self.training
        self.eval()
        logits, _ = self(**tensors)
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        if was_training:
            self.train()
        return probs

    def model_config(self) -> dict[str, Any]:
        """Return config needed to reconstruct this model."""

        return {
            "numeric_dim": self.numeric_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
        }


def load_predictor_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> OpponentPredictorGRU:
    """Load an ``OpponentPredictorGRU`` checkpoint."""

    checkpoint = torch.load(path, map_location=map_location)
    config = checkpoint.get("model_config", {})
    model = OpponentPredictorGRU(**config)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model
