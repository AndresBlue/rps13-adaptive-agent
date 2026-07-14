"""Recurrent actor-critic model for RL experiments."""

from __future__ import annotations

import torch
from torch import nn

from rps13.constants import NUM_ACTIONS, PAD_ACTION, PAD_RESULT_INDEX
from rps13.data.feature_engineering import NUMERIC_FEATURE_KEYS


class ActorCriticGRU(nn.Module):
    """Small recurrent actor-critic prepared for PPO-style training."""

    def __init__(
        self,
        numeric_dim: int = len(NUMERIC_FEATURE_KEYS),
        action_embedding_dim: int = 8,
        result_embedding_dim: int = 4,
        hidden_dim: int = 96,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(NUM_ACTIONS + 1, action_embedding_dim, padding_idx=PAD_ACTION)
        self.ai_action_embedding = nn.Embedding(NUM_ACTIONS + 1, action_embedding_dim, padding_idx=PAD_ACTION)
        self.result_embedding = nn.Embedding(4, result_embedding_dim, padding_idx=PAD_RESULT_INDEX)
        input_dim = action_embedding_dim * 2 + result_embedding_dim + numeric_dim
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.policy_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        human_moves: torch.Tensor,
        ai_moves: torch.Tensor,
        results: torch.Tensor,
        numeric_features: torch.Tensor,
        lengths: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return self.policy_head(last), self.value_head(last).squeeze(-1), hidden_out
