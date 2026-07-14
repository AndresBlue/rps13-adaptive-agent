"""Rollout containers for recurrent RL training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutStep:
    """One policy interaction step."""

    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    reward: float


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """Compute discounted returns for a single episode."""

    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.as_tensor(returns, dtype=torch.float32)
