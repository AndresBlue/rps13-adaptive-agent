"""Loss helpers for simple actor-critic training."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def actor_critic_loss(
    log_probs: torch.Tensor,
    values: torch.Tensor,
    entropies: torch.Tensor,
    returns: torch.Tensor,
    value_coefficient: float,
    entropy_coefficient: float,
) -> torch.Tensor:
    """Compute a compact policy-gradient actor-critic loss."""

    advantages = returns - values.detach()
    policy_loss = -(log_probs * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    entropy_bonus = entropies.mean()
    return policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_bonus
