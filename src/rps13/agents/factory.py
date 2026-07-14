"""Factory for building agents from configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rps13.agents.base import BaseAgent
from rps13.agents.hybrid_agent import HybridAdaptiveAgent
from rps13.agents.mixture_agent import MixtureAdaptiveAgent
from rps13.agents.neural_agent import ActorCriticAgent
from rps13.agents.random_agent import RandomAgent

AGENT_VERSION = "2.0.0"


def build_agent(config: dict[str, Any], project_root: Path | None = None) -> BaseAgent:
    """Instantiate an agent from app/eval configuration."""

    agent_type = str(config.get("agent_type", "hybrid")).lower()
    checkpoint = config.get("agent_checkpoint_path", "models/opponent_predictor.pt")
    if project_root is not None and checkpoint:
        checkpoint_path = project_root / checkpoint if not Path(checkpoint).is_absolute() else Path(checkpoint)
    else:
        checkpoint_path = Path(checkpoint) if checkpoint else None

    target_score = int(config.get("target_score", 13))
    seed = config.get("agent_seed")

    if agent_type == "mixture":
        return MixtureAdaptiveAgent(
            checkpoint_path=checkpoint_path,
            target_score=target_score,
            seed=seed,
            temperature=float(config.get("temperature", 0.5)),
            max_alpha=float(config.get("max_alpha", 0.85)),
            sticky_max_alpha=float(config.get("sticky_max_alpha", 0.95)),
            min_action_prob=float(config.get("min_action_prob", 0.03)),
            sticky_min_action_prob=float(config.get("sticky_min_action_prob", 0.01)),
        )
    if agent_type == "hybrid":
        return HybridAdaptiveAgent(
            checkpoint_path=checkpoint_path,
            target_score=target_score,
            seed=seed,
        )
    if agent_type == "random":
        return RandomAgent()
    if agent_type == "actor_critic":
        if not checkpoint_path:
            raise ValueError("actor_critic requires agent_checkpoint_path")
        return ActorCriticAgent(checkpoint_path)
    raise ValueError(f"Unknown agent_type: {agent_type}")
