"""Dataclasses representing game state and round history."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class RoundRecord:
    """A single completed round."""

    round: int
    human_move: int
    ai_move: int
    result_for_ai: int
    human_score: int
    ai_score: int
    round_winner: str
    is_terminal: bool
    timestamp_or_step: int
    points_awarded: int = 1
    human_bet: bool = False
    ai_bet: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation."""

        data = asdict(self)
        data.update(data.pop("metadata", {}))
        return data


@dataclass(slots=True)
class GameObservation:
    """Observable state returned by the environment."""

    target_score: int
    round_number: int
    human_score: int
    ai_score: int
    done: bool
    history: list[RoundRecord]
    variant: str = "base"
    human_bet_tokens: int = 0
    ai_bet_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable observation."""

        return {
            "target_score": self.target_score,
            "round_number": self.round_number,
            "human_score": self.human_score,
            "ai_score": self.ai_score,
            "done": self.done,
            "variant": self.variant,
            "human_bet_tokens": self.human_bet_tokens,
            "ai_bet_tokens": self.ai_bet_tokens,
            "history": [record.to_dict() for record in self.history],
        }
