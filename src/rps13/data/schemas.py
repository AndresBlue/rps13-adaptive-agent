"""Pydantic schemas for internal datasets."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MatchRoundRow(BaseModel):
    """Internal tabular row schema for one completed round."""

    match_id: str
    player_id: str = "synthetic_human"
    opponent_type: str
    round: int = Field(ge=1)
    human_move: int = Field(ge=0, le=2)
    ai_move: int = Field(ge=0, le=2)
    result_for_ai: int = Field(ge=-1, le=1)
    human_score: int = Field(ge=0)
    ai_score: int = Field(ge=0)
    round_winner: str
    is_terminal: bool
    timestamp_or_step: int
    human_decision_time_ms: int | None = None
    ai_policy_rock: float | None = None
    ai_policy_paper: float | None = None
    ai_policy_scissors: float | None = None
    predictor_confidence: float | None = None
