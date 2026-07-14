"""Deterministic RPS-13 environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rps13.constants import DEFAULT_TARGET_SCORE, Result
from rps13.game.rules import action_name, normalize_action, outcome_for_ai, round_winner
from rps13.game.state import GameObservation, RoundRecord


@dataclass(slots=True)
class StepResult:
    """Return object for environment steps."""

    observation: GameObservation
    reward: float
    done: bool
    info: dict[str, Any]


class RPS13Env:
    """First-to-13 deterministic rock-paper-scissors environment."""

    variant = "base"

    def __init__(self, target_score: int = DEFAULT_TARGET_SCORE) -> None:
        self.target_score = target_score
        self.reset()

    def reset(self) -> GameObservation:
        """Reset the environment and return the initial observation."""

        self.human_score = 0
        self.ai_score = 0
        self.history: list[RoundRecord] = []
        return self.get_observation()

    def step(self, ai_action: int, human_action: int) -> tuple[GameObservation, float, bool, dict[str, Any]]:
        """Play one simultaneous round."""

        if self.is_done():
            raise RuntimeError("Cannot step: match is already done.")
        ai = normalize_action(ai_action)
        human = normalize_action(human_action)
        result = outcome_for_ai(ai, human)
        points = 1
        if result == Result.WIN:
            self.ai_score += points
            reward = 1.0
        elif result == Result.LOSS:
            self.human_score += points
            reward = -1.0
        else:
            reward = 0.0
        done = self.is_done()
        if done:
            reward += 5.0 if self.ai_score >= self.target_score else -5.0
        record = RoundRecord(
            round=len(self.history) + 1,
            human_move=int(human),
            ai_move=int(ai),
            result_for_ai=int(result),
            human_score=self.human_score,
            ai_score=self.ai_score,
            round_winner=round_winner(ai, human),
            is_terminal=done,
            timestamp_or_step=len(self.history) + 1,
            points_awarded=points,
        )
        self.history.append(record)
        obs = self.get_observation()
        info = record.to_dict()
        return obs, reward, done, info

    def get_observation(self) -> GameObservation:
        """Return the current observable state."""

        return GameObservation(
            target_score=self.target_score,
            round_number=len(self.history) + 1,
            human_score=self.human_score,
            ai_score=self.ai_score,
            done=self.is_done(),
            history=list(self.history),
            variant=self.variant,
        )

    def is_done(self) -> bool:
        """Whether either player reached the target score."""

        return self.human_score >= self.target_score or self.ai_score >= self.target_score

    def winner(self) -> str | None:
        """Return ``ai``, ``human`` or ``None``."""

        if self.ai_score >= self.target_score:
            return "ai"
        if self.human_score >= self.target_score:
            return "human"
        return None

    def render_text(self) -> str:
        """Render a compact text summary of the match."""

        lines = [f"RPS-13 score: human {self.human_score} - ai {self.ai_score}"]
        for record in self.history:
            lines.append(
                f"R{record.round}: human={action_name(record.human_move)} "
                f"ai={action_name(record.ai_move)} winner={record.round_winner}"
            )
        if self.is_done():
            lines.append(f"Winner: {self.winner()}")
        return "\n".join(lines)


class RPS13PlusEnv(RPS13Env):
    """Optional RPS13+ variant with three bet tokens per player."""

    variant = "plus"

    def __init__(self, target_score: int = DEFAULT_TARGET_SCORE, bet_tokens: int = 3) -> None:
        self.initial_bet_tokens = bet_tokens
        super().__init__(target_score=target_score)

    def reset(self) -> GameObservation:
        self.human_score = 0
        self.ai_score = 0
        self.human_bet_tokens = self.initial_bet_tokens
        self.ai_bet_tokens = self.initial_bet_tokens
        self.history = []
        return self.get_observation()

    def step(
        self,
        ai_action: int,
        human_action: int,
        ai_bet: bool = False,
        human_bet: bool = False,
    ) -> tuple[GameObservation, float, bool, dict[str, Any]]:
        if self.is_done():
            raise RuntimeError("Cannot step: match is already done.")
        ai = normalize_action(ai_action)
        human = normalize_action(human_action)
        ai_bet = bool(ai_bet and self.ai_bet_tokens > 0)
        human_bet = bool(human_bet and self.human_bet_tokens > 0)
        if ai_bet:
            self.ai_bet_tokens -= 1
        if human_bet:
            self.human_bet_tokens -= 1
        result = outcome_for_ai(ai, human)
        points = 2 if (ai_bet or human_bet) else 1
        if result == Result.WIN:
            self.ai_score += points
            reward = float(points)
        elif result == Result.LOSS:
            self.human_score += points
            reward = -float(points)
        else:
            reward = 0.0
        done = self.is_done()
        if done:
            reward += 5.0 if self.ai_score >= self.target_score else -5.0
        record = RoundRecord(
            round=len(self.history) + 1,
            human_move=int(human),
            ai_move=int(ai),
            result_for_ai=int(result),
            human_score=self.human_score,
            ai_score=self.ai_score,
            round_winner=round_winner(ai, human),
            is_terminal=done,
            timestamp_or_step=len(self.history) + 1,
            points_awarded=points,
            human_bet=human_bet,
            ai_bet=ai_bet,
        )
        self.history.append(record)
        obs = self.get_observation()
        return obs, reward, done, record.to_dict()

    def get_observation(self) -> GameObservation:
        return GameObservation(
            target_score=self.target_score,
            round_number=len(self.history) + 1,
            human_score=self.human_score,
            ai_score=self.ai_score,
            done=self.is_done(),
            history=list(self.history),
            variant=self.variant,
            human_bet_tokens=self.human_bet_tokens,
            ai_bet_tokens=self.ai_bet_tokens,
        )
