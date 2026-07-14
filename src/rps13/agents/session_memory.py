"""Cross-match player memory within a browser session."""

from __future__ import annotations

from dataclasses import dataclass, field

from rps13.game.state import RoundRecord


@dataclass
class SessionPlayerMemory:
    """Accumulates human play patterns across matches in one session."""

    prior_rounds: list[RoundRecord] = field(default_factory=list)
    expert_scores: dict[str, float] = field(default_factory=dict)
    expert_hits: dict[str, list[int]] = field(default_factory=dict)
    matches_played: int = 0
    human_sets: int = 0
    ai_sets: int = 0
    last_set_winner: str | None = None
    set_streak: int = 0
    max_prior_rounds: int = 36
    focus_last_match_rounds: int = 18
    score_carry_decay: float = 0.72

    def archive_match(
        self,
        history: list[RoundRecord],
        *,
        expert_scores: dict[str, float] | None = None,
        expert_hits: dict[str, list[int]] | None = None,
    ) -> None:
        """Store the finished match, emphasizing recent rounds (strategy may shift)."""

        if not history:
            return
        self.matches_played += 1
        focus = history[-self.focus_last_match_rounds :]
        self.prior_rounds = (self.prior_rounds + focus)[-self.max_prior_rounds :]

        if expert_scores:
            for key, value in expert_scores.items():
                self.expert_scores[key] = self.expert_scores.get(key, 0.0) * self.score_carry_decay + value * (
                    1.0 - self.score_carry_decay
                )
        if expert_hits:
            for key, hits in expert_hits.items():
                merged = (self.expert_hits.get(key, []) + hits)[-6:]
                self.expert_hits[key] = merged

    @property
    def prior_human_moves(self) -> list[int]:
        return [int(record.human_move) for record in self.prior_rounds]

    def record_set_win(self, winner: str | None) -> None:
        """Increment session set counters when a player reaches target_score."""

        if winner not in {"human", "ai"}:
            return
        if winner == "human":
            self.human_sets += 1
        else:
            self.ai_sets += 1
        if self.last_set_winner == winner:
            self.set_streak += 1
        else:
            self.set_streak = 1
        self.last_set_winner = winner

    def sets_summary(self) -> dict[str, int | str | None]:
        return {
            "human": self.human_sets,
            "ai": self.ai_sets,
            "total": self.human_sets + self.ai_sets,
            "streak": self.set_streak,
            "streak_holder": self.last_set_winner,
        }

    def summary(self) -> dict[str, int | float]:
        return {
            "matches_played": self.matches_played,
            "prior_rounds": len(self.prior_rounds),
            "prior_human_moves": len(self.prior_human_moves),
        }
