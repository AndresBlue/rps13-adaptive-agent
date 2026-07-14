"""Extended bot population for v2 evaluation and training."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rps13.agents.base import BaseOpponent
from rps13.bots.simple_bots import BiasedActionBot, sample_from_probs
from rps13.constants import Action, Result
from rps13.game.rules import best_response_to, next_cycle_action, previous_cycle_action
from rps13.game.state import GameObservation


class StickyRockBot(BiasedActionBot):
    name = "sticky_rock"

    def __init__(self, seed: int | None = None, noise: float = 0.03, bias: float = 0.92) -> None:
        super().__init__(Action.ROCK, seed=seed, noise=noise, bias=bias)


class StickyPaperBot(BiasedActionBot):
    name = "sticky_paper"

    def __init__(self, seed: int | None = None, noise: float = 0.03, bias: float = 0.92) -> None:
        super().__init__(Action.PAPER, seed=seed, noise=noise, bias=bias)


class StickyScissorsBot(BiasedActionBot):
    name = "sticky_scissors"

    def __init__(self, seed: int | None = None, noise: float = 0.03, bias: float = 0.92) -> None:
        super().__init__(Action.SCISSORS, seed=seed, noise=noise, bias=bias)


class PeriodPatternBot(BaseOpponent):
    """Repeats a fixed move pattern of given period."""

    name = "period_pattern"

    def __init__(self, pattern: list[int], seed: int | None = None, noise: float = 0.02) -> None:
        super().__init__(seed=seed, noise=noise)
        self.pattern = [int(x) % 3 for x in pattern]
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def choose_action(self, observation: GameObservation) -> int:
        action = self.pattern[self.index % len(self.pattern)]
        self.index += 1
        return self.maybe_noise(action)


class Period2ABBot(PeriodPatternBot):
    name = "period2_ab"

    def __init__(self, seed: int | None = None, noise: float = 0.02) -> None:
        super().__init__([Action.ROCK, Action.PAPER], seed=seed, noise=noise)


class Period3ABCBot(PeriodPatternBot):
    name = "period3_abc"

    def __init__(self, seed: int | None = None, noise: float = 0.02) -> None:
        super().__init__([Action.ROCK, Action.PAPER, Action.SCISSORS], seed=seed, noise=noise)


class Period4Bot(PeriodPatternBot):
    name = "period4"

    def __init__(self, seed: int | None = None, noise: float = 0.02) -> None:
        super().__init__(
            [Action.ROCK, Action.PAPER, Action.SCISSORS, Action.ROCK],
            seed=seed,
            noise=noise,
        )


class BrockbankSelfPlusBot(BaseOpponent):
    name = "brockbank_self_plus"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        last = observation.history[-1].human_move
        return self.maybe_noise(int(best_response_to(last)))


class BrockbankSelfMinusBot(BaseOpponent):
    name = "brockbank_self_minus"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        last = observation.history[-1].human_move
        return self.maybe_noise(int(previous_cycle_action(last)))


class BrockbankOppPlusBot(BaseOpponent):
    name = "brockbank_opp_plus"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        last = observation.history[-1].ai_move
        return self.maybe_noise(int(best_response_to(last)))


class BrockbankOppCopyBot(BaseOpponent):
    name = "brockbank_opp_copy"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        return self.maybe_noise(observation.history[-1].ai_move)


class WSLCClassicBot(BaseOpponent):
    name = "wslc_classic"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        last = observation.history[-1]
        action = last.human_move
        if last.result_for_ai == int(Result.WIN):
            action = last.human_move
        elif last.result_for_ai == int(Result.LOSS):
            action = int(next_cycle_action(last.human_move))
        return self.maybe_noise(action)


class WinChangeLoseStayBot(BaseOpponent):
    name = "win_change_lose_stay"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        last = observation.history[-1]
        action = last.human_move
        if last.result_for_ai == int(Result.WIN):
            action = int(next_cycle_action(last.human_move))
        elif last.result_for_ai == int(Result.LOSS):
            action = last.human_move
        return self.maybe_noise(action)


class PseudoRandomBiasedBot(BaseOpponent):
    name = "pseudo_random_biased"

    def choose_action(self, observation: GameObservation) -> int:
        return sample_from_probs(self.rng, [0.40, 0.30, 0.30])


class HumanReplayBot(BaseOpponent):
    """Replays human move sequences extracted from JSONL logs."""

    name = "human_replay"

    def __init__(self, log_dir: str | Path | None = None, seed: int | None = None, noise: float = 0.0) -> None:
        super().__init__(seed=seed, noise=noise)
        self.sequences = self._load_sequences(log_dir)
        self.sequence: list[int] = []
        self.cursor = 0
        self.reset()

    @staticmethod
    def _load_sequences(log_dir: str | Path | None) -> list[list[int]]:
        root = Path(log_dir) if log_dir else Path("data/human_logs")
        sequences: list[list[int]] = []
        if not root.exists():
            return sequences
        for path in sorted(root.glob("*.jsonl")):
            moves: list[int] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("event") == "match_end":
                        continue
                    if "human_move" in row:
                        moves.append(int(row["human_move"]))
            if len(moves) >= 5:
                sequences.append(moves)
        return sequences

    def reset(self) -> None:
        if self.sequences:
            self.sequence = list(self.sequences[int(self.rng.integers(0, len(self.sequences)))])
        else:
            self.sequence = [0, 1, 2, 0, 1, 2]
        self.cursor = 0

    def choose_action(self, observation: GameObservation) -> int:
        action = self.sequence[self.cursor % len(self.sequence)]
        self.cursor += 1
        return self.maybe_noise(action)
