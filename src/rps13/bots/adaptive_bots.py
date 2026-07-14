"""More adaptive and human-like scripted opponents."""

from __future__ import annotations

from collections import Counter

from rps13.agents.base import BaseOpponent
from rps13.bots.simple_bots import (
    AntiLastMoveBot,
    AvoidRepeatBot,
    BiasedPaperBot,
    BiasedRockBot,
    BiasedScissorsBot,
    CopyOpponentBot,
    CycleBot,
    RandomBot,
    ReverseCycleBot,
    WinStayLoseShiftBot,
)
from rps13.constants import Action
from rps13.game.rules import best_response_to
from rps13.game.state import GameObservation


class PressureBot(BaseOpponent):
    """Switches behavior when either player is close to the target score."""

    name = "pressure"

    def choose_action(self, observation: GameObservation) -> int:
        if observation.human_score >= 10 or observation.ai_score >= 10:
            if observation.history:
                return self.maybe_noise(int(best_response_to(observation.history[-1].ai_move)))
            return int(Action.PAPER)
        if self.rng.random() < 0.55:
            return int(Action.ROCK)
        return self.random_action()


class NoisyHumanLikeBot(BaseOpponent):
    """Mixture of plausible human heuristics with random noise."""

    name = "noisy_human_like"

    def __init__(self, seed: int | None = None, noise: float = 0.15) -> None:
        super().__init__(seed=seed, noise=noise)
        self.members = [
            WinStayLoseShiftBot(seed=seed, noise=0.05),
            AntiLastMoveBot(seed=seed, noise=0.08),
            CopyOpponentBot(seed=seed, noise=0.08),
            AvoidRepeatBot(seed=seed, noise=0.08),
            BiasedRockBot(seed=seed, noise=0.1, bias=0.55),
        ]

    def reset(self) -> None:
        for bot in self.members:
            bot.reset()

    def choose_action(self, observation: GameObservation) -> int:
        if self.rng.random() < self.noise:
            return self.random_action()
        bot = self.members[int(self.rng.integers(0, len(self.members)))]
        return bot.choose_action(observation)


class SwitchingStrategyBot(BaseOpponent):
    """Uses one pattern early and another later in the match."""

    name = "switching_strategy"

    def __init__(self, seed: int | None = None, noise: float = 0.05) -> None:
        super().__init__(seed=seed, noise=noise)
        self.early = CycleBot(seed=seed, noise=noise)
        self.late = AntiLastMoveBot(seed=seed, noise=noise)

    def reset(self) -> None:
        self.early.reset()
        self.late.reset()

    def choose_action(self, observation: GameObservation) -> int:
        if len(observation.history) < 12 and observation.human_score < 7 and observation.ai_score < 7:
            return self.early.choose_action(observation)
        return self.late.choose_action(observation)


class AdaptiveCounterBot(BaseOpponent):
    """Detects the AI's most frequent action and counters it."""

    name = "adaptive_counter"

    def choose_action(self, observation: GameObservation) -> int:
        if len(observation.history) < 3:
            return self.random_action()
        counts = Counter(record.ai_move for record in observation.history)
        most_common_ai = counts.most_common(1)[0][0]
        return self.maybe_noise(int(best_response_to(most_common_ai)))


class MixedBiasBot(BaseOpponent):
    """Samples one of the three biased bots at reset time."""

    name = "mixed_bias"

    def __init__(self, seed: int | None = None, noise: float = 0.05) -> None:
        super().__init__(seed=seed, noise=noise)
        self.bot: BaseOpponent = RandomBot(seed=seed)
        self.reset()

    def reset(self) -> None:
        cls = self.rng.choice([BiasedRockBot, BiasedPaperBot, BiasedScissorsBot])
        self.bot = cls(seed=int(self.rng.integers(0, 2**31 - 1)), noise=self.noise)

    def choose_action(self, observation: GameObservation) -> int:
        return self.bot.choose_action(observation)


class ReverseThenCopyBot(BaseOpponent):
    """Reverse cycle at first, then copies the AI's previous move."""

    name = "reverse_then_copy"

    def __init__(self, seed: int | None = None, noise: float = 0.05) -> None:
        super().__init__(seed=seed, noise=noise)
        self.reverse = ReverseCycleBot(seed=seed, noise=noise)
        self.copy = CopyOpponentBot(seed=seed, noise=noise)

    def reset(self) -> None:
        self.reverse.reset()
        self.copy.reset()

    def choose_action(self, observation: GameObservation) -> int:
        if len(observation.history) < 8:
            return self.reverse.choose_action(observation)
        return self.copy.choose_action(observation)
