"""Simple scripted human-side opponent bots."""

from __future__ import annotations

import numpy as np

from rps13.agents.base import BaseOpponent
from rps13.constants import Action, Result
from rps13.game.rules import best_response_to, next_cycle_action, previous_cycle_action
from rps13.game.state import GameObservation


class RandomBot(BaseOpponent):
    name = "random"

    def choose_action(self, observation: GameObservation) -> int:
        return self.random_action()


class BiasedActionBot(BaseOpponent):
    name = "biased_action"

    def __init__(
        self,
        action: int,
        seed: int | None = None,
        noise: float = 0.0,
        bias: float = 0.7,
    ) -> None:
        super().__init__(seed=seed, noise=noise)
        self.action = int(action)
        self.bias = bias

    def choose_action(self, observation: GameObservation) -> int:
        if self.rng.random() < self.bias:
            return self.maybe_noise(self.action)
        return self.random_action()


class BiasedRockBot(BiasedActionBot):
    name = "biased_rock"

    def __init__(self, seed: int | None = None, noise: float = 0.05, bias: float = 0.75) -> None:
        super().__init__(Action.ROCK, seed=seed, noise=noise, bias=bias)


class BiasedPaperBot(BiasedActionBot):
    name = "biased_paper"

    def __init__(self, seed: int | None = None, noise: float = 0.05, bias: float = 0.75) -> None:
        super().__init__(Action.PAPER, seed=seed, noise=noise, bias=bias)


class BiasedScissorsBot(BiasedActionBot):
    name = "biased_scissors"

    def __init__(self, seed: int | None = None, noise: float = 0.05, bias: float = 0.75) -> None:
        super().__init__(Action.SCISSORS, seed=seed, noise=noise, bias=bias)


class CycleBot(BaseOpponent):
    name = "cycle"

    def __init__(self, seed: int | None = None, noise: float = 0.0) -> None:
        super().__init__(seed=seed, noise=noise)
        self.current = Action.ROCK

    def reset(self) -> None:
        self.current = Action.ROCK

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            self.current = Action.ROCK
        else:
            self.current = next_cycle_action(self.current)
        return self.maybe_noise(int(self.current))


class ReverseCycleBot(CycleBot):
    name = "reverse_cycle"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            self.current = Action.ROCK
        else:
            self.current = previous_cycle_action(self.current)
        return self.maybe_noise(int(self.current))


class WinStayLoseShiftBot(BaseOpponent):
    name = "win_stay_lose_shift"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        last = observation.history[-1]
        action = last.human_move
        if last.result_for_ai == int(Result.WIN):
            action = int(next_cycle_action(action))
        elif last.result_for_ai == int(Result.DRAW):
            action = int(next_cycle_action(action)) if self.rng.random() < 0.35 else action
        return self.maybe_noise(action)


class LoseStayWinShiftBot(BaseOpponent):
    name = "lose_stay_win_shift"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        last = observation.history[-1]
        action = last.human_move
        if last.result_for_ai == int(Result.LOSS):
            action = int(next_cycle_action(action))
        elif last.result_for_ai == int(Result.DRAW):
            action = int(previous_cycle_action(action)) if self.rng.random() < 0.35 else action
        return self.maybe_noise(action)


class CopyOpponentBot(BaseOpponent):
    name = "copy_opponent"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        return self.maybe_noise(observation.history[-1].ai_move)


class AntiLastMoveBot(BaseOpponent):
    name = "anti_last_move"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        return self.maybe_noise(int(best_response_to(observation.history[-1].ai_move)))


class AvoidRepeatBot(BaseOpponent):
    name = "avoid_repeat"

    def choose_action(self, observation: GameObservation) -> int:
        if not observation.history:
            return self.random_action()
        last = observation.history[-1].human_move
        choices = [a for a in range(3) if a != last]
        return self.maybe_noise(int(self.rng.choice(choices)))


def sample_from_probs(rng: np.random.Generator, probs: list[float] | np.ndarray) -> int:
    arr = np.asarray(probs, dtype=np.float64)
    arr = arr / arr.sum()
    return int(rng.choice(3, p=arr))
