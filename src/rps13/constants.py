"""Shared constants for RPS-13."""

from __future__ import annotations

from enum import IntEnum


class Action(IntEnum):
    """Rock-paper-scissors action ids."""

    ROCK = 0
    PAPER = 1
    SCISSORS = 2


class Result(IntEnum):
    """Round result from the AI/player-agent perspective."""

    LOSS = -1
    DRAW = 0
    WIN = 1


VALID_ACTIONS = (Action.ROCK, Action.PAPER, Action.SCISSORS)
ACTION_NAMES = {
    Action.ROCK: "ROCK",
    Action.PAPER: "PAPER",
    Action.SCISSORS: "SCISSORS",
}
ACTION_LABELS_ES = {
    Action.ROCK: "Piedra",
    Action.PAPER: "Papel",
    Action.SCISSORS: "Tijera",
}
RESULT_NAMES = {
    Result.WIN: "WIN",
    Result.DRAW: "DRAW",
    Result.LOSS: "LOSS",
}

PAD_ACTION = 3
PAD_RESULT_INDEX = 3
NUM_ACTIONS = 3
DEFAULT_TARGET_SCORE = 13
