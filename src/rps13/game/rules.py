"""Pure rock-paper-scissors rule helpers."""

from __future__ import annotations

from rps13.constants import ACTION_NAMES, Action, Result


def normalize_action(action: int | Action) -> Action:
    """Convert an int-like action to ``Action`` and validate it."""

    try:
        normalized = Action(int(action))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid RPS action: {action!r}") from exc
    return normalized


def outcome_for_ai(ai_action: int | Action, human_action: int | Action) -> Result:
    """Return the round outcome from the AI perspective."""

    ai = normalize_action(ai_action)
    human = normalize_action(human_action)
    if ai == human:
        return Result.DRAW
    if (ai == Action.ROCK and human == Action.SCISSORS) or (
        ai == Action.PAPER and human == Action.ROCK
    ) or (ai == Action.SCISSORS and human == Action.PAPER):
        return Result.WIN
    return Result.LOSS


def round_winner(ai_action: int | Action, human_action: int | Action) -> str:
    """Return ``ai``, ``human`` or ``draw`` for a round."""

    result = outcome_for_ai(ai_action, human_action)
    if result == Result.WIN:
        return "ai"
    if result == Result.LOSS:
        return "human"
    return "draw"


def best_response_to(opponent_action: int | Action) -> Action:
    """Return the action that beats ``opponent_action``."""

    action = normalize_action(opponent_action)
    if action == Action.ROCK:
        return Action.PAPER
    if action == Action.PAPER:
        return Action.SCISSORS
    return Action.ROCK


def losing_response_to(opponent_action: int | Action) -> Action:
    """Return the action that loses to ``opponent_action``."""

    action = normalize_action(opponent_action)
    if action == Action.ROCK:
        return Action.SCISSORS
    if action == Action.PAPER:
        return Action.ROCK
    return Action.PAPER


def next_cycle_action(action: int | Action) -> Action:
    """ROCK -> PAPER -> SCISSORS -> ROCK."""

    action = normalize_action(action)
    return Action((int(action) + 1) % 3)


def previous_cycle_action(action: int | Action) -> Action:
    """ROCK -> SCISSORS -> PAPER -> ROCK."""

    action = normalize_action(action)
    return Action((int(action) - 1) % 3)


def action_name(action: int | Action) -> str:
    """Return the canonical uppercase action name."""

    return ACTION_NAMES[normalize_action(action)]
