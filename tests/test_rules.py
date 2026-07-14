from rps13.constants import Action, Result
from rps13.game.rules import outcome_for_ai


def test_rock_beats_scissors():
    assert outcome_for_ai(Action.ROCK, Action.SCISSORS) == Result.WIN


def test_scissors_beats_paper():
    assert outcome_for_ai(Action.SCISSORS, Action.PAPER) == Result.WIN


def test_paper_beats_rock():
    assert outcome_for_ai(Action.PAPER, Action.ROCK) == Result.WIN


def test_draw():
    assert outcome_for_ai(Action.ROCK, Action.ROCK) == Result.DRAW
