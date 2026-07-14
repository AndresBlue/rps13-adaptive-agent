from rps13.constants import Action
from rps13.game.env import RPS13Env


def test_draw_does_not_score():
    env = RPS13Env()
    env.step(Action.ROCK, Action.ROCK)
    assert env.ai_score == 0
    assert env.human_score == 0


def test_match_ends_at_13():
    env = RPS13Env()
    for _ in range(13):
        env.step(Action.PAPER, Action.ROCK)
    assert env.is_done()
    assert env.winner() == "ai"


def test_reset_clears_state():
    env = RPS13Env()
    env.step(Action.PAPER, Action.ROCK)
    env.reset()
    assert env.ai_score == 0
    assert env.human_score == 0
    assert env.history == []
    assert not env.is_done()
