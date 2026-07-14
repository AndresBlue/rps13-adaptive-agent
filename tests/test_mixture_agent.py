"""Tests for mixture agent and cycle exploitation."""

from __future__ import annotations

import numpy as np

from rps13.agents.mixture_agent import MixtureAdaptiveAgent
from rps13.constants import Action
from rps13.experts.predictors import CycleDetectorExpert
from rps13.game.env import RPS13Env
from rps13.game.state import GameObservation, RoundRecord


def _cycle_history(length: int = 8) -> list[RoundRecord]:
    pattern = [Action.ROCK, Action.PAPER, Action.SCISSORS]
    history: list[RoundRecord] = []
    for idx in range(length):
        history.append(
            RoundRecord(
                round=idx + 1,
                human_move=int(pattern[idx % 3]),
                ai_move=0,
                result_for_ai=0,
                human_score=0,
                ai_score=0,
                round_winner="draw",
                is_terminal=False,
                timestamp_or_step=idx,
            )
        )
    return history


def test_cycle_detector_predicts_next_in_cycle() -> None:
    expert = CycleDetectorExpert(min_match_rate=0.7)
    obs = GameObservation(
        target_score=13,
        round_number=8,
        human_score=0,
        ai_score=0,
        done=False,
        history=_cycle_history(),
    )
    pred = expert.predict(obs)
    assert int(np.argmax(pred.probs)) == int(Action.SCISSORS)
    assert pred.pattern_stability >= 0.7


def test_mixture_exploits_fixed_cycle() -> None:
    agent = MixtureAdaptiveAgent(checkpoint_path=None, seed=123, debug_deterministic=True)
    env = RPS13Env(target_score=13)
    human_pattern = [Action.ROCK, Action.PAPER, Action.SCISSORS]
    wins = 0
    for idx in range(12):
        obs = env.get_observation()
        decision = agent.select_action(obs)
        human_move = int(human_pattern[idx % 3])
        _next, _r, _d, info = env.step(decision.action, human_move)
        agent.observe_round(human_move)
        if info["result_for_ai"] == 1:
            wins += 1
    assert wins >= 8


def test_mixture_policy_is_valid_distribution() -> None:
    agent = MixtureAdaptiveAgent(checkpoint_path=None, seed=1)
    obs = GameObservation(
        target_score=13,
        round_number=8,
        human_score=3,
        ai_score=2,
        done=False,
        history=_cycle_history(),
    )
    probs = agent.policy(obs)
    assert probs.shape == (3,)
    assert np.all(probs >= 0)
    assert abs(float(probs.sum()) - 1.0) < 1e-5


def test_pattern_stability_separate_from_margin() -> None:
    agent = MixtureAdaptiveAgent(checkpoint_path=None, seed=7)
    obs = GameObservation(
        target_score=13,
        round_number=8,
        human_score=0,
        ai_score=0,
        done=False,
        history=_cycle_history(),
    )
    agent.policy(obs)
    assert "pattern_stability" in agent.last_debug
    assert "prediction_margin" in agent.last_debug
