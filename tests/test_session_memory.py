"""Tests for cross-match session learning."""

from __future__ import annotations

from rps13.agents.mixture_agent import MixtureAdaptiveAgent
from rps13.agents.session_memory import SessionPlayerMemory
from rps13.constants import Action
from rps13.game.env import RPS13Env
from rps13.game.state import RoundRecord


def _make_record(round_num: int, human: int) -> RoundRecord:
    return RoundRecord(
        round=round_num,
        human_move=human,
        ai_move=0,
        result_for_ai=0,
        human_score=0,
        ai_score=0,
        round_winner="draw",
        is_terminal=False,
        timestamp_or_step=round_num,
    )


def test_session_memory_archives_and_seeds_agent() -> None:
    memory = SessionPlayerMemory()
    history = [_make_record(i + 1, int(Action.ROCK)) for i in range(10)]
    memory.archive_match(history, expert_scores={"cycle_detector:P.0": 4.0}, expert_hits={"cycle_detector:P.0": [1, 1, 0]})

    agent = MixtureAdaptiveAgent(checkpoint_path=None, seed=1)
    agent.seed_from_session(memory)

    assert agent.session_matches_seen == 1
    assert len(agent.session_prior_history) == 10
    assert agent.scores.get("cycle_detector:P.0", 0) > 0


def test_session_memory_records_sets() -> None:
    memory = SessionPlayerMemory()
    memory.record_set_win("human")
    memory.record_set_win("human")
    memory.record_set_win("ai")

    summary = memory.sets_summary()
    assert summary["human"] == 2
    assert summary["ai"] == 1
    assert summary["total"] == 3
    assert summary["streak"] == 1
    assert summary["streak_holder"] == "ai"


def test_reset_keeps_learning_across_matches() -> None:
    memory = SessionPlayerMemory()
    agent = MixtureAdaptiveAgent(checkpoint_path=None, seed=2)
    env = RPS13Env(target_score=13)
    pattern = [Action.ROCK, Action.PAPER, Action.SCISSORS]
    for idx in range(9):
        obs = env.get_observation()
        decision = agent.select_action(obs)
        human = int(pattern[idx % 3])
        env.step(decision.action, human)
        agent.observe_round(human)

    scores, hits = agent.export_session_learning()
    memory.archive_match(list(env.history), expert_scores=scores, expert_hits=hits)

    agent2 = MixtureAdaptiveAgent(checkpoint_path=None, seed=3)
    agent2.seed_from_session(memory)
    assert len(agent2.session_prior_history) >= 9
