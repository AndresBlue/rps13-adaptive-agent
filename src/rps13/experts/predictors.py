"""Concrete prediction experts for the mixture agent."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from rps13.constants import NUM_ACTIONS, Result
from rps13.experts.base import BaseExpert, ExpertPrediction
from rps13.game.rules import best_response_to, next_cycle_action, previous_cycle_action
from rps13.game.state import GameObservation, RoundRecord
from rps13.models.opponent_predictor import OpponentPredictorGRU, load_predictor_checkpoint


def _human_moves(history: list[RoundRecord]) -> list[int]:
    return [int(record.human_move) for record in history]


def _transition_type(prev: int, current: int) -> str:
    if current == prev:
        return "0"
    if current == int(best_response_to(prev)):
        return "+"
    if current == int(previous_cycle_action(prev)):
        return "-"
    return "?"


class RandomHedgeExpert(BaseExpert):
    name = "random_hedge"

    def predict(self, observation: GameObservation) -> ExpertPrediction:
        probs = np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
        return ExpertPrediction(name=self.name, probs=probs, pattern_stability=0.0)


class StickyFrequencyExpert(BaseExpert):
    name = "sticky_frequency"

    def __init__(self, windows: tuple[int, ...] = (5, 10, 20)) -> None:
        self.windows = windows

    def predict(self, observation: GameObservation) -> ExpertPrediction:
        moves = _human_moves(observation.history)
        if not moves:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)
        counts = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for window in self.windows:
            recent = moves[-window:]
            if not recent:
                continue
            freq = np.bincount(recent, minlength=NUM_ACTIONS).astype(np.float64)
            counts += freq / max(len(recent), 1)
        probs = self.normalize(counts)
        stability = float(np.max(probs) - 1.0 / 3.0) / (2.0 / 3.0)
        return ExpertPrediction(
            name=self.name,
            probs=probs,
            pattern_stability=max(0.0, min(1.0, stability)),
            debug={"dominant": int(np.argmax(probs))},
        )


class CycleDetectorExpert(BaseExpert):
    name = "cycle_detector"

    def __init__(self, periods: tuple[int, ...] = (2, 3, 4, 5, 6), min_match_rate: float = 0.75) -> None:
        self.periods = periods
        self.min_match_rate = min_match_rate

    def predict(self, observation: GameObservation) -> ExpertPrediction:
        moves = _human_moves(observation.history)
        if len(moves) < 4:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        best_period = 0
        best_rate = 0.0
        best_next: int | None = None
        for period in self.periods:
            if len(moves) < period + 2:
                continue
            window = moves[-max(period * 4, 12) :]
            matches = 0
            total = 0
            for idx in range(period, len(window)):
                expected = window[idx - period]
                if window[idx] == expected:
                    matches += 1
                total += 1
            rate = matches / max(total, 1)
            if rate >= self.min_match_rate and rate >= best_rate:
                best_rate = rate
                best_period = period
                best_next = moves[-period]

        if best_next is None:
            last = moves[-1]
            upgrade_hits = sum(1 for i in range(1, len(moves)) if moves[i] == int(next_cycle_action(moves[i - 1])))
            downgrade_hits = sum(
                1 for i in range(1, len(moves)) if moves[i] == int(previous_cycle_action(moves[i - 1]))
            )
            transitions = max(len(moves) - 1, 1)
            up_rate = upgrade_hits / transitions
            down_rate = downgrade_hits / transitions
            if max(up_rate, down_rate) >= self.min_match_rate:
                best_rate = max(up_rate, down_rate)
                best_next = int(next_cycle_action(last)) if up_rate >= down_rate else int(previous_cycle_action(last))
                best_period = 1

        if best_next is None:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        probs = np.zeros(NUM_ACTIONS, dtype=np.float64)
        probs[best_next] = 0.85
        probs[(best_next + 1) % 3] = 0.075
        probs[(best_next + 2) % 3] = 0.075
        return ExpertPrediction(
            name=self.name,
            probs=self.normalize(probs),
            pattern_stability=min(1.0, best_rate),
            debug={"period": best_period, "match_rate": best_rate, "predicted_move": best_next},
        )


class HistoryMatchPPMExpert(BaseExpert):
    name = "history_match_ppm"

    def __init__(self, max_suffix: int = 12) -> None:
        self.max_suffix = max_suffix

    def predict(self, observation: GameObservation) -> ExpertPrediction:
        moves = _human_moves(observation.history)
        if len(moves) < 3:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        best_len = 0
        votes: Counter[int] = Counter()
        for suffix_len in range(min(self.max_suffix, len(moves) - 1), 0, -1):
            suffix = moves[-suffix_len:]
            haystack = moves[:-suffix_len]
            for start in range(len(haystack) - suffix_len):
                if haystack[start : start + suffix_len] == suffix:
                    follow_idx = start + suffix_len
                    if follow_idx < len(haystack):
                        votes[haystack[follow_idx]] += suffix_len
                        best_len = max(best_len, suffix_len)
            if votes:
                break

        if not votes:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        total = sum(votes.values())
        probs = np.array([votes.get(i, 0) / total for i in range(NUM_ACTIONS)], dtype=np.float64)
        stability = min(1.0, best_len / max(self.max_suffix, 1))
        return ExpertPrediction(
            name=self.name,
            probs=self.normalize(probs),
            pattern_stability=stability,
            debug={"suffix_len": best_len, "votes": dict(votes)},
        )


class MarkovLagNExpert(BaseExpert):
    name = "markov_lag_n"

    def __init__(self, order: int = 1) -> None:
        self.order = order

    def predict(self, observation: GameObservation) -> ExpertPrediction:
        moves = _human_moves(observation.history)
        if len(moves) <= self.order:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        for idx in range(self.order, len(moves)):
            state = tuple(moves[idx - self.order : idx])
            counts[state][moves[idx]] += 1

        state = tuple(moves[-self.order :])
        if state not in counts:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        total = sum(counts[state].values())
        probs = np.array([counts[state].get(i, 0) / total for i in range(NUM_ACTIONS)], dtype=np.float64)
        return ExpertPrediction(
            name=self.name,
            probs=self.normalize(probs),
            pattern_stability=float(np.max(probs) - 1.0 / 3.0) / (2.0 / 3.0),
            debug={"order": self.order, "state": state},
        )


class OutcomeMarkovExpert(BaseExpert):
    name = "outcome_markov"

    def predict(self, observation: GameObservation) -> ExpertPrediction:
        history = observation.history
        if len(history) < 2:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
        for idx in range(1, len(history)):
            prev = history[idx - 1]
            key = (int(prev.human_move), int(prev.result_for_ai))
            counts[key][int(history[idx].human_move)] += 1

        last = history[-1]
        key = (int(last.human_move), int(last.result_for_ai))
        if key not in counts:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        total = sum(counts[key].values())
        probs = np.array([counts[key].get(i, 0) / total for i in range(NUM_ACTIONS)], dtype=np.float64)
        return ExpertPrediction(
            name=self.name,
            probs=self.normalize(probs),
            pattern_stability=float(np.max(probs) - 1.0 / 3.0) / (2.0 / 3.0),
            debug={"context": key},
        )


class TransitionBrockbankExpert(BaseExpert):
    name = "transition_brockbank"

    def __init__(self, mode: str = "self") -> None:
        self.mode = mode

    def predict(self, observation: GameObservation) -> ExpertPrediction:
        history = observation.history
        if len(history) < 2:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        transition_counts: Counter[str] = Counter()
        for idx in range(1, len(history)):
            prev = history[idx - 1]
            cur = history[idx]
            ref = int(prev.human_move if self.mode == "self" else prev.ai_move)
            transition_counts[_transition_type(ref, int(cur.human_move))] += 1

        if not transition_counts:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        dominant = transition_counts.most_common(1)[0][0]
        last = history[-1]
        base = int(last.human_move if self.mode == "self" else last.ai_move)
        if dominant == "+":
            nxt = int(best_response_to(base))
        elif dominant == "-":
            nxt = int(previous_cycle_action(base))
        elif dominant == "0":
            nxt = base
        else:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)

        probs = np.full(NUM_ACTIONS, 0.05, dtype=np.float64)
        probs[nxt] = 0.85
        total = transition_counts.total()
        stability = transition_counts[dominant] / max(total, 1)
        return ExpertPrediction(
            name=self.name,
            probs=self.normalize(probs),
            pattern_stability=float(stability),
            debug={"mode": self.mode, "transition": dominant},
        )


class NeuralGRUExpert(BaseExpert):
    name = "neural_gru"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        predictor: OpponentPredictorGRU | None = None,
        sequence_length: int = 20,
        target_score: int = 13,
    ) -> None:
        self.sequence_length = sequence_length
        self.target_score = target_score
        self.predictor = predictor
        if self.predictor is None and checkpoint_path and Path(checkpoint_path).exists():
            self.predictor = load_predictor_checkpoint(checkpoint_path)

    def predict(self, observation: GameObservation) -> ExpertPrediction:
        if self.predictor is None:
            return ExpertPrediction(name=self.name, probs=np.ones(3) / 3.0)
        probs = self.predictor.predict_proba(
            observation.history,
            sequence_length=self.sequence_length,
            target_score=self.target_score,
        )
        probs = self.normalize(probs)
        margin = (float(np.max(probs)) - 1.0 / 3.0) / (2.0 / 3.0)
        return ExpertPrediction(
            name=self.name,
            probs=probs,
            pattern_stability=max(0.0, min(1.0, margin * 0.5)),
            debug={"source": "gru"},
        )


def build_default_experts(checkpoint_path: str | Path | None = None, target_score: int = 13) -> list[BaseExpert]:
    """Construct the default expert ensemble."""

    return [
        RandomHedgeExpert(),
        StickyFrequencyExpert(),
        CycleDetectorExpert(),
        HistoryMatchPPMExpert(),
        MarkovLagNExpert(order=1),
        MarkovLagNExpert(order=2),
        MarkovLagNExpert(order=3),
        OutcomeMarkovExpert(),
        TransitionBrockbankExpert(mode="self"),
        TransitionBrockbankExpert(mode="opp"),
        NeuralGRUExpert(checkpoint_path=checkpoint_path, target_score=target_score),
    ]


def apply_meta_level(probs: np.ndarray, meta_level: str) -> np.ndarray:
    """Apply Iocaine-style meta counterfactual to human move probabilities."""

    probs = BaseExpert.normalize(probs)
    if meta_level == "P.0":
        return probs
    if meta_level == "P.1":
        return BaseExpert.normalize(np.array([probs[2], probs[0], probs[1]]))
    if meta_level == "P'.0":
        return BaseExpert.normalize(np.array([probs[1], probs[2], probs[0]]))
    return probs
