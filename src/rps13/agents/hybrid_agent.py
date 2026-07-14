"""Hybrid adaptive agent: predictor + smoothed best response + Nash fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rps13.agents.base import AgentDecision, BaseAgent
from rps13.constants import NUM_ACTIONS
from rps13.data.feature_engineering import compute_history_features
from rps13.game.rules import best_response_to, next_cycle_action, previous_cycle_action
from rps13.game.state import GameObservation
from rps13.models.opponent_predictor import OpponentPredictorGRU, load_predictor_checkpoint


class HybridAdaptiveAgent(BaseAgent):
    """Adaptive agent that exploits detected patterns without becoming deterministic."""

    def __init__(
        self,
        predictor: OpponentPredictorGRU | None = None,
        checkpoint_path: str | Path | None = None,
        sequence_length: int = 20,
        target_score: int = 13,
        temperature: float = 0.6,
        max_alpha: float = 0.85,
        min_action_prob: float = 0.03,
        seed: int | None = None,
        debug_deterministic: bool = False,
    ) -> None:
        self.sequence_length = sequence_length
        self.target_score = target_score
        self.temperature = temperature
        self.max_alpha = max_alpha
        self.min_action_prob = min_action_prob
        self.rng = np.random.default_rng(seed)
        self.debug_deterministic = debug_deterministic
        self.predictor = predictor
        if self.predictor is None and checkpoint_path and Path(checkpoint_path).exists():
            self.predictor = load_predictor_checkpoint(checkpoint_path)
        self.last_debug: dict[str, Any] = {}

    def reset(self) -> None:
        self.last_debug = {}

    def estimate_human_probs(self, observation: GameObservation) -> tuple[np.ndarray, str]:
        """Estimate the next human move distribution."""

        if self.predictor is not None:
            probs = self.predictor.predict_proba(
                observation.history,
                sequence_length=self.sequence_length,
                target_score=self.target_score,
            )
            return self._normalize(probs), "neural_predictor"
        return self._heuristic_probs(observation), "heuristic_fallback"

    def _heuristic_probs(self, observation: GameObservation) -> np.ndarray:
        features = compute_history_features(observation.history, target_score=self.target_score)
        probs = np.array(
            [
                features["human_freq_rock"],
                features["human_freq_paper"],
                features["human_freq_scissors"],
            ],
            dtype=np.float64,
        )
        recent = np.array(
            [
                features["recent5_rock"],
                features["recent5_paper"],
                features["recent5_scissors"],
            ],
            dtype=np.float64,
        )
        probs = 0.45 * probs + 0.35 * recent + 0.20 * (np.ones(3) / 3.0)
        if observation.history:
            last = observation.history[-1]
            pattern_votes = np.zeros(3, dtype=np.float64)
            pattern_votes[last.ai_move] += float(features["copy_last_ai_rate"])
            pattern_votes[int(best_response_to(last.ai_move))] += float(features["beats_last_ai_rate"])
            pattern_votes[int(next_cycle_action(last.human_move))] += float(features["cycle_rate"])
            pattern_votes[int(previous_cycle_action(last.human_move))] += float(features["reverse_cycle_rate"])
            if pattern_votes.sum() > 0:
                probs = 0.75 * probs + 0.25 * (pattern_votes / pattern_votes.sum())
        return self._normalize(probs)

    def policy(self, observation: GameObservation) -> np.ndarray:
        human_probs, source = self.estimate_human_probs(observation)
        ev = np.array(
            [
                human_probs[2] - human_probs[1],
                human_probs[0] - human_probs[2],
                human_probs[1] - human_probs[0],
            ],
            dtype=np.float64,
        )
        soft_best = self._softmax(ev / max(self.temperature, 1e-6))
        confidence = self._confidence(human_probs)
        alpha = self.max_alpha * confidence
        uniform = np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
        mixed = (1.0 - alpha) * uniform + alpha * soft_best
        if not self.debug_deterministic:
            mixed = np.maximum(mixed, self.min_action_prob)
        mixed = self._normalize(mixed)
        self.last_debug = {
            "source": source,
            "predicted_probs": human_probs.tolist(),
            "expected_values": ev.tolist(),
            "confidence": float(confidence),
            "alpha": float(alpha),
            "chosen_policy": mixed.tolist(),
            "reasoning_debug": (
                "fallback_uniform" if confidence < 0.1 else "smoothed_best_response_to_predicted_human_bias"
            ),
        }
        return mixed.astype(np.float32)

    def select_action(self, observation: GameObservation) -> AgentDecision:
        probs = self.policy(observation)
        if self.debug_deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(self.rng.choice(NUM_ACTIONS, p=probs))
        debug = dict(self.last_debug)
        debug["selected_action"] = action
        self.last_debug = debug
        return AgentDecision(action=action, policy=probs.tolist(), debug=debug)

    @staticmethod
    def _normalize(probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        probs = np.maximum(probs, 0.0)
        total = probs.sum()
        if total <= 0:
            return np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
        return probs / total

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = values - np.max(values)
        exp = np.exp(values)
        return exp / exp.sum()

    @staticmethod
    def _confidence(probs: np.ndarray) -> float:
        probs = HybridAdaptiveAgent._normalize(probs)
        max_margin = (float(np.max(probs)) - 1.0 / 3.0) / (2.0 / 3.0)
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12)))) / np.log(3.0)
        entropy_conf = 1.0 - entropy
        return float(np.clip(0.65 * max_margin + 0.35 * entropy_conf, 0.0, 1.0))
