"""Mixture-of-experts adaptive agent with Iocaine-style virtual scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rps13.agents.base import AgentDecision, BaseAgent
from rps13.constants import NUM_ACTIONS
from rps13.agents.session_memory import SessionPlayerMemory
from rps13.experts.base import BaseExpert, ExpertPrediction
from rps13.experts.predictors import apply_meta_level, build_default_experts
from rps13.game.state import GameObservation


class MixtureAdaptiveAgent(BaseAgent):
    """Ensemble agent: multiple experts + decayed virtual scoring + Nash gate."""

    STICKY_EXPERTS = {"cycle_detector", "history_match_ppm", "sticky_frequency"}

    def __init__(
        self,
        experts: list[BaseExpert] | None = None,
        checkpoint_path: str | Path | None = None,
        target_score: int = 13,
        temperature: float = 0.5,
        max_alpha: float = 0.85,
        sticky_max_alpha: float = 0.95,
        min_action_prob: float = 0.03,
        sticky_min_action_prob: float = 0.01,
        score_decay: float = 0.93,
        sticky_hit_window: int = 6,
        sticky_hit_threshold: int = 4,
        sticky_stability_threshold: float = 0.72,
        meta_levels: tuple[str, ...] = ("P.0", "P.1", "P'.0"),
        seed: int | None = None,
        debug_deterministic: bool = False,
    ) -> None:
        self.target_score = target_score
        self.temperature = temperature
        self.max_alpha = max_alpha
        self.sticky_max_alpha = sticky_max_alpha
        self.min_action_prob = min_action_prob
        self.sticky_min_action_prob = sticky_min_action_prob
        self.score_decay = score_decay
        self.sticky_hit_window = sticky_hit_window
        self.sticky_hit_threshold = sticky_hit_threshold
        self.sticky_stability_threshold = sticky_stability_threshold
        self.meta_levels = meta_levels
        self.rng = np.random.default_rng(seed)
        self.debug_deterministic = debug_deterministic
        self.experts = experts or build_default_experts(checkpoint_path=checkpoint_path, target_score=target_score)
        self.scores: dict[str, float] = {}
        self.recent_hits: dict[str, list[int]] = {}
        self.session_prior_history: list = []
        self.session_matches_seen: int = 0
        self.session_prior_max: int = 36
        self.session_score_decay: float = 0.72
        self.last_predictions: list[ExpertPrediction] = []
        self.last_debug: dict[str, Any] = {}

    def reset(self) -> None:
        """Reset per-match state only (session prior is kept separately)."""

        self.scores = {}
        self.recent_hits = {}
        self.last_predictions = []
        self.last_debug = {}
        for expert in self.experts:
            expert.reset()

    def seed_from_session(self, memory: SessionPlayerMemory) -> None:
        """Warm-start the agent from prior matches in this browser session."""

        self.session_prior_history = list(memory.prior_rounds)
        self.session_matches_seen = memory.matches_played
        self.scores = {key: value * self.session_score_decay for key, value in memory.expert_scores.items()}
        self.recent_hits = {key: list(hits) for key, hits in memory.expert_hits.items()}
        self.last_debug = {
            "session_seeded": True,
            "session_matches_seen": self.session_matches_seen,
            "session_prior_rounds": len(self.session_prior_history),
        }

    def export_session_learning(self) -> tuple[dict[str, float], dict[str, list[int]]]:
        """Return expert state to archive into session memory."""

        return dict(self.scores), {key: list(hits) for key, hits in self.recent_hits.items()}

    def _effective_observation(self, observation: GameObservation) -> GameObservation:
        if not self.session_prior_history:
            return observation
        prior = self.session_prior_history[-self.session_prior_max :]
        merged = prior + list(observation.history)
        return GameObservation(
            target_score=observation.target_score,
            round_number=observation.round_number,
            human_score=observation.human_score,
            ai_score=observation.ai_score,
            done=observation.done,
            history=merged,
            variant=observation.variant,
            human_bet_tokens=observation.human_bet_tokens,
            ai_bet_tokens=observation.ai_bet_tokens,
        )

    def _expert_key(self, prediction: ExpertPrediction) -> str:
        return f"{prediction.name}:{prediction.meta_level}"

    def observe_round(self, human_move: int) -> None:
        """Update virtual expert scores after the human move is revealed."""

        for prediction in self.last_predictions:
            key = self._expert_key(prediction)
            hit = int(np.argmax(prediction.probs) == int(human_move))
            self.recent_hits.setdefault(key, []).append(hit)
            self.recent_hits[key] = self.recent_hits[key][-self.sticky_hit_window :]
            delta = 1.0 if hit else -1.0
            self.scores[key] = self.scores.get(key, 0.0) * self.score_decay + delta

    def _collect_predictions(self, observation: GameObservation) -> list[ExpertPrediction]:
        effective = self._effective_observation(observation)
        predictions: list[ExpertPrediction] = []
        for expert in self.experts:
            base = expert.predict(effective)
            for meta in self.meta_levels:
                meta_probs = apply_meta_level(base.probs, meta)
                predictions.append(
                    ExpertPrediction(
                        name=base.name,
                        probs=meta_probs,
                        pattern_stability=base.pattern_stability,
                        meta_level=meta,
                        debug=dict(base.debug),
                    )
                )
        return predictions

    def _select_prediction(self, predictions: list[ExpertPrediction]) -> ExpertPrediction:
        hedge_score = self.scores.get("random_hedge:P.0", 0.0)
        best = predictions[0]
        best_score = float("-inf")
        for prediction in predictions:
            key = self._expert_key(prediction)
            score = self.scores.get(key, 0.0)
            if prediction.pattern_stability >= self.sticky_stability_threshold:
                score += 0.35 * prediction.pattern_stability
            if score > best_score:
                best_score = score
                best = prediction
        if best_score <= hedge_score and best.name != "random_hedge":
            for prediction in predictions:
                if prediction.name == "random_hedge":
                    return prediction
        return best

    def _sticky_boost(self, prediction: ExpertPrediction) -> tuple[float, float, bool]:
        key = self._expert_key(prediction)
        hits = self.recent_hits.get(key, [])
        recent_hit_rate = float(np.mean(hits)) if hits else 0.0
        sticky = (
            prediction.name in self.STICKY_EXPERTS
            and (
                prediction.pattern_stability >= self.sticky_stability_threshold
                or (
                    len(hits) >= self.sticky_hit_threshold
                    and sum(hits[-self.sticky_hit_window :]) >= self.sticky_hit_threshold
                )
            )
        )
        alpha_cap = self.sticky_max_alpha if sticky else self.max_alpha
        min_prob = self.sticky_min_action_prob if sticky else self.min_action_prob
        return alpha_cap, min_prob, sticky

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

    def _confidence(self, probs: np.ndarray, pattern_stability: float) -> tuple[float, float, float]:
        probs = self._normalize(probs)
        max_margin = (float(np.max(probs)) - 1.0 / 3.0) / (2.0 / 3.0)
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12)))) / np.log(3.0)
        prediction_margin = float(np.clip(max_margin, 0.0, 1.0))
        entropy_conf = 1.0 - entropy
        blended = 0.45 * pattern_stability + 0.35 * prediction_margin + 0.20 * entropy_conf
        return float(np.clip(blended, 0.0, 1.0)), pattern_stability, prediction_margin

    def policy(self, observation: GameObservation) -> np.ndarray:
        predictions = self._collect_predictions(observation)
        self.last_predictions = predictions
        chosen = self._select_prediction(predictions)
        human_probs = self._normalize(chosen.probs)
        alpha_cap, min_prob, sticky = self._sticky_boost(chosen)

        ev = np.array(
            [
                human_probs[2] - human_probs[1],
                human_probs[0] - human_probs[2],
                human_probs[1] - human_probs[0],
            ],
            dtype=np.float64,
        )
        soft_best = self._softmax(ev / max(self.temperature, 1e-6))
        confidence, pattern_stability, prediction_margin = self._confidence(human_probs, chosen.pattern_stability)
        alpha = alpha_cap * confidence
        uniform = np.ones(NUM_ACTIONS, dtype=np.float64) / NUM_ACTIONS
        mixed = (1.0 - alpha) * uniform + alpha * soft_best
        if not self.debug_deterministic:
            mixed = np.maximum(mixed, min_prob)
        mixed = self._normalize(mixed)

        self.last_debug = {
            "source": "mixture_adaptive",
            "expert_chosen": chosen.name,
            "meta_level": chosen.meta_level,
            "predicted_probs": human_probs.tolist(),
            "expected_values": ev.tolist(),
            "confidence": confidence,
            "pattern_stability": pattern_stability,
            "prediction_margin": prediction_margin,
            "alpha": float(alpha),
            "sticky_boost": sticky,
            "pattern_flags": {
                "sticky": sticky,
                "cycle_expert_active": chosen.name == "cycle_detector",
                "ppm_expert_active": chosen.name == "history_match_ppm",
                "session_prior_rounds": len(self.session_prior_history),
                "session_matches_seen": self.session_matches_seen,
            },
            "expert_scores": {k: round(v, 3) for k, v in sorted(self.scores.items())},
            "chosen_policy": mixed.tolist(),
            "reasoning_debug": "sticky_overexploit" if sticky else "mixture_best_expert",
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
