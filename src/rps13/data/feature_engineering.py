"""Feature engineering for logs, bots and neural model inputs."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from typing import Any

import numpy as np

from rps13.constants import PAD_ACTION, PAD_RESULT_INDEX, Result
from rps13.game.rules import best_response_to, next_cycle_action, previous_cycle_action

HistoryItem = Any

NUMERIC_FEATURE_KEYS = [
    "score_diff_norm",
    "round_number_norm",
    "human_score_norm",
    "ai_score_norm",
    "human_freq_rock",
    "human_freq_paper",
    "human_freq_scissors",
    "recent3_rock",
    "recent3_paper",
    "recent3_scissors",
    "recent5_rock",
    "recent5_paper",
    "recent5_scissors",
    "recent10_rock",
    "recent10_paper",
    "recent10_scissors",
    "streak_norm",
    "repeat_after_win",
    "change_after_loss",
    "change_after_draw",
    "beats_last_ai_rate",
    "copy_last_ai_rate",
    "cycle_rate",
    "reverse_cycle_rate",
]


def _get(item: HistoryItem, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _safe_rate(num: int, den: int, default: float = 1.0 / 3.0) -> float:
    return float(num / den) if den else default


def _move_counts(moves: Sequence[int]) -> list[float]:
    total = len(moves)
    return [_safe_rate(sum(1 for move in moves if move == action), total) for action in range(3)]


def current_streak(history: Sequence[HistoryItem]) -> int:
    """Return signed current streak from the AI perspective."""

    if not history:
        return 0
    last_result = int(_get(history[-1], "result_for_ai", 0))
    if last_result == 0:
        return 0
    streak = 0
    for record in reversed(history):
        result = int(_get(record, "result_for_ai", 0))
        if result != last_result:
            break
        streak += 1
    return streak if last_result == int(Result.WIN) else -streak


def compute_history_features(
    history: Sequence[HistoryItem],
    target_score: int = 13,
) -> dict[str, float | int]:
    """Compute cumulative and pattern features from completed rounds."""

    if not history:
        base: dict[str, float | int] = {
            "round_number": 1,
            "human_score": 0,
            "ai_score": 0,
            "score_diff": 0,
            "last_human_move": PAD_ACTION,
            "last_ai_move": PAD_ACTION,
            "streak": 0,
        }
        for key in NUMERIC_FEATURE_KEYS:
            base[key] = 0.0 if key not in {"human_freq_rock", "human_freq_paper", "human_freq_scissors"} else 1.0 / 3.0
        for window in (3, 5, 10):
            for action_name in ("rock", "paper", "scissors"):
                base[f"recent{window}_{action_name}"] = 1.0 / 3.0
        return base

    last = history[-1]
    human_moves = [int(_get(record, "human_move")) for record in history]
    ai_moves = [int(_get(record, "ai_move")) for record in history]
    human_score = int(_get(last, "human_score", 0))
    ai_score = int(_get(last, "ai_score", 0))
    features: dict[str, float | int] = {
        "round_number": len(history) + 1,
        "human_score": human_score,
        "ai_score": ai_score,
        "score_diff": ai_score - human_score,
        "last_human_move": human_moves[-1],
        "last_ai_move": ai_moves[-1],
        "streak": current_streak(history),
    }
    features["score_diff_norm"] = (ai_score - human_score) / max(target_score, 1)
    features["round_number_norm"] = min(len(history), target_score * 3) / max(target_score * 3, 1)
    features["human_score_norm"] = human_score / max(target_score, 1)
    features["ai_score_norm"] = ai_score / max(target_score, 1)
    features["streak_norm"] = float(features["streak"]) / max(target_score, 1)

    all_freq = _move_counts(human_moves)
    for name, value in zip(("rock", "paper", "scissors"), all_freq, strict=True):
        features[f"human_freq_{name}"] = value
    for window in (3, 5, 10):
        freq = _move_counts(human_moves[-window:])
        for name, value in zip(("rock", "paper", "scissors"), freq, strict=True):
            features[f"recent{window}_{name}"] = value

    repeat_after_win = change_after_loss = change_after_draw = 0
    win_count = loss_count = draw_count = 0
    beats_last_ai = copy_last_ai = cycle = reverse_cycle = 0
    transitions = max(len(history) - 1, 0)
    for idx in range(1, len(history)):
        prev = history[idx - 1]
        cur = history[idx]
        prev_human = int(_get(prev, "human_move"))
        cur_human = int(_get(cur, "human_move"))
        prev_ai = int(_get(prev, "ai_move"))
        prev_result = int(_get(prev, "result_for_ai"))
        if prev_result == int(Result.LOSS):
            win_count += 1
            repeat_after_win += int(cur_human == prev_human)
        elif prev_result == int(Result.WIN):
            loss_count += 1
            change_after_loss += int(cur_human != prev_human)
        else:
            draw_count += 1
            change_after_draw += int(cur_human != prev_human)
        beats_last_ai += int(cur_human == int(best_response_to(prev_ai)))
        copy_last_ai += int(cur_human == prev_ai)
        cycle += int(cur_human == int(next_cycle_action(prev_human)))
        reverse_cycle += int(cur_human == int(previous_cycle_action(prev_human)))

    features["repeat_after_win"] = _safe_rate(repeat_after_win, win_count, 0.0)
    features["change_after_loss"] = _safe_rate(change_after_loss, loss_count, 0.0)
    features["change_after_draw"] = _safe_rate(change_after_draw, draw_count, 0.0)
    features["beats_last_ai_rate"] = _safe_rate(beats_last_ai, transitions, 0.0)
    features["copy_last_ai_rate"] = _safe_rate(copy_last_ai, transitions, 0.0)
    features["cycle_rate"] = _safe_rate(cycle, transitions, 0.0)
    features["reverse_cycle_rate"] = _safe_rate(reverse_cycle, transitions, 0.0)

    for key in NUMERIC_FEATURE_KEYS:
        value = float(features.get(key, 0.0))
        features[key] = value if isfinite(value) else 0.0
    return features


def numeric_feature_vector(history: Sequence[HistoryItem], target_score: int = 13) -> np.ndarray:
    """Return the fixed-order numeric feature vector for the current state."""

    features = compute_history_features(history, target_score=target_score)
    return np.asarray([float(features[key]) for key in NUMERIC_FEATURE_KEYS], dtype=np.float32)


def encode_result_index(result_for_ai: int) -> int:
    """Map result values -1/0/1 to embedding indices 0/1/2."""

    if int(result_for_ai) == int(Result.LOSS):
        return 0
    if int(result_for_ai) == int(Result.DRAW):
        return 1
    if int(result_for_ai) == int(Result.WIN):
        return 2
    return PAD_RESULT_INDEX


def build_sequence_arrays(
    history: Sequence[HistoryItem],
    sequence_length: int,
    target_score: int = 13,
) -> dict[str, np.ndarray]:
    """Build padded neural input arrays from prior completed rounds."""

    selected = list(history)[-sequence_length:]
    pad = sequence_length - len(selected)
    human_moves = np.full(sequence_length, PAD_ACTION, dtype=np.int64)
    ai_moves = np.full(sequence_length, PAD_ACTION, dtype=np.int64)
    results = np.full(sequence_length, PAD_RESULT_INDEX, dtype=np.int64)
    numeric = np.zeros((sequence_length, len(NUMERIC_FEATURE_KEYS)), dtype=np.float32)
    prefix: list[HistoryItem] = []
    for idx, record in enumerate(selected, start=pad):
        prefix.append(record)
        human_moves[idx] = int(_get(record, "human_move", PAD_ACTION))
        ai_moves[idx] = int(_get(record, "ai_move", PAD_ACTION))
        results[idx] = encode_result_index(int(_get(record, "result_for_ai", 0)))
        numeric[idx] = numeric_feature_vector(prefix, target_score=target_score)
    return {
        "human_moves": human_moves,
        "ai_moves": ai_moves,
        "results": results,
        "numeric": numeric,
        "length": np.asarray(len(selected), dtype=np.int64),
    }
