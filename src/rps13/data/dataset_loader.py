"""Load and normalize local external RPS datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from rps13.game.rules import outcome_for_ai, round_winner

INTERNAL_COLUMNS = [
    "match_id",
    "player_id",
    "opponent_type",
    "round",
    "human_move",
    "ai_move",
    "result_for_ai",
    "human_score",
    "ai_score",
    "round_winner",
    "is_terminal",
    "timestamp_or_step",
]


def _parse_move(value: Any, value_map: dict[Any, int] | None = None) -> int:
    if value_map and value in value_map:
        return int(value_map[value])
    if pd.isna(value):
        raise ValueError("Move value cannot be null")
    text = str(value).strip().lower()
    aliases = {
        "0": 0,
        "rock": 0,
        "r": 0,
        "piedra": 0,
        "1": 1,
        "paper": 1,
        "p": 1,
        "papel": 1,
        "2": 2,
        "scissors": 2,
        "scissor": 2,
        "s": 2,
        "tijera": 2,
        "tijeras": 2,
    }
    if text not in aliases:
        raise ValueError(f"Cannot parse RPS move value {value!r}")
    return aliases[text]


def normalize_rps_dataset(
    input_path: str | Path,
    output_path: str | Path,
    mapping_config: dict[str, Any],
) -> pd.DataFrame:
    """Normalize a local CSV into the internal round format.

    ``mapping_config`` maps internal column names to source column names. It may
    include ``move_value_map`` to map custom move labels to 0/1/2.
    """

    df = pd.read_csv(input_path)
    move_value_map = mapping_config.get("move_value_map")
    out = pd.DataFrame()
    for internal in ("match_id", "player_id", "opponent_type", "round", "timestamp_or_step"):
        source = mapping_config.get(internal)
        if source and source in df.columns:
            out[internal] = df[source]
    if "match_id" not in out:
        out["match_id"] = "external_0"
    if "player_id" not in out:
        out["player_id"] = "external_human"
    if "opponent_type" not in out:
        out["opponent_type"] = mapping_config.get("opponent_type_value", "external")
    if "round" not in out:
        out["round"] = out.groupby("match_id").cumcount() + 1
    if "timestamp_or_step" not in out:
        out["timestamp_or_step"] = out["round"]

    human_col = mapping_config["human_move"]
    ai_col = mapping_config["ai_move"]
    out["human_move"] = df[human_col].map(lambda value: _parse_move(value, move_value_map))
    out["ai_move"] = df[ai_col].map(lambda value: _parse_move(value, move_value_map))

    result_col = mapping_config.get("result_for_ai")
    if result_col and result_col in df.columns:
        out["result_for_ai"] = df[result_col].astype(int)
    else:
        out["result_for_ai"] = [
            int(outcome_for_ai(ai, human)) for ai, human in zip(out["ai_move"], out["human_move"], strict=True)
        ]
    out["round_winner"] = [
        round_winner(ai, human) for ai, human in zip(out["ai_move"], out["human_move"], strict=True)
    ]

    if mapping_config.get("human_score") in df.columns and mapping_config.get("ai_score") in df.columns:
        out["human_score"] = df[mapping_config["human_score"]].astype(int)
        out["ai_score"] = df[mapping_config["ai_score"]].astype(int)
    else:
        human_scores: list[int] = []
        ai_scores: list[int] = []
        for _match_id, group in out.groupby("match_id", sort=False):
            human_score = 0
            ai_score = 0
            for _, row in group.iterrows():
                if row["result_for_ai"] == 1:
                    ai_score += 1
                elif row["result_for_ai"] == -1:
                    human_score += 1
                human_scores.append(human_score)
                ai_scores.append(ai_score)
        out["human_score"] = human_scores
        out["ai_score"] = ai_scores

    terminal_col = mapping_config.get("is_terminal")
    if terminal_col and terminal_col in df.columns:
        out["is_terminal"] = df[terminal_col].astype(bool)
    else:
        out["is_terminal"] = False
        last_indices = out.groupby("match_id").tail(1).index
        out.loc[last_indices, "is_terminal"] = True

    out = out[INTERNAL_COLUMNS]
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def load_internal_dataset(path: str | Path) -> pd.DataFrame:
    """Load an internal-format dataset with stable dtypes."""

    df = pd.read_csv(path)
    missing = [column for column in INTERNAL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    for column in ("round", "human_move", "ai_move", "result_for_ai", "human_score", "ai_score"):
        df[column] = df[column].astype(int)
    df["is_terminal"] = df["is_terminal"].astype(bool)
    return df
