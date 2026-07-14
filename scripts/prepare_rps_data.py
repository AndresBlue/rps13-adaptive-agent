from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rps13.game.rules import outcome_for_ai, round_winner


MOVE_MAP = {"rock": 0, "paper": 1, "scissors": 2}


def parse_move(value: Any) -> int | None:
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan"}:
        return None
    if text not in MOVE_MAP:
        raise ValueError(f"Unknown move value: {value!r}")
    return MOVE_MAP[text]


def append_segmented_rows(
    rows: list[dict[str, Any]],
    source_rounds: list[dict[str, Any]],
    match_prefix: str,
    player_id: str,
    opponent_type: str,
    human_key: str,
    ai_key: str,
    target_score: int,
) -> None:
    segment = 0
    round_in_segment = 0
    human_score = 0
    ai_score = 0
    for source_round in sorted(source_rounds, key=lambda row: int(row.get("round_index", 0))):
        if human_key not in source_round or ai_key not in source_round:
            continue
        human_move = parse_move(source_round[human_key])
        ai_move = parse_move(source_round[ai_key])
        if human_move is None or ai_move is None:
            continue
        result_for_ai = int(outcome_for_ai(ai_move, human_move))
        if result_for_ai == 1:
            ai_score += 1
        elif result_for_ai == -1:
            human_score += 1
        round_in_segment += 1
        is_terminal = human_score >= target_score or ai_score >= target_score
        rows.append(
            {
                "match_id": f"{match_prefix}_seg{segment:03d}",
                "player_id": player_id,
                "opponent_type": opponent_type,
                "round": round_in_segment,
                "human_move": human_move,
                "ai_move": ai_move,
                "result_for_ai": result_for_ai,
                "human_score": human_score,
                "ai_score": ai_score,
                "round_winner": round_winner(ai_move, human_move),
                "is_terminal": is_terminal,
                "timestamp_or_step": int(source_round.get("round_begin_ts") or source_round.get("round_index", round_in_segment)),
                "human_decision_time_ms": source_round.get("player1_rt") if human_key == "player1_move" else source_round.get("player2_rt"),
                "source_version": source_round.get("version", ""),
            }
        )
        if is_terminal:
            segment += 1
            round_in_segment = 0
            human_score = 0
            ai_score = 0


def convert_rps_json_dataset(
    input_root: Path,
    output_path: Path,
    target_score: int = 13,
    versions: list[str] | None = None,
    rdata_root: Path | None = None,
    include_rdata: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    json_versions = versions or ["v1", "v2", "v3"]
    for version_dir in json_versions:
        data_dir = input_root / version_dir
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.glob("*.json")):
            name = path.name
            if "freeResp" in name or "sliderData" in name:
                continue
            with path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
            rounds = payload.get("rounds")
            if not rounds:
                continue
            game_id = str(payload.get("game_id", path.stem))
            if version_dir == "v1":
                append_segmented_rows(
                    rows,
                    rounds,
                    match_prefix=f"rps_v1_{game_id}_p1",
                    player_id=str(payload.get("player1_id", "player1")),
                    opponent_type="dyad_human_as_player2",
                    human_key="player1_move",
                    ai_key="player2_move",
                    target_score=target_score,
                )
                append_segmented_rows(
                    rows,
                    rounds,
                    match_prefix=f"rps_v1_{game_id}_p2",
                    player_id=str(payload.get("player2_id", "player2")),
                    opponent_type="dyad_human_as_player1",
                    human_key="player2_move",
                    ai_key="player1_move",
                    target_score=target_score,
                )
            else:
                strategy = payload.get("player2_bot_strategy", "bot")
                append_segmented_rows(
                    rows,
                    rounds,
                    match_prefix=f"rps_{version_dir}_{game_id}",
                    player_id=str(payload.get("player1_id", "player1")),
                    opponent_type=f"{version_dir}_{strategy}",
                    human_key="player1_move",
                    ai_key="player2_move",
                    target_score=target_score,
                )
    if include_rdata:
        if rdata_root is None:
            raise ValueError("--include-rdata requires --rdata-root")
        append_rdata_rows(rows, rdata_root, target_score=target_score)
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def append_rdata_rows(rows: list[dict[str, Any]], rdata_root: Path, target_score: int = 13) -> None:
    """Append v2/v3 processed RData rows. Expects one human and one bot row per round."""

    import rdata

    for path in sorted(rdata_root.glob("rps_v*_data.RData")):
        payload = rdata.read_rda(path)
        if "bot_data" not in payload:
            continue
        df = payload["bot_data"].copy()
        df.columns = [str(column) for column in df.columns]
        df = df[df["player_move"].notna()].copy()
        df["is_bot"] = df["is_bot"].astype(float).astype(int)
        version = str(int(float(df["version"].dropna().iloc[0]))) if not df.empty else path.stem
        for game_id, game_df in df.groupby("game_id", sort=False):
            rounds: list[dict[str, Any]] = []
            for _round_index, round_df in game_df.groupby("round_index", sort=True):
                human = round_df[round_df["is_bot"] == 0]
                bot = round_df[round_df["is_bot"] == 1]
                if human.empty or bot.empty:
                    continue
                human_row = human.iloc[0]
                bot_row = bot.iloc[0]
                rounds.append(
                    {
                        "game_id": str(game_id),
                        "version": version,
                        "round_index": int(float(human_row["round_index"])),
                        "player1_id": str(human_row["player_id"]),
                        "player2_id": str(bot_row["player_id"]),
                        "round_begin_ts": human_row.get("round_begin_ts"),
                        "player1_move": human_row["player_move"],
                        "player2_move": bot_row["player_move"],
                        "player1_rt": human_row.get("player_rt"),
                        "player2_rt": bot_row.get("player_rt"),
                    }
                )
            if rounds:
                strategy = str(game_df["bot_strategy"].dropna().iloc[0]) if game_df["bot_strategy"].notna().any() else "bot"
                append_segmented_rows(
                    rows,
                    rounds,
                    match_prefix=f"rps_rdata_v{version}_{game_id}",
                    player_id=str(rounds[0]["player1_id"]),
                    opponent_type=f"rdata_v{version}_{strategy}",
                    human_key="player1_move",
                    ai_key="player2_move",
                    target_score=target_score,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert downloaded RPS JSON data to the internal CSV format.")
    parser.add_argument("--input-root", default="data/rps_data")
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    parser.add_argument("--rdata-root", default=None)
    parser.add_argument("--include-rdata", action="store_true")
    parser.add_argument("--out", default="data/processed/rps_real_matches.csv")
    parser.add_argument("--target-score", type=int, default=13)
    args = parser.parse_args()
    df = convert_rps_json_dataset(
        Path(args.input_root),
        Path(args.out),
        target_score=args.target_score,
        versions=args.versions,
        rdata_root=Path(args.rdata_root) if args.rdata_root else None,
        include_rdata=args.include_rdata,
    )
    print(f"Wrote {len(df)} rows, {df['match_id'].nunique() if not df.empty else 0} matches to {args.out}")
    if not df.empty:
        print(df.groupby("opponent_type").size().sort_values(ascending=False).head(20).to_string())


if __name__ == "__main__":
    main()
