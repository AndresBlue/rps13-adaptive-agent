"""Aggregate human JSONL logs into normalized CSV and summary stats."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def aggregate_logs(log_dir: Path, out_csv: Path, out_summary: Path) -> pd.DataFrame:
    rows: list[dict] = []
    match_rows: list[dict] = []
    files = sorted(log_dir.glob("*.jsonl"))
    for path in tqdm(files, desc="Procesando logs", unit="archivo", dynamic_ncols=True):
        match_id = path.stem
        rounds = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            event = row.get("event", "round")
            if event == "match_end":
                match_rows.append(row)
                continue
            if "human_move" not in row:
                continue
            rows.append(
                {
                    "match_id": row.get("match_id", match_id),
                    "session_id": row.get("session_id"),
                    "round": row.get("round", rounds + 1),
                    "human_move": row["human_move"],
                    "ai_move": row.get("ai_move"),
                    "result_for_ai": row.get("result_for_ai"),
                    "human_score": row.get("human_score"),
                    "ai_score": row.get("ai_score"),
                    "agent_version": row.get("agent_version"),
                    "agent_type": row.get("agent_type"),
                    "expert_chosen": row.get("expert_chosen"),
                    "timestamp": row.get("timestamp"),
                }
            )
            rounds += 1

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    completed = [m for m in match_rows if m.get("event") == "match_end"]
    incomplete = len(files) - len({m.get("match_id") for m in completed})
    human_wins = sum(1 for m in completed if m.get("winner") == "human")
    ai_wins = sum(1 for m in completed if m.get("winner") == "ai")
    summary = {
        "log_files": len(files),
        "round_rows": len(df),
        "completed_matches": len(completed),
        "incomplete_matches_estimate": max(0, incomplete),
        "human_win_rate": human_wins / max(len(completed), 1),
        "ai_win_rate": ai_wins / max(len(completed), 1),
        "avg_rounds_completed": float(pd.Series([m.get("rounds_played", 0) for m in completed]).mean())
        if completed
        else 0.0,
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate human match logs.")
    parser.add_argument("--log-dir", default="data/human_logs")
    parser.add_argument("--out-csv", default="data/processed/human_matches.csv")
    parser.add_argument("--out-summary", default="reports/metrics/human_logs_summary.json")
    args = parser.parse_args()
    df = aggregate_logs(Path(args.log_dir), Path(args.out_csv), Path(args.out_summary))
    print(f"Wrote {len(df)} round rows to {args.out_csv}")
    print(f"Summary -> {args.out_summary}")


if __name__ == "__main__":
    main()
