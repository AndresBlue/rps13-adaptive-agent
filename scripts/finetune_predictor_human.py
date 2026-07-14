"""Prepare mixed human+synthetic dataset and fine-tune GRU expert."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_mixed_dataset(
    human_csv: Path,
    synthetic_csv: Path,
    out_csv: Path,
    human_weight: float = 2.0,
    max_synthetic_rows: int = 100_000,
) -> Path:
    frames: list[pd.DataFrame] = []
    if synthetic_csv.exists():
        synth = pd.read_csv(synthetic_csv)
        if len(synth) > max_synthetic_rows:
            synth = synth.sample(n=max_synthetic_rows, random_state=42)
        frames.append(synth)
    if human_csv.exists():
        human = pd.read_csv(human_csv)
        if not human.empty:
            repeats = max(1, int(human_weight))
            frames.extend([human] * repeats)
    if not frames:
        raise SystemExit("No datasets available for fine-tuning.")
    mixed = pd.concat(frames, ignore_index=True)
    if "match_id" in mixed.columns:
        mixed = mixed.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    mixed.to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune GRU on human+synthetic mix.")
    parser.add_argument("--human-csv", default="data/processed/human_matches.csv")
    parser.add_argument("--synthetic-csv", default="data/synthetic/synthetic_matches.csv")
    parser.add_argument("--out-csv", default="data/processed/human_synthetic_mix.csv")
    parser.add_argument("--config", default="configs/train_predictor_human.yaml")
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()

    if not args.skip_aggregate:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "aggregate_human_logs.py")],
            check=True,
            cwd=ROOT,
        )

    out = build_mixed_dataset(Path(args.human_csv), Path(args.synthetic_csv), Path(args.out_csv))
    print(f"Mixed dataset -> {out} ({len(pd.read_csv(out))} rows)")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_predictor.py"), "--config", args.config],
        check=True,
        cwd=ROOT,
    )
    print("Fine-tune complete. Use models/opponent_predictor_human_ft.pt as NeuralGRU expert checkpoint.")


if __name__ == "__main__":
    main()
