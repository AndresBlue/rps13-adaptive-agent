"""Print evaluation metrics summary for analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_csv(name: str) -> pd.DataFrame | None:
    path = ROOT / "reports" / "metrics" / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show evaluation metrics tables.")
    parser.add_argument("--file", default="evaluation_v2.csv")
    args = parser.parse_args()

    df = load_csv(args.file)
    if df is None:
        print(f"No se encontró reports/metrics/{args.file}")
        sys.exit(1)

    cols = ["bot", "win_rate", "round_win_rate", "average_score_diff", "uniform_deviation", "prediction_accuracy"]
    present = [c for c in cols if c in df.columns]
    print(f"\n=== {args.file} ({int(df['matches'].iloc[0]) if 'matches' in df.columns else '?'} matches/bot) ===\n")
    print(df[present].sort_values("bot").to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if "random" in df["bot"].values:
        rnd = df[df["bot"] == "random"].iloc[0]
        print(f"\nRandom baseline: win_rate={rnd['win_rate']:.3f}, uniform_dev={rnd.get('uniform_deviation', float('nan')):.3f}")

    exploitable = df[~df["bot"].isin(["random", "pseudo_random_biased"])]
    if not exploitable.empty:
        print(f"Win rate medio vs no-aleatorios: {exploitable['win_rate'].mean():.3f}")
        print(f"Win rate mediana vs no-aleatorios: {exploitable['win_rate'].median():.3f}")


if __name__ == "__main__":
    main()
