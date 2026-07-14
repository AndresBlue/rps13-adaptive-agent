from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine internal-format CSV datasets into one shuffled corpus.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default="data/processed/rps_train_full.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    frames = []
    for input_path in args.inputs:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows, {df['match_id'].nunique()} matches to {out}")


if __name__ == "__main__":
    main()
