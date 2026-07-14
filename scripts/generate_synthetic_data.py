from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rps13.data.synthetic_generator import generate_synthetic_matches


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic RPS-13 matches.")
    parser.add_argument("--matches", type=int, default=10000)
    parser.add_argument("--out", default="data/synthetic/synthetic_matches.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = generate_synthetic_matches(matches=args.matches, out_path=args.out, seed=args.seed)
    print(f"Wrote {len(df)} rounds to {args.out}")


if __name__ == "__main__":
    main()
