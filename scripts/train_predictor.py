from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rps13.training.train_predictor import train_predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the GRU opponent predictor.")
    parser.add_argument("--config", default="configs/train_predictor.yaml")
    args = parser.parse_args()
    metrics = train_predictor(args.config)
    print(json.dumps(metrics["final"], indent=2))


if __name__ == "__main__":
    main()
