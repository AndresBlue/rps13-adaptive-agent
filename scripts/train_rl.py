from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rps13.training.train_rl import train_rl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the recurrent actor-critic baseline.")
    parser.add_argument("--config", default="configs/train_rl.yaml")
    args = parser.parse_args()
    metrics = train_rl(args.config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
