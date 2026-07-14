from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rps13.evaluation.evaluate_agents import evaluate_agent
from rps13.utils.io import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an RPS-13 agent against bot population.")
    parser.add_argument("--agent", default="hybrid", choices=["hybrid", "mixture", "random", "actor_critic"])
    parser.add_argument("--matches", type=int, default=1000)
    parser.add_argument("--checkpoint", default="models/opponent_predictor.pt")
    parser.add_argument("--out", default="reports/metrics/evaluation.csv")
    parser.add_argument("--config", default=None, help="Optional YAML config (e.g. configs/eval_v2.yaml)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_yaml(args.config) if args.config else {}
    agent = config.get("agent", args.agent)
    matches = int(config.get("matches", args.matches))
    checkpoint = config.get("checkpoint", args.checkpoint)
    out = config.get("out", args.out)
    seed = int(config.get("seed", args.seed))
    bot_names = config.get("bots")

    df = evaluate_agent(
        agent_name=agent,
        matches=matches,
        checkpoint_path=checkpoint,
        out_path=out,
        seed=seed,
        target_score=int(config.get("target_score", 13)),
        bot_names=bot_names,
        agent_config={"agent_type": agent, "agent_checkpoint_path": checkpoint},
    )
    print(df[["bot", "win_rate", "round_win_rate", "average_score_diff", "prediction_accuracy"]].to_string(index=False))
    print(f"Wrote metrics to {out}")


if __name__ == "__main__":
    main()
