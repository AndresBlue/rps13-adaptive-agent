"""Evaluate agents against the scripted bot population."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rps13.agents.factory import build_agent
from rps13.agents.neural_agent import ActorCriticAgent
from rps13.agents.random_agent import RandomAgent
from rps13.bots.bot_population import BOT_REGISTRY, build_bot
from rps13.evaluation.metrics import policy_uniform_deviation
from rps13.evaluation.plots import plot_evaluation
from rps13.game.env import RPS13Env


def build_eval_agent(agent: str, checkpoint_path: str | None = None, seed: int = 42, config: dict[str, Any] | None = None):
    """Build an agent by CLI name."""

    cfg = dict(config or {})
    if checkpoint_path:
        cfg.setdefault("agent_checkpoint_path", checkpoint_path)
    cfg.setdefault("agent_type", agent)
    cfg.setdefault("agent_seed", seed)

    if agent == "random":
        return RandomAgent()
    if agent in {"hybrid", "mixture"}:
        cfg["agent_type"] = agent
        return build_agent(cfg)
    if agent == "actor_critic":
        if not checkpoint_path:
            raise ValueError("actor_critic evaluation requires --checkpoint")
        return ActorCriticAgent(checkpoint_path)
    raise ValueError(f"Unknown agent: {agent}")


def evaluate_agent(
    agent_name: str = "hybrid",
    matches: int = 1000,
    checkpoint_path: str | None = "models/opponent_predictor.pt",
    out_path: str | Path = "reports/metrics/evaluation.csv",
    figures_dir: str | Path = "reports/figures",
    seed: int = 42,
    target_score: int = 13,
    bot_names: list[str] | None = None,
    agent_config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate one agent against registered bots."""

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    selected_bots = bot_names or list(BOT_REGISTRY)
    for bot_idx, bot_name in enumerate(tqdm(selected_bots, desc="evaluating bots", unit="bot", dynamic_ncols=True)):
        agent = build_eval_agent(agent_name, checkpoint_path=checkpoint_path, seed=seed + bot_idx, config=agent_config)
        wins = draws = round_wins = round_losses = round_draws = 0
        score_diffs: list[int] = []
        lengths: list[int] = []
        deviations: list[float] = []
        pred_hits: list[int] = []
        for _match_idx in tqdm(range(matches), desc=f"{bot_name}", unit="match", dynamic_ncols=True, leave=False):
            bot = build_bot(bot_name, seed=int(rng.integers(0, 2**31 - 1)))
            bot.reset()
            agent.reset()
            env = RPS13Env(target_score=target_score)
            while not env.is_done():
                obs = env.get_observation()
                decision = agent.select_action(obs)
                human_action = bot.choose_action(obs)
                next_obs, _reward, _done, info = env.step(decision.action, human_action)
                if hasattr(agent, "observe_round"):
                    agent.observe_round(human_action)
                bot.observe_round(next_obs.history[-1])
                if info["result_for_ai"] == 1:
                    round_wins += 1
                elif info["result_for_ai"] == -1:
                    round_losses += 1
                else:
                    round_draws += 1
                deviations.append(policy_uniform_deviation(decision.policy))
                predicted = decision.debug.get("predicted_probs")
                if predicted is not None:
                    pred_hits.append(int(np.argmax(predicted) == human_action))
            wins += int(env.winner() == "ai")
            draws += int(env.winner() is None)
            score_diffs.append(env.ai_score - env.human_score)
            lengths.append(len(env.history))
        total_rounds = max(round_wins + round_losses + round_draws, 1)
        rows.append(
            {
                "agent": agent_name,
                "bot": bot_name,
                "matches": matches,
                "win_rate": wins / matches,
                "match_draw_rate": draws / matches,
                "round_win_rate": round_wins / total_rounds,
                "round_loss_rate": round_losses / total_rounds,
                "draw_rate": round_draws / total_rounds,
                "average_score_diff": float(np.mean(score_diffs)),
                "average_match_length": float(np.mean(lengths)),
                "uniform_deviation": float(np.mean(deviations)) if deviations else 0.0,
                "exploitability_proxy_vs_random": float(np.mean(deviations)) if bot_name == "random" else np.nan,
                "prediction_accuracy": float(np.mean(pred_hits)) if pred_hits else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    plot_evaluation(df, figures_dir)
    return df
