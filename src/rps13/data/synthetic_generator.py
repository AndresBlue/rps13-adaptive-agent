"""Synthetic match generation against the bot population."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from rps13.agents.random_agent import RandomAgent
from rps13.bots.bot_population import DEFAULT_BOT_NAMES, build_bot
from rps13.game.env import RPS13Env


def generate_synthetic_matches(
    matches: int,
    out_path: str | Path,
    bot_names: list[str] | None = None,
    seed: int = 42,
    target_score: int = 13,
) -> pd.DataFrame:
    """Generate synthetic RPS-13 matches and save them as CSV."""

    rng = np.random.default_rng(seed)
    names = bot_names or DEFAULT_BOT_NAMES
    rows: list[dict[str, Any]] = []
    ai_agent = RandomAgent()
    for match_idx in tqdm(range(matches), desc="generating synthetic matches", unit="match", dynamic_ncols=True):
        bot_name = str(rng.choice(names))
        bot = build_bot(bot_name, seed=int(rng.integers(0, 2**31 - 1)))
        env = RPS13Env(target_score=target_score)
        bot.reset()
        ai_agent.reset()
        match_id = f"synthetic_{match_idx:08d}"
        while not env.is_done():
            obs = env.get_observation()
            ai_policy = ai_agent.policy(obs)
            ai_move = int(rng.choice(3, p=ai_policy))
            human_move = bot.choose_action(obs)
            next_obs, _reward, _done, info = env.step(ai_move, human_move)
            bot.observe_round(next_obs.history[-1])
            rows.append(
                {
                    "match_id": match_id,
                    "player_id": "synthetic_human",
                    "opponent_type": bot_name,
                    "round": info["round"],
                    "human_move": info["human_move"],
                    "ai_move": info["ai_move"],
                    "result_for_ai": info["result_for_ai"],
                    "human_score": info["human_score"],
                    "ai_score": info["ai_score"],
                    "round_winner": info["round_winner"],
                    "is_terminal": info["is_terminal"],
                    "timestamp_or_step": info["timestamp_or_step"],
                    "human_decision_time_ms": None,
                    "ai_policy_rock": float(ai_policy[0]),
                    "ai_policy_paper": float(ai_policy[1]),
                    "ai_policy_scissors": float(ai_policy[2]),
                    "predictor_confidence": 0.0,
                }
            )
    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df
