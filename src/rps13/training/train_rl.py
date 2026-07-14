"""Functional recurrent actor-critic training loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical
from tqdm.auto import tqdm

from rps13.bots.bot_population import build_bot_population
from rps13.data.feature_engineering import NUMERIC_FEATURE_KEYS, build_sequence_arrays
from rps13.game.env import RPS13Env
from rps13.models.actor_critic import ActorCriticGRU
from rps13.training.buffers import RolloutStep, discounted_returns
from rps13.training.losses import actor_critic_loss
from rps13.utils.io import ensure_parent, load_yaml, write_json
from rps13.utils.seed import set_global_seed


def _resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    require_cuda = bool(config.get("require_cuda", False))
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests CUDA, but this environment has CPU-only PyTorch. "
            "Install a CUDA-enabled torch build before RL training."
        )
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Config require_cuda=true, but CUDA is not available.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return device


def train_rl(config_path: str | Path) -> dict[str, Any]:
    """Train a simple recurrent actor-critic against the configured bot population."""

    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    target_score = int(config.get("target_score", 13))
    sequence_length = int(config.get("sequence_length", 20))
    bots = build_bot_population(config.get("bot_population"), seed=seed)
    device = _resolve_device(config)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = ActorCriticGRU(
        hidden_dim=int(config.get("hidden_dim", 96)),
        num_layers=int(config.get("num_layers", 1)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 5e-4)))
    gamma = float(config.get("gamma", 0.97))
    value_coef = float(config.get("value_coefficient", 0.5))
    entropy_coef = float(config.get("entropy_coefficient", 0.02))
    episode_rewards: list[float] = []
    win_history: list[int] = []
    num_episodes = int(config.get("num_episodes", 2000))

    progress = tqdm(range(1, num_episodes + 1), desc="training rl", unit="episode", dynamic_ncols=True)
    for episode in progress:
        bot = bots[int(rng.integers(0, len(bots)))]
        bot.reset()
        env = RPS13Env(target_score=target_score)
        steps: list[RolloutStep] = []
        done = False
        while not done:
            obs = env.get_observation()
            arrays = build_sequence_arrays(obs.history, sequence_length, target_score)
            logits, value, _hidden = model(
                torch.as_tensor(arrays["human_moves"], dtype=torch.long, device=device).unsqueeze(0),
                torch.as_tensor(arrays["ai_moves"], dtype=torch.long, device=device).unsqueeze(0),
                torch.as_tensor(arrays["results"], dtype=torch.long, device=device).unsqueeze(0),
                torch.as_tensor(arrays["numeric"], dtype=torch.float32, device=device).unsqueeze(0),
                torch.as_tensor([max(int(arrays["length"]), 1)], dtype=torch.long, device=device),
            )
            dist = Categorical(logits=logits)
            action = dist.sample()
            human_action = bot.choose_action(obs)
            next_obs, reward, done, _info = env.step(int(action.item()), human_action)
            bot.observe_round(next_obs.history[-1])
            steps.append(
                RolloutStep(
                    log_prob=dist.log_prob(action).squeeze(0),
                    value=value.squeeze(0),
                    entropy=dist.entropy().squeeze(0),
                    reward=reward,
                )
            )
        returns = discounted_returns([step.reward for step in steps], gamma).to(device)
        log_probs = torch.stack([step.log_prob for step in steps])
        values = torch.stack([step.value for step in steps])
        entropies = torch.stack([step.entropy for step in steps])
        loss = actor_critic_loss(log_probs, values, entropies, returns, value_coef, entropy_coef)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        episode_rewards.append(float(sum(step.reward for step in steps)))
        win_history.append(1 if env.winner() == "ai" else 0)
        if episode == 1 or episode % 10 == 0:
            progress.set_postfix(
                reward=f"{np.mean(episode_rewards[-50:]):.3f}",
                win_rate=f"{np.mean(win_history[-50:]):.3f}",
            )

    checkpoint_path = ensure_parent(config.get("checkpoint_path", "models/actor_critic.pt"))
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "model_config": {
                "numeric_dim": len(NUMERIC_FEATURE_KEYS),
                "hidden_dim": int(config.get("hidden_dim", 96)),
                "num_layers": int(config.get("num_layers", 1)),
            },
            "sequence_length": sequence_length,
        },
        checkpoint_path,
    )
    metrics = {
        "episodes": len(episode_rewards),
        "mean_episode_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "last_100_reward": float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0,
        "win_rate": float(np.mean(win_history)) if win_history else 0.0,
        "last_100_win_rate": float(np.mean(win_history[-100:])) if win_history else 0.0,
        "note": "Simple recurrent actor-critic baseline; PPO clipping/GAE can be added here.",
    }
    write_json("reports/metrics/rl_metrics.json", metrics)
    return metrics
