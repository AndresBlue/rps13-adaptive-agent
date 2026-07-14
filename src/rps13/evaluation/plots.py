"""Matplotlib evaluation plots."""

from __future__ import annotations

import os
from pathlib import Path

_mpl_config_dir = Path("reports") / ".matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import matplotlib.pyplot as plt
import pandas as pd


def plot_evaluation(df: pd.DataFrame, figures_dir: str | Path) -> None:
    """Create simple evaluation charts."""

    out = Path(figures_dir)
    out.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    ordered = df.sort_values("bot")
    plt.figure(figsize=(10, 4))
    plt.bar(ordered["bot"], ordered["win_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("win rate")
    plt.tight_layout()
    plt.savefig(out / "eval_win_rate_by_bot.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.bar(ordered["bot"], ordered["average_score_diff"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("avg AI-human score diff")
    plt.tight_layout()
    plt.savefig(out / "eval_score_diff_by_bot.png")
    plt.close()

    if "prediction_accuracy" in ordered.columns:
        pred = ordered.dropna(subset=["prediction_accuracy"])
        if not pred.empty:
            plt.figure(figsize=(10, 4))
            plt.bar(pred["bot"], pred["prediction_accuracy"])
            plt.xticks(rotation=45, ha="right")
            plt.ylim(0, 1)
            plt.ylabel("prediction accuracy")
            plt.tight_layout()
            plt.savefig(out / "eval_predictor_accuracy_by_bot.png")
            plt.close()
