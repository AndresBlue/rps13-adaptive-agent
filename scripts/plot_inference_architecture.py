"""Generate a detailed inference-only architecture figure for MixtureAdaptiveAgent (v2).

Produces:
  reports/figures/mixture_inference_architecture.png
  reports/figures/mixture_inference_architecture.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

# Visual system
BG = "#F7F4EF"
INK = "#1C1A17"
MUTED = "#5C564C"
ACCENT = "#0B6E4F"
ACCENT2 = "#B85C38"
PANEL = "#FFFFFF"
PANEL_ALT = "#EFEAE2"
NEURAL = "#1F4E79"
META = "#6B4C9A"
GATE = "#8B4513"
FEEDBACK = "#A33B3B"


def _box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    fc: str = PANEL,
    ec: str = INK,
    lw: float = 1.2,
    fontsize: float = 8.5,
    weight: str = "regular",
    color: str = INK,
    ha: str = "center",
    va: str = "center",
    radius: float = 0.02,
    zorder: int = 3,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha=ha,
        va=va,
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        zorder=zorder + 1,
        linespacing=1.25,
        wrap=False,
    )
    return patch


def _arrow(ax, x1: float, y1: float, x2: float, y2: float, *, color: str = INK, lw: float = 1.3, style: str = "-|>") -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=12,
            linewidth=lw,
            color=color,
            zorder=2,
        )
    )


def _section(ax, x: float, y: float, w: float, h: float, title: str, *, ec: str = MUTED, fc: str = PANEL_ALT) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.0,
            linestyle="-",
            zorder=0,
            alpha=0.95,
        )
    )
    ax.text(x + 0.08, y + h - 0.18, title, fontsize=10, fontweight="bold", color=ec, va="top", ha="left", zorder=1)


def _label(ax, x: float, y: float, text: str, *, color: str = MUTED, fontsize: float = 7.5, ha: str = "left") -> None:
    ax.text(x, y, text, fontsize=fontsize, color=color, ha=ha, va="center", zorder=5)


EXPERTS = [
    ("random_hedge", "Uniform U  (safety)"),
    ("sticky_frequency", "Freq windows 5/10/20"),
    ("cycle_detector", "Period 2–6 / ±1 cycle"),
    ("history_match_ppm", "Longest suffix → next"),
    ("markov_lag_1", "Order-1 Markov"),
    ("markov_lag_2", "Order-2 Markov"),
    ("markov_lag_3", "Order-3 Markov"),
    ("outcome_markov", "P(next | move, result)"),
    ("brockbank_self", "Trans. vs own move"),
    ("brockbank_opp", "Trans. vs AI move"),
    ("neural_gru", "OpponentPredictorGRU"),
]


def build_figure(out_dir: Path) -> tuple[Path, Path]:
    steps = [
        "layout",
        "inputs",
        "experts",
        "meta",
        "scoring",
        "policy",
        "gru_detail",
        "feedback",
        "export",
    ]
    bar = tqdm(total=len(steps), desc="Arquitectura inferencia", unit="paso")

    fig = plt.figure(figsize=(22, 14), facecolor=BG)
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.94])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 14)
    ax.axis("off")
    ax.set_facecolor(BG)

    bar.update(1)  # layout
    ax.text(
        11,
        13.55,
        "MixtureAdaptiveAgent  ·  Inferencia (v2 HMOP)",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        11,
        13.15,
        "Solo forward / online adaptation  ·  sin gradientes  ·  π = (1−α)·Nash + α·soft-BR(P̂_humano)",
        ha="center",
        va="center",
        fontsize=10,
        color=MUTED,
    )

    # ── 1. Inputs ──────────────────────────────────────────────────────────
    bar.update(1)
    _section(ax, 0.25, 10.55, 6.4, 2.35, "1. Entrada de decisión", ec=ACCENT)
    _box(
        ax,
        0.45,
        11.55,
        2.9,
        1.05,
        "GameObservation\n"
        "round, scores, history[]\n"
        "RoundRecord(h, a, result)",
        fc="#E8F5EF",
        ec=ACCENT,
        fontsize=8,
        weight="regular",
    )
    _box(
        ax,
        3.5,
        11.55,
        2.9,
        1.05,
        "SessionPlayerMemory\n"
        "prior ≤36 rondas\n"
        "scores/hits (decay 0.72)",
        fc="#FFF1E8",
        ec=ACCENT2,
        fontsize=8,
        weight="regular",
    )
    _box(
        ax,
        1.5,
        10.7,
        3.9,
        0.65,
        "effective_history = prior[-36:]  ∪  match.history",
        fc=PANEL,
        ec=INK,
        fontsize=8,
        weight="bold",
    )
    _arrow(ax, 1.9, 11.55, 2.7, 11.35, color=ACCENT)
    _arrow(ax, 4.95, 11.55, 4.2, 11.35, color=ACCENT2)

    # ── 2. Experts ─────────────────────────────────────────────────────────
    bar.update(1)
    _section(ax, 0.25, 4.35, 6.4, 5.95, "2. Expertos base (11)  →  cada uno P̂ ∈ Δ²", ec=NEURAL)
    expert_y0 = 9.55
    for i, (name, desc) in enumerate(EXPERTS):
        y = expert_y0 - i * 0.45
        is_neural = name == "neural_gru"
        _box(
            ax,
            0.45,
            y,
            5.95,
            0.40,
            f"{i + 1:>2}.  {name:<18}  ·  {desc}",
            fc="#E7EEF6" if is_neural else PANEL,
            ec=NEURAL if is_neural else MUTED,
            lw=1.6 if is_neural else 0.9,
            fontsize=7.6,
            ha="center",
            weight="bold" if is_neural else "regular",
            color=NEURAL if is_neural else INK,
        )

    # ── 3. Meta levels ─────────────────────────────────────────────────────
    bar.update(1)
    _section(ax, 6.95, 7.55, 4.7, 5.35, "3. Meta Iocaine  (×3)", ec=META)
    _box(
        ax,
        7.15,
        11.85,
        4.3,
        0.7,
        "Por cada experto: aplicar meta_levels\n"
        "→ hasta 33 hipótesis  name:meta",
        fc="#F3EEF8",
        ec=META,
        fontsize=8,
        weight="bold",
    )
    metas = [
        ("P.0", "identidad\n[P_R, P_P, P_S]", "#EDE4F5"),
        ("P.1", "BR humano\n[P_S, P_R, P_P]", "#E4D7F0"),
        ("P'.0", "contrafactual\n[P_P, P_S, P_R]", "#D9C8EB"),
    ]
    for i, (title, body, fc) in enumerate(metas):
        x = 7.2 + i * 1.5
        _box(ax, x, 9.85, 1.4, 1.7, f"{title}\n\n{body}", fc=fc, ec=META, fontsize=8, weight="bold")
    _box(
        ax,
        7.15,
        7.75,
        4.3,
        1.85,
        "Salida por hipótesis\n"
        "ExpertPrediction:\n"
        "  · probs ∈ R³\n"
        "  · pattern_stability\n"
        "  · meta_level\n"
        "  · debug{}",
        fc=PANEL,
        ec=META,
        fontsize=8,
    )
    _arrow(ax, 6.65, 7.5, 7.15, 10.5, color=META, lw=1.5)

    # ── 4. Virtual scoring ─────────────────────────────────────────────────
    bar.update(1)
    _section(ax, 6.95, 4.35, 4.7, 2.95, "4. Selección online (virtual scores)", ec=GATE)
    _box(
        ax,
        7.15,
        5.55,
        4.3,
        1.45,
        "score_k ← 0.93·score_k ± 1  (hit/miss)\n"
        "+ 0.35·stability  si stability ≥ 0.72\n"
        "elegir argmax; si ≤ hedge → random_hedge\n"
        "sticky boost: cycle / ppm / sticky_freq",
        fc="#FBF3EB",
        ec=GATE,
        fontsize=8,
        weight="regular",
    )
    _box(
        ax,
        7.15,
        4.55,
        4.3,
        0.8,
        "P̂_humano elegido  ∈ Δ²\n"
        "α_cap ∈ {0.85, 0.95}   min_π ∈ {0.03, 0.01}",
        fc=PANEL,
        ec=GATE,
        fontsize=8,
        weight="bold",
    )
    _arrow(ax, 9.3, 7.75, 9.3, 7.0, color=GATE)

    # ── 5. Policy gate ─────────────────────────────────────────────────────
    bar.update(1)
    _section(ax, 12.0, 4.35, 9.7, 8.55, "5. Soft best-response + mezcla Nash  →  acción AI", ec=ACCENT)
    _box(
        ax,
        12.25,
        11.55,
        4.4,
        1.05,
        "EV = [P_S−P_P,  P_R−P_S,  P_P−P_R]\n"
        "soft_best = softmax(EV / T)   T=0.5",
        fc="#E8F5EF",
        ec=ACCENT,
        fontsize=8.5,
        weight="bold",
    )
    _box(
        ax,
        16.9,
        11.55,
        4.5,
        1.05,
        "confidence =\n"
        "0.45·stability + 0.35·margin + 0.20·(1−H)",
        fc="#FFF8E8",
        ec=ACCENT2,
        fontsize=8.5,
        weight="bold",
    )
    _box(
        ax,
        12.25,
        9.85,
        9.15,
        1.4,
        "α = α_cap · confidence\n"
        "π = (1−α)·U + α·soft_best\n"
        "π ← max(π, min_prob)  →  renormalizar  →  sample a_AI ~ π\n"
        "(debug_deterministic: argmax)",
        fc=PANEL,
        ec=ACCENT,
        lw=1.8,
        fontsize=9,
        weight="bold",
    )
    _arrow(ax, 11.65, 5.0, 12.25, 10.5, color=ACCENT, lw=1.6)

    # Policy math panel
    _box(
        ax,
        12.25,
        7.55,
        9.15,
        2.0,
        "Parámetros de producción (defaults factory / MixtureAdaptiveAgent)\n"
        "agent_type=mixture · checkpoint=opponent_predictor.pt · target=13\n"
        "temperature=0.5 · max_alpha=0.85 · sticky_max_alpha=0.95\n"
        "min_action_prob=0.03 · sticky_min_action_prob=0.01 · score_decay=0.93",
        fc="#F3F8F5",
        ec=MUTED,
        fontsize=8.2,
    )

    # Output
    _box(
        ax,
        14.0,
        5.7,
        5.7,
        1.5,
        "Salida de inferencia\n"
        "AgentDecision(action ∈ {R,P,S}, policy[3], debug)\n"
        "debug: expert_chosen, meta, α, sticky, scores…",
        fc="#E8F5EF",
        ec=ACCENT,
        lw=2.0,
        fontsize=9,
        weight="bold",
    )
    _arrow(ax, 16.8, 9.85, 16.8, 7.2, color=ACCENT, lw=1.8)

    # API loop note
    _box(
        ax,
        12.25,
        4.55,
        9.15,
        0.9,
        "App loop: obs → select_action → env.step(ai, human) → observe_round(human)",
        fc=PANEL,
        ec=MUTED,
        fontsize=8.5,
    )

    # ── 6. GRU detail ──────────────────────────────────────────────────────
    bar.update(1)
    _section(ax, 0.25, 0.25, 14.3, 3.85, "6. Detalle NeuralGRUExpert  ·  OpponentPredictorGRU (solo forward)", ec=NEURAL)
    # Input tensors
    _box(
        ax,
        0.45,
        2.35,
        3.5,
        1.45,
        "build_sequence_arrays\n"
        "T=20  ·  pad=3\n"
        "human_moves [B,T]\n"
        "ai_moves    [B,T]\n"
        "results     [B,T]\n"
        "numeric     [B,T,26]",
        fc="#E7EEF6",
        ec=NEURAL,
        fontsize=7.8,
        weight="regular",
    )
    _box(
        ax,
        4.15,
        2.55,
        2.6,
        1.05,
        "Embeddings\n"
        "act 4→8  ·  AI 4→8\n"
        "result 4→4",
        fc=PANEL,
        ec=NEURAL,
        fontsize=7.8,
    )
    _box(
        ax,
        6.95,
        2.55,
        2.5,
        1.05,
        "concat\n"
        "input_dim=46\n"
        "[B, T, 46]",
        fc=PANEL,
        ec=NEURAL,
        fontsize=8,
        weight="bold",
    )
    _box(
        ax,
        9.65,
        2.55,
        2.4,
        1.05,
        "GRU\n"
        "hidden=96\n"
        "layers=1",
        fc="#D6E4F0",
        ec=NEURAL,
        fontsize=8.5,
        weight="bold",
    )
    _box(
        ax,
        12.25,
        2.35,
        2.05,
        1.45,
        "último t\n"
        "válido\n"
        "[B, 96]",
        fc=PANEL,
        ec=NEURAL,
        fontsize=8,
    )
    _arrow(ax, 3.95, 3.05, 4.15, 3.05, color=NEURAL)
    _arrow(ax, 6.75, 3.05, 6.95, 3.05, color=NEURAL)
    _arrow(ax, 9.45, 3.05, 9.65, 3.05, color=NEURAL)
    _arrow(ax, 12.05, 3.05, 12.25, 3.05, color=NEURAL)

    _box(
        ax,
        0.45,
        0.45,
        9.0,
        1.65,
        "Head: LayerNorm(96) → Linear(96→96) → ReLU → Dropout(0.1) → Linear(96→3)\n"
        "softmax → P̂(human next) ∈ R³\n"
        "26 features: score norms, freqs (all/3/5/10), streak, win/loss/draw rates,\n"
        "beats_last_ai, copy_last_ai, cycle_rate, reverse_cycle_rate",
        fc="#E7EEF6",
        ec=NEURAL,
        fontsize=7.8,
    )
    _box(
        ax,
        9.7,
        0.45,
        4.6,
        1.65,
        "Checkpoint\n"
        "models/opponent_predictor.pt\n"
        "eval() · torch.no_grad()\n"
        "→ NeuralGRUExpert.predict",
        fc=PANEL,
        ec=NEURAL,
        fontsize=8,
        weight="bold",
    )
    _arrow(ax, 13.25, 2.35, 12.0, 2.1, color=NEURAL)

    # ── 7. Feedback loop ───────────────────────────────────────────────────
    bar.update(1)
    _section(ax, 14.8, 0.25, 6.9, 3.85, "7. Adaptación post-reveal (online)", ec=FEEDBACK)
    _box(
        ax,
        15.05,
        2.55,
        6.4,
        1.2,
        "observe_round(human_move)\n"
        "para cada hipótesis en last_predictions:\n"
        "hit = argmax(P)==human  →  score ±1 · decay 0.93",
        fc="#FCECEC",
        ec=FEEDBACK,
        fontsize=8,
        weight="regular",
    )
    _box(
        ax,
        15.05,
        1.35,
        6.4,
        0.95,
        "recent_hits ventana=6  ·  sticky si ≥4 hits\n"
        "o pattern_stability ≥ 0.72",
        fc=PANEL,
        ec=FEEDBACK,
        fontsize=8,
    )
    _box(
        ax,
        15.05,
        0.45,
        6.4,
        0.7,
        "Fin de match → export scores/hits → SessionPlayerMemory",
        fc="#FCECEC",
        ec=FEEDBACK,
        fontsize=8,
        weight="bold",
    )

    # Cross arrows: neural expert highlight to GRU section
    ax.annotate(
        "",
        xy=(3.4, 4.35),
        xytext=(3.4, 4.75),
        arrowprops=dict(arrowstyle="-|>", color=NEURAL, lw=1.4),
    )
    _label(ax, 3.55, 4.55, "detalle GRU ↓", color=NEURAL, fontsize=7.5)

    # Feedback from output to scoring
    ax.annotate(
        "feedback online",
        xy=(11.65, 5.9),
        xytext=(15.0, 3.75),
        fontsize=7.5,
        color=FEEDBACK,
        arrowprops=dict(arrowstyle="-|>", color=FEEDBACK, lw=1.3, connectionstyle="arc3,rad=0.25"),
    )

    # Legend / scope note
    ax.add_patch(Rectangle((0.25, 13.85), 0.01, 0.01, visible=False))
    ax.text(
        21.7,
        0.08,
        "Fuera de figura: HybridAdaptiveAgent (v1) · ActorCriticAgent · entrenamiento supervisado/RL",
        ha="right",
        va="bottom",
        fontsize=7,
        color=MUTED,
        style="italic",
    )

    # ── Export ─────────────────────────────────────────────────────────────
    bar.update(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "mixture_inference_architecture.png"
    pdf = out_dir / "mixture_inference_architecture.pdf"
    fig.savefig(png, dpi=200, facecolor=BG, bbox_inches="tight")
    fig.savefig(pdf, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    bar.close()
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MixtureAdaptiveAgent inference architecture.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "reports" / "figures",
        help="Output directory for PNG/PDF",
    )
    args = parser.parse_args()
    png, pdf = build_figure(args.out_dir)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
