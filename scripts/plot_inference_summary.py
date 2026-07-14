"""Generate a summary flowchart of MixtureAdaptiveAgent inference.

Produces:
  reports/figures/mixture_inference_summary.png
  reports/figures/mixture_inference_summary.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

BG = "#F7F4EF"
INK = "#1C1A17"
MUTED = "#5C564C"
PANEL = "#FFFFFF"
ACCENT = "#0B6E4F"
NEURAL = "#1F4E79"
META = "#6B4C9A"
GATE = "#8B4513"
FEEDBACK = "#A33B3B"
SOFT = "#B85C38"


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
    lw: float = 1.4,
    fontsize: float = 10,
    weight: str = "regular",
    color: str = INK,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.015,rounding_size=0.04",
            facecolor=fc,
            edgecolor=ec,
            linewidth=lw,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=color,
        zorder=4,
        linespacing=1.3,
    )


def _arrow(ax, x1: float, y1: float, x2: float, y2: float, *, color: str = INK, lw: float = 1.8) -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=lw,
            color=color,
            zorder=2,
        )
    )


def build_figure(out_dir: Path) -> tuple[Path, Path]:
    steps = ["canvas", "pipeline", "feedback", "export"]
    bar = tqdm(total=len(steps), desc="Resumen inferencia", unit="paso")

    fig = plt.figure(figsize=(16, 7.2), facecolor=BG)
    ax = fig.add_axes([0.03, 0.06, 0.94, 0.88])
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    ax.set_facecolor(BG)

    bar.update(1)
    ax.text(
        8,
        6.85,
        "Inferencia del agente adaptativo — resumen",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        8,
        6.4,
        "MixtureAdaptiveAgent (v2)  ·  estimar P̂(humano) → explotar con soft-BR → mezclar con Nash",
        ha="center",
        va="center",
        fontsize=11,
        color=MUTED,
    )

    # Horizontal pipeline of 5 main steps
    bar.update(1)
    boxes = [
        (
            0.35,
            "1. Observar",
            "Historial del match\n+ memoria de sesión\n→ effective_history",
            "#E8F5EF",
            ACCENT,
        ),
        (
            3.35,
            "2. Predecir",
            "11 expertos\n(heurísticos + GRU)\ncada uno → P̂ ∈ R³",
            "#E7EEF6",
            NEURAL,
        ),
        (
            6.35,
            "3. Meta + elegir",
            "×3 metas Iocaine\nvirtual scores\n→ mejor hipótesis",
            "#F3EEF8",
            META,
        ),
        (
            9.35,
            "4. Política",
            "EV → soft-BR\nα · exploit + (1−α)·Nash\nπ ∈ Δ²",
            "#FFF1E8",
            SOFT,
        ),
        (
            12.35,
            "5. Actuar",
            "sample a_AI ~ π\n(o argmax en debug)\n→ jugada AI",
            "#E8F5EF",
            ACCENT,
        ),
    ]

    y, w, h = 3.55, 2.85, 2.35
    for x, title, body, fc, ec in boxes:
        _box(ax, x, y, w, h, f"{title}\n\n{body}", fc=fc, ec=ec, lw=2.0, fontsize=10, weight="bold")

    # Arrows between steps
    for i in range(4):
        x1 = boxes[i][0] + w
        x2 = boxes[i + 1][0]
        _arrow(ax, x1 + 0.02, y + h / 2, x2 - 0.02, y + h / 2, color=MUTED, lw=2.0)

    # Formula strip under pipeline
    _box(
        ax,
        3.35,
        2.55,
        8.85,
        0.75,
        "π = (1 − α) · U  +  α · softmax(EV / T)     con     α = α_cap · confidence",
        fc=PANEL,
        ec=GATE,
        lw=1.6,
        fontsize=11,
        weight="bold",
        color=GATE,
    )
    _arrow(ax, 10.8, 3.55, 8.0, 3.3, color=GATE, lw=1.4)

    # Feedback loop
    bar.update(1)
    _box(
        ax,
        0.35,
        0.35,
        15.3,
        1.85,
        "Después de revelar la jugada humana\n"
        "observe_round: actualizar scores de cada hipótesis  (±1, decay 0.93)\n"
        "→ elige mejor el próximo experto  ·  sticky boost si hay patrón estable  ·  scores pasan a la siguiente partida",
        fc="#FCECEC",
        ec=FEEDBACK,
        lw=1.8,
        fontsize=10.5,
        weight="regular",
        color=INK,
    )
    # curved-ish feedback annotation from step 5 back toward start
    ax.annotate(
        "",
        xy=(1.8, 2.2),
        xytext=(13.8, 3.55),
        arrowprops=dict(
            arrowstyle="-|>",
            color=FEEDBACK,
            lw=1.6,
            connectionstyle="arc3,rad=0.28",
        ),
    )
    ax.text(11.2, 2.25, "feedback online", fontsize=9, color=FEEDBACK, style="italic")

    bar.update(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "mixture_inference_summary.png"
    pdf = out_dir / "mixture_inference_summary.pdf"
    fig.savefig(png, dpi=200, facecolor=BG, bbox_inches="tight")
    fig.savefig(pdf, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    bar.close()
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot summary inference flowchart.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "reports" / "figures")
    args = parser.parse_args()
    png, pdf = build_figure(args.out_dir)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
