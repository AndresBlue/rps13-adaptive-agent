"""Render Mermaid inference pipeline: exact graph, dark theme, forced 1.8:1 fill."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
MMD = ROOT / "docs" / "inference_pipeline.mmd"
SVG = ROOT / "docs" / "inference_pipeline.svg"
PNG = ROOT / "docs" / "inference_pipeline.png"
ASPECT = 1.8
WIDTH = 2200
BG = "#141414"


def force_svg_aspect(svg_text: str, aspect: float = ASPECT) -> tuple[str, float, float, float, float]:
    """Keep content geometry; set outer viewBox/size to exact aspect (letterbox)."""
    m = re.search(r'viewBox="([\d.\-]+) ([\d.\-]+) ([\d.\-]+) ([\d.\-]+)"', svg_text)
    if not m:
        raise ValueError("SVG viewBox not found")
    x, y, w, h = map(float, m.groups())
    if w / max(h, 1e-9) > aspect:
        new_w, new_h = w, w / aspect
        dx, dy = 0.0, (new_h - h) / 2
    else:
        new_w, new_h = h * aspect, h
        dx, dy = (new_w - w) / 2, 0.0
    new_x, new_y = x - dx, y - dy
    m_svg = re.search(r"<svg\b[^>]*>", svg_text)
    if not m_svg:
        raise ValueError("root <svg> not found")
    tag = m_svg.group(0)
    tag = re.sub(r'viewBox="[^"]+"', f'viewBox="{new_x:.3f} {new_y:.3f} {new_w:.3f} {new_h:.3f}"', tag, count=1)
    if re.search(r'\bwidth="', tag):
        tag = re.sub(r'\bwidth="[^"]+"', f'width="{new_w:.0f}"', tag, count=1)
    else:
        tag = tag.replace("<svg", f'<svg width="{new_w:.0f}"', 1)
    if re.search(r'\bheight="', tag):
        tag = re.sub(r'\bheight="[^"]+"', f'height="{new_h:.0f}"', tag, count=1)
    else:
        tag = tag.replace("<svg", f'<svg height="{new_h:.0f}"', 1)
    if "style=" in tag:
        tag = re.sub(r'style="[^"]*"', f'style="background-color:{BG};"', tag, count=1)
    else:
        tag = tag.replace("<svg", f'<svg style="background-color:{BG};"', 1)
    return svg_text[: m_svg.start()] + tag + svg_text[m_svg.end() :], w, h, new_w, new_h


def run_mmdc(out: Path, width: int) -> None:
    cmd = [
        "npx",
        "--yes",
        "@mermaid-js/mermaid-cli",
        "-i",
        str(MMD),
        "-o",
        str(out),
        "-b",
        BG,
        "-w",
        str(width),
        "-s",
        "2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)


def force_png_fill(path: Path, out_w: int, aspect: float = ASPECT) -> tuple[int, int, int, int]:
    """Resize content to exactly fill out_w x out_w/aspect (may mildly stretch)."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    out_h = max(1, int(round(out_w / aspect)))
    filled = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
    filled.save(path, optimize=True)
    return w, h, out_w, out_h


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Mermaid inference pipeline (dark, forced 1.8:1).")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--aspect", type=float, default=ASPECT)
    parser.add_argument(
        "--mode",
        choices=("letterbox", "fill"),
        default="letterbox",
        help="letterbox = keep proportions in 1.8:1; fill = stretch to cover",
    )
    args = parser.parse_args()

    bar = tqdm(total=4, desc="Mermaid dark 1.8:1", unit="paso")

    bar.update(1)
    run_mmdc(SVG, args.width)

    bar.update(1)
    svg = SVG.read_text(encoding="utf-8")
    m = re.search(r'viewBox="([\d.\-]+) ([\d.\-]+) ([\d.\-]+) ([\d.\-]+)"', svg)
    if not m:
        raise SystemExit("SVG viewBox missing")
    _, _, cw, ch = map(float, m.groups())
    padded, _, _, nw, nh = force_svg_aspect(svg, args.aspect)
    SVG.write_text(padded, encoding="utf-8")
    print(f"SVG content: {cw:.0f}x{ch:.0f} ({cw/max(ch,1):.2f})")
    print(f"SVG canvas:  {nw:.0f}x{nh:.0f} ({nw/nh:.2f})")

    bar.update(1)
    run_mmdc(PNG, args.width)

    bar.update(1)
    if args.mode == "fill":
        pw, ph, pnw, pnh = force_png_fill(PNG, args.width, args.aspect)
    else:
        img = Image.open(PNG).convert("RGB")
        pw, ph = img.size
        if pw / max(ph, 1) > args.aspect:
            pnw, pnh = pw, int(round(pw / args.aspect))
        else:
            pnw, pnh = int(round(ph * args.aspect)), ph
        canvas = Image.new("RGB", (pnw, pnh), BG)
        canvas.paste(img, ((pnw - pw) // 2, (pnh - ph) // 2))
        canvas.save(PNG, optimize=True)
    print(f"PNG content: {pw}x{ph} ({pw/max(ph,1):.2f})")
    print(f"PNG canvas:  {pnw}x{pnh} ({pnw/pnh:.2f}) mode={args.mode}")

    bar.close()
    print(f"Wrote {SVG}")
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
