"""Render Mermaid inference pipeline to a rectangular (~2:1) SVG/PNG."""

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
ASPECT = 2.0
WIDTH = 2000
BG = (247, 244, 239)  # #F7F4EF


def pad_svg_to_aspect(svg_text: str, aspect: float = ASPECT) -> tuple[str, float, float, float, float]:
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
    return svg_text[: m_svg.start()] + tag + svg_text[m_svg.end() :], w, h, new_w, new_h


def run_mmdc(out: Path, width: int, height: int | None = None) -> None:
    cmd = [
        "npx",
        "--yes",
        "@mermaid-js/mermaid-cli",
        "-i",
        str(MMD),
        "-o",
        str(out),
        "-b",
        "#F7F4EF",
        "-w",
        str(width),
        "-s",
        "2",
    ]
    if height is not None:
        cmd.extend(["-H", str(height)])
    proc = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)


def pad_png_to_aspect(path: Path, aspect: float = ASPECT) -> tuple[int, int, int, int]:
    img = Image.open(path).convert("RGBA")
    w, h = img.size
    if w / max(h, 1) > aspect:
        new_w, new_h = w, int(round(w / aspect))
    else:
        new_w, new_h = int(round(h * aspect)), h
    canvas = Image.new("RGBA", (new_w, new_h), BG + (255,))
    canvas.paste(img, ((new_w - w) // 2, (new_h - h) // 2), img)
    canvas.convert("RGB").save(path)
    return w, h, new_w, new_h


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Mermaid inference pipeline (~2:1).")
    parser.add_argument("--width", type=int, default=WIDTH)
    args = parser.parse_args()

    bar = tqdm(total=4, desc="Mermaid inference", unit="paso")

    bar.update(1)
    run_mmdc(SVG, args.width, None)

    bar.update(1)
    padded, cw, ch, nw, nh = pad_svg_to_aspect(SVG.read_text(encoding="utf-8"), ASPECT)
    SVG.write_text(padded, encoding="utf-8")
    print(f"SVG content: {cw:.0f}x{ch:.0f} ({cw/max(ch,1):.2f})")
    print(f"SVG canvas:  {nw:.0f}x{nh:.0f} ({nw/nh:.2f})")

    bar.update(1)
    # Tight PNG render (no forced height), then letterbox to 2:1.
    run_mmdc(PNG, args.width, None)

    bar.update(1)
    pw, ph, pnw, pnh = pad_png_to_aspect(PNG, ASPECT)
    print(f"PNG content: {pw}x{ph} ({pw/max(ph,1):.2f})")
    print(f"PNG canvas:  {pnw}x{pnh} ({pnw/pnh:.2f})")

    bar.close()
    print(f"Wrote {SVG}")
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
