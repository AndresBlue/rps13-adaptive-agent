"""Compile docs/inference_pipeline.d2 to SVG with exact 2:1 canvas."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D2_SRC = ROOT / "docs" / "inference_pipeline.d2"
SVG_OUT = ROOT / "docs" / "inference_pipeline.svg"
ASPECT = 2.0


def pad_viewbox_to_aspect(svg: str, aspect: float = ASPECT) -> tuple[str, float, float]:
    m = re.search(r'viewBox="([\d.\-]+) ([\d.\-]+) ([\d.\-]+) ([\d.\-]+)"', svg)
    if not m:
        raise ValueError("SVG viewBox not found")
    x, y, w, h = map(float, m.groups())
    if w / h > aspect:
        new_w, new_h = w, w / aspect
        dx, dy = 0.0, (new_h - h) / 2
    else:
        new_w, new_h = h * aspect, h
        dx, dy = (new_w - w) / 2, 0.0
    new_x, new_y = x - dx, y - dy

    m_svg = re.search(r"<svg\b[^>]*>", svg)
    if not m_svg:
        raise ValueError("root <svg> not found")
    tag = m_svg.group(0)
    tag = re.sub(r'viewBox="[^"]+"', f'viewBox="{new_x:.3f} {new_y:.3f} {new_w:.3f} {new_h:.3f}"', tag, count=1)
    tag = re.sub(r'\bwidth="[^"]+"', f'width="{new_w:.0f}"', tag, count=1)
    tag = re.sub(r'\bheight="[^"]+"', f'height="{new_h:.0f}"', tag, count=1)
    return svg[: m_svg.start()] + tag + svg[m_svg.end() :], new_w, new_h


def main() -> None:
    cmd = ["d2", "--pad", "48", "--watch=false", str(D2_SRC), str(SVG_OUT)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    raw = SVG_OUT.read_text(encoding="utf-8")
    m = re.search(r'viewBox="([\d.\-]+) ([\d.\-]+) ([\d.\-]+) ([\d.\-]+)"', raw)
    _, _, w, h = map(float, m.groups())
    print(f"content: {w:.0f}x{h:.0f} ratio={w/h:.3f}")

    padded, nw, nh = pad_viewbox_to_aspect(raw, ASPECT)
    SVG_OUT.write_text(padded, encoding="utf-8")
    print(f"canvas:  {nw:.0f}x{nh:.0f} ratio={nw/nh:.3f}")
    print(f"Wrote {SVG_OUT}")


if __name__ == "__main__":
    main()
