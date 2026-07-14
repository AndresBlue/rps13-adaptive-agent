"""Small IO helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Return the repository root when running from an installed or source tree."""

    return Path(__file__).resolve().parents[3]


def ensure_parent(path: str | Path) -> Path:
    """Ensure parent directory exists and return ``Path``."""

    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    out = ensure_parent(path)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def append_jsonl(path: str | Path, data: dict[str, Any]) -> None:
    out = ensure_parent(path)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(data, sort_keys=True) + "\n")
