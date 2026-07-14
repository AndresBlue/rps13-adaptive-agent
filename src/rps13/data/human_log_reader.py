"""Read human app logs as datasets."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def read_human_logs(log_dir: str | Path) -> pd.DataFrame:
    """Read JSONL human logs from a directory."""

    rows: list[dict] = []
    for path in sorted(Path(log_dir).glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)
