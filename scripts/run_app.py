from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import uvicorn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ACG FastAPI web app.")
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    os.environ["RPS13_CONFIG"] = str(config_path)
    root_path = str(config.get("root_path", "") or "").strip()
    if root_path:
        os.environ["RPS13_ROOT_PATH"] = root_path
    else:
        os.environ.pop("RPS13_ROOT_PATH", None)
    host = args.host or config.get("host", "127.0.0.1")
    port = int(args.port or config.get("port", 8000))
    uvicorn.run("rps13.app.main:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
