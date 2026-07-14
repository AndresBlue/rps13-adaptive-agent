"""Empaqueta una versión congelada del agente ACG para producción."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_VERSION = "2.0.0"
DEFAULT_CHECKPOINT = ROOT / "models" / "opponent_predictor.pt"
DEFAULT_CONFIG = ROOT / "configs" / "app.web.yaml"

METRIC_FILES = [
    "predictor_metrics.json",
    "predictor_robust_metrics.json",
    "predictor_human_ft_metrics.json",
    "rl_metrics.json",
    "evaluation.csv",
    "evaluation_robust.csv",
    "evaluation_v2.csv",
    "evaluation_actor_critic.csv",
    "human_logs_summary.json",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_with_progress(src: Path, dst: Path, desc: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = src.stat().st_size
    bar = tqdm(total=total, unit="B", unit_scale=True, desc=desc, leave=False)
    with src.open("rb") as reader, dst.open("wb") as writer:
        while True:
            chunk = reader.read(1024 * 1024)
            if not chunk:
                break
            writer.write(chunk)
            bar.update(len(chunk))
    bar.close()


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(version: str, checkpoint: Path, config: Path, release_dir: Path) -> dict:
    predictor_metrics = load_json(ROOT / "reports" / "metrics" / "predictor_metrics.json")
    eval_robust = (ROOT / "reports" / "metrics" / "evaluation_robust.csv").read_text(encoding="utf-8")
    config_data = load_json(config) if config.suffix == ".json" else {}

    if config.suffix in {".yaml", ".yml"}:
        import yaml

        config_data = yaml.safe_load(config.read_text(encoding="utf-8"))

    checkpoint_rel = "opponent_predictor.pt"
    manifest = {
        "name": "acg",
        "version": version,
        "codename": "mixture-hmop-v2" if version.startswith("2") else "hybrid-synthetic-v1",
        "released_at": datetime.now(timezone.utc).isoformat(),
        "agent": {
            "type": config_data.get("agent_type", "HybridAdaptiveAgent"),
            "checkpoint": checkpoint_rel,
            "checkpoint_sha256": sha256_file(release_dir / checkpoint_rel),
            "checkpoint_bytes": (release_dir / checkpoint_rel).stat().st_size,
            "hyperparameters": {
                "sequence_length": 20,
                "target_score": 13,
                "temperature": config_data.get("temperature", 0.5),
                "max_alpha": config_data.get("max_alpha", 0.85),
                "sticky_max_alpha": config_data.get("sticky_max_alpha", 0.95),
                "min_action_prob": config_data.get("min_action_prob", 0.03),
                "sticky_min_action_prob": config_data.get("sticky_min_action_prob", 0.01),
            },
        },
        "predictor_training": {
            "config": "configs/train_predictor.yaml",
            "dataset": "data/synthetic/synthetic_matches.csv",
            "val_accuracy": predictor_metrics.get("final", {}).get("accuracy"),
            "baseline_random": predictor_metrics.get("baseline_random_accuracy"),
            "baseline_majority": predictor_metrics.get("baseline_majority_accuracy"),
            "architecture": {
                "hidden_dim": 96,
                "num_layers": 1,
                "dropout": 0.1,
            },
        },
        "deploy": config_data,
        "deploy_url": "https://poblete.servehttp.com/acg/",
        "evaluation_source": "reports/metrics/evaluation_robust.csv",
        "notes": [
            "Versión congelada desplegada en producción (julio 2026).",
            "Usa predictor GRU entrenado en datos sintéticos + política híbrida Nash.",
            "Ver MODEL_CARD.md para análisis completo vs estrategias.",
        ],
    }
    return manifest


def package_release(version: str, checkpoint: Path, config: Path, out_root: Path) -> Path:
    release_dir = out_root / f"acg-v{version}"
    if release_dir.exists():
        shutil.rmtree(release_dir)
    release_dir.mkdir(parents=True)

    tasks = [
        (checkpoint, release_dir / "opponent_predictor.pt", "checkpoint"),
        (config, release_dir / "app.web.yaml", "config"),
    ]
    for name in METRIC_FILES:
        src = ROOT / "reports" / "metrics" / name
        if src.exists():
            tasks.append((src, release_dir / "metrics" / name, name))

    card_src = ROOT / "releases" / "templates" / f"MODEL_CARD-v{version}.md"
    if card_src.exists():
        tasks.append((card_src, release_dir / "MODEL_CARD.md", "MODEL_CARD.md"))

    for src, dst, desc in tqdm(tasks, desc="Empaquetando release", unit="archivo"):
        copy_with_progress(src, dst, desc)

    manifest = build_manifest(version, checkpoint, config, release_dir)
    manifest_path = release_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    readme = release_dir / "README.txt"
    readme.write_text(
        "\n".join(
            [
                f"ACG release v{version}",
                "",
                "Contenido:",
                "  opponent_predictor.pt  — checkpoint de producción",
                "  app.web.yaml           — config FastAPI desplegada",
                "  MANIFEST.json          — metadatos y checksum",
                "  MODEL_CARD.md          — análisis de métricas y estrategias",
                "  metrics/               — JSON/CSV de entrenamiento y evaluación",
                "",
                "Restaurar en el proyecto:",
                f"  copy opponent_predictor.pt ..\\models\\",
                f"  copy app.web.yaml ..\\configs\\",
                "",
                f"URL: {manifest['deploy_url']}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return release_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Empaqueta una release de ACG.")
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path, default=ROOT / "releases")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint no encontrado: {args.checkpoint}")
    if not args.config.exists():
        raise SystemExit(f"Config no encontrada: {args.config}")

    release_dir = package_release(args.version, args.checkpoint, args.config, args.out)
    print(f"Release empaquetada en: {release_dir}")
    print(f"Checksum checkpoint: {json.loads((release_dir / 'MANIFEST.json').read_text())['agent']['checkpoint_sha256'][:16]}...")


if __name__ == "__main__":
    main()
