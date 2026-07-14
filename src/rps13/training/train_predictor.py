"""Supervised training for the GRU opponent predictor."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_mpl_config_dir = Path("reports") / ".matplotlib"
_mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_config_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from rps13.data.dataset_loader import load_internal_dataset
from rps13.data.feature_engineering import NUMERIC_FEATURE_KEYS, build_sequence_arrays
from rps13.models.opponent_predictor import OpponentPredictorGRU, load_predictor_checkpoint
from rps13.utils.io import ensure_parent, load_yaml, write_json
from rps13.utils.seed import set_global_seed


class PredictorSequenceDataset(Dataset):
    """Examples predicting the next human move from previous rounds."""

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int = 20,
        target_score: int = 13,
        desc: str = "building dataset",
    ) -> None:
        self.sequence_length = sequence_length
        self.size = len(df)
        numeric_dim = len(NUMERIC_FEATURE_KEYS)
        self.human_moves = np.full((self.size, sequence_length), 3, dtype=np.int64)
        self.ai_moves = np.full((self.size, sequence_length), 3, dtype=np.int64)
        self.results = np.full((self.size, sequence_length), 3, dtype=np.int64)
        self.numeric = np.zeros((self.size, sequence_length, numeric_dim), dtype=np.float32)
        self.lengths = np.ones(self.size, dtype=np.int64)
        self.targets = np.zeros(self.size, dtype=np.int64)

        cursor = 0
        groups = df.sort_values(["match_id", "round"]).groupby("match_id", sort=False)
        progress = tqdm(total=self.size, desc=desc, unit="rows", dynamic_ncols=True)
        for _match_id, group in groups:
            records = group.to_dict("records")
            for idx in range(len(records)):
                history_start = max(0, idx - sequence_length)
                arrays = build_sequence_arrays(records[history_start:idx], sequence_length, target_score=target_score)
                self.human_moves[cursor] = arrays["human_moves"]
                self.ai_moves[cursor] = arrays["ai_moves"]
                self.results[cursor] = arrays["results"]
                self.numeric[cursor] = arrays["numeric"]
                self.lengths[cursor] = max(int(arrays["length"]), 1)
                self.targets[cursor] = int(records[idx]["human_move"])
                cursor += 1
            progress.update(len(records))
        progress.close()

        self.human_moves_t = torch.from_numpy(self.human_moves)
        self.ai_moves_t = torch.from_numpy(self.ai_moves)
        self.results_t = torch.from_numpy(self.results)
        self.numeric_t = torch.from_numpy(self.numeric)
        self.lengths_t = torch.from_numpy(self.lengths)
        self.targets_t = torch.from_numpy(self.targets)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "human_moves": self.human_moves_t[idx],
            "ai_moves": self.ai_moves_t[idx],
            "results": self.results_t[idx],
            "numeric": self.numeric_t[idx],
            "length": self.lengths_t[idx],
            "target": self.targets_t[idx],
        }


def _split_by_match(df: pd.DataFrame, validation_split: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    match_ids = np.array(df["match_id"].unique())
    rng.shuffle(match_ids)
    val_count = max(1, int(len(match_ids) * validation_split)) if len(match_ids) > 1 else 0
    val_ids = set(match_ids[:val_count])
    train = df[~df["match_id"].isin(val_ids)].copy()
    val = df[df["match_id"].isin(val_ids)].copy()
    if val.empty:
        val = train.copy()
    return train, val


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device, non_blocking: bool) -> dict[str, torch.Tensor]:
    return {
        "human_moves": batch["human_moves"].to(device, non_blocking=non_blocking),
        "ai_moves": batch["ai_moves"].to(device, non_blocking=non_blocking),
        "results": batch["results"].to(device, non_blocking=non_blocking),
        "numeric": batch["numeric"].to(device, non_blocking=non_blocking),
        "length": batch["length"].to(device, non_blocking=non_blocking),
        "target": batch["target"].to(device, non_blocking=non_blocking),
    }


def _evaluate(
    model: OpponentPredictorGRU,
    loader: DataLoader,
    device: torch.device,
    non_blocking: bool = False,
    desc: str = "validating",
) -> dict[str, Any]:
    model.eval()
    targets: list[int] = []
    preds: list[int] = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False):
            batch = _move_batch(batch, device, non_blocking=non_blocking)
            logits, _ = model(
                batch["human_moves"],
                batch["ai_moves"],
                batch["results"],
                batch["numeric"],
                batch["length"],
            )
            target = batch["target"]
            loss = criterion(logits, target)
            total_loss += float(loss.item()) * target.numel()
            preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
            targets.extend(target.cpu().tolist())
    accuracy = accuracy_score(targets, preds) if targets else 0.0
    per_class = {}
    for cls in range(3):
        cls_indices = [idx for idx, target in enumerate(targets) if target == cls]
        if cls_indices:
            per_class[str(cls)] = float(np.mean([preds[idx] == targets[idx] for idx in cls_indices]))
        else:
            per_class[str(cls)] = None
    return {
        "loss": total_loss / max(len(targets), 1),
        "accuracy": float(accuracy),
        "accuracy_by_class": per_class,
        "targets": targets,
        "preds": preds,
    }


def _resolve_device(config: dict[str, Any]) -> torch.device:
    requested = str(config.get("device", "auto")).lower()
    require_cuda = bool(config.get("require_cuda", False))
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests CUDA, but this environment has CPU-only PyTorch. "
            "Install a CUDA-enabled torch build before training on the RTX GPU."
        )
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Config require_cuda=true, but CUDA is not available.")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    return device


def _make_grad_scaler(use_amp: bool) -> torch.amp.GradScaler:
    try:
        return torch.amp.GradScaler("cuda", enabled=use_amp)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=use_amp)


def _save_checkpoint(
    path: str | Path,
    model: OpponentPredictorGRU,
    sequence_length: int,
    metrics: dict[str, Any],
) -> None:
    checkpoint_path = ensure_parent(path)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "model_config": model.model_config(),
            "sequence_length": sequence_length,
            "feature_keys": NUMERIC_FEATURE_KEYS,
            "metrics": metrics,
        },
        checkpoint_path,
    )


def train_predictor(config_path: str | Path) -> dict[str, Any]:
    """Train the supervised opponent predictor from a YAML config."""

    config = load_yaml(config_path)
    seed = int(config.get("seed", 42))
    set_global_seed(seed)
    device = _resolve_device(config)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    dataset_path = Path(config.get("dataset_path", "data/synthetic/synthetic_matches.csv"))
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    print(f"Loading dataset: {dataset_path}")
    df = load_internal_dataset(dataset_path)
    print(f"Loaded {len(df):,} rows across {df['match_id'].nunique():,} matches")
    train_df, val_df = _split_by_match(df, float(config.get("validation_split", 0.2)), seed)
    sequence_length = int(config.get("sequence_length", 20))
    print(f"Train rows: {len(train_df):,}; validation rows: {len(val_df):,}; sequence_length={sequence_length}")
    train_ds = PredictorSequenceDataset(train_df, sequence_length=sequence_length, desc="building train dataset")
    val_ds = PredictorSequenceDataset(val_df, sequence_length=sequence_length, desc="building val dataset")
    batch_size = int(config.get("batch_size", 64))
    num_workers = int(config.get("num_workers", 0))
    pin_memory = bool(config.get("pin_memory", False))
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(config.get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(config.get("prefetch_factor", 2))
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    non_blocking = device.type == "cuda" and pin_memory
    use_amp = bool(config.get("use_amp", False)) and device.type == "cuda"
    scaler = _make_grad_scaler(use_amp)
    if device.type == "cuda":
        print(f"AMP mixed precision: {use_amp}")
    print(f"batch_size={batch_size}; num_workers={num_workers}; pin_memory={pin_memory}")
    init_checkpoint = config.get("init_checkpoint_path")
    if init_checkpoint:
        init_path = Path(init_checkpoint)
        if init_path.exists():
            model = load_predictor_checkpoint(init_path, map_location=device).to(device)
            print(f"Loaded init checkpoint: {init_path}")
        else:
            model = OpponentPredictorGRU(
                hidden_dim=int(config.get("hidden_dim", 96)),
                num_layers=int(config.get("num_layers", 1)),
                dropout=float(config.get("dropout", 0.1)),
            ).to(device)
    else:
        model = OpponentPredictorGRU(
            hidden_dim=int(config.get("hidden_dim", 96)),
            num_layers=int(config.get("num_layers", 1)),
            dropout=float(config.get("dropout", 0.1)),
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.get("learning_rate", 1e-3)))
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    epochs = int(config.get("epochs", 8))
    checkpoint_every = int(config.get("checkpoint_every", 0))
    latest_checkpoint_path = config.get("latest_checkpoint_path")
    best_accuracy = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", unit="batch", dynamic_ncols=True)
        for batch in progress:
            batch = _move_batch(batch, device, non_blocking=non_blocking)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, _ = model(
                    batch["human_moves"],
                    batch["ai_moves"],
                    batch["results"],
                    batch["numeric"],
                    batch["length"],
                )
                target = batch["target"]
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.item()) * target.numel()
            seen += target.numel()
            progress.set_postfix(loss=total_loss / max(seen, 1), seen=f"{seen:,}")
        val_metrics = _evaluate(
            model,
            val_loader,
            device,
            non_blocking=non_blocking,
            desc=f"validating {epoch}/{epochs}",
        )
        history["train_loss"].append(total_loss / max(seen, 1))
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        epoch_metrics = {
            "epoch": epoch,
            "epochs": epochs,
            "train_loss": history["train_loss"][-1],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "history": history,
        }
        print(
            f"epoch {epoch}/{epochs}: "
            f"train_loss={epoch_metrics['train_loss']:.4f} "
            f"val_loss={epoch_metrics['val_loss']:.4f} "
            f"val_accuracy={epoch_metrics['val_accuracy']:.4f}"
        )
        write_json(config.get("metrics_path", "reports/metrics/predictor_metrics.json"), epoch_metrics)
        if latest_checkpoint_path and checkpoint_every > 0 and epoch % checkpoint_every == 0:
            _save_checkpoint(latest_checkpoint_path, model, sequence_length, epoch_metrics)
            model.to(device)
        if val_metrics["accuracy"] > best_accuracy:
            best_accuracy = float(val_metrics["accuracy"])
            best_metrics = dict(epoch_metrics)
            best_metrics["best_epoch"] = epoch
            _save_checkpoint(config.get("checkpoint_path", "models/opponent_predictor.pt"), model, sequence_length, best_metrics)
            model.to(device)

    final_metrics = _evaluate(model, val_loader, device, non_blocking=non_blocking, desc="final validation")
    val_targets = final_metrics.pop("targets")
    final_metrics.pop("preds")
    train_targets = train_ds.targets.tolist()
    counts = (
        np.bincount(np.asarray(train_targets, dtype=np.int64), minlength=3)
        if train_targets
        else np.zeros(3, dtype=np.int64)
    )
    majority = int(np.argmax(counts)) if counts.sum() else 0
    majority_acc = float(np.mean(np.asarray(val_targets) == majority)) if val_targets else 0.0
    metrics = {
        "final": final_metrics,
        "baseline_random_accuracy": 1.0 / 3.0,
        "baseline_majority_accuracy": majority_acc,
        "majority_class": majority,
        "num_train_examples": len(train_ds),
        "num_val_examples": len(val_ds),
        "history": history,
        "feature_keys": NUMERIC_FEATURE_KEYS,
        "best_validation_accuracy": best_accuracy,
    }

    _save_checkpoint(config.get("checkpoint_path", "models/opponent_predictor.pt"), model, sequence_length, metrics)
    write_json(config.get("metrics_path", "reports/metrics/predictor_metrics.json"), metrics)
    _plot_history(history, config.get("figures_dir", "reports/figures"))
    return metrics


def _plot_history(history: dict[str, list[float]], figures_dir: str | Path) -> None:
    out_dir = Path(figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "predictor_loss.png")
    plt.close()
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, history["val_accuracy"], label="val_accuracy")
    plt.xlabel("epoch")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "predictor_accuracy.png")
    plt.close()
