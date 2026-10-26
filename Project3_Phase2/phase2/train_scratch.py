"""Train random-initialization (pretrained=False) scratch baselines on shoplifting videos.

Key Methodological Fix:
Overriding Week 5 Frozen-BN: Scratch models must train BatchNorm running statistics and affine
parameters from random initialization to prevent crippling internal covariate shift.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure paths
PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PHASE2_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from phase2.cached_dataset import FastShopliftingDataset
from phase2.config import (
    CHECKPOINTS_DIR,
    DEVICE,
    LOGS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    SPLITS_PATH,
)
from phase2.metrics import evaluate_model
from src.models.cnn_rnn import ResNetBiGRUAttention
from src.models.cnn3d import VideoCNN3D


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ScratchResNetBiGRU(ResNetBiGRUAttention):
    """ResNet-18 + BiGRU + Temporal Attention initialized with random weights and trainable BatchNorm."""

    def __init__(
        self,
        num_classes: int = 2,
        hidden_dim: int = 256,
        gru_layers: int = 2,
        gru_dropout: float = 0.3,
        head_dropout: float = 0.4,
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained=False,
            hidden_dim=hidden_dim,
            gru_layers=gru_layers,
            gru_dropout=gru_dropout,
            head_dropout=head_dropout,
        )
        for p in self.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        return nn.Module.train(self, mode)


class ScratchVideoCNN3D(VideoCNN3D):
    """R(2+1)D-18 3D CNN initialized with random weights and trainable BatchNorm."""

    def __init__(
        self,
        num_classes: int = 2,
        head_dropout: float = 0.4,
    ):
        super().__init__(
            num_classes=num_classes,
            pretrained=False,
            head_dropout=head_dropout,
        )
        for p in self.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        return nn.Module.train(self, mode)


def train_one_scratch_model(
    model_name: str,
    epochs: int = 20,
    batch_size: int = 8,
    grad_accum_steps: int = 2,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    patience: int = 6,
    force_retrain: bool = False,
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    """Train a single scratch baseline model (or evaluate existing checkpoint)."""
    image_size = 224 if model_name == "scratch_cnn_rnn" else 112

    checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pt"
    history_csv_path = LOGS_DIR / f"history_{model_name}.csv"

    # Fast-path: Check if already completed and checkpoint exists
    if not force_retrain and checkpoint_path.exists() and history_csv_path.exists():
        print(f"\n[CACHE HIT] Found existing trained checkpoint: {checkpoint_path.name}", flush=True)
        print(f"Loading history from {history_csv_path.name} and evaluating on Test Set...", flush=True)

        history_df = pd.read_csv(history_csv_path)
        history = history_df.to_dict(orient="records")

        # Instantiate model architecture
        if model_name == "scratch_cnn_rnn":
            model = ScratchResNetBiGRU(num_classes=2, hidden_dim=256, gru_layers=2, gru_dropout=0.3, head_dropout=0.4)
        else:
            model = ScratchVideoCNN3D(num_classes=2, head_dropout=0.4)

        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()

        test_ds = FastShopliftingDataset(SPLITS_PATH, split="test", T=16, image_size=image_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
        criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

        test_metrics_single, _, _ = evaluate_model(model, test_loader, device=device, criterion=criterion, use_tta=False)
        test_metrics_tta, _, _ = evaluate_model(model, test_loader, device=device, criterion=criterion, use_tta=True)

        best_epoch = ckpt.get("epoch", int(history_df.loc[history_df["val_f1_macro"].idxmax()]["epoch"]))
        best_val = ckpt.get("val_metrics", history[-1])

        print(f"--- TEST RESULTS ({model_name} | Best Epoch {best_epoch}) ---", flush=True)
        print(f"Single-View: Acc={test_metrics_single['accuracy']:.4f} | F1={test_metrics_single['f1_macro']:.4f} | Recall(Shop)={test_metrics_single['recall_shoplifting']:.4f} | AUC={test_metrics_single['roc_auc']:.4f}", flush=True)
        print(f"Flip-TTA:    Acc={test_metrics_tta['accuracy']:.4f} | F1={test_metrics_tta['f1_macro']:.4f} | Recall(Shop)={test_metrics_tta['recall_shoplifting']:.4f} | AUC={test_metrics_tta['roc_auc']:.4f}", flush=True)

        return {
            "model_name": model_name,
            "best_epoch": int(best_epoch),
            "total_train_time_s": float(history_df["epoch_time_s"].sum()),
            "val_metrics": best_val,
            "test_metrics_single": test_metrics_single,
            "test_metrics_tta": test_metrics_tta,
            "history": history,
        }

    # Training flow
    print("\n" + "=" * 80, flush=True)
    print(f"TRAINING SCRATCH BASELINE: {model_name.upper()} (pretrained=False, Trainable BN, image_size={image_size})", flush=True)
    print(f"Epochs: {epochs} | Micro-Batch: {batch_size} | Accum Steps: {grad_accum_steps} (Effective: {batch_size * grad_accum_steps})", flush=True)
    print(f"LR: {lr} | Weight Decay: {weight_decay} | Early Stopping Patience: {patience}", flush=True)
    print("=" * 80, flush=True)

    set_seed(42)

    # 1. In-memory cached datasets
    train_ds = FastShopliftingDataset(SPLITS_PATH, split="train", T=16, image_size=image_size)
    val_ds = FastShopliftingDataset(SPLITS_PATH, split="val", T=16, image_size=image_size)
    test_ds = FastShopliftingDataset(SPLITS_PATH, split="test", T=16, image_size=image_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"), drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))

    # 2. Build Model
    if model_name == "scratch_cnn_rnn":
        model = ScratchResNetBiGRU(num_classes=2, hidden_dim=256, gru_layers=2, gru_dropout=0.3, head_dropout=0.4)
    elif model_name == "scratch_cnn3d":
        model = ScratchVideoCNN3D(num_classes=2, head_dropout=0.4)
    else:
        raise ValueError(f"Unknown scratch model name: {model_name}")

    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: Total={total_params:,} | Trainable={trainable_params:,}", flush=True)

    # 3. Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Training Loop
    best_key = (-1.0, -1.0, float("-inf"))
    best_epoch = -1
    best_val_metrics: Dict[str, Any] = {}
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    start_train_time = time.time()
    num_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss_total = 0.0
        train_samples = 0
        optimizer.zero_grad()

        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            b_size = clips.size(0)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(clips)
                loss = criterion(logits, labels)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()
            train_loss_total += loss.item() * b_size
            train_samples += b_size

            is_accum_step = ((batch_idx + 1) % grad_accum_steps == 0) or (batch_idx + 1 == num_batches)
            if is_accum_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        scheduler.step()
        train_loss = train_loss_total / max(train_samples, 1)

        val_metrics, _, _ = evaluate_model(model, val_loader, device=device, criterion=criterion, use_tta=False)
        current_lr = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 4),
            "val_f1_macro": round(val_metrics["f1_macro"], 4),
            "val_recall_shop": round(val_metrics["recall_shoplifting"], 4),
            "val_roc_auc": round(val_metrics["roc_auc"], 4),
            "epoch_time_s": round(epoch_time, 2),
        }
        history.append(record)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val F1: {val_metrics['f1_macro']:.4f} | "
            f"Val Recall(Shop): {val_metrics['recall_shoplifting']:.4f} | "
            f"Val AUC: {val_metrics['roc_auc']:.4f} | "
            f"Time: {epoch_time:.1f}s",
            flush=True,
        )

        cand_key = (
            round(val_metrics["f1_macro"], 4),
            round(val_metrics["roc_auc"], 4),
            -round(val_metrics["loss"], 4),
        )

        if cand_key > best_key:
            best_key = cand_key
            best_epoch = epoch
            best_val_metrics = val_metrics.copy()
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": best_val_metrics,
                    "history": history,
                    "model_name": model_name,
                    "arch": model.__class__.__name__,
                },
                checkpoint_path,
            )
            print(f"  >>> Best model saved at epoch {epoch} with Val F1={best_val_metrics['f1_macro']:.4f}, Val AUC={best_val_metrics['roc_auc']:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={patience}).", flush=True)
                break

    total_train_time = time.time() - start_train_time
    print(f"Training completed in {total_train_time:.1f}s. Best epoch: {best_epoch}", flush=True)

    pd.DataFrame(history).to_csv(history_csv_path, index=False)
    print(f"Saved history CSV to {history_csv_path}", flush=True)

    # 5. Evaluate Best Checkpoint on Test Set (98 clips)
    print(f"\nEvaluating Best Checkpoint ({checkpoint_path.name}) on Test Set (98 clips)...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    test_metrics_single, _, _ = evaluate_model(model, test_loader, device=device, criterion=criterion, use_tta=False)
    test_metrics_tta, _, _ = evaluate_model(model, test_loader, device=device, criterion=criterion, use_tta=True)

    print("\n--- TEST RESULTS (Single-View tau=0.5) ---", flush=True)
    print(f"Accuracy: {test_metrics_single['accuracy']:.4f}", flush=True)
    print(f"Macro F1: {test_metrics_single['f1_macro']:.4f}", flush=True)
    print(f"Recall (Shoplifting): {test_metrics_single['recall_shoplifting']:.4f}", flush=True)
    print(f"Precision (Shoplifting): {test_metrics_single['precision_shoplifting']:.4f}", flush=True)
    print(f"ROC-AUC: {test_metrics_single['roc_auc']:.4f}", flush=True)
    print(f"Confusion Matrix: {test_metrics_single['confusion_matrix']}", flush=True)

    print("\n--- TEST RESULTS (Flip-TTA tau=0.5) ---", flush=True)
    print(f"Accuracy: {test_metrics_tta['accuracy']:.4f}", flush=True)
    print(f"Macro F1: {test_metrics_tta['f1_macro']:.4f}", flush=True)
    print(f"Recall (Shoplifting): {test_metrics_tta['recall_shoplifting']:.4f}", flush=True)
    print(f"ROC-AUC: {test_metrics_tta['roc_auc']:.4f}", flush=True)

    return {
        "model_name": model_name,
        "best_epoch": best_epoch,
        "total_train_time_s": round(total_train_time, 2),
        "val_metrics": best_val_metrics,
        "test_metrics_single": test_metrics_single,
        "test_metrics_tta": test_metrics_tta,
        "history": history,
    }


def plot_scratch_curves(histories: Dict[str, List[Dict[str, Any]]], output_path: Path) -> None:
    """Plot training and validation curves for scratch models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    styles = {
        "scratch_cnn_rnn": {"color": "tab:blue", "label": "Scratch ResNet-BiGRU"},
        "scratch_cnn3d": {"color": "tab:orange", "label": "Scratch VideoCNN3D"},
    }

    for name, hist in histories.items():
        df = pd.DataFrame(hist)
        cfg = styles.get(name, {"color": "tab:gray", "label": name})

        axes[0].plot(df["epoch"], df["train_loss"], label=f"{cfg['label']} (Train)", linestyle="--", color=cfg["color"], alpha=0.7)
        axes[0].plot(df["epoch"], df["val_loss"], label=f"{cfg['label']} (Val)", linestyle="-", color=cfg["color"], linewidth=2)

        axes[1].plot(df["epoch"], df["val_f1_macro"], label=f"{cfg['label']} Val F1", color=cfg["color"], linewidth=2)
        axes[2].plot(df["epoch"], df["val_roc_auc"], label=f"{cfg['label']} Val AUC", color=cfg["color"], linewidth=2)

    axes[0].set_title("Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Validation Macro-F1 Score")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Macro-F1")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].set_title("Validation ROC-AUC")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("ROC-AUC")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.suptitle("Scratch Baselines (pretrained=False, Trainable BN) Training Curves", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning curves to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Train scratch baseline models.")
    parser.add_argument("--model", type=str, default="all", choices=["scratch_cnn_rnn", "scratch_cnn3d", "all"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--force_retrain", action="store_true")
    args = parser.parse_args()

    models_to_train = ["scratch_cnn_rnn", "scratch_cnn3d"] if args.model == "all" else [args.model]

    results_all = {}
    histories_all = {}

    for m_name in models_to_train:
        res = train_one_scratch_model(
            model_name=m_name,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.lr,
            patience=args.patience,
            force_retrain=args.force_retrain,
        )
        results_all[m_name] = res
        histories_all[m_name] = res["history"]

    results_json_path = METRICS_DIR / "scratch_results.json"
    with open(results_json_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nSaved full scratch results to {results_json_path}", flush=True)

    plot_scratch_curves(histories_all, PLOTS_DIR / "learning_curves_scratch.png")


if __name__ == "__main__":
    main()
