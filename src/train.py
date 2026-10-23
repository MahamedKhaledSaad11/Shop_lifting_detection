"""Training pipeline for Shoplifting Video Classification models."""

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import (
    CHECKPOINTS_DIR,
    DEVICE,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SPLITS_PATH,
    TrainConfig,
)
from src.data.dataset import ShopliftingDataset, create_dataloader
from src.data.transforms import get_transforms
from src.models.cnn_rnn import ResNetBiGRUAttention
from src.models.cnn3d import VideoCNN3D
from src.utils import save_checkpoint, set_seed, setup_logger

logger = setup_logger("train")


def build_model(model_name: str, num_classes: int = 2) -> nn.Module:
    """Build model instance by name."""
    if model_name == "cnn_rnn":
        model = ResNetBiGRUAttention(
            num_classes=num_classes,
            pretrained=True,
            hidden_dim=256,
            gru_layers=2,
            gru_dropout=0.3,
            head_dropout=0.5,
        )
    elif model_name == "cnn3d":
        model = VideoCNN3D(
            num_classes=num_classes,
            pretrained=True,
            head_dropout=0.5,
        )
    else:
        raise ValueError(f"Unknown model architecture '{model_name}'. Choose 'cnn_rnn' or 'cnn3d'.")
    return model


def evaluate_split(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model on a dataset split and return metrics dictionary."""
    model.eval()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []

    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits = model(clips)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item() * len(labels)
            all_targets.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_probs.extend(probs.cpu().tolist())

    n_samples = max(len(all_targets), 1)
    avg_loss = total_loss / n_samples
    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    f1_shoplifting = f1_score(all_targets, all_preds, pos_label=1, zero_division=0)

    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        roc_auc = 0.5

    return {
        "loss": float(avg_loss),
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_macro": float(f1_macro),
        "f1_shoplifting": float(f1_shoplifting),
        "roc_auc": float(roc_auc),
    }


def train_model(
    model_name: str = "cnn_rnn",
    epochs: int = 15,
    batch_size: int = 8,
    lr_head: float = 2e-4,
    lr_backbone: float = 2e-5,
    warmup_epochs: int = 2,
    weight_decay: float = 1e-4,
    label_smoothing: float = 0.05,
    use_amp: bool = True,
    early_stopping_patience: int = 5,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Execute complete staged training pipeline for a given model."""
    set_seed(seed)
    device = DEVICE
    logger.info("Initializing training for [%s] on device: %s", model_name, device)

    # 1. Datasets & Loaders (Deduplicated unique events)
    image_size = 224 if model_name == "cnn_rnn" else 112
    train_transform = get_transforms("train")
    val_transform = get_transforms("val")

    # Override image_size if 3D CNN
    train_transform.image_size = image_size
    val_transform.image_size = image_size

    train_ds = ShopliftingDataset(SPLITS_PATH, split="train", T=16, transform=train_transform, deduplicate=True)
    val_ds = ShopliftingDataset(SPLITS_PATH, split="val", T=16, transform=val_transform, deduplicate=True)

    logger.info("Train dataset size: %d clips | Val dataset size: %d clips", len(train_ds), len(val_ds))

    train_loader = create_dataloader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = create_dataloader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    # 2. Build Model
    model = build_model(model_name).to(device)

    # 3. Warmup stage setup: freeze backbone initially
    model.freeze_backbone()
    logger.info("Epochs 1-%d: Backbone frozen (warmup training of temporal aggregator and head)", warmup_epochs)

    # Param groups
    param_groups = model.get_param_groups(lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay)
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # 4. Training Loop
    history: List[Dict[str, float]] = []
    best_score_tuple = (-1.0, -1.0, -1e9)
    best_val_f1 = -1.0
    patience_counter = 0

    best_checkpoint_path = CHECKPOINTS_DIR / f"best_{model_name}.pt"
    last_checkpoint_path = CHECKPOINTS_DIR / f"last_{model_name}.pt"

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Check unfreeze schedule
        if epoch == warmup_epochs + 1:
            logger.info("Epoch %d: Unfreezing deeper backbone layers for end-to-end fine-tuning", epoch)
            model.unfreeze_backbone(["layer3", "layer4"])
            param_groups = model.get_param_groups(lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay)
            optimizer = torch.optim.AdamW(param_groups)
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)

        # Training Phase
        model.train()
        train_loss = 0.0
        train_targets: List[int] = []
        train_preds: List[int] = []

        for clips, labels in train_loader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
                logits = model(clips)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            train_targets.extend(labels.cpu().tolist())
            train_preds.extend(preds.cpu().tolist())

        scheduler.step()

        train_loss_avg = train_loss / len(train_targets)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="macro", zero_division=0)

        # Validation Phase
        val_metrics = evaluate_split(model, val_loader, criterion, device)
        epoch_duration = time.time() - epoch_start

        current_lr = optimizer.param_groups[-1]["lr"]
        logger.info(
            "Epoch [%02d/%02d] (%.1fs) | Train Loss: %.4f, Acc: %.3f, F1: %.3f | "
            "Val Loss: %.4f, Acc: %.3f, F1: %.3f, AUC: %.3f | LR: %.2e",
            epoch,
            epochs,
            epoch_duration,
            train_loss_avg,
            train_acc,
            train_f1,
            val_metrics["loss"],
            val_metrics["accuracy"],
            val_metrics["f1_macro"],
            val_metrics["roc_auc"],
            current_lr,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss_avg,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_f1_shoplifting": val_metrics["f1_shoplifting"],
            "val_roc_auc": val_metrics["roc_auc"],
            "lr": current_lr,
        })

        # Model Checkpoint with tie-breaking: (Macro-F1, ROC-AUC, -Loss)
        state = {
            "epoch": epoch,
            "model_name": model_name,
            "model_state_dict": model.state_dict(),
            "val_metrics": val_metrics,
            "history": history,
        }
        save_checkpoint(state, last_checkpoint_path)

        score_tuple = (
            round(val_metrics["f1_macro"], 4),
            round(val_metrics["roc_auc"], 4),
            -round(val_metrics["loss"], 4),
        )

        if score_tuple > best_score_tuple:
            best_score_tuple = score_tuple
            best_val_f1 = val_metrics["f1_macro"]
            patience_counter = 0
            save_checkpoint(state, best_checkpoint_path, is_best=True, best_path=best_checkpoint_path)
            logger.info(
                ">>> New Best Checkpoint (Epoch %d): Macro-F1=%.4f, AUC=%.4f, Loss=%.4f (saved to %s)",
                epoch,
                val_metrics["f1_macro"],
                val_metrics["roc_auc"],
                val_metrics["loss"],
                best_checkpoint_path.name,
            )
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience and epoch > warmup_epochs:
                logger.info("Early stopping triggered after %d epochs without improvement.", patience_counter)
                break

    total_training_time = time.time() - start_time
    logger.info("Training completed in %.1f minutes. Best Val Macro-F1: %.4f", total_training_time / 60.0, best_val_f1)

    # 5. Save History DataFrame and Learning Curves
    df_hist = pd.DataFrame(history)
    csv_path = OUTPUTS_DIR / f"training_history_{model_name}.csv"
    df_hist.to_csv(csv_path, index=False)
    logger.info("Saved training history to %s", csv_path)

    plot_training_curves(df_hist, model_name, OUTPUTS_DIR / f"training_curves_{model_name}.png")

    return {
        "model_name": model_name,
        "best_val_f1": best_val_f1,
        "history": df_hist,
        "best_checkpoint": best_checkpoint_path,
    }


def plot_training_curves(df: pd.DataFrame, model_name: str, save_path: Path) -> None:
    """Generate high-resolution learning curves figure."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Loss Curves
    axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train Loss", color="#2b5c8f", lw=2.5)
    axes[0, 0].plot(df["epoch"], df["val_loss"], label="Val Loss", color="#c0392b", lw=2.5)
    axes[0, 0].set_title(f"{model_name.upper()}: Loss Progression", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Cross-Entropy Loss")
    axes[0, 0].legend()

    # 2. Accuracy Curves
    axes[0, 1].plot(df["epoch"], df["train_acc"], label="Train Acc", color="#2b5c8f", lw=2.5)
    axes[0, 1].plot(df["epoch"], df["val_acc"], label="Val Acc", color="#27ae60", lw=2.5)
    axes[0, 1].set_title(f"{model_name.upper()}: Accuracy Progression", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()

    # 3. Macro-F1 & ROC-AUC
    axes[1, 0].plot(df["epoch"], df["train_f1"], label="Train F1 (Macro)", color="#2b5c8f", lw=2.0, ls="--")
    axes[1, 0].plot(df["epoch"], df["val_f1_macro"], label="Val F1 (Macro)", color="#e67e22", lw=2.5)
    axes[1, 0].plot(df["epoch"], df["val_roc_auc"], label="Val ROC-AUC", color="#8e44ad", lw=2.5)
    axes[1, 0].set_title(f"{model_name.upper()}: Validation F1 & ROC-AUC", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Metric Score")
    axes[1, 0].legend()

    # 4. Learning Rate Schedule
    axes[1, 1].plot(df["epoch"], df["lr"], label="Learning Rate", color="#34495e", lw=2.0)
    axes[1, 1].set_title("Cosine Annealing LR Schedule", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_yscale("log")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info("Saved training curves plot to %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="Train video classification models for shoplifting detection.")
    parser.add_argument("--model", type=str, default="cnn_rnn", choices=["cnn_rnn", "cnn3d", "both"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    args = parser.parse_args()

    models_to_train = ["cnn_rnn", "cnn3d"] if args.model == "both" else [args.model]

    for m in models_to_train:
        train_model(
            model_name=m,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_head=args.lr_head,
            lr_backbone=args.lr_backbone,
            warmup_epochs=args.warmup_epochs,
        )


if __name__ == "__main__":
    main()
