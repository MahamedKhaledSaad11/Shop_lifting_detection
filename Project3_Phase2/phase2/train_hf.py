"""Fine-tuning Hugging Face VideoMAE models (VideoMAE-Small and VideoMAE-Base) on shoplifting videos.

Features:
- In-memory cached dataset (FastShopliftingDataset) for ultra-fast training (~6-12s/epoch).
- Layer-wise Learning Rate Decay (LLRD) with layer_decay=0.75, head_lr_mult=10.0, wd=0.0 on bias/norm.
- Per-optimizer-step linear warmup (first 2 epochs) + cosine annealing decay.
- Head dropout (0.2), label smoothing cross-entropy (0.1), AMP fp16, and gradient clipping (1.0).
- Tie-breaking checkpoint selection: (val_f1_macro, val_roc_auc, -val_loss).
- Full artifact logging: checkpoints, per-epoch CSVs, learning curves, and test metrics JSON.
"""

import argparse
import json
import math
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
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from phase2.cached_dataset import FastShopliftingDataset
from phase2.config import (
    CHECKPOINTS_DIR,
    DEVICE,
    LOGS_DIR,
    METRICS_DIR,
    PLOTS_DIR,
    SPLITS_PATH,
    VIDEOMAE_BASE_ID,
    VIDEOMAE_SMALL_ID,
)
from phase2.hf_models import build_hf_model, get_llrd_param_groups
from phase2.metrics import evaluate_model


def set_seed(seed: int = 42) -> None:
    """Set random seed across all libraries for deterministic reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.02,
) -> LambdaLR:
    """Construct a per-optimizer-step linear warmup followed by cosine annealing decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)

    return LambdaLR(optimizer, lr_lambda)


def train_one_hf_model(
    model_name: str,
    epochs: int = 15,
    micro_batch_size: int = 4,
    grad_accum_steps: int = 4,
    base_lr: float = 5e-5,
    head_lr_mult: float = 10.0,
    layer_decay: float = 0.75,
    weight_decay: float = 0.05,
    head_dropout: float = 0.2,
    gradient_checkpointing: bool = False,
    patience: int = 5,
    warmup_epochs: int = 2,
    train_ds: Optional[FastShopliftingDataset] = None,
    val_ds: Optional[FastShopliftingDataset] = None,
    test_ds: Optional[FastShopliftingDataset] = None,
    device: torch.device = DEVICE,
) -> Dict[str, Any]:
    """Fine-tune a Hugging Face VideoMAE model and evaluate on the test split."""
    if model_name == "videomae_small":
        hf_repo_id = VIDEOMAE_SMALL_ID
        ckpt_filename = "best_videomae_small.pt"
        log_filename = "history_videomae_small.csv"
        display_name = "VideoMAE-Small (~22M)"
    elif model_name == "videomae_base":
        hf_repo_id = VIDEOMAE_BASE_ID
        ckpt_filename = "best_videomae_base.pt"
        log_filename = "history_videomae_base.csv"
        display_name = "VideoMAE-Base (~86M)"
    else:
        raise ValueError(f"Unknown HF model name: {model_name}")

    print("\n" + "=" * 80, flush=True)
    print(f"FINE-TUNING HUGGING FACE MODEL: {display_name}", flush=True)
    print(f"HF Repo: {hf_repo_id}", flush=True)
    print(f"Epochs: {epochs} | Micro-Batch: {micro_batch_size} | Accum Steps: {grad_accum_steps} (Effective: {micro_batch_size * grad_accum_steps})", flush=True)
    print(f"Base LR: {base_lr} | Head Mult: {head_lr_mult} | Layer Decay: {layer_decay} | Weight Decay: {weight_decay}", flush=True)
    print(f"Head Dropout: {head_dropout} | Gradient Checkpointing: {gradient_checkpointing} | Patience: {patience}", flush=True)
    print("=" * 80, flush=True)

    set_seed(42)

    # 1. In-memory cached datasets (reusing pre-instantiated caches if passed)
    if train_ds is None:
        train_ds = FastShopliftingDataset(SPLITS_PATH, split="train", T=16, image_size=224)
    if val_ds is None:
        val_ds = FastShopliftingDataset(SPLITS_PATH, split="val", T=16, image_size=224)
    if test_ds is None:
        test_ds = FastShopliftingDataset(SPLITS_PATH, split="test", T=16, image_size=224)

    train_loader = DataLoader(
        train_ds,
        batch_size=micro_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=micro_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    print(f"Dataset sizes -> Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}", flush=True)

    # 2. Build Model with adapted head and head dropout
    model = build_hf_model(
        model_name=hf_repo_id,
        num_labels=2,
        drop_path_rate=0.1,
        head_dropout=head_dropout,
        gradient_checkpointing=gradient_checkpointing,
        verify_processor=True,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: Total={total_params:,} | Trainable={trainable_params:,}", flush=True)

    # 3. LLRD Parameter Groups & Optimization
    param_groups = get_llrd_param_groups(
        model=model,
        base_lr=base_lr,
        head_lr_mult=head_lr_mult,
        layer_decay=layer_decay,
        weight_decay=weight_decay,
    )
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    num_micro_batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(num_micro_batches_per_epoch / grad_accum_steps)
    total_optimizer_steps = optimizer_steps_per_epoch * epochs
    warmup_steps = optimizer_steps_per_epoch * warmup_epochs

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
        min_lr_ratio=0.02,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # 4. Training Loop
    best_key = (-1.0, -1.0, float("-inf"))
    best_epoch = -1
    best_val_metrics: Dict[str, Any] = {}
    patience_counter = 0
    history: List[Dict[str, Any]] = []

    checkpoint_path = CHECKPOINTS_DIR / ckpt_filename
    history_csv_path = LOGS_DIR / log_filename

    start_train_time = time.time()
    global_step = 0

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

            is_accum_step = ((batch_idx + 1) % grad_accum_steps == 0) or (
                batch_idx + 1 == num_micro_batches_per_epoch
            )

            if is_accum_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        train_loss = train_loss_total / max(train_samples, 1)

        val_metrics, _, _ = evaluate_model(
            model=model,
            dataloader=val_loader,
            device=device,
            criterion=criterion,
            use_tta=False,
            threshold=0.5,
        )

        current_head_lr = optimizer.param_groups[0]["lr"]
        current_base_lr = optimizer.param_groups[2]["lr"] if len(optimizer.param_groups) > 2 else current_head_lr
        epoch_time = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "head_lr": round(current_head_lr, 7),
            "base_lr": round(current_base_lr, 7),
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
            f"Head LR: {current_head_lr:.2e} | "
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
                    "hf_repo_id": hf_repo_id,
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

    test_metrics_single, _, _ = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        criterion=criterion,
        use_tta=False,
        threshold=0.5,
    )

    test_metrics_tta, _, _ = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        criterion=criterion,
        use_tta=True,
        threshold=0.5,
    )

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
        "hf_repo_id": hf_repo_id,
        "best_epoch": best_epoch,
        "total_train_time_s": round(total_train_time, 2),
        "val_metrics": best_val_metrics,
        "test_metrics_single": test_metrics_single,
        "test_metrics_tta": test_metrics_tta,
        "history": history,
    }


def plot_hf_curves(histories: Dict[str, List[Dict[str, Any]]], output_path: Path) -> None:
    """Plot training and validation curves for fine-tuned Hugging Face models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    styles = {
        "videomae_small": {"color": "tab:green", "label": "VideoMAE-Small (~22M)"},
        "videomae_base": {"color": "tab:purple", "label": "VideoMAE-Base (~86M)"},
    }

    for name, hist in histories.items():
        df = pd.DataFrame(hist)
        cfg = styles.get(name, {"color": "tab:gray", "label": name})

        axes[0].plot(df["epoch"], df["train_loss"], label=f"{cfg['label']} (Train)", linestyle="--", color=cfg["color"], alpha=0.7)
        axes[0].plot(df["epoch"], df["val_loss"], label=f"{cfg['label']} (Val)", linestyle="-", color=cfg["color"], linewidth=2)

        axes[1].plot(df["epoch"], df["val_f1_macro"], label=f"{cfg['label']} Val F1", color=cfg["color"], linewidth=2)
        axes[2].plot(df["epoch"], df["val_roc_auc"], label=f"{cfg['label']} Val AUC", color=cfg["color"], linewidth=2)

    axes[0].set_title("Cross-Entropy Loss (Label Smoothing 0.1)")
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

    plt.suptitle("Hugging Face VideoMAE Fine-Tuning Curves (LLRD + Cosine Warmup)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved learning curves to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Hugging Face VideoMAE models.")
    parser.add_argument("--model", type=str, default="all", choices=["videomae_small", "videomae_base", "all"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--micro_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--base_lr", type=float, default=5e-5)
    parser.add_argument("--head_lr_mult", type=float, default=10.0)
    parser.add_argument("--layer_decay", type=float, default=0.75)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    args = parser.parse_args()

    models_to_train = ["videomae_small", "videomae_base"] if args.model == "all" else [args.model]

    # Pre-cache datasets once so VideoMAE-Small and VideoMAE-Base share the exact same in-memory cache!
    print("Initializing shared in-memory video caches for VideoMAE...", flush=True)
    shared_train_ds = FastShopliftingDataset(SPLITS_PATH, split="train", T=16, image_size=224)
    shared_val_ds = FastShopliftingDataset(SPLITS_PATH, split="val", T=16, image_size=224)
    shared_test_ds = FastShopliftingDataset(SPLITS_PATH, split="test", T=16, image_size=224)

    results_all = {}
    histories_all = {}

    for m_name in models_to_train:
        grad_ckpt = True if m_name == "videomae_base" else False

        res = train_one_hf_model(
            model_name=m_name,
            epochs=args.epochs,
            micro_batch_size=args.micro_batch_size,
            grad_accum_steps=args.grad_accum_steps,
            base_lr=args.base_lr,
            head_lr_mult=args.head_lr_mult,
            layer_decay=args.layer_decay,
            gradient_checkpointing=grad_ckpt,
            patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            train_ds=shared_train_ds,
            val_ds=shared_val_ds,
            test_ds=shared_test_ds,
        )
        results_all[m_name] = res
        histories_all[m_name] = res["history"]

    results_json_path = METRICS_DIR / "hf_training_summary.json"
    with open(results_json_path, "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\nSaved full HF training summary to {results_json_path}", flush=True)

    plot_hf_curves(histories_all, PLOTS_DIR / "learning_curves_hf.png")


if __name__ == "__main__":
    main()
