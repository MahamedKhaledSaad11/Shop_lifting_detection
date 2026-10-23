"""Evaluation pipeline for test set benchmarking, TTA, threshold tuning, and ensembling."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
import torch.nn as nn
from torch.amp import autocast

from src.config import (
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    DEVICE,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SPLITS_PATH,
)
from src.data.dataset import ShopliftingDataset, create_dataloader
from src.data.transforms import get_transforms
from src.models.cnn_rnn import ResNetBiGRUAttention
from src.models.cnn3d import VideoCNN3D
from src.utils import set_seed, setup_logger

logger = setup_logger("evaluate")


def load_model(model_name: str, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load model architecture and weights from checkpoint."""
    if model_name == "cnn_rnn":
        model = ResNetBiGRUAttention(num_classes=2, pretrained=False)
    elif model_name == "cnn3d":
        model = VideoCNN3D(num_classes=2, pretrained=False)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded %s checkpoint from %s (Epoch %d)", model_name, checkpoint_path.name, checkpoint.get("epoch", -1))
    return model


def predict_dataset(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_tta: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Generate probability predictions and ground truth labels with optional TTA."""
    model.eval()
    all_probs: List[float] = []
    all_labels: List[int] = []
    all_filenames: List[str] = []

    with torch.no_grad():
        for batch_idx, (clips, labels) in enumerate(loader):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                logits_orig = model(clips)
                probs_orig = torch.softmax(logits_orig, dim=1)[:, 1]

                if use_tta:
                    # Test-Time Augmentation: horizontally flipped clip
                    clips_flipped = torch.flip(clips, dims=[-1])
                    logits_flip = model(clips_flipped)
                    probs_flip = torch.softmax(logits_flip, dim=1)[:, 1]
                    probs = (probs_orig + probs_flip) / 2.0
                else:
                    probs = probs_orig

            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    return np.array(all_probs), np.array(all_labels)


def find_optimal_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """Find decision threshold maximizing F1-score on validation set."""
    best_thresh = 0.5
    best_f1 = -1.0
    for thresh in np.linspace(0.1, 0.9, 81):
        preds = (probs >= thresh).astype(int)
        f1 = f1_score(targets, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return float(best_thresh)


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute complete classification metric suite."""
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec = recall_score(targets, preds, zero_division=0)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    f1_normal = f1_score(targets, preds, pos_label=0, zero_division=0)
    f1_shop = f1_score(targets, preds, pos_label=1, zero_division=0)

    try:
        roc_auc = roc_auc_score(targets, probs)
    except Exception:
        roc_auc = 0.5

    prec_curve, rec_curve, _ = precision_recall_curve(targets, probs)
    pr_auc = auc(rec_curve, prec_curve)
    cm = confusion_matrix(targets, preds).tolist()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_macro": float(f1_macro),
        "f1_normal": float(f1_normal),
        "f1_shoplifting": float(f1_shop),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "threshold": float(threshold),
    }


def plot_confusion_matrices(
    results: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot side-by-side confusion matrices for evaluated models."""
    sns.set_theme(style="white", font_scale=1.1)
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    labels = ["Normal (0)", "Shoplifting (1)"]

    for ax, (model_name, res) in zip(axes, results.items()):
        cm = np.array(res["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            annot_kws={"size": 14, "fontweight": "bold"},
        )
        ax.set_title(f"{model_name.upper()}\nAcc: {res['accuracy']:.3f} | F1: {res['f1_macro']:.3f}", fontweight="bold")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved confusion matrix plot to %s", output_path)


def plot_roc_pr_curves(
    model_curves: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot ROC curves and Precision-Recall curves side by side."""
    sns.set_theme(style="whitegrid", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    colors = {"cnn_rnn": "#2b5c8f", "cnn3d": "#e67e22", "ensemble": "#27ae60"}

    # 1. ROC Curves
    for model_name, data in model_curves.items():
        fpr, tpr, _ = roc_curve(data["targets"], data["probs"])
        roc_auc = data["metrics"]["roc_auc"]
        axes[0].plot(
            fpr,
            tpr,
            label=f"{model_name.upper()} (AUC = {roc_auc:.3f})",
            color=colors.get(model_name, "#34495e"),
            lw=2.5,
        )

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance")
    axes[0].set_title("Test Set: ROC Curves", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    # 2. Precision-Recall Curves
    for model_name, data in model_curves.items():
        prec, rec, _ = precision_recall_curve(data["targets"], data["probs"])
        pr_auc = data["metrics"]["pr_auc"]
        axes[1].plot(
            rec,
            prec,
            label=f"{model_name.upper()} (PR-AUC = {pr_auc:.3f})",
            color=colors.get(model_name, "#34495e"),
            lw=2.5,
        )

    axes[1].set_title("Test Set: Precision-Recall Curves", fontweight="bold")
    axes[1].set_xlabel("Recall (Shoplifting)")
    axes[1].set_ylabel("Precision (Shoplifting)")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved ROC and PR curves to %s", output_path)


def main():
    set_seed(RANDOM_SEED)
    device = DEVICE
    logger.info("Initializing comprehensive evaluation on device: %s", device)

    # 1. Prepare Datasets & Loaders
    # Test split: 98 unique video clips
    val_tf_2d = get_transforms("val")
    val_tf_2d.image_size = 224

    val_tf_3d = get_transforms("val")
    val_tf_3d.image_size = 112

    # Validation loaders for threshold tuning
    val_ds_2d = ShopliftingDataset(SPLITS_PATH, split="val", T=16, transform=val_tf_2d, deduplicate=True)
    val_ds_3d = ShopliftingDataset(SPLITS_PATH, split="val", T=16, transform=val_tf_3d, deduplicate=True)
    val_loader_2d = create_dataloader(val_ds_2d, batch_size=8, shuffle=False, num_workers=2)
    val_loader_3d = create_dataloader(val_ds_3d, batch_size=8, shuffle=False, num_workers=2)

    # Test loaders
    test_ds_2d = ShopliftingDataset(SPLITS_PATH, split="test", T=16, transform=val_tf_2d, deduplicate=True)
    test_ds_3d = ShopliftingDataset(SPLITS_PATH, split="test", T=16, transform=val_tf_3d, deduplicate=True)
    test_loader_2d = create_dataloader(test_ds_2d, batch_size=8, shuffle=False, num_workers=2)
    test_loader_3d = create_dataloader(test_ds_3d, batch_size=8, shuffle=False, num_workers=2)

    test_filenames = [p.name for p in test_ds_2d.paths]
    test_targets = test_ds_2d.labels

    logger.info("Test dataset size: %d clips (49 normal, 49 shoplifting)", len(test_ds_2d))

    # 2. Evaluate CNN-RNN
    ckpt_2d = CHECKPOINTS_DIR / "best_cnn_rnn.pt"
    model_2d = load_model("cnn_rnn", ckpt_2d, device)

    # Val predictions for threshold tuning
    val_probs_2d, val_targets = predict_dataset(model_2d, val_loader_2d, device, use_tta=True)
    best_thresh_2d = find_optimal_threshold(val_probs_2d, val_targets)
    logger.info("Optimal validation threshold for CNN-RNN: %.3f", best_thresh_2d)

    # Test predictions
    test_probs_2d, _ = predict_dataset(model_2d, test_loader_2d, device, use_tta=True)
    metrics_2d = compute_metrics(test_probs_2d, test_targets, threshold=0.5)
    metrics_2d_opt = compute_metrics(test_probs_2d, test_targets, threshold=best_thresh_2d)

    # 3. Evaluate CNN-3D
    ckpt_3d = CHECKPOINTS_DIR / "best_cnn3d.pt"
    model_3d = load_model("cnn3d", ckpt_3d, device)

    val_probs_3d, _ = predict_dataset(model_3d, val_loader_3d, device, use_tta=True)
    best_thresh_3d = find_optimal_threshold(val_probs_3d, val_targets)
    logger.info("Optimal validation threshold for CNN-3D: %.3f", best_thresh_3d)

    test_probs_3d, _ = predict_dataset(model_3d, test_loader_3d, device, use_tta=True)
    metrics_3d = compute_metrics(test_probs_3d, test_targets, threshold=0.5)
    metrics_3d_opt = compute_metrics(test_probs_3d, test_targets, threshold=best_thresh_3d)

    # 4. Soft-Voting Ensemble
    val_probs_ens = 0.5 * (val_probs_2d + val_probs_3d)
    best_thresh_ens = find_optimal_threshold(val_probs_ens, val_targets)
    logger.info("Optimal validation threshold for Ensemble: %.3f", best_thresh_ens)

    test_probs_ens = 0.5 * (test_probs_2d + test_probs_3d)
    metrics_ens = compute_metrics(test_probs_ens, test_targets, threshold=0.5)
    metrics_ens_opt = compute_metrics(test_probs_ens, test_targets, threshold=best_thresh_ens)

    # Summary Output
    all_results = {
        "cnn_rnn": metrics_2d,
        "cnn_rnn_optimal_threshold": metrics_2d_opt,
        "cnn3d": metrics_3d,
        "cnn3d_optimal_threshold": metrics_3d_opt,
        "ensemble": metrics_ens,
        "ensemble_optimal_threshold": metrics_ens_opt,
    }

    # Save metrics JSON
    metrics_json_path = OUTPUTS_DIR / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved all evaluation metrics to %s", metrics_json_path)

    # Plots
    cm_results = {
        "cnn_rnn (TTA)": metrics_2d,
        "cnn3d (TTA)": metrics_3d,
        "ensemble (TTA)": metrics_ens,
    }
    plot_confusion_matrices(cm_results, OUTPUTS_DIR / "confusion_matrix.png")

    curve_data = {
        "cnn_rnn": {"probs": test_probs_2d, "targets": test_targets, "metrics": metrics_2d},
        "cnn3d": {"probs": test_probs_3d, "targets": test_targets, "metrics": metrics_3d},
        "ensemble": {"probs": test_probs_ens, "targets": test_targets, "metrics": metrics_ens},
    }
    plot_roc_pr_curves(curve_data, OUTPUTS_DIR / "roc_pr_curves.png")

    # Error Analysis: Misclassified test clips
    test_preds_ens = (test_probs_ens >= 0.5).astype(int)
    errors = []
    for fn, target, prob, pred in zip(test_filenames, test_targets, test_probs_ens, test_preds_ens):
        if target != pred:
            errors.append({
                "filename": fn,
                "true_label": CLASS_NAMES[target],
                "pred_label": CLASS_NAMES[pred],
                "shoplifting_prob": round(float(prob), 4),
            })

    df_errors = pd.DataFrame(errors)
    error_csv = OUTPUTS_DIR / "error_analysis.csv"
    df_errors.to_csv(error_csv, index=False)
    logger.info("Saved error analysis (%d misclassified clips) to %s", len(df_errors), error_csv)

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION SUMMARY (Held-out 98 Unique Clips)")
    print("=" * 60)
    summary_df = pd.DataFrame([
        {
            "Model": "ResNet18-BiGRU-Attention (TTA)",
            "Threshold": "0.50",
            "Accuracy": f"{metrics_2d['accuracy'] * 100:.2f}%",
            "Macro F1": f"{metrics_2d['f1_macro']:.4f}",
            "Shoplifting F1": f"{metrics_2d['f1_shoplifting']:.4f}",
            "ROC-AUC": f"{metrics_2d['roc_auc']:.4f}",
            "PR-AUC": f"{metrics_2d['pr_auc']:.4f}",
        },
        {
            "Model": "R(2+1)D-18 3D CNN (TTA)",
            "Threshold": "0.50",
            "Accuracy": f"{metrics_3d['accuracy'] * 100:.2f}%",
            "Macro F1": f"{metrics_3d['f1_macro']:.4f}",
            "Shoplifting F1": f"{metrics_3d['f1_shoplifting']:.4f}",
            "ROC-AUC": f"{metrics_3d['roc_auc']:.4f}",
            "PR-AUC": f"{metrics_3d['pr_auc']:.4f}",
        },
        {
            "Model": "Soft-Voting Ensemble (TTA)",
            "Threshold": "0.50",
            "Accuracy": f"{metrics_ens['accuracy'] * 100:.2f}%",
            "Macro F1": f"{metrics_ens['f1_macro']:.4f}",
            "Shoplifting F1": f"{metrics_ens['f1_shoplifting']:.4f}",
            "ROC-AUC": f"{metrics_ens['roc_auc']:.4f}",
            "PR-AUC": f"{metrics_ens['pr_auc']:.4f}",
        },
        {
            "Model": "Soft-Voting Ensemble (Tuned)",
            "Threshold": f"{best_thresh_ens:.2f}",
            "Accuracy": f"{metrics_ens_opt['accuracy'] * 100:.2f}%",
            "Macro F1": f"{metrics_ens_opt['f1_macro']:.4f}",
            "Shoplifting F1": f"{metrics_ens_opt['f1_shoplifting']:.4f}",
            "ROC-AUC": f"{metrics_ens_opt['roc_auc']:.4f}",
            "PR-AUC": f"{metrics_ens_opt['pr_auc']:.4f}",
        },
    ])
    print(summary_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
