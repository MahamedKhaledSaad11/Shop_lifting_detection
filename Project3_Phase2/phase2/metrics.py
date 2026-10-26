"""Shared evaluation metrics module for Phase 2 video classification models."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch
import torch.nn as nn


def compute_classification_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute complete classification metrics suite given probabilities and binary targets."""
    preds = (probs >= threshold).astype(int)
    acc = accuracy_score(targets, preds)
    f1_macro = f1_score(targets, preds, average="macro", zero_division=0)
    f1_shop = f1_score(targets, preds, pos_label=1, zero_division=0)
    f1_norm = f1_score(targets, preds, pos_label=0, zero_division=0)
    rec_macro = recall_score(targets, preds, average="macro", zero_division=0)
    rec_shop = recall_score(targets, preds, pos_label=1, zero_division=0)
    rec_norm = recall_score(targets, preds, pos_label=0, zero_division=0)
    prec_macro = precision_score(targets, preds, average="macro", zero_division=0)
    prec_shop = precision_score(targets, preds, pos_label=1, zero_division=0)
    prec_norm = precision_score(targets, preds, pos_label=0, zero_division=0)

    try:
        roc_auc = roc_auc_score(targets, probs)
    except Exception:
        roc_auc = 0.5

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(targets, probs)
        pr_auc = auc(recall_curve, precision_curve)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(targets, preds, labels=[0, 1]).tolist()
    brier = float(np.mean((probs - targets) ** 2))

    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_shoplifting": float(f1_shop),
        "f1_normal": float(f1_norm),
        "recall_macro": float(rec_macro),
        "recall_shoplifting": float(rec_shop),
        "recall_normal": float(rec_norm),
        "precision_macro": float(prec_macro),
        "precision_shoplifting": float(prec_shop),
        "precision_normal": float(prec_norm),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": brier,
        "confusion_matrix": cm,
        "threshold": float(threshold),
    }


def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    use_tta: bool = False,
    threshold: float = 0.5,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    """Evaluate model on a dataloader with optional TTA (horizontal flip).
    
    Returns:
        metrics: Dictionary of classification metrics (and loss if criterion provided).
        probs: Array of predicted probabilities for positive class (shoplifting).
        targets: Array of true class labels.
    """
    model.eval()
    all_probs = []
    all_targets = []
    total_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for clips, labels in dataloader:
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = clips.size(0)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(clips)
                if criterion is not None:
                    loss = criterion(logits, labels)
                    total_loss += loss.item() * batch_size

                probs_orig = torch.softmax(logits, dim=1)[:, 1]

                if use_tta:
                    clips_flip = torch.flip(clips, dims=[-1])
                    logits_flip = model(clips_flip)
                    probs_flip = torch.softmax(logits_flip, dim=1)[:, 1]
                    probs = (probs_orig + probs_flip) / 2.0
                else:
                    probs = probs_orig

            all_probs.extend(probs.cpu().float().numpy().tolist())
            all_targets.extend(labels.cpu().numpy().tolist())
            num_samples += batch_size

    probs_arr = np.array(all_probs)
    targets_arr = np.array(all_targets)
    metrics = compute_classification_metrics(probs_arr, targets_arr, threshold=threshold)
    if criterion is not None and num_samples > 0:
        metrics["loss"] = float(total_loss / num_samples)
    else:
        metrics["loss"] = None

    return metrics, probs_arr, targets_arr
