"""Unified evaluation script across all 6 models (+ VideoMAE ensemble) on the shoplifting test set.

Computes:
- Single-View and Flip-TTA metrics at tau=0.50
- Validation-tuned and max-margin threshold tau*
- 95% Bootstrap Confidence Intervals (2,000 resamples)
- Pairwise McNemar's exact binomial statistical tests
- Visualizations: 2x3 confusion matrix grid, ROC/PR curves, F1/Acc CI bar chart
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure paths
PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PHASE2_ROOT, PROJECT_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    auc,
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
from torch.utils.data import DataLoader

from phase2.cached_dataset import FastShopliftingDataset
from phase2.config import (
    CHECKPOINTS_DIR,
    CLASS_NAMES,
    DEVICE,
    METRICS_DIR,
    PLOTS_DIR,
    SPLITS_PATH,
    VIDEOMAE_BASE_ID,
    VIDEOMAE_SMALL_ID,
)
from phase2.hf_models import build_hf_model
from phase2.metrics import compute_classification_metrics, evaluate_model
from phase2.train_scratch import ScratchResNetBiGRU, ScratchVideoCNN3D
from src.models.cnn_rnn import ResNetBiGRUAttention
from src.models.cnn3d import VideoCNN3D


def compute_bootstrap_ci(
    targets: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    n_bootstraps: int = 2000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% empirical bootstrap confidence intervals for key classification metrics."""
    rng = np.random.RandomState(seed)
    n = len(targets)
    boot_acc, boot_f1, boot_rec_shop, boot_auc = [], [], [], []

    for _ in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        t_sample = targets[idx]
        p_sample = probs[idx]
        pred_sample = (p_sample >= threshold).astype(int)

        boot_acc.append(accuracy_score(t_sample, pred_sample))
        boot_f1.append(f1_score(t_sample, pred_sample, average="macro", zero_division=0))
        boot_rec_shop.append(recall_score(t_sample, pred_sample, pos_label=1, zero_division=0))
        if len(np.unique(t_sample)) == 2:
            boot_auc.append(roc_auc_score(t_sample, p_sample))

    def get_ci(arr):
        if not arr:
            return [None, None]
        return [round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)]

    return {
        "accuracy_ci95": get_ci(boot_acc),
        "f1_macro_ci95": get_ci(boot_f1),
        "recall_shoplifting_ci95": get_ci(boot_rec_shop),
        "roc_auc_ci95": get_ci(boot_auc),
    }


def compute_mcnemar_test(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    """Perform pairwise McNemar's exact binomial test between Model A and Model B."""
    corr_a = (preds_a == targets)
    corr_b = (preds_b == targets)

    # Discordant pairs
    b = int(np.sum(corr_a & (~corr_b)))  # Model A correct, Model B incorrect
    c = int(np.sum((~corr_a) & corr_b))  # Model A incorrect, Model B correct

    n_discordant = b + c
    if n_discordant == 0:
        p_value = 1.0
    else:
        # Exact two-sided binomial test with p0=0.5
        res = binomtest(min(b, c), n_discordant, 0.5, alternative="two-sided")
        p_value = float(res.pvalue)

    return {
        "b_model_a_only": b,
        "c_model_b_only": c,
        "n_discordant": n_discordant,
        "p_value": round(p_value, 5),
        "statistically_significant_05": bool(p_value < 0.05),
    }


def find_optimal_threshold(val_probs: np.ndarray, val_targets: np.ndarray) -> float:
    """Find validation-tuned or max-margin threshold tau*."""
    # Check if perfectly separable
    neg_probs = val_probs[val_targets == 0]
    pos_probs = val_probs[val_targets == 1]
    max_neg = float(np.max(neg_probs))
    min_pos = float(np.min(pos_probs))

    if min_pos > max_neg:
        # Linearly separable: choose exact max-margin midpoint
        tau_margin = (max_neg + min_pos) / 2.0
        return float(round(tau_margin, 4))

    # Otherwise grid search for threshold maximizing validation macro-F1
    best_thresh = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.15, 0.85, 71):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(val_targets, preds, average="macro", zero_division=0)
        rec = recall_score(val_targets, preds, pos_label=1, zero_division=0)
        # Prioritize high shoplifting recall (>= 0.95)
        if rec >= 0.95 and f1 > best_f1:
            best_f1 = f1
            best_thresh = t
        elif best_f1 < 0 and f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return float(round(best_thresh, 4))


def load_all_models(device: torch.device) -> Dict[str, Tuple[nn.Module, int]]:
    """Load all 6 model architectures and their respective trained weights."""
    models = {}

    # 1. Scratch CNN-RNN
    ckpt_scratch_rnn = CHECKPOINTS_DIR / "best_scratch_cnn_rnn.pt"
    m_scratch_rnn = ScratchResNetBiGRU(num_classes=2, hidden_dim=256, gru_layers=2, gru_dropout=0.3, head_dropout=0.4)
    if ckpt_scratch_rnn.exists():
        sd = torch.load(ckpt_scratch_rnn, map_location=device)["model_state_dict"]
        m_scratch_rnn.load_state_dict(sd)
    models["scratch_cnn_rnn"] = (m_scratch_rnn, 224)

    # 2. Scratch CNN3D
    ckpt_scratch_3d = CHECKPOINTS_DIR / "best_scratch_cnn3d.pt"
    m_scratch_3d = ScratchVideoCNN3D(num_classes=2, head_dropout=0.4)
    if ckpt_scratch_3d.exists():
        sd = torch.load(ckpt_scratch_3d, map_location=device)["model_state_dict"]
        m_scratch_3d.load_state_dict(sd)
    models["scratch_cnn3d"] = (m_scratch_3d, 112)

    # 3. Pretrained CNN-RNN (Week 5 Checkpoint)
    ckpt_wk5_rnn = PROJECT_ROOT / "outputs" / "checkpoints" / "best_cnn_rnn.pt"
    m_wk5_rnn = ResNetBiGRUAttention(num_classes=2, pretrained=False, hidden_dim=256, gru_layers=2, gru_dropout=0.3, head_dropout=0.5)
    if ckpt_wk5_rnn.exists():
        sd = torch.load(ckpt_wk5_rnn, map_location=device)["model_state_dict"]
        m_wk5_rnn.load_state_dict(sd)
    models["pretrained_cnn_rnn"] = (m_wk5_rnn, 224)

    # 4. Pretrained CNN3D (Week 5 Checkpoint)
    ckpt_wk5_3d = PROJECT_ROOT / "outputs" / "checkpoints" / "best_cnn3d.pt"
    m_wk5_3d = VideoCNN3D(num_classes=2, pretrained=False, head_dropout=0.5)
    if ckpt_wk5_3d.exists():
        sd = torch.load(ckpt_wk5_3d, map_location=device)["model_state_dict"]
        m_wk5_3d.load_state_dict(sd)
    models["pretrained_cnn3d"] = (m_wk5_3d, 112)

    # 5. VideoMAE-Small (Phase 2 Fine-Tuned)
    ckpt_vmae_s = CHECKPOINTS_DIR / "best_videomae_small.pt"
    m_vmae_s = build_hf_model(VIDEOMAE_SMALL_ID, num_labels=2, head_dropout=0.2, pretrained=False, verify_processor=False)
    if ckpt_vmae_s.exists():
        sd = torch.load(ckpt_vmae_s, map_location=device)["model_state_dict"]
        m_vmae_s.load_state_dict(sd)
    models["videomae_small"] = (m_vmae_s, 224)

    # 6. VideoMAE-Base (Phase 2 Fine-Tuned)
    ckpt_vmae_b = CHECKPOINTS_DIR / "best_videomae_base.pt"
    m_vmae_b = build_hf_model(VIDEOMAE_BASE_ID, num_labels=2, head_dropout=0.2, pretrained=False, verify_processor=False)
    if ckpt_vmae_b.exists():
        sd = torch.load(ckpt_vmae_b, map_location=device)["model_state_dict"]
        m_vmae_b.load_state_dict(sd)
    models["videomae_base"] = (m_vmae_b, 224)

    for m, _ in models.values():
        m.to(device)
        m.eval()

    return models


def run_unified_evaluation() -> Dict[str, Any]:
    """Execute unified evaluation protocol across all 6 models and ensemble."""
    print("\n" + "=" * 80, flush=True)
    print("STARTING UNIFIED EVALUATION ON EXACT 98 TEST CLIPS", flush=True)
    print("=" * 80, flush=True)

    device = DEVICE

    # 1. Load cached datasets for 224 and 112 resolutions
    print("Loading in-memory cached datasets (val & test)...", flush=True)
    val_ds_224 = FastShopliftingDataset(SPLITS_PATH, split="val", T=16, image_size=224)
    test_ds_224 = FastShopliftingDataset(SPLITS_PATH, split="test", T=16, image_size=224)

    val_ds_112 = FastShopliftingDataset(SPLITS_PATH, split="val", T=16, image_size=112)
    test_ds_112 = FastShopliftingDataset(SPLITS_PATH, split="test", T=16, image_size=112)

    val_loaders = {
        224: DataLoader(val_ds_224, batch_size=8, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")),
        112: DataLoader(val_ds_112, batch_size=8, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")),
    }
    test_loaders = {
        224: DataLoader(test_ds_224, batch_size=8, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")),
        112: DataLoader(test_ds_112, batch_size=8, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda")),
    }

    models_dict = load_all_models(device)

    # Containers for outputs
    raw_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    model_metrics: Dict[str, Dict[str, Any]] = {}
    bootstrap_cis: Dict[str, Dict[str, Any]] = {}

    criterion = nn.CrossEntropyLoss()

    for name, (model, res) in models_dict.items():
        print(f"\nEvaluating: {name.upper()} (resolution={res}x{res})...", flush=True)

        v_loader = val_loaders[res]
        t_loader = test_loaders[res]

        # 1. Get Val probabilities (for finding optimal tau*)
        _, val_probs, val_targets = evaluate_model(model, v_loader, device=device, criterion=criterion, use_tta=False)
        tau_star = find_optimal_threshold(val_probs, val_targets)
        print(f"  [Validation] Optimal / Max-Margin Threshold tau* = {tau_star:.4f}", flush=True)

        # 2. Get Test Single-View predictions
        test_m_single, test_probs_single, test_targets = evaluate_model(model, t_loader, device=device, criterion=criterion, use_tta=False, threshold=0.5)

        # 3. Get Test Flip-TTA predictions
        test_m_tta, test_probs_tta, _ = evaluate_model(model, t_loader, device=device, criterion=criterion, use_tta=True, threshold=0.5)

        # 4. Get Test predictions at tau*
        test_m_tuned = compute_classification_metrics(test_probs_single, test_targets, threshold=tau_star)
        test_m_tuned_tta = compute_classification_metrics(test_probs_tta, test_targets, threshold=tau_star)

        # 5. Compute 95% Bootstrap CIs for Single-View tau=0.5
        ci_95 = compute_bootstrap_ci(test_targets, test_probs_single, threshold=0.5, n_bootstraps=2000, seed=42)

        raw_predictions[name] = {
            "val_probs": val_probs,
            "test_probs_single": test_probs_single,
            "test_probs_tta": test_probs_tta,
            "targets": test_targets,
            "tau_star": tau_star,
        }

        model_metrics[name] = {
            "tau_star": tau_star,
            "test_single_tau05": test_m_single,
            "test_tta_tau05": test_m_tta,
            "test_single_taustar": test_m_tuned,
            "test_tta_taustar": test_m_tuned_tta,
            "bootstrap_ci95": ci_95,
        }

        bootstrap_cis[name] = ci_95

        print(f"  [Test Single tau=0.5] Acc: {test_m_single['accuracy']:.4f} | F1: {test_m_single['f1_macro']:.4f} [{ci_95['f1_macro_ci95'][0]:.4f}, {ci_95['f1_macro_ci95'][1]:.4f}] | AUC: {test_m_single['roc_auc']:.4f}", flush=True)
        print(f"  [Test Flip-TTA]      Acc: {test_m_tta['accuracy']:.4f} | F1: {test_m_tta['f1_macro']:.4f} | AUC: {test_m_tta['roc_auc']:.4f}", flush=True)
        print(f"  [Test tau* ({tau_star:.2f})]   Acc: {test_m_tuned['accuracy']:.4f} | F1: {test_m_tuned['f1_macro']:.4f} | Rec(Shop): {test_m_tuned['recall_shoplifting']:.4f}", flush=True)

    # 2. Evaluate VideoMAE Ensemble (Small + Base)
    print("\nEvaluating: VIDEOMAE_ENSEMBLE (Small + Base)...", flush=True)
    p_s_single = raw_predictions["videomae_small"]["test_probs_single"]
    p_b_single = raw_predictions["videomae_base"]["test_probs_single"]
    p_ens_single = 0.5 * (p_s_single + p_b_single)

    p_s_tta = raw_predictions["videomae_small"]["test_probs_tta"]
    p_b_tta = raw_predictions["videomae_base"]["test_probs_tta"]
    p_ens_tta = 0.5 * (p_s_tta + p_b_tta)

    test_targets = raw_predictions["videomae_base"]["targets"]

    val_ens = 0.5 * (raw_predictions["videomae_small"]["val_probs"] + raw_predictions["videomae_base"]["val_probs"])
    tau_star_ens = find_optimal_threshold(val_ens, val_targets)

    ens_single = compute_classification_metrics(p_ens_single, test_targets, threshold=0.5)
    ens_tta = compute_classification_metrics(p_ens_tta, test_targets, threshold=0.5)
    ens_tuned = compute_classification_metrics(p_ens_single, test_targets, threshold=tau_star_ens)
    ens_tuned_tta = compute_classification_metrics(p_ens_tta, test_targets, threshold=tau_star_ens)
    ens_ci = compute_bootstrap_ci(test_targets, p_ens_single, threshold=0.5, n_bootstraps=2000, seed=42)

    raw_predictions["videomae_ensemble"] = {
        "test_probs_single": p_ens_single,
        "test_probs_tta": p_ens_tta,
        "targets": test_targets,
        "tau_star": tau_star_ens,
    }

    model_metrics["videomae_ensemble"] = {
        "tau_star": tau_star_ens,
        "test_single_tau05": ens_single,
        "test_tta_tau05": ens_tta,
        "test_single_taustar": ens_tuned,
        "test_tta_taustar": ens_tuned_tta,
        "bootstrap_ci95": ens_ci,
    }
    bootstrap_cis["videomae_ensemble"] = ens_ci

    print(f"  [Ensemble Single] Acc: {ens_single['accuracy']:.4f} | F1: {ens_single['f1_macro']:.4f} [{ens_ci['f1_macro_ci95'][0]:.4f}, {ens_ci['f1_macro_ci95'][1]:.4f}] | AUC: {ens_single['roc_auc']:.4f}", flush=True)
    print(f"  [Ensemble TTA]    Acc: {ens_tta['accuracy']:.4f} | F1: {ens_tta['f1_macro']:.4f} | AUC: {ens_tta['roc_auc']:.4f}", flush=True)

    # 3. Pairwise McNemar's Statistical Significance Tests
    print("\n" + "=" * 80, flush=True)
    print("COMPUTING PAIRWISE MCNEMAR'S EXACT BINOMIAL TESTS", flush=True)
    print("=" * 80, flush=True)

    mcnemar_results = {}
    preds_map = {k: (v["test_probs_single"] >= 0.5).astype(int) for k, v in raw_predictions.items()}

    pairs_to_test = [
        ("videomae_base", "scratch_cnn3d"),
        ("videomae_base", "scratch_cnn_rnn"),
        ("videomae_base", "pretrained_cnn_rnn"),
        ("videomae_base", "pretrained_cnn3d"),
        ("videomae_base", "videomae_small"),
        ("videomae_small", "scratch_cnn3d"),
        ("videomae_small", "scratch_cnn_rnn"),
        ("videomae_small", "pretrained_cnn_rnn"),
    ]

    for m_a, m_b in pairs_to_test:
        test_res = compute_mcnemar_test(preds_map[m_a], preds_map[m_b], test_targets)
        pair_key = f"{m_a}_vs_{m_b}"
        mcnemar_results[pair_key] = test_res
        print(f"  {m_a} vs {m_b}: Discordant={test_res['n_discordant']} (b={test_res['b_model_a_only']}, c={test_res['c_model_b_only']}) | p-value={test_res['p_value']:.4f} | Sig(p<0.05)={test_res['statistically_significant_05']}", flush=True)

    # 4. Generate Master Comparison Table
    rows = []
    display_names = {
        "scratch_cnn3d": ("Scratch VideoCNN3D", "Scratch (Random)"),
        "scratch_cnn_rnn": ("Scratch ResNet-BiGRU", "Scratch (Random)"),
        "pretrained_cnn_rnn": ("Pretrained CNN-RNN (Wk5)", "Transfer (ImageNet)"),
        "pretrained_cnn3d": ("Pretrained R(2+1)D-18 (Wk5)", "Transfer (Kinetics-400)"),
        "videomae_small": ("VideoMAE-Small", "Pretrained (Kinetics-400)"),
        "videomae_base": ("VideoMAE-Base", "Pretrained (Kinetics-400)"),
        "videomae_ensemble": ("VideoMAE-Ensemble (S+B)", "Ensemble"),
    }

    for name in ["scratch_cnn3d", "scratch_cnn_rnn", "pretrained_cnn_rnn", "pretrained_cnn3d", "videomae_small", "videomae_base", "videomae_ensemble"]:
        m = model_metrics[name]
        s = m["test_single_tau05"]
        tta = m["test_tta_tau05"]
        ci = m["bootstrap_ci95"]
        d_name, init_type = display_names[name]

        f1_str = f"{s['f1_macro']:.4f} [{ci['f1_macro_ci95'][0]:.4f}, {ci['f1_macro_ci95'][1]:.4f}]"
        acc_str = f"{s['accuracy']:.4f} [{ci['accuracy_ci95'][0]:.4f}, {ci['accuracy_ci95'][1]:.4f}]"

        rows.append({
            "model_key": name,
            "model_name": d_name,
            "init_type": init_type,
            "test_accuracy": round(s["accuracy"], 4),
            "test_accuracy_ci": acc_str,
            "test_f1_macro": round(s["f1_macro"], 4),
            "test_f1_macro_ci": f1_str,
            "recall_shoplifting": round(s["recall_shoplifting"], 4),
            "precision_shoplifting": round(s["precision_shoplifting"], 4),
            "roc_auc": round(s["roc_auc"], 4),
            "pr_auc": round(s["pr_auc"], 4),
            "brier_score": round(s["brier_score"], 4),
            "flip_tta_accuracy": round(tta["accuracy"], 4),
            "flip_tta_f1_macro": round(tta["f1_macro"], 4),
            "flip_tta_roc_auc": round(tta["roc_auc"], 4),
            "optimal_tau": m["tau_star"],
            "test_f1_at_tau_star": round(m["test_single_taustar"]["f1_macro"], 4),
        })

    df_comp = pd.DataFrame(rows)
    csv_path = METRICS_DIR / "comparison_table.csv"
    df_comp.to_csv(csv_path, index=False)
    print(f"\nSaved comparison CSV to {csv_path}", flush=True)

    # 5. Save Complete Metrics JSON
    final_output = {
        "models": model_metrics,
        "mcnemar_tests": mcnemar_results,
        "comparison_table": rows,
    }
    json_path = METRICS_DIR / "metrics_all.json"
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"Saved full unified metrics JSON to {json_path}", flush=True)

    # 6. Plots
    generate_evaluation_plots(raw_predictions, model_metrics, df_comp)

    return final_output


def generate_evaluation_plots(
    raw_predictions: Dict[str, Dict[str, Any]],
    model_metrics: Dict[str, Dict[str, Any]],
    df_comp: pd.DataFrame,
) -> None:
    """Generate annotated confusion matrices, ROC/PR curves, and F1/Acc CI bar charts."""
    print("\nGenerating publication-grade evaluation plots...", flush=True)
    sns.set_theme(style="whitegrid", font_scale=1.0)

    model_order = [
        "scratch_cnn3d", "scratch_cnn_rnn",
        "pretrained_cnn3d", "pretrained_cnn_rnn",
        "videomae_small", "videomae_base",
    ]
    titles = [
        "Scratch VideoCNN3D (112x112)", "Scratch ResNet-BiGRU (224x224)",
        "Pretrained R(2+1)D-18 (Wk5)", "Pretrained CNN-RNN (Wk5)",
        "VideoMAE-Small (Phase 2)", "VideoMAE-Base (Phase 2)",
    ]

    # --- PLOT 1: 2x3 Confusion Matrix Grid ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (m_key, title) in enumerate(zip(model_order, titles)):
        cm = np.array(model_metrics[m_key]["test_single_tau05"]["confusion_matrix"])
        ax = axes[idx]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Normal", "Shoplifting"],
            yticklabels=["Normal", "Shoplifting"],
            annot_kws={"size": 14, "weight": "bold"},
        )
        f1_val = model_metrics[m_key]["test_single_tau05"]["f1_macro"]
        acc_val = model_metrics[m_key]["test_single_tau05"]["accuracy"]
        ax.set_title(f"{title}\nAcc: {acc_val:.1%} | F1: {f1_val:.4f}", fontsize=12, pad=10)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)

    plt.suptitle("Test Set Confusion Matrices Across Architectures (tau = 0.50, N = 98)", fontsize=15, y=0.99, weight="bold")
    plt.tight_layout()
    cm_plot_path = PLOTS_DIR / "confusion_matrices_all.png"
    fig.savefig(cm_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved confusion matrices plot to {cm_plot_path}", flush=True)

    # --- PLOT 2: Side-by-Side ROC and PR Curves ---
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(16, 7))

    colors = {
        "scratch_cnn3d": "#d62728",
        "scratch_cnn_rnn": "#ff7f0e",
        "pretrained_cnn3d": "#1f77b4",
        "pretrained_cnn_rnn": "#aec7e8",
        "videomae_small": "#2ca02c",
        "videomae_base": "#9467bd",
        "videomae_ensemble": "#17becf",
    }
    curve_labels = {
        "scratch_cnn3d": "Scratch VideoCNN3D",
        "scratch_cnn_rnn": "Scratch ResNet-BiGRU",
        "pretrained_cnn3d": "Pretrained R(2+1)D-18 (Wk5)",
        "pretrained_cnn_rnn": "Pretrained CNN-RNN (Wk5)",
        "videomae_small": "VideoMAE-Small",
        "videomae_base": "VideoMAE-Base",
        "videomae_ensemble": "VideoMAE-Ensemble (S+B)",
    }

    test_targets = raw_predictions["videomae_base"]["targets"]

    for m_key in list(curve_labels.keys()):
        probs = raw_predictions[m_key]["test_probs_single"]
        c = colors[m_key]
        lbl = curve_labels[m_key]

        # ROC Curve
        fpr, tpr, _ = roc_curve(test_targets, probs)
        roc_auc = auc(fpr, tpr)
        lw = 2.5 if "videomae" in m_key else 1.5
        ax_roc.plot(fpr, tpr, color=c, lw=lw, label=f"{lbl} (AUC = {roc_auc:.4f})")

        # PR Curve
        precision, recall, _ = precision_recall_curve(test_targets, probs)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, color=c, lw=lw, label=f"{lbl} (AUC = {pr_auc:.4f})")

    ax_roc.plot([0, 1], [0, 1], color="navy", lw=1.2, linestyle="--", alpha=0.6)
    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel("False Positive Rate", fontsize=12)
    ax_roc.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, weight="bold")
    ax_roc.legend(loc="lower right", fontsize=10)
    ax_roc.grid(True, alpha=0.3)

    ax_pr.set_xlim([-0.02, 1.02])
    ax_pr.set_ylim([-0.02, 1.02])
    ax_pr.set_xlabel("Recall (Shoplifting)", fontsize=12)
    ax_pr.set_ylabel("Precision (Shoplifting)", fontsize=12)
    ax_pr.set_title("Precision-Recall (PR) Curves", fontsize=13, weight="bold")
    ax_pr.legend(loc="lower left", fontsize=10)
    ax_pr.grid(True, alpha=0.3)

    plt.suptitle("Test Set Discriminative Performance: ROC and Precision-Recall Curves (N = 98)", fontsize=15, y=0.98, weight="bold")
    plt.tight_layout()
    roc_plot_path = PLOTS_DIR / "roc_pr_all.png"
    fig.savefig(roc_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved ROC & PR curves plot to {roc_plot_path}", flush=True)

    # --- PLOT 3: Grouped Bar Chart with 95% Bootstrap CI Error Bars ---
    fig, ax = plt.subplots(figsize=(14, 6))

    plot_df = df_comp[df_comp["model_key"].isin(model_order)].copy()
    x = np.arange(len(plot_df))
    width = 0.35

    acc_vals = plot_df["test_accuracy"].to_numpy()
    f1_vals = plot_df["test_f1_macro"].to_numpy()

    # Extract error bars from bootstrap CI
    acc_err_low = []
    acc_err_high = []
    f1_err_low = []
    f1_err_high = []

    for _, r in plot_df.iterrows():
        ci = model_metrics[r["model_key"]]["bootstrap_ci95"]
        a = r["test_accuracy"]
        f = r["test_f1_macro"]
        acc_err_low.append(max(0.0, a - ci["accuracy_ci95"][0]))
        acc_err_high.append(max(0.0, ci["accuracy_ci95"][1] - a))
        f1_err_low.append(max(0.0, f - ci["f1_macro_ci95"][0]))
        f1_err_high.append(max(0.0, ci["f1_macro_ci95"][1] - f))

    acc_yerr = [acc_err_low, acc_err_high]
    f1_yerr = [f1_err_low, f1_err_high]

    rects1 = ax.bar(x - width/2, acc_vals, width, yerr=acc_yerr, capsize=4, label="Accuracy", color="#4a7bb0", alpha=0.9)
    rects2 = ax.bar(x + width/2, f1_vals, width, yerr=f1_yerr, capsize=4, label="Macro-F1", color="#e87a5d", alpha=0.9)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Test Accuracy and Macro-F1 with 95% Bootstrap Confidence Intervals (2,000 Resamples)", fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model_name"], rotation=15, ha="right", fontsize=11)
    ax.set_ylim([0.75, 1.02])
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate values
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9, weight="bold")

    plt.tight_layout()
    ci_plot_path = PLOTS_DIR / "f1_accuracy_comparison.png"
    fig.savefig(ci_plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved F1 & Accuracy CI bar chart to {ci_plot_path}", flush=True)


if __name__ == "__main__":
    run_unified_evaluation()
