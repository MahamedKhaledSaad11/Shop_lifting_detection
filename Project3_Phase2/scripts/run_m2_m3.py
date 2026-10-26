"""Master execution script for Milestone M2 (Scratch Baselines) and Milestone M3 (HF Fine-Tuning).

1. Evaluates existing scratch_cnn_rnn checkpoint and trains scratch_cnn3d (image_size=112).
2. Fine-tunes videomae_small and videomae_base (image_size=224).
3. Produces all learning curve plots and comprehensive metrics JSONs.
"""

import json
import sys
import time
from pathlib import Path

# Path setup
PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PHASE2_ROOT, PROJECT_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from phase2.cached_dataset import FastShopliftingDataset
from phase2.config import METRICS_DIR, PLOTS_DIR, SPLITS_PATH
from phase2.train_scratch import plot_scratch_curves, train_one_scratch_model
from phase2.train_hf import plot_hf_curves, train_one_hf_model


def execute_m2_scratch() -> dict:
    """Execute Milestone M2: True scratch baselines."""
    print("\n" + "#" * 80, flush=True)
    print("STEP 1: EXECUTING MILESTONE M2 (SCRATCH BASELINES)", flush=True)
    print("#" * 80, flush=True)

    scratch_results = {}
    scratch_histories = {}

    for model_name in ["scratch_cnn_rnn", "scratch_cnn3d"]:
        res = train_one_scratch_model(
            model_name=model_name,
            epochs=20,
            batch_size=8,
            grad_accum_steps=2,
            lr=3e-4,
            patience=6,
            force_retrain=False,  # preserve scratch_cnn_rnn
        )
        scratch_results[model_name] = res
        scratch_histories[model_name] = res["history"]

    scratch_json_path = METRICS_DIR / "scratch_results.json"
    with open(scratch_json_path, "w") as f:
        json.dump(scratch_results, f, indent=2)
    print(f"\n[M2] Saved scratch metrics to {scratch_json_path}", flush=True)

    scratch_plot_path = PLOTS_DIR / "learning_curves_scratch.png"
    plot_scratch_curves(scratch_histories, scratch_plot_path)

    return scratch_results


def execute_m3_hf() -> dict:
    """Execute Milestone M3: Hugging Face VideoMAE fine-tuning."""
    print("\n" + "#" * 80, flush=True)
    print("STEP 2: EXECUTING MILESTONE M3 (VIDEOMAE FINE-TUNING)", flush=True)
    print("#" * 80, flush=True)

    # Initialize shared in-memory video caches for 224x224
    print("\nInitializing shared 224x224 in-memory video cache for VideoMAE models...", flush=True)
    shared_train_ds = FastShopliftingDataset(SPLITS_PATH, split="train", T=16, image_size=224)
    shared_val_ds = FastShopliftingDataset(SPLITS_PATH, split="val", T=16, image_size=224)
    shared_test_ds = FastShopliftingDataset(SPLITS_PATH, split="test", T=16, image_size=224)

    hf_results = {}
    hf_histories = {}

    # 1. VideoMAE-Small: micro-batch 4, accum 4, gradient_checkpointing=False
    res_s = train_one_hf_model(
        model_name="videomae_small",
        epochs=15,
        micro_batch_size=4,
        grad_accum_steps=4,
        base_lr=5e-5,
        head_lr_mult=10.0,
        layer_decay=0.75,
        gradient_checkpointing=False,
        head_dropout=0.2,
        patience=5,
        warmup_epochs=2,
        train_ds=shared_train_ds,
        val_ds=shared_val_ds,
        test_ds=shared_test_ds,
    )
    hf_results["videomae_small"] = res_s
    hf_histories["videomae_small"] = res_s["history"]

    # 2. VideoMAE-Base: micro-batch 4, accum 4, gradient_checkpointing=True
    res_b = train_one_hf_model(
        model_name="videomae_base",
        epochs=15,
        micro_batch_size=4,
        grad_accum_steps=4,
        base_lr=5e-5,
        head_lr_mult=10.0,
        layer_decay=0.75,
        gradient_checkpointing=True,
        head_dropout=0.2,
        patience=5,
        warmup_epochs=2,
        train_ds=shared_train_ds,
        val_ds=shared_val_ds,
        test_ds=shared_test_ds,
    )
    hf_results["videomae_base"] = res_b
    hf_histories["videomae_base"] = res_b["history"]

    hf_json_path = METRICS_DIR / "hf_training_summary.json"
    with open(hf_json_path, "w") as f:
        json.dump(hf_results, f, indent=2)
    print(f"\n[M3] Saved HF fine-tuning metrics to {hf_json_path}", flush=True)

    hf_plot_path = PLOTS_DIR / "learning_curves_hf.png"
    plot_hf_curves(hf_histories, hf_plot_path)

    return hf_results


def main():
    total_start = time.time()

    # 1. Run M2 Scratch Training
    scratch_res = execute_m2_scratch()

    # 2. Run M3 HF Fine-Tuning
    hf_res = execute_m3_hf()

    # 3. Compile unified summary
    combined = {
        "scratch": scratch_res,
        "huggingface": hf_res,
        "total_wall_time_s": round(time.time() - total_start, 2),
    }

    summary_file = METRICS_DIR / "m2_m3_summary.json"
    with open(summary_file, "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print(f"ALL M2 & M3 RUNS COMPLETED IN {time.time() - total_start:.1f}s", flush=True)
    print(f"Summary saved to: {summary_file}", flush=True)
    print("=" * 80, flush=True)

    # Print preliminary comparison table
    print("\n### Test-Set Comparison Table (Single-View tau=0.5)", flush=True)
    print("| Model | Type | Best Epoch | Test Acc | Macro F1 | Recall (Shop) | Precision (Shop) | ROC-AUC |", flush=True)
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |", flush=True)
    for name, r in scratch_res.items():
        tm = r["test_metrics_single"]
        print(f"| {name} | Scratch (rand init) | {r['best_epoch']} | {tm['accuracy']:.4f} | {tm['f1_macro']:.4f} | {tm['recall_shoplifting']:.4f} | {tm['precision_shoplifting']:.4f} | {tm['roc_auc']:.4f} |", flush=True)
    for name, r in hf_res.items():
        tm = r["test_metrics_single"]
        print(f"| {name} | Pretrained HF | {r['best_epoch']} | {tm['accuracy']:.4f} | {tm['f1_macro']:.4f} | {tm['recall_shoplifting']:.4f} | {tm['precision_shoplifting']:.4f} | {tm['roc_auc']:.4f} |", flush=True)

    print("\n### Test-Set Comparison Table (Flip-TTA tau=0.5)", flush=True)
    print("| Model | Type | Best Epoch | Test Acc | Macro F1 | Recall (Shop) | ROC-AUC |", flush=True)
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |", flush=True)
    for name, r in scratch_res.items():
        tm = r["test_metrics_tta"]
        print(f"| {name} | Scratch (rand init) | {r['best_epoch']} | {tm['accuracy']:.4f} | {tm['f1_macro']:.4f} | {tm['recall_shoplifting']:.4f} | {tm['roc_auc']:.4f} |", flush=True)
    for name, r in hf_res.items():
        tm = r["test_metrics_tta"]
        print(f"| {name} | Pretrained HF | {r['best_epoch']} | {tm['accuracy']:.4f} | {tm['f1_macro']:.4f} | {tm['recall_shoplifting']:.4f} | {tm['roc_auc']:.4f} |", flush=True)


if __name__ == "__main__":
    main()
