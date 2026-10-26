"""VRAM benchmark script for Milestone M1.

Measures peak GPU memory for a full fp16 training step (forward + backward + optimizer step)
across micro-batch sizes (2, 4) and gradient checkpointing (False, True) for VideoMAE-S and VideoMAE-B.
"""

import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure Phase 2 and Project3 root are on sys.path
PHASE2_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
for p in (PHASE2_ROOT, PROJECT_ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

import torch
import torch.nn as nn

from phase2.config import (
    OUTPUTS_DIR,
    VIDEOMAE_BASE_ID,
    VIDEOMAE_SMALL_ID,
)
from phase2.hf_models import build_hf_model, get_llrd_param_groups


def measure_vram_train_step(
    model_name: str,
    batch_size: int,
    gradient_checkpointing: bool,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Measure peak VRAM in MB for a single fp16 train step."""
    if not torch.cuda.is_available():
        return {
            "model": model_name,
            "batch_size": batch_size,
            "grad_checkpointing": gradient_checkpointing,
            "peak_vram_mb": None,
            "peak_vram_gb": None,
            "status": "CUDA not available",
        }

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    result = {
        "model": model_name,
        "batch_size": batch_size,
        "grad_checkpointing": gradient_checkpointing,
        "peak_vram_mb": None,
        "peak_vram_gb": None,
        "status": "SUCCESS",
    }

    try:
        # Build model and optimizer
        model = build_hf_model(
            model_name=model_name,
            num_labels=2,
            gradient_checkpointing=gradient_checkpointing,
            verify_processor=False,
        ).to(device)
        model.train()

        param_groups = get_llrd_param_groups(model)
        optimizer = torch.optim.AdamW(param_groups)
        scaler = torch.amp.GradScaler("cuda")
        criterion = nn.CrossEntropyLoss()

        x = torch.randn(batch_size, 16, 3, 224, 224, device=device)
        y = torch.randint(0, 2, (batch_size,), device=device)

        # 1. Warm-up step
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 2. Reset peak memory and run measured step
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()

        peak_bytes = torch.cuda.max_memory_allocated()
        peak_mb = peak_bytes / (1024 ** 2)
        peak_gb = peak_bytes / (1024 ** 3)

        result["peak_vram_mb"] = round(peak_mb, 2)
        result["peak_vram_gb"] = round(peak_gb, 3)

    except torch.cuda.OutOfMemoryError as e:
        result["status"] = "OOM (>8GB)"
    except Exception as e:
        result["status"] = f"ERROR: {str(e)}"
    finally:
        del model, optimizer, scaler, criterion, x, y, logits, loss
        gc.collect()
        torch.cuda.empty_cache()

    return result


def run_benchmark() -> List[Dict[str, Any]]:
    """Run full benchmark matrix for VideoMAE-S and VideoMAE-B."""
    matrix = [
        # (Model ID, Label, batch_size, grad_ckpt)
        (VIDEOMAE_SMALL_ID, "VideoMAE-S (~22M)", 2, False),
        (VIDEOMAE_SMALL_ID, "VideoMAE-S (~22M)", 2, True),
        (VIDEOMAE_SMALL_ID, "VideoMAE-S (~22M)", 4, False),
        (VIDEOMAE_SMALL_ID, "VideoMAE-S (~22M)", 4, True),
        (VIDEOMAE_BASE_ID, "VideoMAE-B (~86M)", 2, False),
        (VIDEOMAE_BASE_ID, "VideoMAE-B (~86M)", 2, True),
        (VIDEOMAE_BASE_ID, "VideoMAE-B (~86M)", 4, False),
        (VIDEOMAE_BASE_ID, "VideoMAE-B (~86M)", 4, True),
    ]

    results = []
    print("\n" + "=" * 80)
    print("VRAM BENCHMARK FOR MILESTONE M1 (FP16 TRAIN STEP: FORWARD + BACKWARD + OPTIM)")
    print("=" * 80)

    for model_id, model_label, bsz, grad_ckpt in matrix:
        print(f"Testing {model_label} | Micro-Batch: {bsz} | Grad Checkpoint: {grad_ckpt} ...", end=" ", flush=True)
        res = measure_vram_train_step(model_id, bsz, grad_ckpt)
        res["model_label"] = model_label
        results.append(res)
        if res["status"] == "SUCCESS":
            print(f"Peak VRAM: {res['peak_vram_mb']:.1f} MB ({res['peak_vram_gb']:.2f} GB)")
        else:
            print(f"Result: {res['status']}")

    # Save to JSON
    output_path = OUTPUTS_DIR / "vram_benchmark_m1.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print markdown table
    print("\n### VRAM Benchmark Results Table")
    print("| Model | Params | Micro-Batch | Grad Checkpoint | Peak VRAM (MB) | Peak VRAM (GB) | Status |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for r in results:
        vram_mb = f"{r['peak_vram_mb']:.1f}" if r["peak_vram_mb"] is not None else "N/A"
        vram_gb = f"{r['peak_vram_gb']:.2f}" if r["peak_vram_gb"] is not None else "N/A"
        print(f"| {r['model_label']} | - | {r['batch_size']} | {r['grad_checkpointing']} | {vram_mb} | {vram_gb} | {r['status']} |")

    print(f"\nSaved benchmark results to {output_path}")
    return results


if __name__ == "__main__":
    run_benchmark()
