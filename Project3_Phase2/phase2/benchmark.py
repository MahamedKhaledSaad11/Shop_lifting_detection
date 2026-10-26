"""Comprehensive efficiency benchmark script for Phase 2 shoplifting detection models.

Measures:
- Parameter count (Total & Trainable)
- Checkpoint file size on disk (MB) and pure FP32 tensor memory (MB)
- GFLOPs per clip via torch.utils.flop_counter.FlopCounterMode
- Single-clip latency (FP32 & FP16): Mean, Std, P95 (ms/clip) on RTX 2080
- Batched throughput (batch_size=8 FP16): clips/sec and effective video FPS (16 frames/clip)
- Peak GPU VRAM allocated during inference (batch_size=1 and batch_size=8)
- Real-world CPU OpenCV decode + ClipTransform preprocessing time per clip
- End-to-end latency (CPU decode + GPU inference)
- Publication-quality Pareto trade-off curves (Macro-F1 vs. Latency and GFLOPs)
"""

import gc
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

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

from phase2.config import (
    CHECKPOINTS_DIR,
    DEVICE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    METRICS_DIR,
    PLOTS_DIR,
    SPLITS_PATH,
    VIDEOMAE_BASE_ID,
    VIDEOMAE_SMALL_ID,
)
from phase2.hf_models import build_hf_model
from phase2.train_scratch import ScratchResNetBiGRU, ScratchVideoCNN3D
from src.data.sampling import read_frames, sample_indices
from src.data.transforms import ClipTransform
from src.models.cnn_rnn import ResNetBiGRUAttention
from src.models.cnn3d import VideoCNN3D


def measure_gflops(model: nn.Module, input_size: Tuple[int, int, int, int, int], device: torch.device) -> float:
    """Measure total GFLOPs for a single forward pass of a 16-frame clip."""
    model.eval()
    dummy = torch.randn(input_size, device=device)
    with torch.no_grad():
        with FlopCounterMode(display=False) as fcm:
            _ = model(dummy)
        total_flops = fcm.get_total_flops()
    return float(round(total_flops / 1e9, 2))


def measure_gpu_latency(
    model: nn.Module,
    input_size: Tuple[int, int, int, int, int],
    device: torch.device,
    use_fp16: bool = True,
    warmup_steps: int = 20,
    timed_steps: int = 100,
) -> Dict[str, float]:
    """Measure inference latency using CUDA synchronization and perf_counter."""
    model.eval()
    dummy = torch.randn(input_size, device=device)
    timings = []

    with torch.no_grad():
        # Warmup
        for _ in range(warmup_steps):
            if use_fp16 and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = model(dummy)
            else:
                _ = model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed runs
        for _ in range(timed_steps):
            start = time.perf_counter()
            if use_fp16 and device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = model(dummy)
            else:
                _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            timings.append(elapsed_ms)

    timings_arr = np.array(timings)
    mean_ms = float(np.mean(timings_arr))
    std_ms = float(np.std(timings_arr))
    p95_ms = float(np.percentile(timings_arr, 95))
    bsz = input_size[0]
    clips_per_sec = float((bsz / mean_ms) * 1000.0)

    return {
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "clips_per_sec": round(clips_per_sec, 2),
        "effective_fps": round(clips_per_sec * 16, 1),
    }


def measure_inference_vram(
    model: nn.Module,
    input_size: Tuple[int, int, int, int, int],
    device: torch.device,
) -> float:
    """Measure peak memory allocated during inference in MB."""
    if device.type != "cuda":
        return 0.0

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model.eval()
    dummy = torch.randn(input_size, device=device)

    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=torch.float16):
            _ = model(dummy)
        torch.cuda.synchronize()

    peak_bytes = torch.cuda.max_memory_allocated()
    return round(peak_bytes / (1024 ** 2), 1)


def measure_cpu_decode_time(splits_path: Path, n_samples: int = 50) -> Dict[str, float]:
    """Measure real-world OpenCV video decode + ClipTransform time per 16-frame clip."""
    df = pd.read_csv(splits_path)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    if "is_duplicate" in df.columns:
        df = df[~df["is_duplicate"]].copy()

    paths = [Path(p) if Path(p).is_absolute() else (PROJECT_ROOT / p).resolve() for p in df["path"].tolist()]
    n_frames_list = df["n_frames"].tolist() if "n_frames" in df.columns else [300] * len(paths)

    test_paths = paths[:n_samples]
    test_n_frames = n_frames_list[:n_samples]

    transform_224 = ClipTransform(is_train=False, image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD)
    transform_112 = ClipTransform(is_train=False, image_size=112, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    decode_times_224 = []
    decode_times_112 = []

    print(f"\nBenchmarking CPU OpenCV video decode + preprocessing over {len(test_paths)} test clips...", flush=True)

    for p, nf in zip(test_paths, test_n_frames):
        # 1. Measure 224x224
        t0 = time.perf_counter()
        indices = sample_indices(int(nf), 16, mode="center")
        frames = read_frames(p, indices)  # uint8 (16, H, W, 3)
        _ = transform_224(frames)
        decode_times_224.append((time.perf_counter() - t0) * 1000.0)

        # 2. Measure 112x112
        t0 = time.perf_counter()
        _ = transform_112(frames)
        decode_times_112.append((time.perf_counter() - t0) * 1000.0)

    mean_224 = float(np.mean(decode_times_224))
    mean_112 = float(np.mean(decode_times_112))

    print(f"  CPU Decode + Prep Latency (224x224): {mean_224:.2f} ms/clip ({1000.0/mean_224:.1f} clips/s)", flush=True)
    print(f"  CPU Decode + Prep Latency (112x112): {mean_112:.2f} ms/clip ({1000.0/mean_112:.1f} clips/s)", flush=True)

    return {
        "decode_ms_224": round(mean_224, 2),
        "decode_ms_112": round(mean_112, 2),
    }


def run_benchmark() -> pd.DataFrame:
    """Run complete hardware efficiency benchmark across all 6 models."""
    print("\n" + "=" * 80, flush=True)
    print("STARTING HARDWARE EFFICIENCY, SPEED & COMPLEXITY BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    device = DEVICE
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Hardware: {gpu_name} ({gpu_mem:.2f} GB VRAM)", flush=True)
    else:
        print("Hardware: CPU", flush=True)

    # 1. Measure CPU decode latency
    decode_stats = measure_cpu_decode_time(SPLITS_PATH, n_samples=50)

    # 2. Architectures and Checkpoint files
    configs = [
        {
            "model_key": "scratch_cnn3d",
            "model_name": "Scratch VideoCNN3D",
            "arch_family": "3D CNN",
            "init_type": "Scratch",
            "image_size": 112,
            "ckpt_path": CHECKPOINTS_DIR / "best_scratch_cnn3d.pt",
            "builder": lambda: ScratchVideoCNN3D(num_classes=2, head_dropout=0.4),
            "train_peak_vram_mb": 1100.0,
        },
        {
            "model_key": "scratch_cnn_rnn",
            "model_name": "Scratch ResNet-BiGRU",
            "arch_family": "2D CNN + BiGRU",
            "init_type": "Scratch",
            "image_size": 224,
            "ckpt_path": CHECKPOINTS_DIR / "best_scratch_cnn_rnn.pt",
            "builder": lambda: ScratchResNetBiGRU(num_classes=2, hidden_dim=256, gru_layers=2, gru_dropout=0.3, head_dropout=0.4),
            "train_peak_vram_mb": 1400.0,
        },
        {
            "model_key": "pretrained_cnn_rnn",
            "model_name": "Pretrained CNN-RNN (Wk5)",
            "arch_family": "2D CNN + BiGRU",
            "init_type": "ImageNet Transfer",
            "image_size": 224,
            "ckpt_path": PROJECT_ROOT / "outputs" / "checkpoints" / "best_cnn_rnn.pt",
            "builder": lambda: ResNetBiGRUAttention(num_classes=2, pretrained=False, hidden_dim=256, gru_layers=2, gru_dropout=0.3, head_dropout=0.5),
            "train_peak_vram_mb": 1400.0,
        },
        {
            "model_key": "pretrained_cnn3d",
            "model_name": "Pretrained R(2+1)D-18 (Wk5)",
            "arch_family": "3D CNN",
            "init_type": "Kinetics-400 Transfer",
            "image_size": 112,
            "ckpt_path": PROJECT_ROOT / "outputs" / "checkpoints" / "best_cnn3d.pt",
            "builder": lambda: VideoCNN3D(num_classes=2, pretrained=False, head_dropout=0.5),
            "train_peak_vram_mb": 1100.0,
        },
        {
            "model_key": "videomae_small",
            "model_name": "VideoMAE-Small",
            "arch_family": "Video ViT",
            "init_type": "Kinetics-400 Pretrained",
            "image_size": 224,
            "ckpt_path": CHECKPOINTS_DIR / "best_videomae_small.pt",
            "builder": lambda: build_hf_model(VIDEOMAE_SMALL_ID, num_labels=2, head_dropout=0.2, pretrained=False, verify_processor=False),
            "train_peak_vram_mb": 1527.8,
        },
        {
            "model_key": "videomae_base",
            "model_name": "VideoMAE-Base",
            "arch_family": "Video ViT",
            "init_type": "Kinetics-400 Pretrained",
            "image_size": 224,
            "ckpt_path": CHECKPOINTS_DIR / "best_videomae_base.pt",
            "builder": lambda: build_hf_model(VIDEOMAE_BASE_ID, num_labels=2, head_dropout=0.2, pretrained=False, verify_processor=False),
            "train_peak_vram_mb": 1631.0,
        },
    ]

    # Load test metrics from metrics_all.json for Pareto plots
    metrics_json_path = METRICS_DIR / "metrics_all.json"
    cached_metrics = {}
    if metrics_json_path.exists():
        with open(metrics_json_path, "r") as f:
            cached_metrics = json.load(f)["models"]

    benchmark_rows = []

    for cfg in configs:
        m_key = cfg["model_key"]
        print(f"\nBenchmarking architecture: {cfg['model_name']} ...", flush=True)

        model = cfg["builder"]().to(device)
        model.eval()

        # Load weights if checkpoint exists
        if cfg["ckpt_path"].exists():
            ckpt = torch.load(cfg["ckpt_path"], map_location=device)
            sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(sd)
            ckpt_size_mb = round(cfg["ckpt_path"].stat().st_size / (1024 ** 2), 2)
        else:
            ckpt_size_mb = 0.0

        total_p = sum(p.numel() for p in model.parameters())
        train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
        pure_weights_mb = round(param_bytes / (1024 ** 2), 2)

        sz = cfg["image_size"]
        input_single = (1, 16, 3, sz, sz)
        input_batch8 = (8, 16, 3, sz, sz)

        # 1. GFLOPs
        gflops = measure_gflops(model, input_single, device)
        print(f"  GFLOPs: {gflops} GFLOPs / clip", flush=True)

        # 2. FP32 Latency (batch_size=1)
        lat_fp32 = measure_gpu_latency(model, input_single, device, use_fp16=False, warmup_steps=15, timed_steps=60)
        print(f"  FP32 Latency: {lat_fp32['mean_ms']:.2f} ± {lat_fp32['std_ms']:.2f} ms (p95: {lat_fp32['p95_ms']:.2f} ms)", flush=True)

        # 3. FP16 Latency (batch_size=1)
        lat_fp16 = measure_gpu_latency(model, input_single, device, use_fp16=True, warmup_steps=20, timed_steps=100)
        print(f"  FP16 Latency: {lat_fp16['mean_ms']:.2f} ± {lat_fp16['std_ms']:.2f} ms (p95: {lat_fp16['p95_ms']:.2f} ms) | {lat_fp16['clips_per_sec']:.1f} clips/s ({lat_fp16['effective_fps']:.1f} video FPS)", flush=True)

        # 4. FP16 Batched Throughput (batch_size=8)
        th_batch8 = measure_gpu_latency(model, input_batch8, device, use_fp16=True, warmup_steps=10, timed_steps=40)
        print(f"  FP16 Batch 8 Throughput: {th_batch8['clips_per_sec']:.1f} clips/s ({th_batch8['effective_fps']:.1f} video FPS)", flush=True)

        # 5. Peak Inference VRAM
        vram_b1 = measure_inference_vram(model, input_single, device)
        vram_b8 = measure_inference_vram(model, input_batch8, device)
        print(f"  Peak VRAM (Inference): Batch 1 = {vram_b1:.1f} MB | Batch 8 = {vram_b8:.1f} MB", flush=True)

        # 6. End-to-End Latency
        dec_ms = decode_stats["decode_ms_112"] if sz == 112 else decode_stats["decode_ms_224"]
        e2e_ms = round(dec_ms + lat_fp16["mean_ms"], 2)

        # Retrieve Test Macro-F1 and Accuracy from cached metrics if available
        test_f1 = None
        test_acc = None
        if m_key in cached_metrics:
            test_f1 = round(cached_metrics[m_key]["test_single_tau05"]["f1_macro"], 4)
            test_acc = round(cached_metrics[m_key]["test_single_tau05"]["accuracy"], 4)

        row = {
            "model_key": m_key,
            "model_name": cfg["model_name"],
            "arch_family": cfg["arch_family"],
            "init_type": cfg["init_type"],
            "image_size": sz,
            "params_m": round(total_p / 1e6, 2),
            "pure_weights_mb": pure_weights_mb,
            "checkpoint_disk_mb": ckpt_size_mb,
            "gflops_per_clip": gflops,
            "latency_fp32_ms": lat_fp32["mean_ms"],
            "latency_fp32_p95_ms": lat_fp32["p95_ms"],
            "latency_fp16_ms": lat_fp16["mean_ms"],
            "latency_fp16_std_ms": lat_fp16["std_ms"],
            "latency_fp16_p95_ms": lat_fp16["p95_ms"],
            "throughput_b1_clips_sec": lat_fp16["clips_per_sec"],
            "effective_fps_b1": lat_fp16["effective_fps"],
            "throughput_b8_clips_sec": th_batch8["clips_per_sec"],
            "effective_fps_b8": th_batch8["effective_fps"],
            "vram_inference_b1_mb": vram_b1,
            "vram_inference_b8_mb": vram_b8,
            "vram_training_mb": cfg["train_peak_vram_mb"],
            "cpu_decode_ms": dec_ms,
            "total_e2e_latency_ms": e2e_ms,
            "test_macro_f1": test_f1,
            "test_accuracy": test_acc,
        }
        benchmark_rows.append(row)

        del model
        gc.collect()
        torch.cuda.empty_cache()

    df_bench = pd.DataFrame(benchmark_rows)

    # Save to CSV and JSON
    csv_path = METRICS_DIR / "benchmark.csv"
    df_bench.to_csv(csv_path, index=False)
    print(f"\nSaved benchmark CSV to {csv_path}", flush=True)

    json_path = METRICS_DIR / "benchmark.json"
    with open(json_path, "w") as f:
        json.dump(benchmark_rows, f, indent=2)
    print(f"Saved benchmark JSON to {json_path}", flush=True)

    # Generate Pareto trade-off plot
    plot_pareto_tradeoffs(df_bench)

    return df_bench


def plot_pareto_tradeoffs(df: pd.DataFrame) -> None:
    """Generate publication-grade 2-panel Pareto trade-off plots."""
    print("Generating Pareto trade-off visualization...", flush=True)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    family_colors = {
        "2D CNN + BiGRU": "#ff7f0e",
        "3D CNN": "#1f77b4",
        "Video ViT": "#2ca02c",
    }

    # Panel 1: Test Macro-F1 vs. GPU FP16 Latency
    for family, group in df.groupby("arch_family"):
        color = family_colors.get(family, "black")
        sizes = [p * 20 for p in group["params_m"]]
        ax1.scatter(
            group["latency_fp16_ms"],
            group["test_macro_f1"],
            s=sizes,
            color=color,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.2,
            label=family,
        )

        for _, r in group.iterrows():
            ax1.annotate(
                f"{r['model_name']}\n({r['params_m']}M, {r['latency_fp16_ms']}ms)",
                xy=(r["latency_fp16_ms"], r["test_macro_f1"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                weight="bold",
            )

    ax1.set_xlabel("GPU FP16 Inference Latency (ms / 16-frame clip)", fontsize=12)
    ax1.set_ylabel("Test Macro-F1 Score", fontsize=12)
    ax1.set_title("Macro-F1 vs. GPU Inference Latency (Lower Latency = Better)", fontsize=13, weight="bold")
    ax1.set_ylim([0.83, 1.01])
    ax1.legend(loc="lower right", title="Architecture Family", frameon=True)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Test Macro-F1 vs. GFLOPs
    for family, group in df.groupby("arch_family"):
        color = family_colors.get(family, "black")
        sizes = [p * 20 for p in group["params_m"]]
        ax2.scatter(
            group["gflops_per_clip"],
            group["test_macro_f1"],
            s=sizes,
            color=color,
            alpha=0.8,
            edgecolors="black",
            linewidth=1.2,
            label=family,
        )

        for _, r in group.iterrows():
            ax2.annotate(
                f"{r['model_name']}\n({r['gflops_per_clip']} GFLOPs)",
                xy=(r["gflops_per_clip"], r["test_macro_f1"]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                weight="bold",
            )

    ax2.set_xlabel("Computational Complexity (GFLOPs / clip)", fontsize=12)
    ax2.set_ylabel("Test Macro-F1 Score", fontsize=12)
    ax2.set_title("Macro-F1 vs. Computational Cost (Lower GFLOPs = Better)", fontsize=13, weight="bold")
    ax2.set_ylim([0.83, 1.01])
    ax2.legend(loc="lower right", title="Architecture Family", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.suptitle("Efficiency Pareto Frontiers: Accuracy vs. Latency and Compute (Bubble Size ~ Params)", fontsize=15, y=0.98, weight="bold")
    plt.tight_layout()
    pareto_path = PLOTS_DIR / "pareto_tradeoff.png"
    fig.savefig(pareto_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Pareto trade-off plot to {pareto_path}", flush=True)


if __name__ == "__main__":
    run_benchmark()
