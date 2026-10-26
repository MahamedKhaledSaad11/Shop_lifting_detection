"""Configuration and hyperparameters for Phase 2 fine-tuning and evaluation."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple
import torch

from phase2._paths import PROJECT_ROOT, PHASE2_ROOT

# Data paths
DATA_DIR = PROJECT_ROOT / "Shop DataSet"
SPLITS_PATH = PROJECT_ROOT / "outputs" / "splits.csv"

# Phase 2 outputs
OUTPUTS_DIR = PHASE2_ROOT / "outputs"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PLOTS_DIR = OUTPUTS_DIR / "plots"
LOGS_DIR = OUTPUTS_DIR / "logs"

for dir_path in (OUTPUTS_DIR, CHECKPOINTS_DIR, METRICS_DIR, PLOTS_DIR, LOGS_DIR):
    dir_path.mkdir(parents=True, exist_ok=True)

# Hugging Face model IDs
VIDEOMAE_BASE_ID = "MCG-NJU/videomae-base-finetuned-kinetics"
VIDEOMAE_SMALL_ID = "MCG-NJU/videomae-small-finetuned-kinetics"
TIMESFORMER_BASE_ID = "facebook/timesformer-base-finetuned-k400"

# Target classes
CLASS_NAMES: Dict[int, str] = {0: "normal", 1: "shoplifting"}
CLASS_TO_IDX: Dict[str, int] = {"normal": 0, "shoplifting": 1}
ID2LABEL: Dict[int, str] = {0: "normal", 1: "shoplifting"}
LABEL2ID: Dict[str, int] = {"normal": 0, "shoplifting": 1}
NUM_CLASSES: int = 2

# Standard ImageNet normalization used by VideoMAE and ClipTransform
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

# Default hardware settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2  # Stable multiprocessing on Windows


@dataclass
class Phase2Config:
    """Hyperparameters for VideoMAE fine-tuning on shoplifting videos."""

    # Video architecture
    model_name: str = VIDEOMAE_BASE_ID
    num_frames: int = 16
    image_size: int = 224
    num_labels: int = 2

    # Training schedule
    max_epochs: int = 20
    warmup_epochs: int = 2
    patience: int = 5
    seed: int = 42

    # Batching & Memory
    micro_batch_size: int = 4
    grad_accum_steps: int = 4  # Effective batch = micro_batch_size * grad_accum_steps = 16
    gradient_checkpointing: bool = True
    amp_fp16: bool = True

    # Optimization & Regularization
    base_lr: float = 5e-5
    head_lr_mult: float = 10.0
    layer_decay: float = 0.75
    weight_decay: float = 0.05
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    drop_path_rate: float = 0.1

    # Augmentation
    crop_scale: Tuple[float, float] = (0.85, 1.0)
    crop_ratio: Tuple[float, float] = (0.9, 1.35)
    flip_prob: float = 0.5
    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.05
