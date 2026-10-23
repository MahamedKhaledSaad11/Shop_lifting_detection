"""Configuration module for Shoplifting Detection project."""

import os
from dataclasses import dataclass
from pathlib import Path
import torch

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "Shop DataSet"
NON_SHOPLIFTERS_DIR = DATA_DIR / "non shop lifters"
SHOPLIFTERS_DIR = DATA_DIR / "shop lifters"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
METADATA_PATH = OUTPUTS_DIR / "metadata.csv"
SPLITS_PATH = OUTPUTS_DIR / "splits.csv"

# Ensure essential output directories exist
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Class mappings
CLASS_NAMES = {0: "normal", 1: "shoplifting"}
CLASS_TO_IDX = {"normal": 0, "shoplifting": 1}

# Hardware
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2  # Balanced for Windows DataLoader stability

# Reproducibility
RANDOM_SEED = 42

# Default Data & Training Config
@dataclass
class DataConfig:
    num_frames: int = 16  # T frames sampled uniformly
    image_size: int = 224  # 224 for 2D ResNet, 112 for 3D CNN
    crop_scale: tuple = (0.85, 1.0)
    crop_ratio: tuple = (0.9, 1.35)  # Preserves wide PAL CCTV 704x576 field of view
    jitter_brightness: float = 0.2
    jitter_contrast: float = 0.2
    jitter_saturation: float = 0.2
    jitter_hue: float = 0.05
    flip_prob: float = 0.5
    mean: tuple = (0.485, 0.456, 0.406)  # ImageNet default
    std: tuple = (0.229, 0.224, 0.225)
    kinetics_mean: tuple = (0.43216, 0.394666, 0.37645)  # Kinetics-400 default
    kinetics_std: tuple = (0.22803, 0.22145, 0.216989)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

@dataclass
class TrainConfig:
    batch_size: int = 8
    epochs: int = 25
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    gradient_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    use_amp: bool = True
