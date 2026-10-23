"""Clip-consistent spatial and photometric transformations for video tensors."""

import random
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ColorJitter, RandomResizedCrop
import torchvision.transforms.functional as F

from src.config import (
    DataConfig,
    OUTPUTS_DIR,
)
from src.data.sampling import read_frames, sample_indices
from src.utils import setup_logger

logger = setup_logger("transforms")


class ClipTransform:
    """Applies clip-consistent spatial and photometric transforms across all T frames of a video.
    
    Guarantees:
    - Exactly the same crop coordinates (i, j, h, w) are applied to all frames.
    - Exactly the same horizontal flip decision is applied to all frames.
    - Exactly the same color jitter factors (brightness, contrast, saturation, hue) are applied to all frames.
    - Returns a float32 tensor of shape (T, C, H, W) normalized by dataset statistics.
    """

    def __init__(
        self,
        is_train: bool = True,
        image_size: int = 224,
        crop_scale: Tuple[float, float] = (0.7, 1.0),
        flip_prob: float = 0.5,
        color_jitter: Optional[Tuple[float, float, float, float]] = (0.2, 0.2, 0.2, 0.05),
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ):
        self.is_train = is_train
        self.image_size = image_size
        self.crop_scale = crop_scale
        self.flip_prob = flip_prob
        self.mean = list(mean)
        self.std = list(std)

        if color_jitter is not None:
            self.jitter = ColorJitter(
                brightness=color_jitter[0],
                contrast=color_jitter[1],
                saturation=color_jitter[2],
                hue=color_jitter[3],
            )
        else:
            self.jitter = None

    def __call__(self, frames: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Transform input frames into a normalized float32 tensor.
        
        Args:
            frames: Either:
                - np.ndarray of shape (T, H, W, C) uint8 RGB
                - torch.Tensor of shape (T, C, H, W) uint8 or float32
                
        Returns:
            torch.Tensor: Normalized tensor of shape (T, C, H, W) with dtype float32.
        """
        # Convert np.ndarray (T, H, W, C) -> torch.Tensor (T, C, H, W)
        if isinstance(frames, np.ndarray):
            # Transpose (T, H, W, C) -> (T, C, H, W)
            frames = torch.from_numpy(frames).permute(0, 3, 1, 2)
        elif isinstance(frames, torch.Tensor) and frames.ndim == 4 and frames.shape[-1] == 3:
            # (T, H, W, C) Tensor -> (T, C, H, W)
            frames = frames.permute(0, 3, 1, 2)

        if self.is_train:
            # 1. Random Resized Crop with consistent parameters across all frames
            i, j, h, w = RandomResizedCrop.get_params(
                frames, scale=self.crop_scale, ratio=(3.0 / 4.0, 4.0 / 3.0)
            )
            frames = F.resized_crop(
                frames,
                top=i,
                left=j,
                height=h,
                width=w,
                size=[self.image_size, self.image_size],
                antialias=True,
            )

            # 2. Consistent Horizontal Flip
            if random.random() < self.flip_prob:
                frames = F.hflip(frames)

            # 3. Consistent Color Jitter
            if self.jitter is not None:
                fn_idx, b, c, s, h_factor = self.jitter.get_params(
                    self.jitter.brightness,
                    self.jitter.contrast,
                    self.jitter.saturation,
                    self.jitter.hue,
                )
                for fn_id in fn_idx:
                    if fn_id == 0 and b is not None:
                        frames = F.adjust_brightness(frames, b)
                    elif fn_id == 1 and c is not None:
                        frames = F.adjust_contrast(frames, c)
                    elif fn_id == 2 and s is not None:
                        frames = F.adjust_saturation(frames, s)
                    elif fn_id == 3 and h_factor is not None:
                        frames = F.adjust_hue(frames, h_factor)

        else:
            # Validation / Test: Deterministic Center Crop pipeline
            # Resize shorter side to maintain aspect ratio, then center crop
            # Original aspect is 704x576 (PAL ~ 1.22 ratio)
            scale_factor = 256.0 / 224.0
            short_side = int(round(self.image_size * scale_factor))
            # H: short_side, W: int(round(short_side * 704 / 576))
            frames = F.resize(frames, size=[short_side], antialias=True)
            frames = F.center_crop(frames, output_size=[self.image_size, self.image_size])

        # Normalize to float32 in [0, 1] then standardize
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0
        elif frames.max() > 1.0:
            frames = frames.float() / 255.0
        else:
            frames = frames.float()

        frames = F.normalize(frames, mean=self.mean, std=self.std)
        return frames


def get_transforms(
    split: str = "train",
    config: Optional[DataConfig] = None,
) -> ClipTransform:
    """Factory helper to construct ClipTransform for a given dataset split."""
    cfg = config if config is not None else DataConfig()
    is_train = (split == "train")

    return ClipTransform(
        is_train=is_train,
        image_size=cfg.image_size,
        crop_scale=cfg.crop_scale,
        flip_prob=cfg.flip_prob if is_train else 0.0,
        color_jitter=(
            (
                cfg.jitter_brightness,
                cfg.jitter_contrast,
                cfg.jitter_saturation,
                cfg.jitter_hue,
            )
            if is_train
            else None
        ),
        mean=cfg.mean,
        std=cfg.std,
    )


def save_augmentation_check(
    video_path: Union[str, Path],
    output_path: Path = OUTPUTS_DIR / "aug_check.png",
    T: int = 8,
    image_size: int = 224,
) -> None:
    """Generate visual verification grid comparing original frames with augmented frames."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indices = sample_indices(300, T, mode="center")
    raw_frames = read_frames(video_path, indices)  # (T, H, W, 3) uint8

    # Apply train augmentation
    transform = ClipTransform(is_train=True, image_size=image_size)
    aug_tensor = transform(raw_frames.copy())  # (T, 3, H, W) normalized

    # Denormalize augmented frames for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    denorm = aug_tensor * std + mean
    denorm = torch.clamp(denorm, 0.0, 1.0).permute(0, 2, 3, 1).cpu().numpy()

    fig, axes = plt.subplots(2, T, figsize=(2.5 * T, 5))
    for t in range(T):
        # Top row: original sampled frame (resized for visual display)
        axes[0, t].imshow(raw_frames[t])
        axes[0, t].set_title(f"Orig t={t}", fontsize=10)
        axes[0, t].axis("off")

        # Bottom row: augmented frame
        axes[1, t].imshow(denorm[t])
        axes[1, t].set_title(f"Aug t={t}", fontsize=10)
        axes[1, t].axis("off")

    axes[0, 0].set_ylabel("ORIGINAL", fontsize=12, fontweight="bold", labelpad=10)
    axes[1, 0].set_ylabel("AUGMENTED", fontsize=12, fontweight="bold", labelpad=10)
    fig.suptitle(
        "Clip-Consistent Augmentation Verification (Identical Crop, Flip & Jitter across time)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info("Saved augmentation visual check to %s", output_path)
