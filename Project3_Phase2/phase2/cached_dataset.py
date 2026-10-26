"""In-memory cached dataset wrapper for high-throughput video training and instantaneous evaluation.

Reuses Week 5 modules:
- src.data.sampling.sample_indices
- src.data.sampling.read_frames
- src.data.transforms.ClipTransform
- outputs/splits.csv (deduplicate=True -> 446 train / 93 val / 98 test)
"""

import concurrent.futures
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from phase2._paths import PROJECT_ROOT
from phase2.config import IMAGENET_MEAN, IMAGENET_STD, SPLITS_PATH
from src.data.sampling import read_frames, sample_indices
from src.data.transforms import ClipTransform


class FastShopliftingDataset(Dataset):
    """In-memory dataset that pre-caches decoded frames in RAM.
    
    - For val and test (deterministic): precomputes and caches exact (16, C, H, W) float32 tensors.
    - For train (stochastic): pre-caches 32 uniformly spaced uint8 frames in RAM, then applies
      random temporal jitter + ClipTransform on-the-fly in RAM with zero disk I/O.
    """

    def __init__(
        self,
        data: Path = SPLITS_PATH,
        split: str = "train",
        T: int = 16,
        image_size: int = 224,
        is_train: Optional[bool] = None,
        cache_frames_count: int = 32,
        num_decode_workers: int = 8,
    ):
        df = pd.read_csv(data)
        if "split" in df.columns:
            df = df[df["split"] == split].copy()
        if "is_duplicate" in df.columns:
            df = df[~df["is_duplicate"]].copy()
        df = df.reset_index(drop=True)

        self.df = df
        self.split = split
        self.T = T
        self.image_size = image_size
        self.is_train = (split == "train") if is_train is None else is_train
        self.cache_frames_count = cache_frames_count

        # Build clip transform using Week 5 ClipTransform
        self.transform = ClipTransform(
            is_train=self.is_train,
            image_size=image_size,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )

        # Resolve paths
        self.paths: List[Path] = []
        for p in self.df["path"].tolist():
            p_obj = Path(p)
            if not p_obj.is_absolute():
                p_obj = (PROJECT_ROOT / p_obj).resolve()
            self.paths.append(p_obj)

        self.labels = self.df["label"].to_numpy(dtype=np.int64)
        self.n_frames = self.df["n_frames"].to_numpy(dtype=np.int64)

        # Pre-cache into RAM
        print(f"[{split.upper()} CACHE] Pre-extracting {len(self.paths)} clips (image_size={image_size}, is_train={self.is_train})...", flush=True)

        if not self.is_train:
            # Deterministic: extract and transform all tensors once
            self.cached_tensors: List[torch.Tensor] = [None] * len(self.paths)

            def decode_val_clip(idx: int):
                p = self.paths[idx]
                nf = int(self.n_frames[idx])
                indices = sample_indices(nf, self.T, mode="center")
                frames = read_frames(p, indices)  # (T, H, W, 3) uint8
                tensor = self.transform(frames)   # (T, 3, H, W) float32
                return idx, tensor

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_decode_workers) as executor:
                for idx, tensor in executor.map(decode_val_clip, range(len(self.paths))):
                    self.cached_tensors[idx] = tensor

            print(f"[{split.upper()} CACHE] Successfully cached {len(self.cached_tensors)} tensors in RAM.", flush=True)

        else:
            # Training: pre-cache cache_frames_count (32) uint8 frames per video
            pre_size = 256 if image_size == 224 else 144
            self.cached_uint8_frames: List[np.ndarray] = [None] * len(self.paths)

            def decode_train_clip(idx: int):
                p = self.paths[idx]
                nf = int(self.n_frames[idx])
                indices = sample_indices(nf, self.cache_frames_count, mode="center")
                frames = read_frames(p, indices)  # (K, H_orig, W_orig, 3) uint8
                resized = np.empty((frames.shape[0], pre_size, pre_size, 3), dtype=np.uint8)
                for f_idx in range(frames.shape[0]):
                    resized[f_idx] = cv2.resize(frames[f_idx], (pre_size, pre_size), interpolation=cv2.INTER_LINEAR)
                return idx, resized

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_decode_workers) as executor:
                for idx, frames_arr in executor.map(decode_train_clip, range(len(self.paths))):
                    self.cached_uint8_frames[idx] = frames_arr

            total_mb = sum(arr.nbytes for arr in self.cached_uint8_frames) / (1024 ** 2)
            print(f"[{split.upper()} CACHE] Successfully cached {len(self.cached_uint8_frames)} videos ({total_mb:.1f} MB in RAM).", flush=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if not self.is_train:
            return self.cached_tensors[idx], label

        # Training mode: sample 16 frames from the 32 cached frames with random jitter
        frames_32 = self.cached_uint8_frames[idx]
        indices = sample_indices(len(frames_32), self.T, mode="random")
        sampled_frames = frames_32[indices]  # (16, pre_size, pre_size, 3) uint8
        clip_tensor = self.transform(sampled_frames)  # (16, 3, image_size, image_size) float32
        return clip_tensor, label
