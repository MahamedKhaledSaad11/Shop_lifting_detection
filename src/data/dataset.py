"""PyTorch Dataset and DataLoader constructors for Shoplifting Detection."""

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.config import PROJECT_ROOT
from src.data.sampling import read_frames, sample_indices
from src.data.transforms import ClipTransform, get_transforms
from src.utils import setup_logger

logger = setup_logger("dataset")


class ShopliftingDataset(Dataset):
    """PyTorch Dataset for shoplifting video classification.
    
    Loads video clips and returns:
        clip: torch.Tensor of shape (T, C, H, W) with dtype float32
        label: torch.Tensor scalar of dtype torch.long (0 = normal, 1 = shoplifting)
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, str, Path],
        split: Optional[str] = None,
        T: int = 16,
        transform: Optional[Callable] = None,
        is_train: Optional[bool] = None,
        deduplicate: bool = False,
    ):
        """
        Args:
            data: DataFrame or path to CSV containing metadata/splits.
            split: Filter by split ('train', 'val', 'test', or None for all).
            T: Number of frames to sample per video.
            transform: Transformation callable applied to (T, H, W, C) uint8 frames.
            is_train: Whether in training mode (uses random sampling jitter if True).
                      If None, inferred from split == 'train'.
            deduplicate: If True, filters out exact duplicate files (is_duplicate == True).
        """
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        else:
            df = data.copy()

        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].copy()

        if deduplicate and "is_duplicate" in df.columns:
            df = df[~df["is_duplicate"]].copy()

        df = df.reset_index(drop=True)
        self.df = df
        self.T = T
        self.split = split

        if is_train is None:
            self.is_train = (split == "train")
        else:
            self.is_train = is_train

        # Setup transform if not provided
        if transform is not None:
            self.transform = transform
        else:
            self.transform = get_transforms("train" if self.is_train else "val")

        # Resolve paths
        self.paths = []
        for p in self.df["path"].tolist():
            path_obj = Path(p)
            if not path_obj.is_absolute():
                path_obj = (PROJECT_ROOT / path_obj).resolve()
            self.paths.append(path_obj)

        self.labels = self.df["label"].to_numpy(dtype=np.int64)
        if "n_frames" in self.df.columns:
            self.n_frames = self.df["n_frames"].to_numpy(dtype=np.int64)
        else:
            # Fallback: probe video frame count
            import cv2
            counts = []
            for p in self.paths:
                cap = cv2.VideoCapture(str(p))
                nf = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 100
                cap.release()
                counts.append(max(nf, 1))
            self.n_frames = np.array(counts, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        total_frames = int(self.n_frames[idx])
        mode = "random" if self.is_train else "center"

        # 1. Sample uniform indices
        indices = sample_indices(total_frames, self.T, mode=mode)

        # 2. Sequential frame decoding
        frames_rgb = read_frames(path, indices)  # (T, H, W, 3) uint8

        # 3. Apply clip-consistent transformations
        clip = self.transform(frames_rgb)  # (T, 3, H, W) float32

        # 4. Target label
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return clip, label

    def get_class_counts(self) -> Tuple[int, int]:
        """Return (count_normal, count_shoplifting)."""
        counts = np.bincount(self.labels, minlength=2)
        return int(counts[0]), int(counts[1])

    def get_sampler(self) -> WeightedRandomSampler:
        """Create a WeightedRandomSampler for class balance."""
        counts = np.bincount(self.labels, minlength=2)
        weights = 1.0 / np.maximum(counts, 1).astype(np.float64)
        sample_weights = weights[self.labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(self.labels),
            replacement=True,
        )
        return sampler


def create_dataloader(
    dataset: ShopliftingDataset,
    batch_size: int = 8,
    shuffle: Optional[bool] = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
) -> DataLoader:
    """Construct an optimized PyTorch DataLoader for ShopliftingDataset."""
    if use_weighted_sampler and dataset.is_train:
        sampler = dataset.get_sampler()
        shuffle = False
    else:
        sampler = None
        if shuffle is None:
            shuffle = dataset.is_train

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        drop_last=(dataset.is_train and len(dataset) > batch_size),
    )
