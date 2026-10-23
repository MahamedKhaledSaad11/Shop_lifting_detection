"""Unit tests for ShopliftingDataset and DataLoader."""

from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import METADATA_PATH, SPLITS_PATH
from src.data.dataset import ShopliftingDataset, create_dataloader
from src.data.transforms import ClipTransform


@pytest.fixture(scope="module")
def metadata_file():
    """Ensure metadata/splits CSV exists for tests."""
    if not SPLITS_PATH.exists():
        pytest.skip(f"Splits file {SPLITS_PATH} does not exist. Run src.data.index first.")
    return SPLITS_PATH


def test_dataset_single_item_shape_and_dtype(metadata_file):
    """Verify single __getitem__ returns (T, C, H, W) float32 tensor and scalar int64 label."""
    T = 16
    image_size = 224
    transform = ClipTransform(is_train=False, image_size=image_size)

    ds = ShopliftingDataset(
        data=metadata_file,
        split="val",
        T=T,
        transform=transform,
        deduplicate=True,
    )
    assert len(ds) > 0

    # Test first 3 samples of various durations
    for idx in range(min(3, len(ds))):
        clip, label = ds[idx]
        assert isinstance(clip, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert clip.shape == (T, 3, image_size, image_size)
        assert clip.dtype == torch.float32
        assert label.ndim == 0  # scalar
        assert label.dtype == torch.long
        assert label.item() in (0, 1)


def test_dataset_splits_lengths(metadata_file):
    """Verify train, val, and test splits have non-zero lengths and correct subsets."""
    ds_train = ShopliftingDataset(data=metadata_file, split="train", deduplicate=True)
    ds_val = ShopliftingDataset(data=metadata_file, split="val", deduplicate=True)
    ds_test = ShopliftingDataset(data=metadata_file, split="test", deduplicate=True)

    assert len(ds_train) > 0
    assert len(ds_val) > 0
    assert len(ds_test) > 0
    assert len(ds_train) > len(ds_val)
    assert len(ds_train) > len(ds_test)


def test_dataset_deduplication_filtering(metadata_file):
    """Verify deduplicate flag properly excludes exact duplicate clips."""
    ds_all = ShopliftingDataset(data=metadata_file, split="train", deduplicate=False)
    ds_dedup = ShopliftingDataset(data=metadata_file, split="train", deduplicate=True)

    assert len(ds_all) > len(ds_dedup)


def test_dataloader_batch_shape(metadata_file):
    """Verify DataLoader outputs batch tensor of shape (B, T, C, H, W)."""
    B = 4
    T = 8
    image_size = 112
    transform = ClipTransform(is_train=False, image_size=image_size)

    ds = ShopliftingDataset(
        data=metadata_file,
        split="test",
        T=T,
        transform=transform,
        deduplicate=True,
    )

    loader = create_dataloader(ds, batch_size=B, shuffle=False, num_workers=0)
    batch_clips, batch_labels = next(iter(loader))

    assert isinstance(batch_clips, torch.Tensor)
    assert isinstance(batch_labels, torch.Tensor)
    assert batch_clips.shape == (B, T, 3, image_size, image_size)
    assert batch_clips.dtype == torch.float32
    assert batch_labels.shape == (B,)
    assert batch_labels.dtype == torch.long


def test_dataloader_weighted_sampler(metadata_file):
    """Verify DataLoader initialization with WeightedRandomSampler."""
    ds = ShopliftingDataset(data=metadata_file, split="train", T=8, deduplicate=True)
    loader = create_dataloader(ds, batch_size=4, num_workers=0, use_weighted_sampler=True)

    batch_clips, batch_labels = next(iter(loader))
    assert batch_clips.shape[0] == 4
    assert batch_labels.shape[0] == 4
