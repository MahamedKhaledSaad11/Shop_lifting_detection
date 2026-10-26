"""Tests verifying reuse of the Week 5 data pipeline and split file."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor

from phase2.config import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    SPLITS_PATH,
    VIDEOMAE_BASE_ID,
    VIDEOMAE_SMALL_ID,
)
from phase2.hf_models import build_hf_model

# Direct Week 5 imports to verify sys.path linkage
from src.data.dataset import ShopliftingDataset
from src.data.sampling import read_frames, sample_indices
from src.data.transforms import ClipTransform


class TestReusePipeline:
    """Validate seamless reuse of Week 5 modules and data splits."""

    def test_imports_succeed(self):
        """Verify Week 5 modules import without ImportError."""
        assert callable(sample_indices)
        assert callable(read_frames)
        assert issubclass(ClipTransform, object)
        assert issubclass(ShopliftingDataset, object)

    def test_videomae_processor_matches_clip_transform(self):
        """Verify AutoImageProcessor mean/std matches ClipTransform defaults."""
        processor = AutoImageProcessor.from_pretrained(VIDEOMAE_BASE_ID)
        transform = ClipTransform(is_train=False, image_size=224)

        assert np.allclose(processor.image_mean, transform.mean, atol=1e-4)
        assert np.allclose(processor.image_std, transform.std, atol=1e-4)
        assert np.allclose(processor.image_mean, IMAGENET_MEAN, atol=1e-4)
        assert np.allclose(processor.image_std, IMAGENET_STD, atol=1e-4)
        assert getattr(processor, "num_frames", 16) == 16

    def test_split_counts_deduplicated(self):
        """Verify exact split counts per implementation plan: 446 train, 93 val, 98 test."""
        assert SPLITS_PATH.exists(), f"Splits file not found at {SPLITS_PATH}"

        train_ds = ShopliftingDataset(SPLITS_PATH, split="train", deduplicate=True)
        val_ds = ShopliftingDataset(SPLITS_PATH, split="val", deduplicate=True)
        test_ds = ShopliftingDataset(SPLITS_PATH, split="test", deduplicate=True)

        assert len(train_ds) == 446, f"Expected 446 train clips, got {len(train_ds)}"
        assert len(val_ds) == 93, f"Expected 93 val clips, got {len(val_ds)}"
        assert len(test_ds) == 98, f"Expected 98 test clips, got {len(test_ds)}"

    def test_real_batch_from_dataset_feeds_hf_model(self):
        """Load one real batch of shoplifting videos and pass through VideoMAE-Small."""
        dataset = ShopliftingDataset(SPLITS_PATH, split="train", T=16, deduplicate=True)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        clips, labels = next(iter(loader))
        assert clips.shape == (2, 16, 3, 224, 224), f"Unexpected clip shape {clips.shape}"
        assert labels.shape == (2,), f"Unexpected labels shape {labels.shape}"
        assert clips.dtype == torch.float32

        # Build model and feed batch
        model = build_hf_model(
            model_name=VIDEOMAE_SMALL_ID,
            num_labels=2,
            gradient_checkpointing=False,
            verify_processor=False,
        )
        model.eval()

        with torch.no_grad():
            logits = model(clips)

        assert logits.shape == (2, 2), f"Expected logits shape (2, 2), got {logits.shape}"
        assert not torch.isnan(logits).any()
