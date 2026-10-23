"""Unit tests for clip-consistent spatial and photometric video transforms."""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.data.transforms import ClipTransform, get_transforms, save_augmentation_check


def test_clip_transform_identical_input_frames():
    """Verify that identical input frames yield strictly identical output frames across time.
    
    This is the core completion requirement: spatial crop, flip, and color jitter
    must be applied consistently across the entire video clip.
    """
    T = 16
    H, W = 576, 704
    # Single random frame
    single_frame = torch.randint(0, 256, (1, 3, H, W), dtype=torch.uint8)
    # Replicate across T frames
    clip = single_frame.repeat(T, 1, 1, 1)

    # Test across multiple random seeds/initializations
    for seed in range(5):
        torch.manual_seed(seed)
        transform = ClipTransform(
            is_train=True,
            image_size=224,
            crop_scale=(0.6, 0.9),
            flip_prob=0.5,
            color_jitter=(0.3, 0.3, 0.3, 0.1),
        )
        out = transform(clip.clone())

        assert out.shape == (T, 3, 224, 224), f"Unexpected shape {out.shape}"
        assert out.dtype == torch.float32, f"Expected float32, got {out.dtype}"

        # Check every frame matches frame 0
        for k in range(1, T):
            assert torch.allclose(out[0], out[k], atol=1e-5), (
                f"Seed {seed}: Frame {k} differs from Frame 0! Max diff: "
                f"{(out[0] - out[k]).abs().max().item()}"
            )


def test_clip_transform_numpy_input():
    """Verify ClipTransform accepts numpy uint8 (T, H, W, C) arrays."""
    T = 8
    np_clip = np.random.randint(0, 256, size=(T, 576, 704, 3), dtype=np.uint8)
    transform = ClipTransform(is_train=True, image_size=112)
    out = transform(np_clip)

    assert isinstance(out, torch.Tensor)
    assert out.shape == (T, 3, 112, 112)
    assert out.dtype == torch.float32


def test_clip_transform_deterministic_eval():
    """Verify validation/test transforms are completely deterministic."""
    T = 8
    clip = torch.randint(0, 256, (T, 3, 576, 704), dtype=torch.uint8)
    transform = ClipTransform(is_train=False, image_size=224)

    out1 = transform(clip.clone())
    out2 = transform(clip.clone())

    assert torch.allclose(out1, out2), "Evaluation transforms should be deterministic!"


def test_get_transforms_factory():
    """Test transform factory function."""
    train_tf = get_transforms("train")
    assert train_tf.is_train is True

    val_tf = get_transforms("val")
    assert val_tf.is_train is False

    test_tf = get_transforms("test")
    assert test_tf.is_train is False


def test_save_augmentation_check(tmp_path):
    """Test generating visual verification grid on a real video."""
    video_path = Path("Shop DataSet/non shop lifters/shop_lifter_n_215.mp4")
    if not video_path.exists():
        pytest.skip(f"Video {video_path} not found.")

    output_path = tmp_path / "test_aug_check.png"
    save_augmentation_check(video_path, output_path=output_path, T=8, image_size=112)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
