"""Unit tests for uniform temporal frame sampling and resilient sequential video frame reading."""

from pathlib import Path
import numpy as np
import pytest

from src.data.sampling import read_frames, sample_indices


@pytest.mark.parametrize("n_frames", [1, 5, 16, 17, 100, 1000, 12345])
@pytest.mark.parametrize("T", [8, 16, 32])
@pytest.mark.parametrize("mode", ["center", "random"])
def test_sample_indices_properties(n_frames: int, T: int, mode: str):
    """Test mathematical and behavioral invariant properties of sample_indices."""
    indices = sample_indices(n_frames, T, mode=mode)

    # 1. Output length must equal T
    assert len(indices) == T, f"Expected {T} indices, got {len(indices)}"

    # 2. Output dtype must be integer
    assert np.issubdtype(indices.dtype, np.integer)

    # 3. All indices must be within [0, n_frames - 1]
    assert np.all(indices >= 0), f"Negative index found: {indices}"
    assert np.all(indices < n_frames), f"Index >= n_frames: {indices} (n_frames={n_frames})"

    # 4. Indices must be monotonically non-decreasing
    assert np.all(np.diff(indices) >= 0), f"Indices not non-decreasing: {indices}"

    # 5. Boundary proximity checks
    if n_frames >= T:
        skip = n_frames / T
        # First index should be within first segment [0, ceil(skip)]
        assert indices[0] <= int(np.ceil(skip))
        # Last index should be within last segment
        assert indices[-1] >= n_frames - int(np.ceil(skip)) - 1

    # 6. Center mode gap uniformity check when n_frames >= T
    if mode == "center" and n_frames >= T:
        diffs = np.diff(indices)
        spread = diffs.max() - diffs.min()
        # Spread of gaps should be at most 1 (or 2 for rounding edge cases)
        assert spread <= 2, f"Excessive gap spread in center mode: {diffs}"


def test_sample_indices_invalid_inputs():
    """Test that invalid n_frames or T raise ValueError."""
    with pytest.raises(ValueError):
        sample_indices(0, 16)
    with pytest.raises(ValueError):
        sample_indices(-5, 16)
    with pytest.raises(ValueError):
        sample_indices(100, 0)
    with pytest.raises(ValueError):
        sample_indices(100, -1)
    with pytest.raises(ValueError):
        sample_indices(100, 16, mode="invalid_mode")


def test_sample_indices_single_frame():
    """Test edge case where video has only 1 frame."""
    indices = sample_indices(1, 16, mode="center")
    assert len(indices) == 16
    assert np.all(indices == 0)


@pytest.mark.parametrize("video_path, expected_n_frames", [
    (Path("Shop DataSet/non shop lifters/shop_lifter_n_215.mp4"), 75),    # Short
    (Path("Shop DataSet/non shop lifters/shop_lifter_n_114.mp4"), 325),   # Median
    (Path("Shop DataSet/non shop lifters/shop_lifter_n_13.mp4"), 1850),   # Long
])
def test_read_frames_real_videos(video_path: Path, expected_n_frames: int):
    """Test reading frames on real short, median, and long video files."""
    if not video_path.exists():
        pytest.skip(f"Video file {video_path} does not exist.")

    T = 16
    indices = sample_indices(expected_n_frames, T, mode="center")
    frames = read_frames(video_path, indices)

    # Check shape: (T, H, W, 3)
    assert frames.ndim == 4
    assert frames.shape[0] == T
    assert frames.shape[1] == 576  # PAL height
    assert frames.shape[2] == 704  # PAL width
    assert frames.shape[3] == 3    # RGB channels
    assert frames.dtype == np.uint8

    # Ensure frames are not blank / all zeros
    assert frames.sum() > 0


def test_read_frames_graceful_fallback_beyond_length():
    """Test that requesting indices beyond video frame count gracefully repeats last frame."""
    video_path = Path("Shop DataSet/non shop lifters/shop_lifter_n_215.mp4")
    if not video_path.exists():
        pytest.skip(f"Video file {video_path} does not exist.")

    # Request indices far beyond actual 75 frames (e.g. up to frame 200)
    target_indices = [0, 10, 50, 70, 150, 200]
    frames = read_frames(video_path, target_indices)

    assert len(frames) == len(target_indices)
    assert frames.dtype == np.uint8
    # The last two frames (indices 150, 200 beyond EOF) should match the last valid decoded frame
    assert np.array_equal(frames[-1], frames[-2])


def test_read_frames_nonexistent_file_fallback():
    """Test fallback when video file cannot be opened."""
    fake_path = Path("Shop DataSet/non_existent_video_12345.mp4")
    target_indices = [0, 5, 10, 15]
    frames = read_frames(fake_path, target_indices, fallback_size=(224, 224))

    assert frames.shape == (4, 224, 224, 3)
    assert frames.dtype == np.uint8
    assert np.all(frames == 0)
