"""Uniform temporal frame sampling and resilient sequential video frame reading."""

import os
from collections import defaultdict
from pathlib import Path
from typing import Sequence, Union

import cv2
import numpy as np

from src.utils import setup_logger

logger = setup_logger("sampling")


def sample_indices(n_frames: int, T: int, mode: str = "center") -> np.ndarray:
    """Sample T indices uniformly across a video of n_frames.
    
    Args:
        n_frames: Total number of frames in the video (must be >= 1).
        T: Target number of frames to sample (must be >= 1).
        mode: Sampling mode:
            - 'center': deterministic midpoint of each uniform segment (validation/test).
            - 'random': random jittered offset within each uniform segment (TSN training style).
            - 'uniform': alias for 'center'.
            
    Returns:
        np.ndarray: 1D array of length T with non-decreasing integer frame indices in [0, n_frames - 1].
    """
    if n_frames < 1:
        raise ValueError(f"n_frames must be >= 1, got {n_frames}")
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")

    if n_frames == 1:
        return np.zeros(T, dtype=np.int64)

    skip = float(n_frames) / float(T)
    indices = np.empty(T, dtype=np.int64)

    if mode in ("center", "uniform"):
        offset = skip / 2.0
        for i in range(T):
            idx = int(np.floor(i * skip + offset))
            indices[i] = idx
    elif mode == "random":
        for i in range(T):
            # Sample offset uniformly in [0, skip)
            offset = np.random.uniform(0.0, skip)
            idx = int(np.floor(i * skip + offset))
            indices[i] = idx
        # Ensure non-decreasing ordering after random jitter within segments
        indices = np.sort(indices)
    else:
        raise ValueError(f"Unknown sampling mode '{mode}'. Choose 'center' or 'random'.")

    # Clip to valid bounds [0, n_frames - 1]
    indices = np.clip(indices, 0, n_frames - 1)
    return indices


def read_frames(
    path: Union[str, Path],
    target_indices: Sequence[int],
    fallback_size: tuple = (224, 224),
) -> np.ndarray:
    """Read specific frames sequentially from a video file with robust fallback.
    
    Sequential decoding avoids slow and imprecise seeking in MP4/H.264 video containers.
    If EOF is hit before all target indices are reached, the last valid frame is repeated.
    
    Args:
        path: Path to the video file.
        target_indices: Sequence of T frame indices (0-indexed).
        fallback_size: (height, width) for dummy black frame if video is completely unreadable.
        
    Returns:
        np.ndarray: Array of shape (T, H, W, 3) with uint8 RGB values.
    """
    target_indices = np.asarray(target_indices, dtype=np.int64)
    T = len(target_indices)
    if T == 0:
        return np.empty((0, fallback_size[0], fallback_size[1], 3), dtype=np.uint8)

    # Map target frame index -> list of positions in output array
    index_to_positions = defaultdict(list)
    for pos, idx in enumerate(target_indices):
        index_to_positions[int(idx)].append(pos)

    max_target_idx = int(np.max(target_indices))
    frames_out = [None] * T
    last_valid_frame: Optional[np.ndarray] = None

    video_path_str = str(path)
    cap = cv2.VideoCapture(video_path_str)

    if cap.isOpened():
        current_frame_idx = 0
        while cap.isOpened() and current_frame_idx <= max_target_idx:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Video ended early or frame decode failed
                break

            # If current frame index is one of the targets, convert BGR to RGB and store
            if current_frame_idx in index_to_positions:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_valid_frame = frame_rgb
                for pos in index_to_positions[current_frame_idx]:
                    frames_out[pos] = frame_rgb

            current_frame_idx += 1

        cap.release()
    else:
        logger.warning("Could not open video file: %s", video_path_str)

    # Handle missing frames due to early EOF or unreadable video
    missing_positions = [pos for pos, f in enumerate(frames_out) if f is None]
    if missing_positions:
        if last_valid_frame is not None:
            for pos in missing_positions:
                frames_out[pos] = last_valid_frame.copy()
        else:
            # Complete failure: generate black frames
            logger.warning(
                "Zero valid frames decoded from %s. Using black frame fallback.",
                video_path_str,
            )
            black_frame = np.zeros((fallback_size[0], fallback_size[1], 3), dtype=np.uint8)
            for pos in missing_positions:
                frames_out[pos] = black_frame.copy()

    return np.stack(frames_out, axis=0)
