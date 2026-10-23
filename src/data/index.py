"""Dataset indexing, duplicate analysis, metadata extraction, and grouped splits."""

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold

from src.config import (
    DATA_DIR,
    EDA_DIR,
    METADATA_PATH,
    NON_SHOPLIFTERS_DIR,
    OUTPUTS_DIR,
    RANDOM_SEED,
    SHOPLIFTERS_DIR,
    SPLITS_PATH,
)
from src.utils import set_seed, setup_logger

logger = setup_logger("index")


def compute_file_hash(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    """Compute MD5 checksum of a video file."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_video_metadata(path: Path) -> Dict[str, any]:
    """Extract metadata (n_frames, fps, width, height, duration) using OpenCV."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {
            "n_frames": 0,
            "fps": 0.0,
            "width": 0,
            "height": 0,
            "duration_s": 0.0,
            "is_valid": False,
        }

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = (n_frames / fps) if fps > 0 else 0.0

    ret, _ = cap.read()
    cap.release()

    is_valid = bool(ret and n_frames > 0)
    return {
        "n_frames": n_frames,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "duration_s": round(duration_s, 2),
        "is_valid": is_valid,
    }


def build_metadata(
    non_shop_dir: Path = NON_SHOPLIFTERS_DIR,
    shop_dir: Path = SHOPLIFTERS_DIR,
) -> pd.DataFrame:
    """Scan dataset folders, compute hashes, extract OpenCV metadata, and assign group IDs."""
    logger.info("Scanning dataset folders for video files...")
    all_files: List[Tuple[Path, int, str]] = []
    
    for p in sorted(non_shop_dir.glob("*.mp4")):
        all_files.append((p, 0, "normal"))
    for p in sorted(shop_dir.glob("*.mp4")):
        all_files.append((p, 1, "shoplifting"))

    logger.info("Found %d video files (%d normal, %d shoplifting).",
                len(all_files),
                sum(1 for _, l, _ in all_files if l == 0),
                sum(1 for _, l, _ in all_files if l == 1))

    # Identify base names and duplicates in non_shop
    non_shop_stems = {p.stem for p, l, _ in all_files if l == 0}
    non_shop_bases = {
        s[:-2] if (s.endswith("_1") and s[:-2] in non_shop_stems) else s
        for s in non_shop_stems
    }

    records = []
    seen_hashes: Dict[str, str] = {}  # hash -> first encountered filename

    for path, label, label_name in all_files:
        md5_hash = compute_file_hash(path)
        meta = extract_video_metadata(path)
        file_size = path.stat().st_size

        stem = path.stem
        is_clone = False

        if label_name == "normal" and stem.endswith("_1"):
            base_stem = stem[:-2]
            if base_stem in non_shop_bases:
                base_id = base_stem
                is_clone = True
            else:
                base_id = stem
        else:
            base_id = stem

        group_id = f"{label_name}_{base_id}"

        # Exact duplicate detection via hash
        if md5_hash in seen_hashes:
            is_duplicate = True
            duplicate_of = seen_hashes[md5_hash]
        else:
            is_duplicate = False
            duplicate_of = None
            seen_hashes[md5_hash] = path.name

        rel_path = path.as_posix()

        records.append({
            "filename": path.name,
            "path": rel_path,
            "label": label,
            "label_name": label_name,
            "group_id": group_id,
            "is_duplicate": is_duplicate,
            "duplicate_of": duplicate_of,
            "md5": md5_hash,
            "file_size": file_size,
            "n_frames": meta["n_frames"],
            "fps": meta["fps"],
            "duration_s": meta["duration_s"],
            "width": meta["width"],
            "height": meta["height"],
            "is_valid": meta["is_valid"],
        })

    df = pd.DataFrame(records)
    logger.info("Metadata extraction completed. Total records: %d", len(df))
    logger.info("Exact duplicate files detected by MD5: %d", df["is_duplicate"].sum())
    logger.info("Unique video groups: %d (%d normal, %d shoplifting)",
                df["group_id"].nunique(),
                df[df.label == 0]["group_id"].nunique(),
                df[df.label == 1]["group_id"].nunique())
    return df


def split_dataset(
    df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Split dataset into 70% train, 15% val, 15% test using StratifiedGroupKFold.
    
    Guarantees:
    - Zero data leakage: all clips sharing the same group_id are assigned to the exact same split.
    - Balanced stratification across classes.
    """
    set_seed(seed)
    # Using 20 splits for exact 70% (14 folds), 15% (3 folds), 15% (3 folds)
    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    
    folds = np.zeros(len(df), dtype=int)
    for fold_idx, (_, val_idx) in enumerate(sgkf.split(df, df["label"], df["group_id"])):
        folds[val_idx] = fold_idx

    df = df.copy()
    df["fold"] = folds
    df["split"] = "train"
    df.loc[df["fold"].isin([14, 15, 16]), "split"] = "val"
    df.loc[df["fold"].isin([17, 18, 19]), "split"] = "test"

    # Verify zero leakage
    train_groups = set(df[df.split == "train"]["group_id"])
    val_groups = set(df[df.split == "val"]["group_id"])
    test_groups = set(df[df.split == "test"]["group_id"])

    assert len(train_groups.intersection(val_groups)) == 0, "Data leakage between train and val!"
    assert len(train_groups.intersection(test_groups)) == 0, "Data leakage between train and test!"
    assert len(val_groups.intersection(test_groups)) == 0, "Data leakage between val and test!"

    logger.info("Split verification passed with zero group leakage.")
    return df


def generate_eda_figures(df: pd.DataFrame, output_dir: Path = EDA_DIR) -> None:
    """Generate high-resolution EDA figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)

    # 1. Class Distribution (Raw vs Unique)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Raw counts
    raw_counts = df["label_name"].value_counts()[["normal", "shoplifting"]]
    sns.barplot(
        x=raw_counts.index,
        y=raw_counts.values,
        hue=raw_counts.index,
        legend=False,
        ax=axes[0],
        palette=["#2b5c8f", "#c0392b"],
    )
    axes[0].set_title(f"Raw Video Clips (Total = {len(df)})", fontweight="bold")
    axes[0].set_ylabel("Number of Video Clips")
    for idx, v in enumerate(raw_counts.values):
        pct = (v / len(df)) * 100
        axes[0].text(idx, v + 8, f"{v} ({pct:.1f}%)", ha="center", fontweight="bold")

    # Unique groups
    unique_df = df[~df["is_duplicate"]]
    uniq_counts = unique_df["label_name"].value_counts()[["normal", "shoplifting"]]
    sns.barplot(
        x=uniq_counts.index,
        y=uniq_counts.values,
        hue=uniq_counts.index,
        legend=False,
        ax=axes[1],
        palette=["#2b5c8f", "#c0392b"],
    )
    axes[1].set_title(f"Unique Video Events (Total = {len(unique_df)})", fontweight="bold")
    axes[1].set_ylabel("Number of Unique Events")
    for idx, v in enumerate(uniq_counts.values):
        pct = (v / len(unique_df)) * 100
        axes[1].text(idx, v + 8, f"{v} ({pct:.1f}%)", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=300)
    plt.close()
    logger.info("Saved class_distribution.png")

    # 2. Duration & Frame Count Distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Frame Count Hist
    sns.histplot(data=df, x="n_frames", hue="label_name", kde=True, bins=30, ax=axes[0, 0],
                 palette={"normal": "#2b5c8f", "shoplifting": "#c0392b"}, alpha=0.6)
    axes[0, 0].set_title("Frame Count Distribution (Histogram & KDE)", fontweight="bold")
    axes[0, 0].set_xlabel("Number of Frames")

    # Frame Count Boxplot
    sns.boxplot(
        data=df,
        x="label_name",
        y="n_frames",
        hue="label_name",
        legend=False,
        ax=axes[0, 1],
        palette={"normal": "#2b5c8f", "shoplifting": "#c0392b"},
    )
    axes[0, 1].set_title("Frame Count Boxplot by Class", fontweight="bold")
    axes[0, 1].set_ylabel("Number of Frames")

    # Duration Hist
    sns.histplot(
        data=df,
        x="duration_s",
        hue="label_name",
        kde=True,
        bins=30,
        ax=axes[1, 0],
        palette={"normal": "#2b5c8f", "shoplifting": "#c0392b"},
        alpha=0.6,
    )
    axes[1, 0].set_title("Video Duration Distribution (Seconds)", fontweight="bold")
    axes[1, 0].set_xlabel("Duration (seconds)")

    # Duration Boxplot
    sns.boxplot(
        data=df,
        x="label_name",
        y="duration_s",
        hue="label_name",
        legend=False,
        ax=axes[1, 1],
        palette={"normal": "#2b5c8f", "shoplifting": "#c0392b"},
    )
    axes[1, 1].set_title("Video Duration Boxplot by Class", fontweight="bold")
    axes[1, 1].set_ylabel("Duration (seconds)")

    plt.tight_layout()
    plt.savefig(output_dir / "duration_distribution.png", dpi=300)
    plt.close()
    logger.info("Saved duration_distribution.png")

    # 3. Splits Breakdown Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    split_order = ["train", "val", "test"]
    raw_split = pd.crosstab(df["split"], df["label_name"]).loc[split_order]
    raw_split.plot(kind="bar", stacked=False, ax=axes[0], color=["#2b5c8f", "#c0392b"])
    axes[0].set_title("Split Breakdown: All Clips (855 total)", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_xticklabels(split_order, rotation=0)

    uniq_split = pd.crosstab(unique_df["split"], unique_df["label_name"]).loc[split_order]
    uniq_split.plot(kind="bar", stacked=False, ax=axes[1], color=["#2b5c8f", "#c0392b"])
    axes[1].set_title("Split Breakdown: Unique Clips (637 total)", fontweight="bold")
    axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(split_order, rotation=0)

    plt.tight_layout()
    plt.savefig(output_dir / "splits_distribution.png", dpi=300)
    plt.close()
    logger.info("Saved splits_distribution.png")

    # 4. Sample Frames Grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    normal_samples = df[df.label == 0].sample(4, random_state=42)
    shop_samples = df[df.label == 1].sample(4, random_state=42)

    for col, (_, row) in enumerate(normal_samples.iterrows()):
        cap = cv2.VideoCapture(row["path"])
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_f // 2)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[0, col].imshow(frame_rgb)
        axes[0, col].set_title(f"Normal: {row['filename'][:16]}...", fontsize=10)
        axes[0, col].axis("off")

    for col, (_, row) in enumerate(shop_samples.iterrows()):
        cap = cv2.VideoCapture(row["path"])
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_f // 2)
        ret, frame = cap.read()
        cap.release()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            axes[1, col].imshow(frame_rgb)
        axes[1, col].set_title(f"Shoplifting: {row['filename'][:16]}...", fontsize=10)
        axes[1, col].axis("off")

    axes[0, 0].set_ylabel("NORMAL", fontsize=14, fontweight="bold", labelpad=15)
    axes[1, 0].set_ylabel("SHOPLIFTING", fontsize=14, fontweight="bold", labelpad=15)
    fig.suptitle("Sample Mid-Frames from Normal and Shoplifting Classes", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "sample_frames_grid.png", dpi=300)
    plt.close()
    logger.info("Saved sample_frames_grid.png")


def main() -> None:
    """Run full Phase 1 pipeline."""
    set_seed(RANDOM_SEED)
    df = build_metadata()
    df = split_dataset(df, seed=RANDOM_SEED)

    # Save outputs
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(METADATA_PATH, index=False)
    logger.info("Saved complete metadata to %s", METADATA_PATH)

    # Splits CSV
    splits_cols = [
        "filename",
        "path",
        "label",
        "label_name",
        "group_id",
        "is_duplicate",
        "n_frames",
        "fps",
        "duration_s",
        "width",
        "height",
        "split",
    ]
    df[splits_cols].to_csv(SPLITS_PATH, index=False)
    logger.info("Saved splits to %s", SPLITS_PATH)

    # Generate EDA figures
    generate_eda_figures(df)
    logger.info("Phase 1 metadata generation and EDA completed successfully.")


if __name__ == "__main__":
    main()
