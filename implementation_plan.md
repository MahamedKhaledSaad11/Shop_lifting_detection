# Shoplifting Detection: Implementation Plan

**Goal:** Binary video classification (normal vs. shoplifting) with a temporal deep model in PyTorch.
**Environment:** conda env `torch_gpu` (PyTorch + CUDA). Video decoding uses OpenCV (`cv2`).

## What's in the dataset so far

| Class | Folder | Clips | Size |
|---|---|---|---|
| Normal (label 0) | `Shop DataSet/non shop lifters` | 531 | ~1.25 GB |
| Shoplifting (label 1) | `Shop DataSet/shop lifters` | 324 | ~0.45 GB |

- All files are `.mp4`. The classes are **imbalanced (~62% / 38%)**.
- Some file names have a suffix, like `shop_lifter_n_0.mp4` and `shop_lifter_n_0_1.mp4`. These are probably clips cut from the same source video. If they end up on both sides of the split, the test score will be too high (**data leakage**). The split must be **grouped by base ID**.

---

## Project layout

```
Project3/
├── Shop DataSet/                 # raw data (git-ignored)
├── notebooks/01_eda.ipynb        # Phase 1
├── src/
│   ├── config.py                 # hyper-params, paths, seed
│   ├── data/
│   │   ├── index.py              # build metadata CSV + grouped splits
│   │   ├── sampling.py           # Phase 2: uniform frame sampler
│   │   ├── transforms.py         # Phase 3: clip-consistent augmentation
│   │   └── dataset.py            # Phase 4: ShopliftingDataset
│   ├── models/
│   │   ├── cnn_rnn.py            # Phase 5b: ResNet + GRU
│   │   └── cnn3d.py              # Phase 5a: R(2+1)D / R3D-18 (optional)
│   ├── train.py                  # Phase 6
│   ├── evaluate.py               # Phase 7
│   └── utils.py                  # seed, logging, checkpoint helpers
├── tests/                        # pytest checks for the completion requirements
├── outputs/                      # checkpoints, plots, metrics.json (git-ignored)
├── requirements.txt / environment.yml
└── README.md
```

---

## Phase 0: Setup
1. Activate `torch_gpu`. Check that `torch.cuda.is_available()` is true and that `torchvision`, `opencv-python`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn`, `tqdm` and `pytest` are installed.
2. Run `git init`. Add a `.gitignore` covering `Shop DataSet/`, `outputs/`, `*.pt` and `__pycache__`.
3. Add a global `set_seed()` covering `random`, `numpy` and `torch`, with `cudnn.deterministic` switched on for reproducibility.

## Phase 1: Explore the data (EDA)
**Method:** read each video's metadata with OpenCV (`CAP_PROP_FRAME_COUNT`, `CAP_PROP_FPS`, width and height), store it in a `pandas` table, and plot it.

Steps:
1. Build `metadata.csv` with columns `path, label, group_id, n_frames, fps, duration_s, width, height`.
   - `group_id` comes from a regex that strips the trailing `_k` suffix.
   - Flag broken videos (0 frames or unreadable) and leave them out.
2. Report and plot:
   - Clips per class (bar chart, imbalance ratio).
   - Duration and frame-count distributions per class (histogram plus box plot).
   - Resolution and FPS frequency tables.
   - A sample frame grid from each class.
3. Use the results to pick settings:
   - **T**: around the 10th percentile of frame count, typically 16.
   - **Input size**: resize to 128 or 112, crop to 112 for 3D CNNs, or use 224 for ResNet.
4. **Split:** `StratifiedGroupKFold` (or `GroupShuffleSplit`) into **70/15/15 train/val/test**, stratified by label and grouped by `group_id`. Save `splits.csv` so the split is fixed.

## Phase 2: Frame sampling (`sampling.py`)
**Method:** uniform temporal sampling with a fractional skip interval.

```python
def sample_indices(n_frames: int, T: int, mode="center") -> np.ndarray:
    # skip = n_frames / T  (float, so it spans the whole video)
    # idx_i = floor(skip * i + offset), offset = skip/2 (center) or random in [0, skip) (train jitter)
    # if n_frames < T: indices repeat (loop/pad) -> still exactly T frames
    # clip to [0, n_frames-1]
```
- Training uses **random offset within each segment** (TSN-style) as extra temporal augmentation. Val and test use the fixed segment centre.
- **Decoding:** read frames in order and keep only the target indices. This is faster and more reliable than seeking in mp4 files. If the real frame count is smaller than the header says, repeat the last valid frame.
- **Optional speed-up:** pre-extract frames to `.npy` or JPEG once, so the GPU isn't waiting on video decoding.

**Verification** (`tests/test_sampling.py`), run on synthetic lengths `{5, 16, 17, 100, 1000, 12345}` plus real short, median and long clips:
- `len(idx) == T`
- indices never decrease and stay within range
- the first and last indices are close to the start and end of the video
- the gaps between indices are nearly equal (spread of at most 1)

## Phase 3: Clip-consistent augmentation (`transforms.py`)
**Method:** draw the random parameters **once per clip**, then apply them to every frame with the functional API (`torchvision.transforms.v2.functional`). Alternatively, stack the frames into one tensor and transform it in a single call.

```python
class ClipTransform:
    def __call__(self, frames):            # frames: (T, C, H, W) uint8
        i, j, h, w = RandomResizedCrop.get_params(frames[0], scale, ratio)
        flip = random.random() < 0.5
        b, c, s, hue = sample_jitter_factors()
        out = [F.resized_crop(f, i, j, h, w, size) for f in frames]
        if flip: out = [F.hflip(f) for f in out]
        out = [color_jitter(f, b, c, s, hue) for f in out]
        return normalize(torch.stack(out))
```
- **Train:** RandomResizedCrop (scale 0.7–1.0), HorizontalFlip (p = 0.5), ColorJitter (0.2, 0.2, 0.2, 0.05), then Normalize (ImageNet or Kinetics mean and std, matching the backbone).
- **Val and test:** Resize, CenterCrop, Normalize. No random steps.

**Verification:**
- Code check (`tests/test_transforms.py`): feed in a clip where every frame is the same image. Every output frame must also be identical (`torch.allclose(out[0], out[k])`).
- Visual check: save a grid of the original and augmented frames for one clip to `outputs/aug_check.png`.

## Phase 4: Dataset class (`dataset.py`)
```python
class ShopliftingDataset(Dataset):
    def __init__(self, df, T=16, transform=None, train=False): ...
    def __getitem__(self, i):
        idx = sample_indices(n, T, mode="random" if train else "center")
        frames = read_frames(path, idx)           # (T, H, W, C) uint8, RGB
        clip = transform(to_tensor(frames))       # (T, C, H, W) float
        return clip, label
```
- Always returns `(T, C, H, W)`. Models that need a different layout (3D CNNs use `(C, T, H, W)`) permute it inside the model.
- **DataLoader:** `num_workers` > 0 (on Windows, guard the entry point with `if __name__ == "__main__"`), `pin_memory=True`, `persistent_workers=True`.
- **Class imbalance:** use `WeightedRandomSampler` **or** class-weighted loss, not both.

**Verification:** `tests/test_dataset.py` loads N samples of different lengths and asserts `clip.shape == (T, 3, H, W)` and `dtype == float32`. It also checks that a batch from the DataLoader has shape `(B, T, C, H, W)`.

## Phase 5: Model architectures
**5b (main): CNN + GRU** (`cnn_rnn.py`)
- **Backbone:** ImageNet-pretrained ResNet-18 or 34 with the classification layer removed, giving one 512-d feature per frame.
- Fold time into the batch for the CNN pass: `(B·T, C, H, W)` → features → reshape to `(B, T, 512)`.
- **Temporal part:** 1–2 layer GRU (hidden size 256, optionally bidirectional). Pool by taking the last hidden state, or by mean or attention over time.
- **Head:** Dropout(0.5) → Linear → 2 logits.
- **Transfer-learning schedule:** freeze the backbone for the first few epochs, then unfreeze `layer4` with a 10× lower learning rate.

**5a (optional / comparison): 3D CNN** (`cnn3d.py`)
- `torchvision.models.video.r3d_18` or `r2plus1d_18`, Kinetics-400 pretrained. Input is `(B, C, T, H, W)` at 112×112 with T = 16. Swap in a new `fc` layer for 2 classes.

## Phase 6: Training (`train.py`)
**Method:** supervised fine-tuning with these settings:

| Item | Choice |
|---|---|
| Loss | CrossEntropy with class weights (or label smoothing 0.1) |
| Optimizer | AdamW (lr 1e-4 for the head and GRU, 1e-5 for the backbone; weight decay 1e-4) |
| Scheduler | Cosine annealing with warm-up (or ReduceLROnPlateau) |
| Mixed precision | `torch.amp.autocast` + `GradScaler` (saves GPU memory) |
| Batch / epochs | 8–16 clips, 20–30 epochs; gradient accumulation if memory is short |
| Regularisation | Dropout, augmentation, gradient clipping (1.0) |
| Model selection | Keep the best **validation macro-F1** checkpoint; early-stop after 5 epochs without improvement |
| Logging | Per-epoch loss, accuracy and F1 for train and val, written to CSV or TensorBoard, plus learning curves |

## Phase 7: Evaluation (`evaluate.py`)
1. Load the best checkpoint and run it on the **held-out test split** (centre sampling). Optionally add test-time augmentation: average over several temporal offsets or a flipped copy.
2. Metrics (`sklearn.metrics`): **accuracy, precision, recall and F1** (per class plus macro), a confusion matrix, ROC-AUC and a PR curve.
3. Optionally tune the decision threshold on the validation set. Recall on the shoplifting class matters most in practice, since a missed theft is worse than a false alarm.
4. Save `outputs/metrics.json`, `confusion_matrix.png` and `roc.png`, and add a short error analysis showing a few misclassified clips.

## Phase 8: Delivery
- `README.md` with an overview, dataset stats from EDA, how to run (`python -m src.train --model cnn_rnn`), the results table, and figures.
- `requirements.txt` / `environment.yml`.
- Run `pytest` so all checks pass, then push to GitHub (dataset and checkpoints excluded).

---

## Requirement → verification map

| Completion requirement | Where it is met | How it is checked |
|---|---|---|
| Samples T frames from any video length | `sampling.py` | `test_sampling.py` on synthetic and real clips of different lengths |
| Same augmentation on every frame of a clip | `transforms.py` | identical-frame test plus saved visual grid |
| Dataset returns `(T, C, H, W)` | `dataset.py` | `test_dataset.py` shape checks |
| A temporal architecture trains without errors | `cnn_rnn.py` (+ `cnn3d.py`) | one-batch overfit smoke test, then full training logs |
| Accuracy and F1 reported on the test set | `evaluate.py` | `metrics.json` plus README table |
| Code on GitHub | repo | pushed commit link |

## Suggested order and checkpoints
1. Phases 0–1: EDA plus fixed grouped split.
2. Phases 2–4: data pipeline with tests passing.
3. Phase 5: smoke test where the model overfits a single batch (loss goes to about 0).
4. Phases 6–7: full training and test metrics.
5. Phase 8: README and push.

## Open questions
1. **Architecture:** CNN+GRU only, or also a 3D CNN (R(2+1)D) for comparison?
2. **Do you have a GitHub repo already?** The workspace is linked to `MahamedKhaledSaad11/Eye_Disease_Classification`. Should this go into a new repo (e.g. `Shoplifting_Detection`)?
3. **Format:** plain `.py` modules plus one EDA notebook, or a single end-to-end notebook?
