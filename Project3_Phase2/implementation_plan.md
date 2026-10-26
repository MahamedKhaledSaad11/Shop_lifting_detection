# Phase 2 (Week 6): Fine-tuning Pretrained Hugging Face Video Models: Implementation Plan

**Goal:** Fine-tune a pretrained Hugging Face video classifier on the same shoplifting data, then compare it against a model trained from scratch with the Week 5 pipeline. The comparison covers F1, accuracy, inference speed and model size.
**Environment:** conda env `torch_gpu` (PyTorch 2.14 + CUDA, RTX 2080, 8 GB VRAM). New dependency: `transformers` (plus `accelerate` if needed).
**Data:** `../Shop DataSet`. Use the **existing Week 5 split file `../outputs/splits.csv`** with `deduplicate=True` (446 train / 93 val / 98 test unique clips). Do **not** re-split. Every model must be scored on exactly the same 98 test clips.

---

## 0. Key methodology decisions (read first)

### 0.1 What counts as "the Week 5 scratch model"
The Week 5 models were **not** trained from scratch. ResNet-18 used ImageNet weights and R(2+1)D-18 used Kinetics-400 weights. R(2+1)D-18 already reached test F1 = 1.000, so no model can score higher. To meet the requirement honestly and correctly:

1. **Scratch baseline (required):** retrain the Week 5 architectures with **random initialisation** (`pretrained=False`). Use the same pipeline and splits, with a longer schedule (30 epochs, no backbone freezing).
   - `scratch_cnn_rnn`: `ResNetBiGRUAttention(pretrained=False)`
   - `scratch_cnn3d`: `VideoCNN3D(pretrained=False)`
2. **Week 5 transfer models (reference only):** reload the existing checkpoints `../outputs/checkpoints/best_cnn_rnn.pt` and `best_cnn3d.pt`. Re-benchmark them for speed and size so the final table is complete. Do **not** retrain them.

### 0.2 Hugging Face model choice
| Candidate | HF id | Params | Frames × res | Notes |
|---|---|---|---|---|
| **VideoMAE-Base (primary)** | `MCG-NJU/videomae-base-finetuned-kinetics` | ~86M | 16 × 224 | Takes `(B, T, C, H, W)` input with ImageNet mean/std, so it matches our dataset output exactly. Strongest data-efficient ViT for video. |
| **VideoMAE-Small (secondary)** | `MCG-NJU/videomae-small-finetuned-kinetics` | ~22M | 16 × 224 | Same interface. Gives the size/speed trade-off point. |
| TimeSformer-Base (optional) | `facebook/timesformer-base-finetuned-k400` | ~121M | 8 × 224 | Divided space-time attention. Only add it if time allows. |
| X3D | not in `transformers` (PyTorchVideo hub) | – | – | **Skipped.** The task asks for Hugging Face Hub models. |

### 0.3 Reuse of the Week 5 pipeline (requirement)
Import the Week 5 modules directly instead of copying them: `src.data.sampling.sample_indices / read_frames`, `src.data.transforms.ClipTransform`, `src.data.dataset.ShopliftingDataset`. Add the Project3 root to `sys.path` in `phase2/__init__.py` or a `_paths.py`. `ClipTransform` already uses ImageNet mean/std, which is what VideoMAE expects. Read these values from `VideoMAEImageProcessor` and assert that they are equal.

---

## 1. Folder layout (`Project3_Phase2/`)
```
Project3_Phase2/
├── implementation_plan.md
├── README.md                      # final report + comparison table + discussion
├── requirements.txt               # adds transformers
├── phase2/
│   ├── __init__.py / _paths.py    # sys.path → ../ (reuse Week 5 src)
│   ├── config.py                  # model ids, hyper-params, output paths
│   ├── hf_models.py               # HF loader + 2-class head adaptation + LLRD param groups
│   ├── train_hf.py                # fine-tune VideoMAE-B / VideoMAE-S
│   ├── train_scratch.py           # Week 5 archs with pretrained=False
│   ├── evaluate_all.py            # same test protocol for every model, bootstrap CIs, McNemar
│   └── benchmark.py               # latency, throughput, params, size, VRAM, GFLOPs
├── tests/
│   ├── test_hf_model.py           # loads from HF, head = 2 logits, forward/backward, 1-batch overfit
│   └── test_reuse_pipeline.py     # dataset → (T,3,224,224) → model accepts it
└── outputs/                       # checkpoints (git-ignored), metrics, plots, comparison.csv/md
```

## 2. Step 1: Load the HF model and adapt the head (`hf_models.py`)
- `VideoMAEForVideoClassification.from_pretrained(id, num_labels=2, id2label={0:"normal",1:"shoplifting"}, label2id=..., ignore_mismatched_sizes=True)`. The 400-class Kinetics head is replaced by a freshly initialised 2-class `nn.Linear`.
- Keep the pretrained `fc_norm` / mean-pooling. Set `hidden_dropout_prob` / classifier dropout to 0.1–0.3 and `drop_path_rate` to 0.1.
- Enable `gradient_checkpointing_enable()` so VideoMAE-B fits in 8 GB.
- Wrap the model so `forward(x)` returns `logits`. That keeps the interface identical to the Week 5 models for evaluation and benchmarking.
- **Tests:** load without errors; `model.classifier.out_features == 2`; `config.num_labels == 2`; output shape `(B, 2)`; gradients reach the head and the last block; a 1-batch overfit drives the loss below 0.05.

## 3. Step 2: Fine-tuning recipe (`train_hf.py`)
| Item | Choice | Why |
|---|---|---|
| Input | T = 16, 224×224, Week 5 `ClipTransform` (train: clip-consistent RRC scale (0.85,1), flip, jitter; eval: full-FOV resize) | Reuses the pipeline and matches VideoMAE pretraining |
| Precision | `torch.amp` fp16 + GradScaler | Needed on 8 GB |
| Batch | micro-batch 2–4, gradient accumulation to an effective 16 | Memory |
| Optimiser | AdamW, wd 0.05 (no decay on bias/norm), betas (0.9, 0.999) | Standard ViT fine-tuning |
| LR | base 5e-5 for the top block, **layer-wise LR decay 0.75** toward the embeddings, head LR 10× | Protects pretrained low-level features on a small dataset |
| Schedule | 1–2 epoch linear warm-up, then cosine to 1e-6; 15–20 epochs; early stop with patience 5 | |
| Loss | CE + label smoothing 0.1 | |
| Regularisation | drop_path 0.1, grad clip 1.0 | |
| Selection | best val (macro-F1, ROC-AUC, −loss), the same tie-breaker as Week 5 | Consistent with Week 5 |
| Seeds | fixed seed 42 (optionally 3 seeds if time allows, reporting mean ± std) | Reproducibility |
Log per-epoch CSVs and plot learning curves.

## 4. Step 3: Scratch baseline (`train_scratch.py`)
Same splits, augmentation, AMP and early stopping. `pretrained=False`, all layers trainable from epoch 1, 30 epochs, lr 3e-4 (AdamW, cosine). **BatchNorm must not stay frozen here.** The Week 5 Frozen-BN override only makes sense for pretrained statistics. With random initialisation, BN must train, so add a flag such as `freeze_bn=False` or subclass the model.

## 5. Step 4: Unified evaluation (`evaluate_all.py`)
For every model (scratch ×2, Week 5 transfer ×2, VideoMAE-S, VideoMAE-B):
- Use the same 98 test clips with centre temporal sampling. Report each metric two ways: **without TTA at τ = 0.5 (primary)**, and with flip-TTA.
- Report accuracy, precision, recall, F1 (per class + macro), ROC-AUC, PR-AUC and the confusion matrix.
- **Statistics:** a 95% bootstrap CI (2000 resamples) for macro-F1 and accuracy, plus a McNemar test of VideoMAE-B against the scratch model. With only 98 clips, a 1–2% gap is noise and must be reported as such.
- Optional threshold tuning on val only, constrained to τ ∈ [0.3, 0.75] with val recall ≥ 0.98.
- Save `outputs/metrics_all.json`, `outputs/confusion_matrices.png` and `outputs/roc_pr_all.png`.

## 6. Step 5: Efficiency benchmark (`benchmark.py`)
For each model on the same GPU:
- **Params** (total / trainable) and **checkpoint size on disk** (MB, fp32 state_dict).
- **GFLOPs per clip** via `torch.utils.flop_counter.FlopCounterMode`.
- **Latency:** batch 1, 20 warm-up runs, then 100 timed runs with `torch.cuda.synchronize()`, in fp32 and fp16. Report the mean and p95 in ms per clip.
- **Throughput:** clips/s at the largest batch that fits.
- **Peak VRAM** (`torch.cuda.max_memory_allocated`).
- Report **video decode + preprocessing time per clip** separately, because it is shared by all models and often dominates end-to-end latency.
- Save the results to `outputs/benchmark.csv`.

## 7. Step 6: Comparison table and discussion (`README.md`)
One table with these columns: Model | Init | Params | Size MB | GFLOPs | Latency fp16 ms | Throughput | Peak VRAM | Test Acc | Macro-F1 [95% CI] | Recall(shop) | ROC-AUC.
Add a discussion section covering:
- Accuracy vs. cost.
- How pretraining helps data efficiency, with the scratch vs. VideoMAE gap as evidence.
- ViT quadratic attention vs. 3D-CNN cost.
- Which model to deploy at the edge vs. on a server.
- Limits: a small test set, a single store and camera, and possible scene shortcuts.

## 8. Completion requirements → verification
| Requirement | Where | Check |
|---|---|---|
| Pretrained HF model loads without errors | `hf_models.py` | `tests/test_hf_model.py` |
| Head adapted to 2 classes | `hf_models.py` | assert `out_features == 2`, logits `(B,2)` |
| Fine-tuned F1 > Week 5 scratch F1 | `evaluate_all.py` | `metrics_all.json`, with CI and McNemar |
| Comparison table | `README.md`, `outputs/comparison.md` | table present |
| Code on GitHub | repo `MahamedKhaledSaad11/Shop_lifting_detection` | commit pushed (done by the user) |

## 9. Order and review checkpoints (the supervisor reviews at each one)
1. **M1:** Environment (`transformers`), `hf_models.py`, tests, and a 1-batch overfit of VideoMAE-S and VideoMAE-B. Report the VRAM used at micro-batch 2/4. → **review**
2. **M2:** Scratch baselines trained, with curves. → **review**
3. **M3:** VideoMAE-S and VideoMAE-B fine-tuned, with curves. → **review**
4. **M4:** Unified evaluation, benchmark, README and comparison table. → **final review**
