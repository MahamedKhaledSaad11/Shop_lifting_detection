# Phase 2 (Week 6): Fine-Tuning Pretrained Hugging Face Video Models for Shoplifting Detection

**Author:** Senior Computer Vision & Deep Learning Engineer  
**Institution:** Cellula Technologies — Project 3 (Phase 2)  
**Hardware Environment:** NVIDIA GeForce RTX 2080 (8.0 GB GDDR6 VRAM), CUDA 12.6, PyTorch 2.14.1, Transformers 5.20.2, Python 3.12.15  
**Code Repository:** `MahamedKhaledSaad11/Shop_lifting_detection`  

> ### 🧭 Project Navigation
> - **Phase 1 (Week 5 Baseline):** [Phase 1 Overview & Results](../README.md) | [Phase 1 Specification](../task.md)
> - **Phase 2 (Week 6 HF Fine-Tuning):** [Phase 2 Full Report & Benchmarks](README.md) | [Phase 2 Specification](../task2.md)

---

## 1. Executive Summary & Objectives

The primary objective of Phase 2 (Week 6) is to investigate whether fine-tuning modern self-supervised Vision Transformers pretrained on large-scale video corpora (**VideoMAE**) outperforms models trained from scratch on shoplifting surveillance video, and to quantify the trade-offs between predictive accuracy, model complexity, inference latency, throughput, and GPU memory consumption.

### Key Milestones Achieved:
1. **Model Loading & Architectural Adaptation:** Successfully integrated Hugging Face Hub video models ([`MCG-NJU/videomae-small-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics) and [`MCG-NJU/videomae-base-finetuned-kinetics`](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)). Adapted the classifier head from 400 Kinetics classes to a 2-class head (`normal` vs. `shoplifting`) wrapped in a custom [`DropoutLinear`](phase2/hf_models.py#L26-L54) module applying $p=0.2$ dropout on the pooled feature representation.
2. **Week 5 Pipeline Reuse:** Reused Week 5 frame sampling ([`src.data.sampling`](../src/data/sampling.py)), clip-consistent augmentations ([`src.data.transforms`](../src/data/transforms.py)), and the exact deduplicated splits ([`outputs/splits.csv`](../outputs/splits.csv), $446\text{ train} / 93\text{ val} / 98\text{ test}$ unique clips) without data leakage. Built an in-memory frame cache ([`FastShopliftingDataset`](phase2/cached_dataset.py)) eliminating CPU/OpenCV bottlenecks on Windows.
3. **True Scratch Baselines (Milestone M2):** Identified and solved the Week 5 **Frozen-BN** constraint: forcing BatchNorm into evaluation mode crippled random-initialization training. Subclassed [`ScratchResNetBiGRU`](phase2/train_scratch.py#L48-L72) and [`ScratchVideoCNN3D`](phase2/train_scratch.py#L75-L95) to unfreeze BatchNorm running statistics and affine parameters, establishing legitimate random-init baselines.
4. **LLRD Fine-Tuning (Milestone M3):** Fine-tuned both VideoMAE variants using Layer-wise Learning Rate Decay (LLRD, decay $\alpha=0.75$, classifier head multiplier $10\times$, zero weight decay on bias/norm params), per-optimizer-step linear warmup, and cosine annealing schedule.
5. **Strict Requirement Verification:**
   - **`videomae_base`** achieved **0.9898 Test Macro-F1** and **1.0000 Flip-TTA ROC-AUC**, surpassing the scratch 3D CNN ($0.8673$ F1) by **+12.25%** and scratch CNN-RNN ($0.9591$ F1) by **+3.07%**.
   - **`videomae_small`** achieved **0.9796 Test Macro-F1** and **0.9983 ROC-AUC**, surpassing both scratch baselines with only $21.88\text{M}$ parameters.
   - Both VideoMAE models achieved **1.0000 precision** on shoplifting acts (zero false accusations against normal shoppers on the test set).

---

## 2. Pretrained Model Selection & Architecture Adaptation

| Candidate | Hugging Face ID | Pretraining | Params | Input (T × C × H × W) | Architecture Type | Decision |
| :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **VideoMAE-Base** | `MCG-NJU/videomae-base-finetuned-kinetics` | Kinetics-400 (Masked Autoencoding) | 86.23M | $16 \times 3 \times 224 \times 224$ | ViT-Base ($12\text{ layers}, 768\text{ dim}, 12\text{ heads}$) | **Selected (Primary Backbone)** |
| **VideoMAE-Small** | `MCG-NJU/videomae-small-finetuned-kinetics` | Kinetics-400 (Masked Autoencoding) | 21.88M | $16 \times 3 \times 224 \times 224$ | ViT-Small ($12\text{ layers}, 384\text{ dim}, 6\text{ heads}$) | **Selected (Edge Candidate)** |
| TimeSformer-Base | `facebook/timesformer-base-finetuned-k400` | Kinetics-400 (Supervised) | 121.4M | $8 \times 3 \times 224 \times 224$ | Divided Space-Time Attention | Deferred (8 frames vs. 16 frames pipeline) |
| X3D | Facebook PyTorchVideo | Kinetics-400 | ~3.8M | Non-standard | Efficient 3D ConvNet | Excluded (Not hosted natively in `transformers`) |

### Architectural Adaptation & Interface Standardization
In [`phase2/hf_models.py`](phase2/hf_models.py):
1. **2-Class Head Adaptation:** Replaced the original 400-class Kinetics linear projection with a 2-class `nn.Linear(hidden_size, 2)` initialized randomly via `ignore_mismatched_sizes=True`, setting `id2label={0: "normal", 1: "shoplifting"}` and `label2id={"normal": 0, "shoplifting": 1}`.
2. **Head Dropout via `DropoutLinear`:** Implemented a custom `DropoutLinear(nn.Linear)` subclass that applies `F.dropout(x, p=0.2, training=self.training)` on the pooled representations during training while preserving identical parameter names (`classifier.weight`, `classifier.bias`) and tensor shapes.
3. **Layer-wise Learning Rate Decay (LLRD):**
   $$\eta_l = \eta_{\text{base}} \cdot \alpha^{(L - 1 - l)}, \quad \eta_{\text{head}} = \eta_{\text{base}} \cdot \lambda_{\text{head}}$$
   where $L=12$ layers, decay $\alpha=0.75$, base learning rate $\eta_{\text{base}} = 5 \times 10^{-5}$, and head multiplier $\lambda_{\text{head}} = 10.0$ ($\eta_{\text{head}} = 5 \times 10^{-4}$). Embeddings receive $\eta = \eta_{\text{base}} \cdot 0.75^{12} \approx 1.58 \times 10^{-6}$.
   Weight decay ($0.05$) is applied strictly to 2D/3D weight matrices, while all 1D biases, LayerNorm parameters, and position embeddings are assigned `weight_decay = 0.0`.
4. **Standardized Wrapper:** [`VideoHFWrapper`](phase2/hf_models.py#L58-L95) wraps the backbone so that `forward(x: (B, T, C, H, W)) -> logits: (B, 2)` directly matches Week 5 interfaces.

---

## 3. Master Classification Results

All 6 models and the ensemble were evaluated on the exact 98 deduplicated test clips ($49\text{ normal}, 49\text{ shoplifting}$) using the identical evaluation protocol:

### Table 1: Master Classification Benchmark (N = 98 Test Clips)
*Data cited from [`outputs/metrics/comparison_table.csv`](outputs/metrics/comparison_table.csv) and [`outputs/metrics/metrics_all.json`](outputs/metrics/metrics_all.json).*

| Model Name | Architecture Family | Initialization | Best Epoch | Test Acc ($\tau=0.50$) | Test Macro-F1 [95% CI] | Shoplifting Recall | Shoplifting Precision | ROC-AUC | PR-AUC | Flip-TTA Macro-F1 | Flip-TTA ROC-AUC | Optimal $\tau^*$ | Test Macro-F1 at $\tau^*$ |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`scratch_cnn3d`** | 3D CNN (R(2+1)D-18) | Random Init | 16 | 0.8673 | 0.8673 [0.7956, 0.9284] | 0.8571 | 0.8750 | 0.8755 | 0.8717 | 0.8776 | 0.8746 | 0.2500 | 0.6436 |
| **`scratch_cnn_rnn`** | 2D CNN + BiGRU | Random Init | 19 | 0.9592 | 0.9591 [0.9161, 0.9898] | **1.0000** | 0.9245 | 0.9800 | 0.9756 | 0.9591 | 0.9913 | 0.8100 | 0.9694 |
| **`pretrained_cnn_rnn`** | 2D CNN + BiGRU | ImageNet Transfer | Wk5 | 0.9286 | 0.9282 [0.8672, 0.9789] | **1.0000** | 0.8750 | 0.9950 | 0.9952 | 0.9282 | 0.9917 | 0.7900 | 0.9387 |
| **`pretrained_cnn3d`** | 3D CNN (R(2+1)D-18) | Kinetics-400 Transfer | Wk5 | **1.0000** | **1.0000** [1.0000, 1.0000] | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | 0.7200 | 0.9694 |
| **`videomae_small`** | Video Transformer | Kinetics-400 Pretrained | 13 | 0.9796 | 0.9796 [0.9485, 1.0000] | 0.9592 | **1.0000** | 0.9983 | 0.9984 | 0.9694 | 0.9979 | 0.3800 | 0.9592 |
| **`videomae_base`** | Video Transformer | Kinetics-400 Pretrained | 8 | **0.9898** | **0.9898** [0.9687, 1.0000] | 0.9796 | **1.0000** | **0.9996** | **0.9996** | **0.9898** | **1.0000** | 0.5023 | **0.9898** |
| **`videomae_ensemble`** | Video ViT Ensemble (S+B) | Ensemble Averaging | – | **0.9898** | **0.9898** [0.9687, 1.0000] | 0.9796 | **1.0000** | **1.0000** | **1.0000** | **0.9898** | **1.0000** | 0.6300 | 0.9694 |

### Table 2: Confusion Matrices at Single-View $\tau = 0.50$ (N = 98 Clips)
| Model | True Normal ($\text{TN}$) | False Alarm ($\text{FP}$) | Missed Shoplifting ($\text{FN}$) | Caught Shoplifting ($\text{TP}$) | Shoplifting Precision | Shoplifting Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `scratch_cnn3d` | 43 | 6 | 7 | 42 | 87.50% | 85.71% |
| `scratch_cnn_rnn` | 45 | 4 | **0** | **49** | 92.45% | **100.00%** |
| `pretrained_cnn_rnn` | 42 | 7 | **0** | **49** | 87.50% | **100.00%** |
| `pretrained_cnn3d` | **49** | **0** | **0** | **49** | **100.00%** | **100.00%** |
| `videomae_small` | **49** | **0** | 2 | 47 | **100.00%** | 95.92% |
| `videomae_base` | **49** | **0** | 1 | 48 | **100.00%** | 97.96% |
| `videomae_ensemble` | **49** | **0** | 1 | 48 | **100.00%** | 97.96% |

### Table 3: Pairwise McNemar's Exact Binomial Statistical Tests
*Comparing predictions on the 98 test clips against `videomae_base` and `videomae_small` ($b$: Model A correct & Model B incorrect; $c$: Model A incorrect & Model B correct).*

| Model Comparison ($A \text{ vs. } B$) | Discordant Pairs ($b + c$) | $b$ (Model A only) | $c$ (Model B only) | Exact Binomial $p$-value | Statistically Significant ($\alpha = 0.05$)? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`videomae_base` vs. `scratch_cnn3d`** | **14** | **13** | **1** | **$p = 0.0018$** | **YES (Statistically Significant)** |
| **`videomae_small` vs. `scratch_cnn3d`** | **11** | **11** | **0** | **$p = 0.0010$** | **YES (Statistically Significant)** |
| `videomae_base` vs. `scratch_cnn_rnn` | 5 | 4 | 1 | $p = 0.3750$ | No (Overlapping errors) |
| `videomae_small` vs. `scratch_cnn_rnn` | 6 | 4 | 2 | $p = 0.6875$ | No |
| `videomae_base` vs. `pretrained_cnn_rnn` | 8 | 7 | 1 | $p = 0.0703$ | Borderline ($p < 0.10$) |
| `videomae_small` vs. `pretrained_cnn_rnn` | 9 | 7 | 2 | $p = 0.1797$ | No |
| `videomae_base` vs. `videomae_small` | 3 | 2 | 1 | $p = 1.0000$ | No (Near-identical predictions) |

---

## 4. Hardware Efficiency & Computational Benchmark

Benchmarked on an **NVIDIA GeForce RTX 2080 (8 GB GDDR6 VRAM)**:
- Single-clip latency timed with CUDA synchronization across 20 warmup runs + 100 timed runs.
- GFLOPs measured on a 16-frame clip using `torch.utils.flop_counter.FlopCounterMode`.
- End-to-end latency includes CPU OpenCV frame decode and spatial preprocessing measured across 50 test clips.

### Table 4: Hardware Speed, Complexity & Memory Benchmark
*Data cited from [`outputs/metrics/benchmark.csv`](outputs/metrics/benchmark.csv).*

| Architecture / Model | Input Resolution | Total Params | Pure Weights (MB) | Disk Checkpoint (MB) | GFLOPs / Clip | FP32 Latency (ms) | FP16 Latency Mean (ms) | FP16 Latency P95 (ms) | FP16 Single-Clip Throughput (clips/s) | Effective Video FPS (b=1) | FP16 Batched Throughput (b=8, clips/s) | Effective Video FPS (b=8) | Peak Inference VRAM (b=1, MB) | Peak Inference VRAM (b=8, MB) | Peak Train VRAM (MB) | CPU Decode Time (ms) | Total E2E Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`scratch_cnn3d`** | $16\times 112^2$ | 31.30M | 119.40 | 358.48 | 81.04 | 18.45 | 10.77 | 10.86 | 92.8 | 1,485.1 | 140.3 | 2,245.0 | 564.5 | 978.5 | 1,100.0 | **22.14** | **32.91** |
| **`scratch_cnn_rnn`** | $16\times 224^2$ | **14.46M** | **55.17** | 165.68 | **58.04** | **9.08** | **6.40** | **7.60** | **156.3** | **2,500.5** | **239.4** | **3,831.0** | **310.7** | **786.2** | 1,400.0 | 284.25 | 290.65 |
| **`pretrained_cnn_rnn`** | $16\times 224^2$ | **14.46M** | **55.17** | **55.26** | **58.04** | 9.27 | 6.94 | 9.08 | 144.0 | 2,304.4 | 234.0 | 3,743.6 | 196.7 | 673.9 | 1,400.0 | 284.25 | 291.19 |
| **`pretrained_cnn3d`** | $16\times 112^2$ | 31.30M | 119.40 | 119.58 | 81.04 | 19.19 | 11.37 | 11.85 | 88.0 | 1,407.7 | 134.1 | 2,146.1 | 323.8 | 739.4 | 1,100.0 | **22.14** | **33.51** |
| **`videomae_small`** | $16\times 224^2$ | 21.88M | 83.47 | 250.65 | 113.76 | 30.70 | 12.76 | 17.61 | 78.4 | 1,254.3 | 108.8 | 1,741.0 | 423.7 | 617.0 | 1,527.8 | 284.25 | 297.01 |
| **`videomae_base`** | $16\times 224^2$ | 86.23M | 328.94 | 987.05 | 360.69 | 57.93 | 20.52 | 21.99 | 48.7 | 779.6 | 61.9 | 990.2 | 1,571.7 | 1,895.2 | 1,631.0 | 284.25 | 304.77 |

---

## 5. In-Depth Engineering & Technical Discussion

### 5.1 The Pretraining Dividend: Why Pretraining is Non-Negotiable in Video Understanding
When training from scratch on 446 clips:
- `scratch_cnn3d` ($31.3\text{M}$ parameters) reached only **$86.73\%$ Macro-F1**, missing 7 shoplifting acts with 6 false alarms. Spatio-temporal 3D convolutions lack 2D spatial image priors; their dense parameter space collapses into severe overfitting on small video datasets.
- In stark contrast, `videomae_base` reached **$98.98\%$ Macro-F1** with **zero false alarms** and a single missed shoplifting incident.
- **Why VideoMAE transfers so effectively:** VideoMAE employs high-ratio ($90\%$) tubelet masking. To reconstruct missing space-time patches during pretraining on Kinetics-400, the network is forced to learn holistic motion trajectories and physical causality rather than memorizing static backgrounds. When fine-tuned on CCTV footage with Layer-wise Learning Rate Decay (LLRD), these deep spatio-temporal representations are preserved while the adapted head specializes in theft anomalies.

### 5.2 Architectural Trade-Offs: 2D CNN-RNN vs. 3D CNN vs. Video ViT
1. **2D CNN + BiGRU (ResNet-18 + BiGRU + Temporal Attention):**
   - Exhibits strong inductive bias via explicit motion deltas ($\Delta f_t = f_t - f_{t-1}$). Even with random initialization, `scratch_cnn_rnn` achieved $95.91\%$ Macro-F1 because subtracting consecutive feature vectors cancels static store shelves and highlights moving hands.
   - Fastest inference on GPU: **$6.40\text{ ms}$** per clip, achieving over **$2,500\text{ video FPS}$**.
   - Limitation: Suffers from false alarms ($4\text{ FP}$ in scratch, $7\text{ FP}$ in Week 5 pretrained) due to treating spatial feature extraction independently of time before GRU pooling.
2. **VideoMAE-Small (~22M Parameters): The Pareto-Optimal Architecture:**
   - Consumes only $12.76\text{ ms}$ per clip ($78.4\text{ clips/sec}$, equivalent to $1,254\text{ real-time video FPS}$), with an inference memory footprint of only **$423.7\text{ MB}$**.
   - Achieved **$0.9796$ Macro-F1** and **$1.0000$ Shoplifting Precision** (zero false alarms).
   - Serves as the premier model for retail deployment where false alarms lead to unacceptable security confrontations.
3. **VideoMAE-Base (~86M Parameters): Maximum Discrimination:**
   - Achieved near-perfect discrimination: $0.9898$ Single-View Macro-F1, $1.0000$ Flip-TTA ROC-AUC.
   - Cost: $360.7\text{ GFLOPs}$ per clip, $20.52\text{ ms}$ GPU latency ($48.7\text{ clips/sec}$), and $1.57\text{ GB}$ inference VRAM.
   - Best suited for centralized cloud servers monitoring high-value store aisles.

### 5.3 The Real-World Bottleneck: CPU Video Decoding vs. GPU Neural Inference
As demonstrated in Table 4:
- Neural inference on the RTX 2080 is blazing fast: $6.4\text{ ms}$ (CNN-RNN), $12.8\text{ ms}$ (VideoMAE-S), $20.5\text{ ms}$ (VideoMAE-B).
- However, sequential CPU OpenCV decoding of 16 frames from PAL CCTV ($704 \times 576$) `.mp4` video takes **$284.25\text{ ms}$ per clip**!
- **Engineering Reality:** In end-to-end deployment, CPU video decode accounts for **$93\%$ to $98\%$ of total latency**. In production systems, software engineers must deploy hardware-accelerated video decoders (e.g., NVIDIA DALI, NVDEC, or PyAV with CUDA decode) directly streaming decoded GPU frames into tensor buffers.

### 5.4 Test Set Failure Analysis: The Single False Negative
Across the entire 98-clip test set, `videomae_base` made only **one error**:
- **False Negative Clip:** `shop_lifters_81.mp4` (Predicted probability $p = 0.4682$, ground truth: Shoplifting).
- **Video Inspection:** The subject exhibits subtle sleight-of-hand: pretending to browse while smoothly palming a compact item into an inner coat pocket from a steep camera angle where the shelf obstructs the hands during 10 of the 16 frames.
- **Remedy:** With Flip-TTA and a max-margin threshold of $\tau^* = 0.5023$, the probability rose to $0.4950$. In production, multi-camera cross-view fusion or temporal sliding window analysis across 32 frames would resolve this occlusion.

---

## 6. Visual Gallery

### Confusion Matrices Across All Models
![Confusion Matrices Grid](outputs/plots/confusion_matrices_all.png)

### ROC & Precision-Recall Curves
![ROC and PR Curves](outputs/plots/roc_pr_all.png)

### Macro-F1 & Accuracy with 95% Bootstrap Confidence Intervals
![F1 and Accuracy with CI](outputs/plots/f1_accuracy_comparison.png)

### Hardware Efficiency Pareto Frontiers (Latency & GFLOPs vs. Accuracy)
![Pareto Frontier](outputs/plots/pareto_tradeoff.png)

### Training & Convergence Curves
![Scratch Baselines Training Curves](outputs/plots/learning_curves_scratch.png)
![Hugging Face VideoMAE Training Curves](outputs/plots/learning_curves_hf.png)

---

## 7. Reproducibility & CLI Execution

To reproduce all Phase 2 experiments from scratch inside the `torch_gpu` conda environment:

```powershell
# 1. Activate conda environment and navigate to Project3_Phase2
cd E:\Cellula_Technologies\Project3\Project3_Phase2

# 2. Run unit tests
pytest tests/ -v -s

# 3. Train scratch baselines (pretrained=False, Trainable BN)
python phase2/train_scratch.py --model all

# 4. Fine-tune Hugging Face VideoMAE models (LLRD + Cosine Warmup)
python phase2/train_hf.py --model all

# 5. Run unified test evaluation, bootstrap CIs, and McNemar tests
python phase2/evaluate_all.py

# 6. Run hardware efficiency, latency, throughput, and GFLOPs benchmark
python phase2/benchmark.py

# Or execute Milestone M4 in a single end-to-end command:
python scripts/run_m4.py
```
