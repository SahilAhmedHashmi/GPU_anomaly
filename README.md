# Spatio-Temporal GNN for GPU Failure Prediction

Predicting GPU-level job failures in large-scale GPU clusters using a Spatio-Temporal Graph Neural Network. Built on the [Microsoft Philly cluster trace](https://github.com/msr-fiddle/philly-traces), the model learns both temporal patterns in GPU telemetry and spatial relationships across the cluster topology to forecast failures within a 30-minute horizon.

---

## Problem

Large GPU clusters experience frequent job failures that waste compute, delay research, and increase operational costs. Reactive approaches — detecting failures only after they occur — lead to lost GPU-hours and expensive job restarts. This project frames failure prediction as a node-level binary classification task over a dynamic graph of GPUs, enabling proactive mitigation such as checkpointing, job migration, or preemptive rescheduling.

## Architecture

The model is a two-stage Spatio-Temporal GNN:

```
Input (per GPU): 60-min telemetry window × 6 features
        │
        ▼
┌─────────────────────┐
│  Temporal Encoder    │   Bidirectional LSTM
│  (per-node)          │   Captures temporal patterns in GPU utilization,
│                      │   job state, and historical failure rates
└────────┬────────────┘
         │  Node embeddings
         ▼
┌─────────────────────┐
│  Spatial Encoder     │   2-layer GATv2 with multi-head attention
│  (graph-level)       │   Propagates information across intra-server
│                      │   and inter-server (job-level) edges
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  MLP Classifier      │   Per-GPU binary prediction:
│                      │   "Will a job on this GPU fail within 30 min?"
└─────────────────────┘
```

### Graph Construction

Each GPU in the cluster is a node. Edges are constructed from two sources:

- **Intra-server edges** — fully connected GPUs within the same physical machine (static).
- **Inter-server edges** — GPUs co-allocated to the same distributed job across machines (dynamic, changes per time point).

Edge features encode the relationship type: `[is_intra, is_inter, is_same_job]`.

### Node Features (6 per timestep)

| Channel | Feature | Normalization |
|---------|---------|---------------|
| 0 | GPU utilization (%) | ÷ 100 |
| 1 | Offline status (binary) | 0 or 1 |
| 2 | Job allocation flag | 0 or 1 |
| 3 | Attempt number | ÷ 10 |
| 4 | Historical failure rate (24h lookback) | [0, 1] |
| 5 | Job duration so far (minutes) | ÷ 1440 |

## Dataset

The project uses the **Microsoft Philly GPU cluster trace** covering **October 15–22, 2017**.

| Statistic | Value |
|-----------|-------|
| Date range | Oct 15 – Oct 22, 2017 |
| Observation window | 60 minutes |
| Prediction horizon | 30 minutes |
| GPU nodes | ~104 (across ~13 machines) |
| Sampling strategy | Failed-job-centric + 3× negative sampling |

**Data sources:**
- `cluster_gpu_util.csv` — Per-GPU utilization readings at regular intervals.
- `cluster_job_log.json` — Job metadata including status, timestamps, and GPU allocations.

> **Note:** The raw trace files are not included in this repository. Download them from the [official Philly traces repo](https://github.com/msr-fiddle/philly-traces).

### Sampling Strategy

Time points are sampled around failed job events (offsets: -30, -15, -10, -5, 0 minutes from failure) to capture the lead-up to failures. Negative samples are drawn from the midpoints of passed jobs at a 3:1 ratio.

## Repository Structure

```
.
├── README.md
├── gpu_anomaly.py         # Full pipeline: preprocessing, model definition, training, and inference
```

### `gpu_anomaly.py`

A single-file pipeline covering both data preparation and model training/inference.

**Preprocessing (top half):**

1. Parses job logs and GPU utilization CSVs.
2. Discovers cluster topology (machines → GPUs → nodes).
3. Builds graph structures (intra-server and inter-server edges).
4. Computes per-GPU historical failure rates.
5. Extracts feature tensors and binary labels at sampled time points.
6. Applies a **temporal split** (70% train / 15% val / 15% test by chronological order).
7. Saves the preprocessed dataset as a pickle file.

**Model & Training (bottom half):**

Deployed on [Modal](https://modal.com) with GPU support:

- `train_model` — Full training loop with mixed-precision, early stopping, and threshold tuning.
- `run_inference` — Single-sample inference.
- `batch_inference` — Batch evaluation on the test set.
- `get_training_history` — Retrieve training metrics from a saved checkpoint.

## Getting Started

### Prerequisites

```
Python 3.10+
torch
torch-geometric
pandas
numpy
scikit-learn
tqdm
modal          # for cloud GPU training
```

### 1. Download the Data

Download the Philly trace from the [official repo](https://github.com/msr-fiddle/philly-traces) and place the files in your working directory:

```
cluster_gpu_util.csv
cluster_job_log.json
```

### 2. Preprocess

Set the file paths (`JOB_LOG_PATH`, `GPU_UTIL_PATH`, `OUTPUT_PATH`) at the top of the script, then run the preprocessing section:

```bash
python gpu_anomaly.py
```

This produces a `preprocessed_data_new.pkl` file containing train/val/test splits and graph metadata.

### 3. Train

Training runs on [Modal](https://modal.com) with GPU acceleration:

```bash
# Check data and GPU availability
modal run gpu_anomaly.py --action check

# Verify preprocessed data integrity
modal run gpu_anomaly.py --action verify

# Train the model
modal run gpu_anomaly.py --action train --epochs 50 --batch-size 32 --hidden-dim 64 --learning-rate 0.0001
```

### 4. Inference

```bash
# Single sample
modal run gpu_anomaly.py --action inference --sample-idx 0

# Batch evaluation
modal run gpu_anomaly.py --action batch_inference --num-samples 100

# View training history
modal run gpu_anomaly.py --action history
```

## Training Configuration

| Hyperparameter | Default |
|----------------|---------|
| Hidden dimension | 64 |
| GAT attention heads | 2 |
| Temporal LSTM layers | 1 |
| Spatial GAT layers | 2 |
| Dropout | 0.5 |
| Learning rate | 5e-5 |
| Weight decay | 1e-3 |
| Batch size | 32 |
| Positive class weight | 3.0 |
| Early stopping patience | 5 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=3) |
| Mixed precision | Enabled (AMP + GradScaler) |

## Known Limitations

- **Label semantics.** Job failure is used as a proxy for GPU failure, but jobs can fail for non-hardware reasons (bugs, OOM, preemption). The model is effectively predicting "job failure involving this GPU" rather than hardware faults.
- **Positive class weight** is hardcoded at 3.0 rather than derived from the actual class imbalance, which is typically well below 1%.
- **Feature normalization** uses manual per-channel scaling rather than learned or standardized normalization.
- **Preprocessing performance.** The pipeline makes multiple passes over the GPU utilization CSV and uses row-level iteration in places where vectorized operations would be significantly faster.

## References

- Jeon, M., Venkataraman, S., Phanishayee, A., Qian, J., Xiao, W., & Yang, F. (2019). *Analysis of Large-Scale Multi-Tenant GPU Clusters for DNN Training Workloads.* USENIX ATC.
- [Philly Traces Repository](https://github.com/msr-fiddle/philly-traces)
- Brody, S., Alon, U., & Yahav, E. (2022). *How Attentive are Graph Attention Networks?* ICLR. (GATv2)

## License

This project is for research and educational purposes. The Philly trace data is subject to its own [license terms](https://github.com/msr-fiddle/philly-traces).
