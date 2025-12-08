# Comparison Experiments Guide

## Overview

Compare three training approaches:
1. **Centralized** - Single model trained on all data
2. **Federated** - Collaborative training across clients
3. **Local-Only** - Each client trains independently

## Quick Start

### Run Complete Experiment

```bash
# Automated experiment (recommended)
python run_comparison_experiment.py

# This will:
# 1. Create preprocessors (if needed)
# 2. Train centralized model on full dataset
# 3. Run federated learning across N clients
# 4. Train N independent local models
# 5. Generate comparison visualizations
```

### Manual Experiment

```bash
# Step 1: Setup
python create_schema_based_preprocessors.py --output_dir federated_data
python split_train_federated.py --data_path train.csv --num_clients 4 --output_dir federated_data

# Step 2: Centralized Training
python train.py \
    --data_path train.csv \
    --output_dir experiments/centralized \
    --epochs 20

# Step 3: Federated Learning
python job.py \
    --n_clients 4 \
    --num_rounds 10 \
    --data_dir federated_data \
    --job_name federated_exp

# Step 4: Local-Only Training
for i in {1..4}; do
    python train.py \
        --data_path federated_data/site-${i}.csv \
        --output_dir experiments/local_site-${i} \
        --epochs 20
done

# Step 5: Compare Results
python compare_results.py --exp_dir experiments/your_experiment_name/
```

## What Gets Compared

### Metrics

- **Validation Loss (MSE)** - Mean squared error on validation set
- **MAE** - Mean absolute error
- **RMSE** - Root mean squared error

### Scenarios

| Scenario | Training Data | Model Count | Data Sharing |
|----------|---------------|-------------|--------------|
| **Centralized** | Full dataset | 1 | Full dataset centralized |
| **Federated** | Distributed | 1 (shared) | No raw data shared |
| **Local-Only** | Per-client | N (independent) | No sharing at all |

## Expected Outcomes

### Typical Performance Ranking

1. **Centralized** (best) - Access to all data
2. **Federated** (middle) - Collaborative without data sharing
3. **Local-Only** (worst) - Limited to local data only

### Why Federated is Better Than Local-Only

- ✅ Benefits from knowledge across all clients
- ✅ Better generalization to unseen patterns
- ✅ No data sharing required (privacy-preserving)
- ✅ Single model deployment

### When Local-Only Might Win

- Site-specific patterns dominate
- Data distributions very heterogeneous
- Communication costs prohibitive

## Visualizations Generated

### 1. Final Performance Comparison

Bar chart comparing final metrics across all three scenarios:
- Shows which approach achieves lowest loss/error
- Highlights best performer with gold border

**File:** `final_performance_comparison.png`

### 2. Per-Client Comparison

Compares federated vs local-only for each individual client:
- Shows if federated helps weaker clients
- Identifies if some clients benefit more than others
- Centralized baseline shown as horizontal line

**File:** `per_client_comparison.png`

### 3. Summary Table

Clean table with all metrics:
- Easy comparison of numbers
- CSV export for further analysis

**Files:** `summary_table.png`, `summary.csv`

## Experiment Configuration

### Customize in `run_comparison_experiment.py`

```python
# Number of federated clients
num_clients = 4

# Federated learning rounds
num_rounds = 10

# Epochs for centralized/local training
epochs = 20
```

### Advanced Configuration

```bash
# Custom model architecture
python job.py \
    --n_clients 4 \
    --num_rounds 10 \
    --d_model 256 \
    --num_layers 4 \
    --nhead 16

# Different batch size
python train.py \
    --data_path train.csv \
    --output_dir experiments/centralized \
    --epochs 20 \
    --batch_size 512
```

## Analyzing Results

### View TensorBoard

```bash
# Centralized
tensorboard --logdir experiments/your_exp/models/centralized/

# Federated
tensorboard --logdir experiments/your_exp/models/federated_exp/

# Local (all clients)
tensorboard --logdir experiments/your_exp/models/local_*
```

### Check Logs

```bash
# All logs
ls experiments/your_exp/logs/

# View specific log
cat experiments/your_exp/logs/01_centralized.log
cat experiments/your_exp/logs/02_federated.log
cat experiments/your_exp/logs/03_local_site-1.log
```

### Load Models for Inference

```python
import torch

# Load centralized model
checkpoint = torch.load('experiments/your_exp/models/centralized/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Load federated model (from any client)
checkpoint = torch.load('experiments/your_exp/models/federated_exp/site-1/model_round_10.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Load local model
checkpoint = torch.load('experiments/your_exp/models/local_site-1/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## Troubleshooting

### "Preprocessors not found"

```bash
python create_schema_based_preprocessors.py --output_dir federated_data
```

### "Client data not found"

```bash
python split_train_federated.py --data_path train.csv --num_clients 4
```

### Experiment takes too long

Reduce epochs or rounds:
```python
# In run_comparison_experiment.py
num_rounds = 5  # Instead of 10
epochs = 10     # Instead of 20
```

### Out of memory

Reduce batch size:
```bash
python train.py --data_path train.csv --batch_size 128
```

## Example Output

```
======================================================================
LUMOS5G TRAINING COMPARISON EXPERIMENT
======================================================================
Experiment: comparison_exp_20250108_143022
Clients: 4
FL Rounds: 10
Local Epochs: 20
======================================================================

SCENARIO 1: CENTRALIZED TRAINING
✓ Completed: Centralized Training (took 300.5s)

SCENARIO 2: FEDERATED LEARNING
✓ Completed: Federated Learning (took 450.2s)

SCENARIO 3: LOCAL-ONLY TRAINING
✓ Completed: Local Training - site-1 (took 75.3s)
✓ Completed: Local Training - site-2 (took 72.1s)
✓ Completed: Local Training - site-3 (took 68.9s)
✓ Completed: Local Training - site-4 (took 71.2s)

======================================================================
SUMMARY RESULTS
======================================================================
    Scenario                      Description  Val Loss   Val MAE  Val RMSE
 Centralized       Single model on full dataset  0.0234    0.1123    0.1530
   Federated          Collaborative (4 clients)  0.0267    0.1245    0.1634
  Local-Only  Independent (4 clients avg)        0.0389    0.1567    0.1973
======================================================================

✓ All plots saved to: experiments/comparison_exp_20250108_143022/plots/
```
