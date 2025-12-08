# Lumos5G Federated Learning

Complete guide for training Transformer models on distributed 5G network data using NVIDIA FLARE with schema-based feature consistency.

## Table of Contents

- [Quick Start](#quick-start)
- [Overview](#overview)
- [Schema-Based Approach](#schema-based-approach)
- [Files](#files)
- [Detailed Setup](#detailed-setup)
- [Advanced Usage](#advanced-usage)
- [Real-World Deployment](#real-world-deployment)
- [Troubleshooting](#troubleshooting)
- [Architecture](#architecture)
- [References](#references)

---

## Quick Start

### Simulation (Single Machine)

```bash
# 1. Create schema-based preprocessors (no data needed)
python create_schema_based_preprocessors.py --output_dir federated_data

# 2. Split training data to simulate multiple clients
python split_train_federated.py --data_path train.csv --num_clients 4 --output_dir federated_data

# 3. Run federated learning simulation
python job.py --n_clients 4 --num_rounds 10

# 4. View results
tensorboard --logdir /tmp/nvflare/simulation/lumos5g-fedavg/
```

### Real Deployment (Distributed)

```bash
# === SERVER/COORDINATOR ===
python create_schema_based_preprocessors.py --output_dir shared_config
# Distribute shared_config/ to all clients
python job.py --n_clients <N> --num_rounds <R> --data_dir /path/to/shared_config

# === CLIENTS ===
# Receive preprocessors and training scripts
# Client script runs automatically via NVFlare
```

---

## Overview

This federated learning setup enables training a Transformer-based time series prediction model across multiple distributed clients (sites) **without sharing raw data**. Each client trains on its local subset of 5G network measurements.

### Key Features

✅ **Privacy-Preserving**: No raw data ever leaves client devices  
✅ **No Centralized Data**: Preprocessors created from domain specifications only  
✅ **Guaranteed Consistency**: All clients have matching input dimensions  
✅ **Easy Scaling**: Add new clients anytime without retraining  
✅ **Domain-Driven**: Uses 5G network specifications (3GPP standards)

### The Challenge

In federated learning, different clients may have different categorical values in their local data (e.g., different cell towers, connection states). Without proper handling, this leads to different input dimensions after one-hot encoding, causing model incompatibility.

---

## Schema-Based Approach

### Why Schema-Based?

Instead of fitting preprocessors on data, we define all possible categorical values based on **domain knowledge**:

1. **Network specifications** tell you valid connection states, mobility modes
2. **Industry standards** define valid categories (3GPP specs)
3. **Data collection protocols** specify allowed values

**Result:** Create preprocessors without any data → distribute to all clients → guaranteed consistency!

### How It Works

```python
# Domain knowledge defines vocabularies
categorical_schemas = {
    'nrStatus': ['CONNECTED', 'NOT_RESTRICTED', 'RESTRICTED', 
                 'UNAVAILABLE', 'NONE', 'UNKNOWN'],
    'mobility_mode': ['stationary', 'walking', 'driving', 'UNKNOWN'],
    'trajectory_direction': ['CW', 'ACW', 'UNKNOWN'],
}

# Each client uses the same encoder
'CONNECTED' → 0 → one-hot [1, 0, 0, 0, 0, 0]
'NONE' → 4 → one-hot [0, 0, 0, 0, 1, 0]
```

All clients create **identical input dimensions** (13 numerical + 13 one-hot categorical = **26 total**).

### Advantages

- ✅ **No data needed**: Create preprocessors before any data collection
- ✅ **Most privacy-preserving**: Zero data sharing required
- ✅ **Fast setup**: No data processing or aggregation needed
- ✅ **Works with dynamic clients**: New clients can join anytime
- ✅ **Predictable**: Fixed input dimension known in advance

### Limitations

- ⚠️ **May include unused categories**: Could result in sparse features
- ⚠️ **Requires domain expertise**: Need to know valid value ranges
- ⚠️ **Static schema**: Adding new categories requires schema update

---

## Files

### Core Scripts

- **`create_schema_based_preprocessors.py`** - Creates preprocessors from domain knowledge (no data)
- **`client.py`** - Federated learning client training script
- **`job.py`** - Federated learning job orchestration
- **`split_train_federated.py`** - Utility to split data for simulation
- **`model.py`** - Transformer model architecture
- **`data.py`** - Dataset loader with one-hot encoding
- **`train.py`** - Centralized training script (for comparison)
- **`test_preprocessors.py`** - Validation utility

---

## Detailed Setup

### Step 1: Create Schema-Based Preprocessors

Create preprocessors using pre-defined feature schemas based on domain knowledge. This ensures all clients use consistent feature encoding **without requiring access to any training data**.

```bash
python create_schema_based_preprocessors.py --output_dir federated_data

# Output:
#   federated_data/scaler.pkl              - StandardScaler (identity for now)
#   federated_data/label_encoders.pkl      - LabelEncoders with 5G vocabularies
#   federated_data/feature_config.txt      - Feature summary (input_dim = 26)
```

**What gets created:**
- **Numerical features (13)**: Signal strength, location, speed, throughput
- **Categorical vocabularies**:
  - `nrStatus`: 6 classes (CONNECTED, NOT_RESTRICTED, etc.)
  - `mobility_mode`: 4 classes (stationary, walking, driving, UNKNOWN)
  - `trajectory_direction`: 3 classes (CW, ACW, UNKNOWN)
- **Total input dimension**: 13 + 6 + 4 + 3 = **26 dimensions**

**Verify preprocessors match your data:**
```bash
python test_preprocessors.py federated_data/site-1.csv
# Should show all ✅ with no warnings
```

### Step 2: Split Training Data (Simulation Only)

For simulation purposes, split your training data into separate files for each federated client. 

⚠️ **Note**: In real-world deployments, each client would already have their own local data.

```bash
python split_train_federated.py --data_path train.csv --num_clients 4 --output_dir federated_data

# Output:
#   federated_data/site-1.csv                  - Client 1 data
#   federated_data/site-2.csv                  - Client 2 data
#   federated_data/site-3.csv                  - Client 3 data
#   federated_data/site-4.csv                  - Client 4 data
#   federated_data/site-run_assignments.txt    - Documentation
```

**Data distribution properties:**
- ✅ **No overlap**: Each run (trajectory) assigned to exactly one client
- ✅ **Balanced**: Runs distributed as evenly as possible
- ✅ **Reproducible**: Use `--seed` for consistent splits

**Validate splits:**
```bash
python split_train_federated.py --data_path train.csv --num_clients 4 --validate
# Checks for overlapping runs
```

### Step 3: Run Federated Learning

Execute the federated learning job:

```bash
python job.py --n_clients 4 --num_rounds 10

# With custom options:
python job.py \
    --n_clients 4 \
    --num_rounds 10 \
    --data_dir /path/to/federated_data \
    --d_model 256 \
    --num_layers 4
```

**What happens:**
1. Server initializes global model with input_dim=26
2. Server distributes model to all clients (site-1, site-2, site-3, site-4)
3. Each client:
   - Loads shared preprocessors
   - Loads local data
   - Trains model for 2 epochs
   - Sends updated model back
4. Server aggregates updates using FedAvg (weighted by sample count)
5. Repeat for specified number of rounds

**Monitor progress:**
```bash
# Real-time logs
tail -f /tmp/nvflare/simulation/lumos5g-fedavg/server/log.txt

# TensorBoard visualization
tensorboard --logdir /tmp/nvflare/simulation/lumos5g-fedavg/
# Open browser to http://localhost:6006
```

### Step 4: View Results

After training completes:

```bash
# Server logs and aggregated metrics
ls /tmp/nvflare/simulation/lumos5g-fedavg/server/

# Individual client logs
ls /tmp/nvflare/simulation/lumos5g-fedavg/site-*/

# Client model checkpoints
ls outputs/site-1/  # model_round_1.pth, model_round_2.pth, ...
ls outputs/site-2/
```

---

## Advanced Usage

### Custom Model Architecture

```bash
python job.py \
    --n_clients 4 \
    --num_rounds 20 \
    --d_model 256 \
    --nhead 16 \
    --num_layers 4 \
    --dim_feedforward 1024 \
    --dropout 0.2
```

### Custom Training Parameters

Modify `job.py` to pass custom arguments to clients:

```python
# In job.py:
train_args = (
    f"--data_dir {data_dir} "
    f"--batch_size 512 "
    f"--epochs_per_round 5 "
    f"--lr 0.0005 "
    f"--weight_decay 0.001"
)
```

### Different Random Seeds

```bash
# Split data with different seed
python split_train_federated.py \
    --data_path train.csv \
    --num_clients 4 \
    --seed 123
```

### Compare with Centralized Training

```bash
# Centralized training (for baseline)
python train.py --data_path train.csv --output_dir outputs/centralized

# Federated learning
python job.py --n_clients 4 --num_rounds 10

# Compare results
```

---

## Real-World Deployment

### Initial Setup (One Time)

```bash
# === SERVER/COORDINATOR ===

# 1. Create schema-based preprocessors
python create_schema_based_preprocessors.py --output_dir shared_config

# 2. Get input dimension
INPUT_DIM=$(grep "Input Dimension:" shared_config/feature_config.txt | cut -d: -f2 | tr -d ' ')
echo "Input dimension: $INPUT_DIM"

# 3. Distribute to all clients (via secure channel):
#    - shared_config/scaler.pkl
#    - shared_config/label_encoders.pkl
#    - shared_config/feature_config.txt
#    - client.py (training script)
#    - model.py, data.py (utilities)
```

### Ongoing Federated Learning

```bash
# === SERVER ===

# Start federated learning job
python job.py \
    --n_clients 10 \
    --num_rounds 50 \
    --input_dim $INPUT_DIM \
    --data_dir /path/to/shared_config

# Server automatically:
# - Sends global model to clients
# - Receives updated models
# - Aggregates using FedAvg
# - Repeats for num_rounds
```

```bash
# === CLIENTS (Edge Devices) ===

# Clients have:
# - Local data (collected on-device)
# - Shared preprocessors (received once)
# - Training scripts (client.py, model.py, data.py)

# Client script runs automatically via NVFlare:
# 1. Loads shared preprocessors
# 2. Loads local data
# 3. Receives global model from server
# 4. Trains on local data
# 5. Sends updated model back
# 6. Repeats for each round

# No manual intervention needed!
```

### Adding New Clients

```bash
# === NEW CLIENT JOINS ===

# 1. Distribute same preprocessor files
# 2. Client loads preprocessors
# 3. Client processes local data
# 4. Client participates in next round

# No retraining or schema updates needed!
```

### Production Considerations

#### 1. Version Control

```bash
# Version your preprocessors
shared_config/
├── scaler_v1.pkl
├── label_encoders_v1.pkl
└── feature_config_v1.txt

# Include version in model checkpoints
checkpoint = {
    'model_state_dict': model.state_dict(),
    'preprocessor_version': 'v1',
    'schema_version': 'v1',
    'input_dim': 26,
    ...
}
```

#### 2. Schema Updates

When updating the schema (e.g., new connection state):

```bash
# 1. Create new schema version
python create_schema_based_preprocessors.py --output_dir shared_config_v2

# 2. Distribute v2 preprocessors to all clients

# 3. Start new FL job with updated input_dim

# 4. Optionally: Convert old models using transfer learning
```

#### 3. Monitoring

Track which categorical values appear in practice:

```python
# Optional: Add to client.py
def log_unseen_categories(dataset):
    """Log which categories map to 'UNKNOWN'"""
    unseen = dataset.get_unseen_categories()
    if unseen:
        logger.info(f"Unseen categories: {unseen}")
    # Report to server for schema refinement
```

#### 4. Handling Unknown Values

All schemas include 'UNKNOWN' category as catch-all:

```python
# In data.py (automatic handling)
if value not in le.classes_:
    value = 'UNKNOWN'  # Maps to UNKNOWN category
```

---

## Troubleshooting

### "Preprocessor files not found"

**Error:** `FileNotFoundError: Preprocessor files not found!`

**Solution:**
```bash
python create_schema_based_preprocessors.py --output_dir federated_data
```

This creates preprocessors using domain knowledge (no training data needed).

### "Data file not found"

**Error:** `FileNotFoundError: Data file not found: federated_data/site-1.csv`

**Solution:**
```bash
# For simulation: split data first
python split_train_federated.py --data_path train.csv --num_clients 4

# For production: ensure clients have local data
python job.py --n_clients 4 --num_rounds 10 --data_dir $(pwd)/federated_data
```

### "Size mismatch" error

**Error:** `RuntimeError: size mismatch for input_embedding.weight: copying a param with shape torch.Size([128, 26]) from checkpoint, the shape in current model is torch.Size([128, 16])`

**Causes:**
1. Preprocessors not regenerated after schema update
2. Clients using different preprocessor versions
3. Input dimension mismatch

**Solution:**
```bash
# 1. Regenerate preprocessors
python create_schema_based_preprocessors.py --output_dir federated_data

# 2. Verify all clients use same files
python test_preprocessors.py federated_data/site-1.csv

# 3. Ensure input_dim matches feature_config.txt
grep "Input Dimension:" federated_data/feature_config.txt
```

### Unknown categorical values

**Warning:** `⚠️ Column 'nrStatus' has unseen values: ['NEW_VALUE']`

**What happens:**
- Unseen values automatically map to 'UNKNOWN' category
- Training continues normally
- Consider adding to schema if common

**Solution (if needed):**
```bash
# Update schema to include new value
# Edit create_schema_based_preprocessors.py
categorical_schemas = {
    'nrStatus': [..., 'NEW_VALUE', 'UNKNOWN'],
}

# Regenerate preprocessors
python create_schema_based_preprocessors.py --output_dir federated_data
```

### Model initialization error

**Error:** `TypeError: TransformerTimeSeriesRegressor.__init__() missing 1 required positional argument: 'input_dim'`

**Solution:** Ensure `model.py` stores initialization parameters as member variables:

```python
# In model.py __init__
self.input_dim = input_dim
self.d_model = d_model
self.nhead = nhead
# ... etc
```

### Client count mismatch

**Error:** Missing client data files

**Solution:**
```bash
# Number of clients must match number of data splits
python split_train_federated.py --data_path train.csv --num_clients 5
python job.py --n_clients 5 --num_rounds 10  # Must match!
```

---

## Architecture

### Model

- **Type**: Transformer Encoder for time series regression
- **Architecture**:
  - Input embedding layer (input_dim → d_model)
  - Learnable positional encoding
  - N transformer encoder layers
  - Feedforward output layers
- **Input**: Sequence of 10 past timesteps (26 features each)
- **Output**: Predicted throughput for next timestep
- **Default parameters**: 
  - d_model=128, nhead=8, num_layers=3
  - dim_feedforward=512, dropout=0.1
  - Total parameters: ~500K

### Features

**Numerical (13 features):**
- Signal strength: `abstractSignalStr`
- Location: `latitude`, `longitude`
- Movement: `movingSpeed`, `compassDirection`
- LTE metrics: `lte_rssi`, `lte_rsrp`, `lte_rsrq`, `lte_rssnr`
- 5G NR metrics: `nr_ssRsrp`, `nr_ssRsrq`, `nr_ssSinr`
- Target: `Throughput`

**Categorical (3 features, 13 one-hot dimensions):**
- `nrStatus`: 6 classes (connection status)
- `mobility_mode`: 4 classes (stationary/walking/driving)
- `trajectory_direction`: 3 classes (CW/ACW)

**Total**: 26 input dimensions

### Federated Learning

- **Algorithm**: FedAvg (Federated Averaging)
- **Aggregation**: Weighted by number of samples per client
- **Communication**: Full model parameters
- **Rounds**: Server coordinates multiple training rounds
- **Client updates**: Typically 2 epochs per round

### Training

- **Optimizer**: AdamW
- **Loss**: MSE (Mean Squared Error)
- **Metrics**: MAE, RMSE
- **Scheduler**: ReduceLROnPlateau
- **Data split**: Run-based (entire trajectories assigned to train/val)

### Privacy

**What is shared:**
- Model architecture
- Model parameters (aggregated updates)
- Number of samples per client
- Training metrics

**What is protected:**
- Raw sensor measurements
- Location data
- Temporal patterns
- Client-specific data distributions

---

## References

- [NVIDIA FLARE Documentation](https://nvidia.github.io/NVFlare/)
- [FedAvg Paper](https://arxiv.org/abs/1602.05629) - McMahan et al., 2017
- [3GPP Specifications](https://www.3gpp.org/) - 5G network standards
- [Lumos5G Dataset](https://github.com/mlab-upenn/lumos5g) - 5G measurements

---

## Quick Reference Card

```bash
# Setup (one time)
python create_schema_based_preprocessors.py --output_dir federated_data
python split_train_federated.py --data_path train.csv --num_clients 4

# Run federated learning
python job.py --n_clients 4 --num_rounds 10

# Monitor
tensorboard --logdir /tmp/nvflare/simulation/lumos5g-fedavg/

# Validate
python test_preprocessors.py federated_data/site-1.csv

# Compare with centralized
python train.py --data_path train.csv --output_dir outputs/centralized
```

---

**For questions or issues, refer to the detailed sections above or check the NVIDIA FLARE documentation.**
