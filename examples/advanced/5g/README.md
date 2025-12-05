# Lumos5G Throughput Prediction with Transformer

This project implements a PyTorch-based Transformer model for **time series prediction** of 5G network throughput using the Lumos5G-v1.0 dataset.

## Features

- **Time Series Prediction**: Uses sequences of past observations to predict future throughput
- **Transformer Architecture**: Multi-head self-attention mechanism for temporal feature learning
- **Comprehensive Data Preprocessing**: Handles missing values, encodes categorical features, normalizes numerical features
- **Sequence-aware Training**: Respects temporal ordering within each run (trajectory)
- **Advanced Training**: Learning rate scheduling, early stopping, model checkpointing
- **Visualization**: Loss curves and prediction scatter plots
- **Reproducibility**: Fixed random seeds for consistent results

## 🎯 Key Features
- **Time Series Modeling**: Predicts future throughput based on historical sequences
- **13 numerical features**: signal metrics, location, speed, direction, past throughput
- **3 categorical features**: network status, mobility mode, trajectory
- **Temporal awareness**: Maintains temporal ordering within trajectories (runs)
- **Configurable prediction**: Adjustable sequence length and prediction horizon
- **Robust preprocessing**: handles missing values and outliers
- **Modern architecture:** Transformer with multi-head attention and positional encoding
- **Production-ready**: includes checkpointing, metrics, and visualization
- **Flexible configuration**: 17+ command-line arguments for customization

## Dataset

The Lumos5G-v1.0 dataset contains 5G network measurements organized as **time series trajectories**:

- **`run_num`**: Trajectory identifier - each run represents a separate measurement session
- **`seq_num`**: Temporal index within each run (acts like a per-second timeline)

The data is sorted by `run_num` and `seq_num` to maintain temporal ordering. **Sequences are created within each run** to avoid mixing data across different trajectories.

**Numerical Features**:
- `abstractSignalStr`: Signal strength indicator
- `latitude`, `longitude`: GPS coordinates
- `movingSpeed`: Speed of movement
- `compassDirection`: Direction in degrees
- `lte_rssi`, `lte_rsrp`, `lte_rsrq`, `lte_rssnr`: LTE signal metrics
- `nr_ssRsrp`, `nr_ssRsrq`, `nr_ssSinr`: 5G NR signal metrics
- `Throughput`: Past throughput values (used as input features in sequences)

**Categorical Features**:
- `nrStatus`: Connection status
- `mobility_mode`: Type of mobility (e.g., driving)
- `trajectory_direction`: Direction of trajectory

**Target**:
- `Throughput`: Network throughput (Mbps) - the value we're predicting at future timesteps

### Time Series Structure

The model uses sequences of observations to make predictions:

**Example with `sequence_length=10`, `prediction_horizon=1`:**
```
Input:  Timesteps [t₀, t₁, t₂, ..., t₉]  → All features including past throughput
Target: Throughput at timestep t₁₀
```

**Example with `sequence_length=10`, `prediction_horizon=3`:**
```
Input:  Timesteps [t₀, t₁, t₂, ..., t₉]
Target: Throughput at timestep t₁₂ (3 steps ahead)
```

For a run with 100 timesteps:
- `sequence_length=10`, `prediction_horizon=1` → Creates 90 sequences
- First sequence doesn't have enough history
- Sequences don't cross run boundaries

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python train.py --data_path Lumos5G-v1.0/Lumos5G-v1.0.csv --output_dir outputs --epochs 50
```

This will train a model with default settings:
- Sequence length: 10 (uses 10 past timesteps)
- Prediction horizon: 1 (predicts 1 timestep ahead)

### Advanced Training with Custom Parameters

```bash
python train.py \
    --data_path Lumos5G-v1.0/Lumos5G-v1.0.csv \
    --output_dir outputs \
    --sequence_length 20 \
    --prediction_horizon 3 \
    --d_model 256 \
    --nhead 16 \
    --num_layers 4 \
    --dim_feedforward 1024 \
    --dropout 0.2 \
    --batch_size 512 \
    --epochs 100 \
    --lr 0.0005 \
    --weight_decay 0.01
```

This configuration:
- Uses 20 past timesteps to predict 3 timesteps ahead
- Larger model with 256-dimensional embeddings
- 16 attention heads and 4 transformer layers

### Command-Line Arguments

**Data Parameters**:
- `--data_path`: Path to the CSV file (default: `Lumos5G-v1.0/Lumos5G-v1.0.csv`)
- `--output_dir`: Directory to save outputs (default: `outputs`)
- `--train_split`: Train/validation split ratio (default: `0.8`)
- `--sequence_length`: Number of past timesteps to use (default: `10`)
- `--prediction_horizon`: Number of timesteps ahead to predict (default: `1`)

**Model Parameters**:
- `--d_model`: Dimension of the model (default: `128`)
- `--nhead`: Number of attention heads (default: `8`)
- `--num_layers`: Number of transformer encoder layers (default: `3`)
- `--dim_feedforward`: Dimension of feedforward network (default: `512`)
- `--dropout`: Dropout rate (default: `0.1`)

**Training Parameters**:
- `--batch_size`: Batch size (default: `256`)
- `--epochs`: Number of epochs (default: `50`)
- `--lr`: Learning rate (default: `0.001`)
- `--weight_decay`: Weight decay for regularization (default: `0.01`)
- `--num_workers`: Number of data loading workers (default: `4`)
- `--seed`: Random seed for reproducibility (default: `42`)

## Model Architecture

The `TransformerTimeSeriesRegressor` model (defined in `model.py`) consists of:

1. **Input Embedding Layer**: Projects input features to model dimension
2. **Positional Encoding**: Learnable positional embeddings for sequence positions
3. **Transformer Encoder**: Multi-layer transformer with self-attention across timesteps
4. **Temporal Aggregation**: Uses the last timestep's representation
5. **Output Layers**: Fully connected layers for throughput prediction

### How It Works

```
Input Sequence (10 timesteps × 16 features)
           ↓
    Embedding Layer
           ↓
  Positional Encoding
           ↓
  Transformer Encoder
  (Self-attention across time)
           ↓
   Last Timestep Output
           ↓
    FC Layers
           ↓
  Predicted Throughput
```

The model learns temporal dependencies and patterns across the sequence to make accurate predictions.

## Why Time Series Prediction?

### Advantages Over Point-wise Prediction

**1. Temporal Dependencies**
- Captures trends and patterns over time
- Models how throughput changes based on recent history
- Better handles transitions and fluctuations in network conditions

**2. Context-Aware Predictions**
- Uses information from multiple timesteps
- Understands trajectory patterns (e.g., moving through areas with varying signal strength)
- More robust to instantaneous noise

**3. Realistic for Deployment**
- Mirrors real-world scenario: predict future based on recent observations
- Adjustable prediction horizon based on use case (1 second ahead vs. 5 seconds ahead)
- Natural fit for network monitoring and proactive resource allocation

**4. Leverages Transformer Strengths**
- Self-attention across time captures long-range dependencies
- Positional encoding preserves temporal information
- Parallel processing of sequences for efficiency

### Use Cases

- **Short-term prediction** (`horizon=1`): Immediate network optimization
- **Medium-term prediction** (`horizon=5`): Resource pre-allocation
- **Long-term prediction** (`horizon=10+`): Capacity planning

## Code Structure

The project is organized into modular components:

- **`model.py`**: Contains the `TransformerTimeSeriesRegressor` model architecture
- **`data.py`**: Contains data loading, preprocessing, and the `Lumos5GTimeSeriesDataset` class
- **`train.py`**: Training script with full pipeline
- **`inference.py`**: Inference script with metrics and visualization

### Key Implementation Details

**Sequence Creation** (`data.py`):
```python
# For each run_num (trajectory):
for i in range(len(run) - sequence_length - prediction_horizon + 1):
    # Input: sequence of past timesteps
    X[i] = features[i:i+sequence_length]
    
    # Target: throughput at future timestep
    y[i] = throughput[i + sequence_length + prediction_horizon - 1]
```

**Model Input/Output**:
- Input shape: `(batch_size, sequence_length, num_features)`
- Output: `(batch_size,)` - single throughput value per sequence
- Features per timestep: 16 (13 numerical + 3 categorical + past throughput)

## Outputs

After training, the following files will be saved in the output directory:

- `best_model.pth`: Best model checkpoint with lowest validation loss
- `loss_plot.png`: Training and validation loss curves
- `predictions_plot.png`: Scatter plot of predictions vs actual values

## Inference

### Running Inference on New Data

Use the `inference.py` script to run predictions on new or existing data:

#### Basic Inference

```bash
python inference.py \
    --checkpoint outputs/best_model.pth \
    --data_path Lumos5G-v1.0/Lumos5G-v1.0.csv \
    --output_dir inference_outputs
```

Or use the quick start script:

```bash
./run_inference.sh
```

#### Inference with Visualization

```bash
python inference.py \
    --checkpoint outputs/best_model.pth \
    --data_path Lumos5G-v1.0/Lumos5G-v1.0.csv \
    --output_dir inference_outputs \
    --batch_size 512 \
    --plot
```

### Inference Arguments

- `--checkpoint`: Path to model checkpoint file (required)
- `--data_path`: Path to CSV file for inference (required)
- `--output_dir`: Directory to save outputs (default: `inference_outputs`)
- `--batch_size`: Batch size for inference (default: `256`)
- `--plot`: Generate comparison plots if ground truth is available (flag)

### Inference Outputs

The inference script generates:

1. **`predictions.csv`**: Original data with added `Predicted_Throughput` column
2. **`metrics.txt`**: Evaluation metrics (if ground truth available)
3. **`inference_comparison.png`**: Comprehensive comparison plots (if `--plot` flag is used)
   - Predictions vs Actual scatter plot
   - Residuals plot
   - Distribution comparison
   - Error distribution histogram
4. **`timeseries_comparison.png`**: Time series visualization (if `--plot` flag is used)
   - Actual vs Predicted throughput over time
   - Prediction error over time
   - Shows temporal patterns and model performance across timesteps

### Inference on Data Without Ground Truth

The script automatically detects if the `Throughput` column is missing and will only generate predictions without computing metrics:

```bash
python inference.py \
    --checkpoint outputs/best_model.pth \
    --data_path new_data_without_labels.csv \
    --output_dir predictions
```

## Model Checkpoint

The saved model checkpoint includes:
- Model state dictionary
- Optimizer state dictionary
- Validation metrics (loss, MAE, RMSE)
- Scaler and label encoders for preprocessing
- Model configuration (d_model, nhead, num_layers, etc.)
- **`sequence_length`**: Number of past timesteps used during training
- **`prediction_horizon`**: Number of steps ahead the model predicts

### Checkpoint Structure
```python
{
    'model_state_dict': ...,
    'optimizer_state_dict': ...,
    'scaler': StandardScaler object,
    'label_encoders': {feature_name: LabelEncoder, ...},
    'input_dim': 16,                    # Features per timestep
    'sequence_length': 10,              # Past timesteps
    'prediction_horizon': 1,            # Future timesteps
    'val_loss': 123.45,
    'val_mae': 12.34,
    'val_rmse': 15.67,
    'model_config': {
        'd_model': 128,
        'nhead': 8,
        'num_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1
    }
}
```

## Loading a Trained Model

```python
import torch
from model import TransformerTimeSeriesRegressor

# Load checkpoint (weights_only=False is required for PyTorch 2.6+
# because we save preprocessing objects like scaler and label encoders)
checkpoint = torch.load('outputs/best_model.pth', weights_only=False)

# Recreate model
model = TransformerTimeSeriesRegressor(
    input_dim=checkpoint['input_dim'],
    **checkpoint['model_config']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Access preprocessing tools and configuration
scaler = checkpoint['scaler']
label_encoders = checkpoint['label_encoders']
sequence_length = checkpoint['sequence_length']
prediction_horizon = checkpoint['prediction_horizon']

print(f"Model predicts {prediction_horizon} step(s) ahead using {sequence_length} past timesteps")
```

**Note**: The `weights_only=False` parameter is required when loading the checkpoint because it contains non-tensor objects (StandardScaler and LabelEncoders). This is safe since you're loading your own trained model.

## Programmatic Inference

For more control over the inference process, you can use the model programmatically:

```python
import torch
import pandas as pd
from model import TransformerTimeSeriesRegressor
from data import preprocess_timeseries_data

# Load checkpoint
checkpoint = torch.load('outputs/best_model.pth', weights_only=False)
sequence_length = checkpoint['sequence_length']

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerTimeSeriesRegressor(
    input_dim=checkpoint['input_dim'],
    **checkpoint['model_config']
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess data (creates sequences)
df = pd.read_csv('your_data.csv')
X, sequence_indices = preprocess_timeseries_data(
    df, 
    checkpoint['scaler'], 
    checkpoint['label_encoders'],
    sequence_length
)

# Run inference
with torch.no_grad():
    X_tensor = torch.from_numpy(X).to(device)
    predictions = model(X_tensor).cpu().numpy()

# Map predictions back to dataframe
df['Predicted_Throughput'] = np.nan
for i, idx in enumerate(sequence_indices):
    df.loc[idx, 'Predicted_Throughput'] = predictions[i]
```

**Note**: Not all rows will have predictions - the first `sequence_length` rows in each run don't have enough history.

## Performance Metrics

The model is evaluated using:
- **MSE (Mean Squared Error)**: Loss function
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Squared Error)**: Standard deviation of prediction errors
- **R² Score**: Coefficient of determination (how well the model explains variance)
- **MAPE (Mean Absolute Percentage Error)**: Percentage error

## Performance Considerations

### Dataset Size
- **Point-wise approach**: N samples → N predictions
- **Time series approach**: N samples → approximately (N - seq_len - horizon) predictions per run
- Slight reduction due to sequence requirements, but predictions are more informed

### Training Time
- Similar per epoch (sequences processed in batches)
- May need slightly more epochs due to increased task complexity
- Batch size can be adjusted based on GPU memory

### Inference
- Requires `sequence_length` previous timesteps for each prediction
- First `sequence_length` rows in each run cannot be predicted (no history)
- Predictions for later timesteps may be more accurate (more context available)
- Predictions don't cross `run_num` boundaries (maintains trajectory integrity)

## Tips for Better Performance

### Model Architecture
1. **Increase model capacity**: Use larger `d_model` and `dim_feedforward`
   ```bash
   --d_model 256 --dim_feedforward 1024
   ```

2. **Add more layers**: Increase `num_layers` for deeper models
   ```bash
   --num_layers 6
   ```

3. **Adjust attention heads**: More heads can capture different patterns
   ```bash
   --nhead 16
   ```

### Time Series Configuration
4. **Longer sequences**: Capture more temporal context
   ```bash
   --sequence_length 20
   ```

5. **Adjust prediction horizon**: Match your use case
   ```bash
   --prediction_horizon 5  # Predict 5 steps ahead
   ```

### Training
6. **Tune batch size**: Larger batches can stabilize training
   ```bash
   --batch_size 512
   ```

7. **Adjust learning rate**: Start with lower rates for larger models
   ```bash
   --lr 0.0001
   ```

8. **More epochs**: Complex temporal patterns may need more training
   ```bash
   --epochs 100
   ```

### Advanced Techniques
9. **Ensemble models**: Train multiple models with different configurations and average predictions
10. **Data augmentation**: Consider adding noise or temporal jittering during training
11. **Feature engineering**: Derive additional features like velocity changes, signal strength derivatives

## Future Enhancements

Potential improvements to the time series model:

1. **Multi-step Prediction**: Predict multiple future timesteps at once
   - Output: `(batch, prediction_horizon)` instead of `(batch,)`
   - Useful for longer-term planning

2. **Attention Visualization**: Visualize which past timesteps are most important
   - Helps understand what the model learns
   - Can guide feature engineering

3. **Variable Sequence Lengths**: Dynamic padding for different trajectory lengths
   - More flexible data handling
   - Better utilize short trajectories

4. **Encoder-Decoder Architecture**: Separate encoder for past, decoder for future
   - More powerful for multi-step prediction
   - Can condition on desired future scenario

5. **Auxiliary Tasks**: Jointly predict other metrics
   - Signal strength prediction
   - Connection status prediction
   - Multi-task learning for better representations

6. **Attention Mechanisms**: Add cross-attention between spatial and temporal features
   - Better model location-throughput relationships
   - Capture spatial-temporal interactions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## License

This project uses the Lumos5G-v1.0 dataset. Please refer to the dataset's LICENSE for usage terms.

