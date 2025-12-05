# Lumos5G Throughput Prediction with Transformer

This project implements a PyTorch-based Transformer model to predict 5G network throughput using the Lumos5G-v1.0 dataset.

## Features

- **Transformer Architecture**: Uses multi-head self-attention mechanism for feature learning
- **Comprehensive Data Preprocessing**: Handles missing values, encodes categorical features, and normalizes numerical features
- **Advanced Training**: Includes learning rate scheduling, early stopping, and model checkpointing
- **Visualization**: Generates loss curves and prediction scatter plots
- **Reproducibility**: Fixed random seeds for consistent results

## 🎯 Key Features
- **13 numerical features:** signal metrics, location, speed, direction
- **3 categorical features:** network status, mobility mode, trajectory
- **Robust preprocessing:** handles missing values and outliers
- **Modern architecture:** Transformer with multi-head attention
- **Production-ready:** includes checkpointing, metrics, and visualization
- **Flexible configuration:** 15+ command-line arguments for customization

## Dataset

The Lumos5G-v1.0 dataset contains 5G network measurements including:

**Numerical Features**:
- `seq_num`: Sequence number
- `abstractSignalStr`: Signal strength indicator
- `latitude`, `longitude`: GPS coordinates
- `movingSpeed`: Speed of movement
- `compassDirection`: Direction in degrees
- `lte_rssi`, `lte_rsrp`, `lte_rsrq`, `lte_rssnr`: LTE signal metrics
- `nr_ssRsrp`, `nr_ssRsrq`, `nr_ssSinr`: 5G NR signal metrics

**Categorical Features**:
- `nrStatus`: Connection status
- `mobility_mode`: Type of mobility (e.g., driving)
- `trajectory_direction`: Direction of trajectory

**Target**:
- `Throughput`: Network throughput (Mbps) - the value we're predicting

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

### Advanced Training with Custom Parameters

```bash
python train.py \
    --data_path Lumos5G-v1.0/Lumos5G-v1.0.csv \
    --output_dir outputs \
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

### Command-Line Arguments

**Data Parameters**:
- `--data_path`: Path to the CSV file (default: `Lumos5G-v1.0/Lumos5G-v1.0.csv`)
- `--output_dir`: Directory to save outputs (default: `outputs`)
- `--train_split`: Train/validation split ratio (default: `0.8`)

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

The `TransformerRegressor` model consists of:

1. **Input Embedding Layer**: Projects input features to model dimension
2. **Positional Encoding**: Learnable positional embeddings
3. **Transformer Encoder**: Multi-layer transformer with self-attention
4. **Output Layers**: Fully connected layers for regression

## Outputs

After training, the following files will be saved in the output directory:

- `best_model.pth`: Best model checkpoint with lowest validation loss
- `loss_plot.png`: Training and validation loss curves
- `predictions_plot.png`: Scatter plot of predictions vs actual values

## Model Checkpoint

The saved model checkpoint includes:
- Model state dictionary
- Optimizer state dictionary
- Validation metrics (loss, MAE, RMSE)
- Scaler and label encoders for preprocessing
- Model configuration

## Loading a Trained Model

```python
import torch
from train import TransformerRegressor

# Load checkpoint (weights_only=False is required for PyTorch 2.6+
# because we save preprocessing objects like scaler and label encoders)
checkpoint = torch.load('outputs/best_model.pth', weights_only=False)

# Recreate model
model = TransformerRegressor(
    input_dim=checkpoint['input_dim'],
    **checkpoint['model_config']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Access preprocessing tools
scaler = checkpoint['scaler']
label_encoders = checkpoint['label_encoders']
```

**Note**: The `weights_only=False` parameter is required when loading the checkpoint because it contains non-tensor objects (StandardScaler and LabelEncoders). This is safe since you're loading your own trained model.

## Performance Metrics

The model is evaluated using:
- **MSE (Mean Squared Error)**: Loss function
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Squared Error)**: Standard deviation of prediction errors

## Tips for Better Performance

1. **Increase model capacity**: Use larger `d_model` and `dim_feedforward`
2. **Add more layers**: Increase `num_layers` for deeper models
3. **Tune batch size**: Larger batches can stabilize training
4. **Adjust learning rate**: Use learning rate finder to optimize
5. **Data augmentation**: Consider adding temporal or spatial features
6. **Ensemble models**: Train multiple models and average predictions

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## License

This project uses the Lumos5G-v1.0 dataset. Please refer to the dataset's LICENSE for usage terms.

