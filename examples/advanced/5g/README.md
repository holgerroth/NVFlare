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

The `TransformerRegressor` model (defined in `model.py`) consists of:

1. **Input Embedding Layer**: Projects input features to model dimension
2. **Positional Encoding**: Learnable positional embeddings
3. **Transformer Encoder**: Multi-layer transformer with self-attention
4. **Output Layers**: Fully connected layers for regression

## Code Structure

The project is organized into modular components:

- **`model.py`**: Contains the `TransformerRegressor` model architecture
- **`data.py`**: Contains data loading, preprocessing, and the `Lumos5GDataset` class
- **`train.py`**: Training script with full pipeline
- **`inference.py`**: Inference script with metrics and visualization
- **`inference_example.py`**: Example of programmatic inference usage

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
- Model configuration

## Loading a Trained Model

```python
import torch
from model import TransformerRegressor

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

## Programmatic Inference

For more control over the inference process, you can use the model programmatically:

```python
import torch
from model import TransformerRegressor
from data import preprocess_data

# Load checkpoint
checkpoint = torch.load('outputs/best_model.pth', weights_only=False)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerRegressor(
    input_dim=checkpoint['input_dim'],
    **checkpoint['model_config']
).to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load and preprocess data
df = pd.read_csv('your_data.csv')
X = preprocess_data(df, checkpoint['scaler'], checkpoint['label_encoders'])

# Run inference
predictions = run_inference(model, X, device, batch_size=256)

# Add predictions to dataframe
df['Predicted_Throughput'] = predictions
```

See `inference_example.py` for a complete working example.

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

