#!/usr/bin/env python3
"""
Evaluate a trained model on validation data.

Usage:
    python evaluate_model.py --model_path path/to/model.pth --data_path val.csv --output_file results.json
"""

import torch
import argparse
import json
import os
import numpy as np
from pathlib import Path

from model import TransformerTimeSeriesRegressor
from data import Lumos5GTimeSeriesDataset
from torch.utils.data import DataLoader


def evaluate_model(model_path, data_path, batch_size=256, num_workers=4):
    """
    Evaluate a model on validation data.
    
    Returns:
        dict with metrics: val_loss, val_mae, val_rmse
    """
    print(f"\nEvaluating model: {model_path}")
    print(f"On data: {data_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Check if this is a federated model (NVFlare saves with 'model' and 'train_conf' keys)
    # vs regular checkpoint with 'model_state_dict', 'input_dim', 'scaler', etc.
    is_federated = isinstance(checkpoint, dict) and 'model' in checkpoint and 'train_conf' in checkpoint and 'input_dim' not in checkpoint
    
    if is_federated:
        # Federated model: checkpoint has 'model' and 'train_conf' keys
        # We need to get metadata from external sources
        print(f"  Detected federated global model (NVFlare format)")
        
        scaler = None
        label_encoders = None
        sequence_length = 10
        prediction_horizon = 1
        input_dim = None
        model_config = {}
        
        # Use config_dir from environment variable (set by main())
        config_dir = Path(os.environ.get('FEDERATED_CONFIG_DIR', 'federated_data'))
        print(f"  Config directory: {config_dir.absolute()}")
        
        config_path = config_dir / "feature_config.txt"
        scaler_path = config_dir / "scaler.pkl"
        encoders_path = config_dir / "label_encoders.pkl"
        
        print(f"  Checking config_path exists: {config_path.exists()}")
        
        if config_path.exists():
            print(f"  Reading: {config_path}")
            with open(config_path, 'r') as f:
                first_line = f.readline()
                if 'Input Dimension:' in first_line:
                    input_dim = int(first_line.split(':')[1].strip())
                    print(f"  ✓ Found input_dim={input_dim}")
        
        if scaler_path.exists():
            import pickle
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"  ✓ Loaded scaler")
        
        if encoders_path.exists():
            import pickle
            with open(encoders_path, 'rb') as f:
                label_encoders = pickle.load(f)
            print(f"  ✓ Loaded label encoders")
        
        if input_dim is None:
            print(f"\n  ERROR: feature_config.txt not found or invalid")
            print(f"  Looked in: {config_dir.absolute()}")
            print(f"  Current directory: {Path.cwd()}")
            raise ValueError(
                f"Cannot determine input_dim for federated model.\n"
                f"Please ensure {config_dir}/feature_config.txt exists.\n"
                f"Run: python create_schema_based_preprocessors.py --output_dir {config_dir}"
            )
        
        # Model config with defaults
        model_config = {
            'd_model': 128,
            'nhead': 8,
            'num_layers': 3,
            'dim_feedforward': 512,
            'dropout': 0.1
        }
        
        # Extract the actual state dict from checkpoint['model']
        model_state_dict = checkpoint['model']
    else:
        # Regular model checkpoint
        model_config = checkpoint.get('model_config', {})
        input_dim = checkpoint.get('input_dim', model_config.get('input_dim'))
        scaler = checkpoint.get('scaler')
        label_encoders = checkpoint.get('label_encoders')
        sequence_length = checkpoint.get('sequence_length', 10)
        prediction_horizon = checkpoint.get('prediction_horizon', 1)
        model_state_dict = checkpoint.get('model_state_dict', checkpoint.get('model'))
        
        if input_dim is None:
            raise ValueError("Cannot determine input_dim from checkpoint")
    
    print(f"  Model: input_dim={input_dim}, sequence_length={sequence_length}")
    
    # Create model
    model = TransformerTimeSeriesRegressor(
        input_dim=input_dim,
        d_model=model_config.get('d_model', 128),
        nhead=model_config.get('nhead', 8),
        num_layers=model_config.get('num_layers', 3),
        dim_feedforward=model_config.get('dim_feedforward', 512),
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Load model weights
    model.load_state_dict(model_state_dict)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  Device: {device}")
    
    # Load validation data
    print(f"  Loading validation data...")
    val_dataset = Lumos5GTimeSeriesDataset(
        data_path,
        sequence_length=sequence_length,
        prediction_horizon=checkpoint.get('prediction_horizon', 1),
        scaler=scaler,
        label_encoders=label_encoders,
        fit_transform=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Evaluate
    print(f"  Running evaluation...")
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    val_loss = total_loss / len(val_loader)
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    metrics = {
        'val_loss': float(val_loss),
        'val_mae': float(mae),
        'val_rmse': float(rmse),
        'num_samples': len(val_dataset),
        'model_path': str(model_path),
        'data_path': str(data_path)
    }
    
    print(f"\n  Results:")
    print(f"    Val Loss: {val_loss:.6f}")
    print(f"    Val MAE:  {mae:.6f}")
    print(f"    Val RMSE: {rmse:.6f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model on validation data')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint (.pth or .pt file)')
    parser.add_argument('--data_path', type=str, default='val.csv',
                       help='Path to validation data (default: val.csv)')
    parser.add_argument('--config_dir', type=str, default='federated_data',
                       help='Directory with preprocessors (default: federated_data)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Path to save results JSON (optional)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for evaluation (default: 256)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    
    args = parser.parse_args()
    
    # Set config_dir as environment variable so evaluate_model can access it
    os.environ['FEDERATED_CONFIG_DIR'] = args.config_dir
    
    # Evaluate
    metrics = evaluate_model(
        args.model_path,
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Save results if output file specified
    if args.output_file:
        output_path = Path(args.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n  Saved results to: {args.output_file}")
    
    return 0


if __name__ == '__main__':
    exit(main())

