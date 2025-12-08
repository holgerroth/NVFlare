#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Federated Learning client for Lumos5G Time Series Prediction
"""

import os
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

import nvflare.client as flare
from nvflare.client.tracking import SummaryWriter

from model import TransformerTimeSeriesRegressor
from data import Lumos5GTimeSeriesDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate additional metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    return total_loss / len(dataloader), mae, rmse


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Federated Learning Client for Lumos5G')
    parser.add_argument('--data_dir', type=str, default='federated_data',
                       help='Directory containing client data files (default: federated_data)')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Number of past timesteps to use for prediction (default: 10)')
    parser.add_argument('--prediction_horizon', type=int, default=1,
                       help='Number of timesteps ahead to predict (default: 1)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for training (default: 256)')
    parser.add_argument('--epochs_per_round', type=int, default=2,
                       help='Number of epochs to train per federated round (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (default: 0.01)')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers (default: 4)')
    args = parser.parse_args()
    
    # Hyperparameters from arguments
    sequence_length = args.sequence_length
    prediction_horizon = args.prediction_horizon
    batch_size = args.batch_size
    epochs_per_round = args.epochs_per_round
    lr = args.lr
    weight_decay = args.weight_decay
    num_workers = args.num_workers
    data_dir = args.data_dir
    
    # Model architecture parameters (fixed for consistency across clients)
    d_model = 128
    nhead = 8
    num_layers = 3
    dim_feedforward = 512
    dropout = 0.1
    
    # Initialize NVFlare
    flare.init()
    sys_info = flare.system_info()
    client_name = sys_info["site_name"]
    
    print(f"=" * 70)
    print(f"Federated Learning Client: {client_name}")
    print(f"=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Construct data path based on client name
    # Assuming client names are like "site-1", "site-2", etc.
    data_path = os.path.join(data_dir, f"{client_name}.csv")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please run split_train_federated.py to generate client data splits.\n"
            f"Example: python split_train_federated.py --data_path train.csv --num_clients N\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Looking for data in: {os.path.abspath(data_dir)}"
        )
    
    print(f"Loading client data from: {data_path}")
    
    # Load shared preprocessors (scaler and label encoders)
    # These must be fitted on the full dataset to ensure consistent feature dimensions
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    encoders_path = os.path.join(data_dir, 'label_encoders.pkl')
    
    if not os.path.exists(scaler_path) or not os.path.exists(encoders_path):
        raise FileNotFoundError(
            f"Preprocessor files not found!\n"
            f"  Expected: {scaler_path}\n"
            f"  Expected: {encoders_path}\n\n"
            f"Please run create_schema_based_preprocessors.py first:\n"
            f"  python create_schema_based_preprocessors.py --output_dir {data_dir}\n\n"
            f"This creates preprocessors using domain knowledge (no training data needed)."
        )
    
    print(f"Loading shared preprocessors...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)
    print(f"  Loaded scaler and label encoders")
    
    # Load dataset with shared preprocessors (fit_transform=False)
    dataset = Lumos5GTimeSeriesDataset(
        data_path,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        scaler=scaler,
        label_encoders=label_encoders,
        fit_transform=False
    )
    
    input_dim = dataset.get_feature_dim()
    print(f"Dataset loaded: {len(dataset)} sequences")
    print(f"Input dimension: {input_dim}, Sequence length: {sequence_length}")
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Model will be created based on received global model architecture
    model = None
    criterion = nn.MSELoss()
    optimizer = None
    
    # Initialize summary writer for tracking
    summary_writer = SummaryWriter()
    
    print("\nStarting federated learning...")
    print("=" * 70)
    
    # Federated learning loop
    while flare.is_running():
        # Receive global model from server
        input_model = flare.receive()
        current_round = input_model.current_round
        
        print(f"\n{'=' * 70}")
        print(f"Round {current_round}: Training on {client_name}")
        print(f"{'=' * 70}")
        
        # On first round, create model matching the global model architecture
        if model is None:
            # Create model with same architecture as global model
            model = TransformerTimeSeriesRegressor(
                input_dim=input_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            
            print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
            print(f"Input dimension: {input_dim}")
            
            # Create optimizer
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Load global model parameters
        model.load_state_dict(input_model.params)
        model.to(device)
        
        # Train for specified number of epochs
        steps = epochs_per_round * len(train_loader)
        for epoch in range(epochs_per_round):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Evaluate on local data after each epoch
            eval_loss, eval_mae, eval_rmse = evaluate(model, train_loader, criterion, device)
            
            print(f"  Epoch {epoch + 1}/{epochs_per_round}:")
            print(f"    Train Loss: {train_loss:.4f}")
            print(f"    Eval Loss: {eval_loss:.4f}, MAE: {eval_mae:.4f}, RMSE: {eval_rmse:.4f}")
            
            # Log metrics at the end of each epoch
            global_step = current_round * steps + (epoch + 1) * len(train_loader)
            summary_writer.add_scalar(tag="train_loss", scalar=train_loss, global_step=global_step)
            summary_writer.add_scalar(tag="eval_loss", scalar=eval_loss, global_step=global_step)
            summary_writer.add_scalar(tag="eval_mae", scalar=eval_mae, global_step=global_step)
            summary_writer.add_scalar(tag="eval_rmse", scalar=eval_rmse, global_step=global_step)
        
        # Final evaluation results for the round
        print(f"\n  Round {current_round} Training Complete:")
        print(f"    Final Loss: {eval_loss:.4f}, MAE: {eval_mae:.4f}, RMSE: {eval_rmse:.4f}")
        
        print(f"\n  Finished training for round {current_round}")
        
        # Save local model checkpoint
        checkpoint_dir = f"outputs/{client_name}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_round_{current_round}.pth")
        torch.save({
            'round': current_round,
            'model_state_dict': model.state_dict(),
            'eval_loss': eval_loss,
            'eval_mae': eval_mae,
            'eval_rmse': eval_rmse,
            'scaler': dataset.get_scaler(),
            'label_encoders': dataset.get_label_encoders(),
            'input_dim': model.input_dim,
            'sequence_length': sequence_length,
            'prediction_horizon': prediction_horizon,
        }, checkpoint_path)
        
        # Prepare output model to send back to server
        output_model = flare.FLModel(
            params=model.cpu().state_dict(),
            meta={
                "NUM_STEPS_CURRENT_ROUND": steps,
                "num_samples": len(dataset),
            },
            metrics={
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                "eval_mae": eval_mae,
                "eval_rmse": eval_rmse,
            },
        )
        
        print(f"  Sending model to server...")
        flare.send(output_model)
        print(f"  Model sent successfully!\n")
    
    print("=" * 70)
    print(f"Federated learning completed for {client_name}")
    print("=" * 70)


if __name__ == "__main__":
    main()

