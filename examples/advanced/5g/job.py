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
Federated Learning Job for Lumos5G Time Series Prediction

This script sets up a federated learning job using FedAvg algorithm
for training a Transformer model on distributed 5G network data.
"""
import argparse
import os

from model import TransformerTimeSeriesRegressor

from nvflare.app_opt.pt.recipes.fedavg import FedAvgRecipe
from nvflare.recipe import SimEnv, add_experiment_tracking


def define_parser():
    parser = argparse.ArgumentParser(
        description='Federated Learning Job for Lumos5G Time Series Prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with 4 clients for 10 rounds
  python job.py --n_clients 4 --num_rounds 10
  
  # Run with custom model architecture
  python job.py --n_clients 3 --num_rounds 20 --d_model 256 --num_layers 4
  
  # Use custom data directory
  python job.py --n_clients 5 --num_rounds 15 --data_dir /path/to/federated_data

Prerequisites:
  1. Split your training data using split_train_federated.py:
     python split_train_federated.py --data_path train.csv --num_clients N
     
  2. Ensure federated_data/ directory contains site-1.csv, site-2.csv, etc.
  
  3. Make sure model.py and data.py are in the same directory
        """
    )

    # Output directory
    parser.add_argument('--output_dir', type=str, default='federated_training',
                       help='Directory to save outputs (default: federated_training)')

    # Federated learning parameters
    parser.add_argument('--n_clients', type=int, default=4,
                       help='Number of federated clients/sites (default: 4)')
    parser.add_argument('--num_rounds', type=int, default=10,
                       help='Number of federated learning rounds (default: 10)')
    parser.add_argument('--epochs_per_round', type=int, default=2,
                       help='Number of epochs to train per federated round (default: 2)')
    
    # Data configuration
    parser.add_argument('--data_dir', type=str, default='federated_data',
                       help='Directory containing client data files (default: federated_data)')
    
    # Model architecture parameters
    parser.add_argument('--input_dim', type=int, default=45,
                       help='Input feature dimension (default: 45, auto-detected from data)')
    parser.add_argument('--d_model', type=int, default=128,
                       help='Dimension of the transformer model (default: 128)')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of transformer encoder layers (default: 3)')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                       help='Dimension of feedforward network (default: 512)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    
    # Job configuration
    parser.add_argument('--job_name', type=str, default='lumos5g-fedavg',
                       help='Name for the federated learning job (default: lumos5g-fedavg)')
    
    return parser.parse_args()


def main():
    args = define_parser()
    
    n_clients = args.n_clients
    num_rounds = args.num_rounds
    
    # Get absolute path to data directory
    data_dir = os.path.abspath(args.data_dir)
    
    # Verify data directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print(f"\nPlease run split_train_federated.py first to create the data splits:")
        print(f"  python split_train_federated.py --data_path train.csv --num_clients {n_clients}")
        return 1
    
    # Check for preprocessor files (schema-based)
    scaler_path = os.path.join(data_dir, 'scaler.pkl')
    encoders_path = os.path.join(data_dir, 'label_encoders.pkl')
    config_path = os.path.join(data_dir, 'feature_config.txt')
    
    if not os.path.exists(scaler_path) or not os.path.exists(encoders_path):
        print(f"ERROR: Preprocessor files not found in {data_dir}")
        print(f"\nPlease run create_schema_based_preprocessors.py first:")
        print(f"  python create_schema_based_preprocessors.py --output_dir {args.data_dir}")
        print(f"\nThis creates preprocessors using domain knowledge (no training data needed).")
        return 1
    
    # Read input dimension from config if available
    detected_input_dim = args.input_dim
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            first_line = f.readline()
            if 'Input Dimension:' in first_line:
                detected_input_dim = int(first_line.split(':')[1].strip())
                print(f"Detected input dimension from feature_config.txt: {detected_input_dim}")
    
    # Use detected dimension if not manually specified
    if args.input_dim == 45 and detected_input_dim != 45:
        print(f"Using auto-detected input dimension: {detected_input_dim}")
        args.input_dim = detected_input_dim
    
    # Verify all client data files exist
    missing_files = []
    for i in range(1, n_clients + 1):
        client_data_file = os.path.join(data_dir, f"site-{i}.csv")
        if not os.path.exists(client_data_file):
            missing_files.append(client_data_file)
    
    if missing_files:
        print(f"ERROR: Missing client data files:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\nPlease run split_train_federated.py to create data splits for {n_clients} clients:")
        print(f"  python split_train_federated.py --data_path train.csv --num_clients {n_clients} --output_dir {args.data_dir}")
        return 1
    
    print("=" * 70)
    print("Lumos5G Federated Learning Job Configuration")
    print("=" * 70)
    print(f"Job Name: {args.job_name}")
    print(f"Number of Clients: {n_clients}")
    print(f"Number of Rounds: {num_rounds}")
    print(f"Data Directory: {data_dir}")
    print(f"\nModel Architecture:")
    print(f"  - Input Dimension: {args.input_dim}")
    print(f"  - Model Dimension (d_model): {args.d_model}")
    print(f"  - Attention Heads: {args.nhead}")
    print(f"  - Transformer Layers: {args.num_layers}")
    print(f"  - Feedforward Dimension: {args.dim_feedforward}")
    print(f"  - Dropout: {args.dropout}")
    print("=" * 70)
    
    # Create initial model with specified architecture
    initial_model = TransformerTimeSeriesRegressor(
        input_dim=args.input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    
    num_params = sum(p.numel() for p in initial_model.parameters())
    print(f"\nInitial model created with {num_params:,} parameters")
    
    # Create train_args to pass data directory to clients
    train_args = f"--data_dir {data_dir} --epochs_per_round {args.epochs_per_round}"
    
    # Create FedAvg recipe
    recipe = FedAvgRecipe(
        name=args.job_name,
        min_clients=n_clients,
        num_rounds=num_rounds,
        initial_model=initial_model,
        train_script="client.py",
        train_args=train_args,
    )
    
    # Add experiment tracking with TensorBoard
    add_experiment_tracking(recipe, tracking_type="tensorboard")
    
    print("\nSetting up simulation environment...")
    print(f"Clients will be named: site-1, site-2, ..., site-{n_clients}")
    print(f"Each client will load data from: {data_dir}/site-<N>.csv")
    
    # Create simulation environment
    env = SimEnv(num_clients=n_clients, workspace_root=args.output_dir)
    
    print("\n" + "=" * 70)
    print("Starting Federated Learning Job")
    print("=" * 70)
    
    # Execute the federated learning job
    run = recipe.execute(env)
    
    # Print results
    print("\n" + "=" * 70)
    print("Federated Learning Job Complete!")
    print("=" * 70)
    print(f"Job Status: {run.get_status()}")
    print(f"Results saved to: {run.get_result()}")
    print("\nNext Steps:")
    print("  1. View training logs in the result directory")
    print("  2. Visualize metrics with TensorBoard:")
    print(f"     tensorboard --logdir {run.get_result()}")
    print("  3. Check individual client models in outputs/site-<N>/")
    print("  4. Evaluate the final aggregated model on test data")
    print("=" * 70)


if __name__ == "__main__":
    main()

