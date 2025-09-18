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

import argparse
import os
import traceback
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# (1) import nvflare client API
import nvflare.client as flare


def create_sample_financial_data(test_size=0.2, random_state=42):
    """
    Create sample financial data for demonstration purposes.

    Args:
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: ((train_features, train_labels), (test_features, test_labels))
    """
    np.random.seed(random_state)

    # Create sample financial features (e.g., transaction amount, time, location, etc.)
    n_samples = 1000
    n_features = 7

    # Generate random features
    features = np.random.randn(n_samples, n_features)

    # Create labels (0: normal transaction, 1: fraudulent transaction)
    # Simple rule: if sum of features > threshold, mark as fraudulent
    threshold = 2.0
    labels = (np.sum(features, axis=1) > threshold).astype(int)

    # Split the data
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    # Scale the features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    # Convert to float32 for TensorFlow/PyTorch
    train_features = train_features.astype(np.float32)
    test_features = test_features.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)

    print(f"Created sample financial data: {len(train_features)} training samples, {len(test_features)} test samples")
    print(f"Feature shape: {train_features.shape[1]}, Number of classes: {len(np.unique(labels))}")

    return (train_features, train_labels), (test_features, test_labels)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="NVFlare Deep Learning Client for Synthetic Data Generation")
    parser.add_argument(
        "--out_data_path",
        type=str,
        help="Path to the CSV dataset dir to save the data",
    )
    parser.add_argument("--size", type=int, default=1000, help="Training batch size (default: 32)")
    parser.add_argument(
        "--feature_names",
        type=str,
        nargs="+",
        default=["amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"],
        help="List of feature column names for the CSV file (default: amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest)"
    )
    args = parser.parse_args()

    # (2) initializes NVFlare client API
    flare.init()

    input_model = flare.receive()
    print(f"current_round={input_model.current_round}")

    # (optional) print system info
    system_info = flare.system_info()
    print(f"NVFlare system info: {system_info}")

    client_name = flare.get_site_name()
    
    # Convert client_name to integer seed using hash
    client_seed = hash(client_name) % (2**31)  # Ensure it's within int32 range
    print(f"Client name: {client_name}, Generated seed: {client_seed}")

    (train_features, train_labels), (test_features, test_labels) = create_sample_financial_data(test_size=0.2, random_state=client_seed)

    # Concatenate training and test data into a single DataFrame
    # Create feature column names
    if args.feature_names is not None:
        if len(args.feature_names) != train_features.shape[1]:
            raise ValueError(f"Number of feature names ({len(args.feature_names)}) must match number of features ({train_features.shape[1]})")
        feature_columns = args.feature_names
    else:
        feature_columns = [f'feature_{i}' for i in range(train_features.shape[1])]
    
    # Create DataFrames for training and test data
    train_df = pd.DataFrame(train_features, columns=feature_columns)
    train_df['isFraud'] = train_labels
    train_df['split'] = 'train'
    
    test_df = pd.DataFrame(test_features, columns=feature_columns)
    test_df['isFraud'] = test_labels
    test_df['split'] = 'test'
    # TODO: split column is not used in the training code.
    
    # Concatenate the DataFrames
    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    
    # Save to CSV if out_data_path is provided
    os.makedirs(args.out_data_path, exist_ok=True)
    csv_path = os.path.join(args.out_data_path, 'data.csv')
    combined_df.to_csv(csv_path, index=False)
    print(f"Combined dataset saved to: {csv_path}")
    print(f"Dataset shape: {combined_df.shape}")
    print(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")

    output_model = flare.FLModel(params={"success": torch.tensor(1)})

    # (7) send model back to NVFlare
    flare.send(output_model)

if __name__ == "__main__":
    main()
