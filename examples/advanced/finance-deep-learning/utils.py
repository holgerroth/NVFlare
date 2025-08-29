# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_csv_data(file_path, feature_columns=None, label_column=None, test_size=0.2, random_state=42):
    """
    Load CSV data for financial analysis and return train/test splits.
    
    Args:
        file_path (str): Path to the CSV file
        feature_columns (list or None): List of column names to use as features. 
                                       If None, uses all columns except the last one.
        label_column (str or None): Name of the column to use as label.
                                   If None, uses the last column.
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: ((train_features, train_labels), (test_features, test_labels))
    """
    try:
        # Load the CSV data
        data = pd.read_csv(file_path)
        
        # Determine feature and label columns
        if feature_columns is None:
            feature_columns = data.columns[:-1].tolist()
        if label_column is None:
            label_column = data.columns[-1]
        
        # Validate that the specified columns exist
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in CSV: {missing_features}")
        
        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV")
        
        # Extract features and labels
        X = data[feature_columns].values
        y = data[label_column].values
        
        print(f"Using feature columns: {feature_columns}")
        print(f"Using label column: {label_column}")
        
        # Split the data into train and test sets
        train_features, test_features, train_labels, test_labels = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale the features
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)
        
        # Convert to float32 for TensorFlow
        train_features = train_features.astype(np.float32)
        test_features = test_features.astype(np.float32)
        train_labels = train_labels.astype(np.int32)
        test_labels = test_labels.astype(np.int32)
        
        print(f"Loaded CSV data: {len(train_features)} training samples, {len(test_features)} test samples")
        print(f"Feature shape: {train_features.shape[1]}, Number of classes: {len(np.unique(y))}")
        
        return (train_features, train_labels), (test_features, test_labels)
        
    except FileNotFoundError:
        print(f"CSV file {file_path} not found. Creating sample financial data...")
        return create_sample_financial_data(test_size, random_state)
    except Exception as e:
        print(f"Error loading CSV data: {e}. Creating sample financial data...")
        return create_sample_financial_data(test_size, random_state)


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
    n_features = 10
    
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
    
    # Convert to float32 for TensorFlow
    train_features = train_features.astype(np.float32)
    test_features = test_features.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    
    print(f"Created sample financial data: {len(train_features)} training samples, {len(test_features)} test samples")
    print(f"Feature shape: {train_features.shape[1]}, Number of classes: {len(np.unique(labels))}")
    
    return (train_features, train_labels), (test_features, test_labels)
