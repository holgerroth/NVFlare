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
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from nvflare.app_common.utils.fl_model_utils import FLModelUtils
from typing import Union
from tensorflow.keras.callbacks import Callback
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.dxo import DXO, DataKind, MetaKey, from_shareable
from nvflare.apis.fl_constant import FLContextKey, FLMetaKey
import os


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
    n_features = 5
    
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


def compute_shapley_values(model, test_features, test_labels, n_samples=100, plot_prefix="", feature_names=None):
    """
    Compute Shapley values for feature importance using SHAP library.
    
    Args:
        model: Trained TensorFlow model
        test_features: Test feature data
        test_labels: Test label data
        n_samples: Number of samples to use for SHAP computation (for performance)
        plot_prefix: Prefix for saved plot files
        feature_names: List of feature names/column names to display in plots
    
    Returns:
        dict: Dictionary containing SHAP metrics
    """
    try:
        # Sample a subset of test data for SHAP computation (for performance)
        if len(test_features) > n_samples:
            indices = np.random.choice(len(test_features), n_samples, replace=False)
            sample_features = test_features[indices]
            sample_labels = test_labels[indices]
        else:
            sample_features = test_features
            sample_labels = test_labels
        
        # Create a background dataset for SHAP (using a subset of the data)
        background_size = min(50, len(sample_features))
        background_indices = np.random.choice(len(sample_features), background_size, replace=False)
        background_data = sample_features[background_indices]
        
        # Create SHAP explainer for the model
        explainer = shap.DeepExplainer(model, background_data)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(sample_features)

        # Plot the SHAP values and save to file
        print("Starting SHAP computation...")
        plt.figure(figsize=(20, 16))
        
        # Create feature names for all features
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(sample_features.shape[1])]
        elif len(feature_names) != sample_features.shape[1]:
            print(f"Warning: feature_names length ({len(feature_names)}) doesn't match number of features ({sample_features.shape[1]})")
            feature_names = [f'Feature_{i}' for i in range(sample_features.shape[1])]
        
        print(f"Using feature names: {feature_names}")
        
        # For multi-output models, we need to specify which output to plot
        print(f"Plotting single SHAP values, shape: {shap_values.shape}")
        print(f"Sample features shape for plotting: {sample_features.shape}")
        print(f"Number of features in sample_features: {sample_features.shape[1]}")
        print(f"Background data shape: {background_data.shape}")
        # Set max_display to show all features and force it
        shap.summary_plot(shap_values, sample_features, feature_names=feature_names, show=False, max_display=sample_features.shape[1])
        plt.tight_layout()
        plt.savefig(f'{plot_prefix}_shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save a bar plot of feature importance
        plt.figure(figsize=(20, 12))
        # Handle case where shap_values is a list (multiple outputs)
        shap_values_for_importance = shap_values
        print(f"Using single SHAP values for importance, shape: {shap_values_for_importance.shape}")
    
        # Check if we need to handle the shape differently
        if len(shap_values_for_importance.shape) == 3:
            # If 3D array (samples, features, classes), take mean across classes
            shap_values_for_importance = np.mean(shap_values_for_importance, axis=2)
        
        feature_importance = np.mean(np.abs(shap_values_for_importance), axis=0)
        # Use the same feature names for the bar plot
        plt.barh(feature_names, feature_importance)
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()
        plt.savefig(f'{plot_prefix}_shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("SHAP plots saved successfully")
       
        # Compute feature importance metrics
        total_importance = np.sum(feature_importance)
        
        # Create metrics dictionary
        shap_metrics = {
            "shap_values": shap_values, 
            "shap_sample_features": sample_features, 
            "shap_feature_names": feature_names,
            "shap_feature_importance": feature_importance,
            "shap_total_importance": float(total_importance),
            "shap_samples_used": len(sample_features)
        }

        #print(f"SHAP metrics: {shap_metrics}")

        # Save the SHAP values to a file using numpy
        np.save(f'{plot_prefix}_shap_metrics.npy', shap_metrics)
        
        return shap_metrics
        
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        # Return default metrics if SHAP computation fails
        return {}


class MLflowCallback(Callback):
    """
    Custom TensorFlow callback for logging training metrics to MLflow.
    
    This callback logs training and validation metrics at the end of each epoch
    to the provided MLflow writer.
    """
    def __init__(self, mlflow_writer):
        super().__init__()
        self.mlflow_writer = mlflow_writer
        self.gobal_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            print(f"Logging training metrics for epoch {self.gobal_epoch}")
            # Log training metrics
            if 'loss' in logs:
                self.mlflow_writer.log_metric("train_loss", logs.get('loss', 0), self.gobal_epoch)
            if 'accuracy' in logs:
                self.mlflow_writer.log_metric("train_accuracy", logs.get('accuracy', 0), self.gobal_epoch)
            
            # Log validation metrics if available
            if 'val_loss' in logs:
                self.mlflow_writer.log_metric("val_loss", logs.get('val_loss', 0), self.gobal_epoch)
            if 'val_accuracy' in logs:
                self.mlflow_writer.log_metric("val_accuracy", logs.get('val_accuracy', 0), self.gobal_epoch)
            
            # Log additional metrics if they exist
            for key, value in logs.items():
                if key not in ['loss', 'accuracy', 'val_loss', 'val_accuracy']:
                    self.mlflow_writer.log_metric(f"train_{key}", value, self.gobal_epoch)

        self.gobal_epoch += 1


class ShapCollectionFilter(DXOFilter):
    def __init__(self):
        super().__init__(supported_data_kinds=[DataKind.WEIGHT_DIFF, DataKind.WEIGHTS], data_kinds_to_filter=None)
        
        # Global dictionary to store shape metrics for each round
        self.all_shap_metrics = {}

    def process_dxo(self, dxo, shareable, fl_ctx) -> Union[None, 'DXO']:
        """
        Process DXO objects, extract FLModels, store them globally, and dump to JSON.
        
        Args:
            dxo: The DXO object received
            shareable: The shareable object
            fl_ctx: The FL context
            
        Returns:
            The processed DXO object
        """
        try:
            # get shap metrics from dxo
            shap_metrics = dxo.meta['initial_metrics']['shap_metrics']
            self.log_info(fl_ctx, f"SHAP metrics {shap_metrics.keys()}")

            current_round = fl_ctx.get_prop(FLMetaKey.CURRENT_ROUND)
            peer_context = fl_ctx.get_peer_context()
            client_name = peer_context.get_identity_name()

            if f"round{current_round}" not in self.all_shap_metrics:
                self.all_shap_metrics[f"round{current_round}"] = {}
            self.all_shap_metrics[f"round{current_round}"][client_name] = shap_metrics
                
            # Dump global dictionary to JSON file
            np.save('shap_values.npy', self.all_shap_metrics)

            self.log_info(fl_ctx, f"Saved SHAP metrics for round {current_round} and client {client_name} at 'shape_values.npy'")
        except Exception as e:
            self.log_error(fl_ctx, f"Error processing DXO in ShapCollectionFilter: {e}")
            
        # Return the DXO unchanged
        return dxo
