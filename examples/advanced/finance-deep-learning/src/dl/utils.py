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
        print(f"Sample features shape: {sample_features.shape}")
        print(f"Background data shape: {background_data.shape}")
        shap_values = explainer.shap_values(sample_features)
        print(f"SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"SHAP values list length: {len(shap_values)}")
            for i, sv in enumerate(shap_values):
                print(f"SHAP values[{i}] shape: {sv.shape}")
        else:
            print(f"SHAP values shape: {shap_values.shape}")

        # Plot the SHAP values and save to file
        print("Starting SHAP plotting...")
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
        # Set max_display to show all features and force it
        shap.summary_plot(shap_values, sample_features, feature_names=feature_names, show=False, max_display=sample_features.shape[1])
        plt.tight_layout()
        plt.savefig(f'{plot_prefix}_shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("SHAP summary plot saved successfully")
        
        # Also save a bar plot of feature importance
        print("Starting feature importance plotting...")
        plt.figure(figsize=(20, 12))
        # Handle case where shap_values is a list (multiple outputs)
        shap_values_for_importance = shap_values
        print(f"Using single SHAP values for importance, shape: {shap_values_for_importance.shape}")
        
        # Debug the shape issue
        print(f"SHAP values for importance shape: {shap_values_for_importance.shape}")
        print(f"Sample features shape: {sample_features.shape}")
        
        # Check if we need to handle the shape differently
        if len(shap_values_for_importance.shape) == 3:
            # If 3D array (samples, features, classes), take mean across classes
            print("3D SHAP array detected, taking mean across classes")
            shap_values_for_importance = np.mean(shap_values_for_importance, axis=2)
            print(f"After taking mean across classes, shape: {shap_values_for_importance.shape}")
        
        feature_importance = np.mean(np.abs(shap_values_for_importance), axis=0)
        print(f"Feature importance shape: {feature_importance.shape}")
        # Use the same feature names for the bar plot
        plt.barh(feature_names, feature_importance)
        plt.xlabel('Mean |SHAP value|')
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()
        plt.savefig(f'{plot_prefix}_shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Feature importance plot saved successfully")
        
        # If model has multiple outputs, take the first one for metrics
        if isinstance(shap_values, list):
            shap_values_for_metrics = shap_values[0]
        else:
            shap_values_for_metrics = shap_values
        
        # Handle 3D SHAP arrays for metrics computation too
        if len(shap_values_for_metrics.shape) == 3:
            print("3D SHAP array detected for metrics, taking mean across classes")
            shap_values_for_metrics = np.mean(shap_values_for_metrics, axis=2)
            print(f"After taking mean across classes for metrics, shape: {shap_values_for_metrics.shape}")
        
        # Compute feature importance metrics
        feature_importance = np.mean(np.abs(shap_values_for_metrics), axis=0)
        total_importance = np.sum(feature_importance)
        
        # Normalize feature importance
        normalized_importance = feature_importance / total_importance if total_importance > 0 else feature_importance
        
        # Create metrics dictionary
        shap_metrics = {
            "shap_feature_importance": normalized_importance.tolist(),
            "shap_total_importance": float(total_importance),
            "shap_mean_abs_value": float(np.mean(np.abs(shap_values_for_metrics))),
            "shap_std_value": float(np.std(shap_values_for_metrics)),
            "shap_samples_used": len(sample_features)
        }

        # Save the SHAP values to a file (convert numpy arrays to lists for JSON serialization)
        if isinstance(shap_values, list):
            # If model has multiple outputs, save each output's SHAP values
            shap_values_for_json = [sv.tolist() if hasattr(sv, 'tolist') else sv for sv in shap_values]
        else:
            # Single output model
            shap_values_for_json = shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values
        
        with open(f'{plot_prefix}_shap_values.json', 'w') as f:
            json.dump(shap_values_for_json, f)
        
        # Save the SHAP values to a file
        with open(f'{plot_prefix}_shap_metrics.json', 'w') as f:
            json.dump(shap_metrics, f)
                

        print(f"SHAP metrics: {shap_metrics}")
        
        return shap_metrics
        
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        # Return default metrics if SHAP computation fails
        return {
            "shap_feature_importance": [],
            "shap_total_importance": 0.0,
            "shap_mean_abs_value": 0.0,
            "shap_std_value": 0.0,
            "shap_samples_used": 0,
            "shap_error": str(e)
        }
