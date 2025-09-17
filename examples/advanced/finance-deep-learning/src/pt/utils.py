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

import fcntl
import os
import traceback
import time
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nvflare.apis.dxo import DXO, DataKind
from nvflare.apis.dxo_filter import DXOFilter
from nvflare.apis.fl_constant import FLMetaKey


def load_csv_data(data_path, feature_columns=None, label_column=None, test_size=0.2, random_state=42):
    """
    Load CSV data for financial analysis and return train/test splits.
    Can load from a single CSV file or concatenate multiple CSV files from a directory.

    Args:
        data_path (str): Path to a CSV file or directory containing CSV files
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
        data_path = Path(data_path)
        
        # Check if path is a file or directory
        if data_path.is_file():
            # Single file case
            csv_files = [data_path]
        elif data_path.is_dir():
            # Directory case - find all CSV files
            csv_files = list(data_path.glob("*.csv"))
            if not csv_files:
                raise ValueError(f"No CSV files found in directory: {data_path}")
        else:
            raise FileNotFoundError(f"Path not found: {data_path}")

        print(f"Found {len(csv_files)} CSV file(s) to process")

        # Load and concatenate all CSV files
        dataframes = []
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                print(f"Loaded {len(df)} rows from {csv_file.name}")
                dataframes.append(df)
            except Exception as e:
                print(f"Warning: Could not load {csv_file.name}: {e}")
                continue

        if not dataframes:
            raise ValueError("No valid CSV files could be loaded")

        # Concatenate all dataframes
        data = pd.concat(dataframes, ignore_index=True)
        print(f"Concatenated data: {len(data)} total rows")

        # Determine feature and label columns from the first dataframe
        if feature_columns is None:
            feature_columns = data.columns[:-1].tolist()
        if label_column is None:
            label_column = data.columns[-1]

        # Validate that the specified columns exist in all dataframes
        missing_features = [col for col in feature_columns if col not in data.columns]
        if missing_features:
            raise ValueError(f"Feature columns not found in CSV data: {missing_features}")

        if label_column not in data.columns:
            raise ValueError(f"Label column '{label_column}' not found in CSV data")

        # Check that all CSV files have the same columns
        for i, df in enumerate(dataframes):
            df_missing_features = [col for col in feature_columns if col not in df.columns]
            if df_missing_features:
                raise ValueError(f"CSV file {csv_files[i].name} missing feature columns: {df_missing_features}")
            if label_column not in df.columns:
                raise ValueError(f"CSV file {csv_files[i].name} missing label column: {label_column}")

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

        print(f"Loaded CSV data from {data_path}")
        print(f"Loaded CSV data: {len(train_features)} training samples, {len(test_features)} test samples")
        print(f"Feature shape: {train_features.shape[1]}, Number of classes: {len(np.unique(y))}")

        return (train_features, train_labels), (test_features, test_labels)

    except FileNotFoundError:
        print(f"Data path {data_path} not found. Creating sample financial data...")
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


def plot_shap_summary(shap_metrics, plot_prefix="", save_fig=False):
    """
    Plot SHAP summary plot from pre-computed metrics.

    Args:
        shap_metrics: Dictionary containing SHAP metrics from compute_shapley_values
        plot_prefix: Prefix for saved plot files
    """
    try:
        shap_values = shap_metrics["shap_values"]
        sample_features = shap_metrics["shap_sample_features"]
        feature_names = shap_metrics["shap_feature_names"]

        shap_values_for_plot = shap_values
        # Check if we need to handle the shape differently
        if len(shap_values_for_plot.shape) == 3:
            # If 3D array (samples, features, classes), take mean across classes
            shap_values_for_plot = np.mean(shap_values_for_plot, axis=2)

        plt.figure(figsize=(20, 16))
        shap.plots.violin(
            shap_values_for_plot, feature_names=feature_names, show=False, max_display=sample_features.shape[1]
        )
        if save_fig:
            save_name = f"{plot_prefix}_shap_summary_plot.png"
            plt.tight_layout()
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"SHAP summary plot saved successfully to {save_name}")
    except Exception as e:
        traceback.print_exc()
        print(f"Error plotting SHAP summary: {e}")


def plot_shap_feature_importance(shap_metrics, plot_prefix="", save_fig=False):
    """
    Plot SHAP feature importance bar chart from pre-computed metrics.

    Args:
        shap_metrics: Dictionary containing SHAP metrics from compute_shapley_values
        plot_prefix: Prefix for saved plot files
    """
    try:
        shap_values = shap_metrics["shap_values"]
        feature_names = shap_metrics["shap_feature_names"]

        plt.figure(figsize=(20, 12))

        # Handle case where shap_values is a list (multiple outputs)
        shap_values_for_importance = shap_values

        # Check if we need to handle the shape differently
        if len(shap_values_for_importance.shape) == 3:
            # If 3D array (samples, features, classes), take mean across classes
            shap_values_for_importance = np.mean(shap_values_for_importance, axis=2)

        feature_importance = np.mean(np.abs(shap_values_for_importance), axis=0)

        # Use the same feature names for the bar plot
        plt.barh(feature_names, feature_importance)
        plt.xlabel("Mean |SHAP value|")
        plt.title("Feature Importance (SHAP)")
        if save_fig:
            save_name = f"{plot_prefix}_shap_feature_importance.png"
            plt.tight_layout()
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"SHAP feature importance plot saved successfully to {save_name}")
    except Exception as e:
        traceback.print_exc()
        print(f"Error plotting SHAP feature importance: {e}")


def plot_all_shap_plots(shap_metrics, plot_prefix="", save_fig=False):
    """
    Generate all SHAP plots from pre-computed metrics.

    Args:
        shap_metrics: Dictionary containing SHAP metrics from compute_shapley_values
        plot_prefix: Prefix for saved plot files
        save_fig: Whether to save the plots
    """
    plot_shap_summary(shap_metrics, plot_prefix, save_fig)
    plot_shap_feature_importance(shap_metrics, plot_prefix, save_fig)


class PyTorchModelWrapper:
    """
    Wrapper class to make PyTorch models compatible with SHAP.
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def __call__(self, x):
        """
        Forward pass for SHAP compatibility.
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            # Convert to numpy for SHAP
            return outputs.cpu().numpy()


def compute_shapley_values(model, test_features, test_labels, n_samples=100, plot_prefix="", feature_names=None):
    """
    Compute Shapley values for feature importance using SHAP library.

    Args:
        model: Trained PyTorch model
        test_features: Test feature data
        test_labels: Test label data
        n_samples: Number of samples to use for SHAP computation (for performance)
        plot_prefix: Prefix for saved plot files
        feature_names: List of feature names/column names to display in plots

    Returns:
        dict: Dictionary containing SHAP metrics
    """
    try:
        # Get device from model
        device = next(model.parameters()).device
        
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

        # Create wrapper for PyTorch model
        model_wrapper = PyTorchModelWrapper(model, device)

        # Create SHAP explainer for the model
        explainer = shap.DeepExplainer(model_wrapper, background_data)

        # Compute SHAP values
        shap_values = explainer.shap_values(sample_features)

        # Create feature names for all features
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(sample_features.shape[1])]
        elif len(feature_names) != sample_features.shape[1]:
            print(
                f"Warning: feature_names length ({len(feature_names)}) doesn't match number of features ({sample_features.shape[1]})"
            )
            feature_names = [f"Feature_{i}" for i in range(sample_features.shape[1])]

        print(f"Using feature names: {feature_names}")

        # For multi-output models, we need to specify which output to plot
        print(f"Plotting single SHAP values, shape: {shap_values.shape}")
        print(f"Sample features shape for plotting: {sample_features.shape}")
        print(f"Number of features in sample_features: {sample_features.shape[1]}")
        print(f"Background data shape: {background_data.shape}")

        # Generate plots using the factored-out plotting functions
        plot_all_shap_plots(
            {"shap_values": shap_values, "shap_sample_features": sample_features, "shap_feature_names": feature_names},
            plot_prefix,
            save_fig=True,
        )

        # Compute feature importance metrics for the return value
        shap_values_for_importance = shap_values
        if len(shap_values_for_importance.shape) == 3:
            # If 3D array (samples, features, classes), take mean across classes
            shap_values_for_importance = np.mean(shap_values_for_importance, axis=2)

        feature_importance = np.mean(np.abs(shap_values_for_importance), axis=0)
        total_importance = np.sum(feature_importance)

        # Create metrics dictionary
        shap_metrics = {
            "shap_values": shap_values,
            "shap_sample_features": sample_features,
            "shap_feature_names": feature_names,
            "shap_feature_importance": feature_importance,
            "shap_total_importance": float(total_importance),
            "shap_samples_used": len(sample_features),
        }

        # print(f"SHAP metrics: {shap_metrics}")

        # Save the SHAP values to a file using numpy
        np.save(f"{plot_prefix}_shap_metrics.npy", shap_metrics)

        return shap_metrics

    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        # Return default metrics if SHAP computation fails
        return {}


class MLflowCallback:
    """
    Custom PyTorch callback for logging training metrics to MLflow.

    This callback logs training and validation metrics at the end of each epoch
    to the provided MLflow writer.
    """

    def __init__(self, mlflow_writer):
        self.mlflow_writer = mlflow_writer
        self.global_epoch = 0

    def log_metrics(self, train_loss, train_accuracy, val_accuracy=None):
        """
        Log training metrics to MLflow.
        
        Args:
            train_loss: Training loss
            train_accuracy: Training accuracy
            val_accuracy: Validation accuracy (optional)
        """
        print(f"Logging training metrics for epoch {self.global_epoch}")
        
        # Log training metrics
        self.mlflow_writer.log_metric("train_loss", train_loss, self.global_epoch)
        self.mlflow_writer.log_metric("train_accuracy", train_accuracy, self.global_epoch)
        
        # Log validation metrics if available
        if val_accuracy is not None:
            self.mlflow_writer.log_metric("val_accuracy", val_accuracy, self.global_epoch)

        self.global_epoch += 1


class ShapCollectionFilter(DXOFilter):
    def __init__(self):
        super().__init__(supported_data_kinds=[DataKind.WEIGHT_DIFF, DataKind.WEIGHTS], data_kinds_to_filter=None)

        # Global dictionary to store shape metrics for each round
        self.all_shap_metrics = {}
        self._save_path = None

    def _safe_save_with_lock(self, data, file_path, max_retries=5, retry_delay=0.1):
        """
        Safely save data to file with file locking to prevent concurrent access.

        Args:
            data: Data to save
            file_path: Path to save the file
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds

        Returns:
            bool: True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Open file for writing with exclusive lock
                with open(file_path, "wb") as f:
                    # Try to acquire exclusive lock (non-blocking)
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

                    # Save the data
                    np.save(f, data)

                    # Lock is automatically released when file is closed

                return True

            except (OSError, IOError) as e:
                if attempt < max_retries - 1:
                    self.log_warning(
                        None,
                        f"Failed to acquire lock for {file_path}, attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay}s...",
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.log_error(None, f"Failed to save {file_path} after {max_retries} attempts: {e}")
                    return False
            except Exception as e:
                self.log_error(None, f"Unexpected error saving {file_path}: {e}")
                return False

        return False

    def process_dxo(self, dxo, shareable, fl_ctx) -> Union[None, "DXO"]:
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
            if self._save_path is None:
                workspace = fl_ctx.get_engine().get_workspace()
                app_root = workspace.get_app_dir(fl_ctx.get_job_id())
                self._save_path = os.path.join(app_root, "shap_values.npy")

            # get shap metrics from dxo
            shap_metrics = dxo.meta["initial_metrics"]["shap_metrics"]
            self.log_info(fl_ctx, f"SHAP metrics {shap_metrics.keys()}")

            current_round = fl_ctx.get_prop(FLMetaKey.CURRENT_ROUND)
            peer_context = fl_ctx.get_peer_context()
            client_name = peer_context.get_identity_name()

            if f"round{current_round}" not in self.all_shap_metrics:
                self.all_shap_metrics[f"round{current_round}"] = {}
            self.all_shap_metrics[f"round{current_round}"][client_name] = shap_metrics

            # Dump global dictionary to file with file locking
            success = self._safe_save_with_lock(self.all_shap_metrics, self._save_path)

            if success:
                self.log_info(
                    fl_ctx,
                    f"Saved SHAP metrics for round {current_round} and client {client_name} at {self._save_path}",
                )
            else:
                self.log_error(
                    fl_ctx,
                    f"Failed to save SHAP metrics for round {current_round} and client {client_name} at {self._save_path}",
                )
        except Exception as e:
            self.log_error(fl_ctx, f"Error processing DXO in ShapCollectionFilter: {e}")

        # Return the DXO unchanged
        return dxo


def load_shap_metrics(file_path):
    """
    Load SHAP metrics from a saved .npy file.

    Args:
        file_path: Path to the saved SHAP metrics file

    Returns:
        dict: Loaded SHAP metrics
    """
    try:
        return np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading SHAP metrics from {file_path}: {e}")
        return None
