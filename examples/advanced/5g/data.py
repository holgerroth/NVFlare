"""
Data loading and preprocessing utilities for Lumos5G dataset
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Lumos5GTimeSeriesDataset(Dataset):
    """Time series dataset class for Lumos5G data
    
    Creates sequences of observations to predict future throughput.
    Each sample consists of a sequence of past observations (features) 
    and a target throughput value to predict.
    """
    
    def __init__(self, csv_path, sequence_length=10, prediction_horizon=1, 
                 scaler=None, label_encoders=None, fit_transform=False):
        """
        Args:
            csv_path: Path to the CSV file
            sequence_length: Number of past timesteps to use for prediction
            prediction_horizon: Number of timesteps ahead to predict (1 = next timestep)
            scaler: StandardScaler for numerical features
            label_encoders: Dictionary of LabelEncoders for categorical features
            fit_transform: Whether to fit the scaler and encoders (True for train, False for val/test)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.df = pd.read_csv(csv_path)
        
        # Sort by run_num and seq_num to ensure temporal ordering
        self.df = self.df.sort_values(['run_num', 'seq_num']).reset_index(drop=True)
        
        # Define feature columns (excluding seq_num as it's just an index)
        self.numerical_features = [
            'abstractSignalStr', 'latitude', 'longitude', 
            'movingSpeed', 'compassDirection', 'lte_rssi', 'lte_rsrp', 
            'lte_rsrq', 'lte_rssnr', 'nr_ssRsrp', 'nr_ssRsrq', 'nr_ssSinr',
            'Throughput'  # Include throughput as a feature for the sequence
        ]
        
        self.categorical_features = [
            'nrStatus', 'mobility_mode', 'trajectory_direction'
        ]
        
        self.target = 'Throughput'
        
        # Handle missing values - replace with median for numerical features
        for col in self.numerical_features:
            if col in self.df.columns:
                # Replace sentinel value 2147483647 with NaN
                self.df[col] = self.df[col].replace(2147483647.0, np.nan)
                # Check if column has any valid values
                if self.df[col].notna().sum() > 0:
                    median_val = self.df[col].median()
                    self.df[col] = self.df[col].fillna(median_val)
                else:
                    # If entire column is NaN, fill with 0
                    self.df[col] = 0.0
        
        # Handle missing values in categorical features
        for col in self.categorical_features:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna('UNKNOWN')
        
        # Encode categorical features
        if fit_transform:
            self.label_encoders = {}
            for col in self.categorical_features:
                if col in self.df.columns:
                    le = LabelEncoder()
                    self.df[col] = le.fit_transform(self.df[col].astype(str))
                    self.label_encoders[col] = le
        else:
            self.label_encoders = label_encoders
            for col in self.categorical_features:
                if col in self.df.columns and col in self.label_encoders:
                    # Handle unseen labels
                    le = self.label_encoders[col]
                    self.df[col] = self.df[col].apply(
                        lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
                    )
        
        # Prepare features
        feature_cols = [col for col in self.numerical_features + self.categorical_features 
                       if col in self.df.columns]
        X = self.df[feature_cols].values.astype(np.float32)
        
        # Replace any remaining NaN or inf values with 0
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        if fit_transform:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
            # Handle any NaN values that might appear after scaling
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)
            # Handle any NaN values that might appear after scaling
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create sequences
        self.sequences = []
        self.targets = []
        self.run_nums = self.df['run_num'].values
        
        # Group by run_num to create sequences within each run
        for run_num in self.df['run_num'].unique():
            run_mask = self.run_nums == run_num
            run_indices = np.where(run_mask)[0]
            
            # Create sequences for this run
            for i in range(len(run_indices) - sequence_length - prediction_horizon + 1):
                seq_indices = run_indices[i:i + sequence_length]
                target_idx = run_indices[i + sequence_length + prediction_horizon - 1]
                
                # Sequence of features (excluding throughput from the last position)
                sequence = X[seq_indices]
                # Target is the throughput at the future timestep
                target = self.df.iloc[target_idx][self.target]
                
                self.sequences.append(sequence)
                self.targets.append(target)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        
        print(f"Created {len(self.sequences)} sequences with length {sequence_length}, "
              f"predicting {prediction_horizon} step(s) ahead")
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.tensor(self.targets[idx])
    
    def get_scaler(self):
        return self.scaler
    
    def get_label_encoders(self):
        return self.label_encoders
    
    def get_sequence_length(self):
        return self.sequence_length
    
    def get_feature_dim(self):
        return self.sequences.shape[2] if len(self.sequences) > 0 else 0


def preprocess_timeseries_data(df, scaler, label_encoders, sequence_length):
    """
    Preprocess time series data using the saved scaler and label encoders
    Creates sequences for inference
    
    Args:
        df: Input dataframe (should be sorted by run_num and seq_num)
        scaler: Fitted StandardScaler
        label_encoders: Dictionary of fitted LabelEncoders
        sequence_length: Number of past timesteps in each sequence
    
    Returns:
        Numpy array of sequences, corresponding indices for mapping back to dataframe
    """
    # Sort by run_num and seq_num to ensure temporal ordering
    df = df.sort_values(['run_num', 'seq_num']).reset_index(drop=True)
    
    # Define feature columns (same as in training)
    numerical_features = [
        'abstractSignalStr', 'latitude', 'longitude', 
        'movingSpeed', 'compassDirection', 'lte_rssi', 'lte_rsrp', 
        'lte_rsrq', 'lte_rssnr', 'nr_ssRsrp', 'nr_ssRsrq', 'nr_ssSinr',
        'Throughput'
    ]
    
    categorical_features = [
        'nrStatus', 'mobility_mode', 'trajectory_direction'
    ]
    
    # Handle missing values - replace with median for numerical features
    for col in numerical_features:
        if col in df.columns:
            # Replace sentinel value 2147483647 with NaN
            df[col] = df[col].replace(2147483647.0, np.nan)
            # Check if column has any valid values
            if df[col].notna().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                # If entire column is NaN, fill with 0
                df[col] = 0.0
    
    # Handle missing values in categorical features
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN')
    
    # Encode categorical features
    for col in categorical_features:
        if col in df.columns and col in label_encoders:
            # Handle unseen labels
            le = label_encoders[col]
            df[col] = df[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
            )
    
    # Prepare features
    feature_cols = [col for col in numerical_features + categorical_features 
                   if col in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    
    # Replace any remaining NaN or inf values with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    X = scaler.transform(X)
    
    # Handle any NaN values that might appear after scaling
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create sequences
    sequences = []
    sequence_indices = []  # To track which rows each sequence corresponds to
    run_nums = df['run_num'].values
    
    # Group by run_num to create sequences within each run
    for run_num in df['run_num'].unique():
        run_mask = run_nums == run_num
        run_indices = np.where(run_mask)[0]
        
        # Create sequences for this run
        for i in range(len(run_indices) - sequence_length + 1):
            seq_indices = run_indices[i:i + sequence_length]
            sequence = X[seq_indices]
            
            sequences.append(sequence)
            # The prediction is for the timestep after the last in the sequence
            if i + sequence_length < len(run_indices):
                sequence_indices.append(run_indices[i + sequence_length])
            else:
                sequence_indices.append(run_indices[-1])  # Last available index
    
    return np.array(sequences, dtype=np.float32), np.array(sequence_indices)


def preprocess_data(df, scaler, label_encoders):
    """
    Preprocess data using the saved scaler and label encoders (legacy function for backward compatibility)
    
    Args:
        df: Input dataframe
        scaler: Fitted StandardScaler
        label_encoders: Dictionary of fitted LabelEncoders
    
    Returns:
        Preprocessed numpy array
    """
    # Define feature columns (same as in training)
    numerical_features = [
        'abstractSignalStr', 'latitude', 'longitude', 
        'movingSpeed', 'compassDirection', 'lte_rssi', 'lte_rsrp', 
        'lte_rsrq', 'lte_rssnr', 'nr_ssRsrp', 'nr_ssRsrq', 'nr_ssSinr'
    ]
    
    categorical_features = [
        'nrStatus', 'mobility_mode', 'trajectory_direction'
    ]
    
    # Handle missing values - replace with median for numerical features
    for col in numerical_features:
        if col in df.columns:
            # Replace sentinel value 2147483647 with NaN
            df[col] = df[col].replace(2147483647.0, np.nan)
            # Check if column has any valid values
            if df[col].notna().sum() > 0:
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
            else:
                # If entire column is NaN, fill with 0
                df[col] = 0.0
    
    # Handle missing values in categorical features
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].fillna('UNKNOWN')
    
    # Encode categorical features
    for col in categorical_features:
        if col in df.columns and col in label_encoders:
            # Handle unseen labels
            le = label_encoders[col]
            df[col] = df[col].apply(
                lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else 0
            )
    
    # Prepare features
    feature_cols = [col for col in numerical_features + categorical_features 
                   if col in df.columns]
    X = df[feature_cols].values.astype(np.float32)
    
    # Replace any remaining NaN or inf values with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale features
    X = scaler.transform(X)
    
    # Handle any NaN values that might appear after scaling
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X


