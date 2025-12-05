"""
Data loading and preprocessing utilities for Lumos5G dataset
"""
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Lumos5GDataset(Dataset):
    """Dataset class for Lumos5G data"""
    
    def __init__(self, csv_path, scaler=None, label_encoders=None, fit_transform=False):
        """
        Args:
            csv_path: Path to the CSV file
            scaler: StandardScaler for numerical features
            label_encoders: Dictionary of LabelEncoders for categorical features
            fit_transform: Whether to fit the scaler and encoders (True for train, False for val/test)
        """
        self.df = pd.read_csv(csv_path)
        
        # Define feature columns
        self.numerical_features = [
            'seq_num', 'abstractSignalStr', 'latitude', 'longitude', 
            'movingSpeed', 'compassDirection', 'lte_rssi', 'lte_rsrp', 
            'lte_rsrq', 'lte_rssnr', 'nr_ssRsrp', 'nr_ssRsrq', 'nr_ssSinr'
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
        
        self.features = torch.from_numpy(X)
        self.targets = torch.from_numpy(self.df[self.target].values.astype(np.float32))
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]
    
    def get_scaler(self):
        return self.scaler
    
    def get_label_encoders(self):
        return self.label_encoders


def preprocess_data(df, scaler, label_encoders):
    """
    Preprocess data using the saved scaler and label encoders
    
    Args:
        df: Input dataframe
        scaler: Fitted StandardScaler
        label_encoders: Dictionary of fitted LabelEncoders
    
    Returns:
        Preprocessed numpy array
    """
    # Define feature columns (same as in training)
    numerical_features = [
        'seq_num', 'abstractSignalStr', 'latitude', 'longitude', 
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

