# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd



def load_csv_data_from_path(
    data_path: str, 
    data_features: List[str], 
    sep: str = r"\s*,\s*",
    engine: str = "python",
    na_values: str = "?"
) -> pd.DataFrame:
    """
    Load CSV data from a single file or directory containing multiple CSV files.
    
    This function handles both single file and directory loading, with comprehensive
    validation to ensure data consistency across multiple CSV files.
    
    Args:
        data_path (str): Path to a CSV file or directory containing CSV files
        data_features (List[str]): List of column names to extract from the CSV files
        sep (str): Separator for CSV parsing (default: r"\s*,\s*")
        engine (str): Pandas CSV parsing engine (default: "python")
        na_values (str): Values to treat as NaN (default: "?")
    
    Returns:
        pd.DataFrame: Concatenated DataFrame containing all loaded data
        
    Raises:
        FileNotFoundError: If the specified path doesn't exist
        ValueError: If no valid CSV files are found or feature consistency issues
    """
    data_path = Path(data_path)
    client_name = "client"
    
    # Check if path is a file or directory
    if data_path.is_file():
        # Single file case
        csv_files = [data_path]
        print(f"Loading data from single file: {data_path}")
    elif data_path.is_dir():
        # Directory case - find all CSV files
        csv_files = list(data_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory: {data_path}")
        print(f"Loading data from directory: {data_path}")
    else:
        raise FileNotFoundError(f"Path not found: {data_path}")

    print(f"Found {len(csv_files)} CSV file(s) to process on client {client_name}")

    # Load and validate all CSV files
    dataframes = []
    expected_columns = None
    
    for csv_file in csv_files:
        try:
            # Load the CSV file
            df = pd.read_csv(
                csv_file, 
                usecols=data_features, 
                sep=sep, 
                engine=engine, 
                na_values=na_values
            )
            
            print(f"Loaded {len(df)} rows from {csv_file.name}")
            
            # Validate feature consistency across files
            if expected_columns is None:
                # First file - set the expected column structure
                expected_columns = list(df.columns)
                print(f"Expected columns from first file: {expected_columns}")
            else:
                # Subsequent files - check for consistency
                current_columns = list(df.columns)
                if current_columns != expected_columns:
                    raise ValueError(
                        f"Column mismatch in {csv_file.name}. "
                        f"Expected: {expected_columns}, Got: {current_columns}"
                    )
                
                # Check for missing features
                missing_features = [col for col in data_features if col not in df.columns]
                if missing_features:
                    raise ValueError(
                        f"Missing features in {csv_file.name}: {missing_features}"
                    )
            
            # Check for empty dataframes
            if len(df) == 0:
                print(f"WARNING: File {csv_file.name} is empty, skipping")
                continue
                
            dataframes.append(df)
            
        except Exception as e:
            print(f"WARNING: Could not load {csv_file.name}: {e}")
            continue

    if not dataframes:
        raise ValueError("No valid CSV files could be loaded")

    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Successfully concatenated {len(dataframes)} files into {len(combined_df)} total rows")
    
    # Final validation
    if len(combined_df) == 0:
        raise ValueError("Combined dataset is empty")
    
    # Log summary statistics
    print(f"Final dataset shape: {combined_df.shape} on client {client_name}")
    print(f"Columns: {list(combined_df.columns)}")
    
    return combined_df


def validate_data_features(df: pd.DataFrame, data_features: List[str]) -> None:
    """
    Validate that all required features exist in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        data_features (List[str]): List of required feature names
        
    Raises:
        ValueError: If any required features are missing
    """
    missing_features = [col for col in data_features if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Check for completely empty features
    empty_features = []
    for feature in data_features:
        if df[feature].isna().all():
            empty_features.append(feature)
    
    if empty_features:
        raise ValueError(f"Features with all NaN values: {empty_features}")


def split_data_for_statistics(df: pd.DataFrame, train_frac: float = 0.8, random_state: int = 200) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into train and test sets for statistics computation.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        train_frac (float): Fraction of data to use for training (default: 0.8)
        random_state (int): Random seed for reproducibility (default: 200)
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_data, test_data)
    """
    train = df.sample(frac=train_frac, random_state=random_state)
    test = df.drop(train.index).sample(frac=1.0, random_state=random_state)
    
    return train, test
