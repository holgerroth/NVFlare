#!/usr/bin/env python3
"""
Create preprocessors with pre-defined feature schema for federated learning.

This approach doesn't require access to the full dataset. Instead, it uses
domain knowledge or data collection standards to define the feature vocabulary.
"""

import pickle
import os
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np


def create_preprocessors_from_schema(output_dir='federated_data'):
    """
    Create preprocessors using pre-defined feature schemas.
    
    This is suitable for real-world federated learning where you don't have
    access to the full dataset but know the feature vocabulary from:
    - Network specifications (e.g., valid frequency bands)
    - Data collection standards
    - Domain expertise
    """
    
    print("Creating preprocessors from pre-defined schema...")
    print("=" * 70)
    
    # Define numerical features (no preprocessing needed for schema)
    numerical_features = [
        'abstractSignalStr', 'latitude', 'longitude', 
        'movingSpeed', 'compassDirection', 'lte_rssi', 'lte_rsrp', 
        'lte_rsrq', 'lte_rssnr', 'nr_ssRsrp', 'nr_ssRsrq', 'nr_ssSinr',
        'Throughput'  # Included as feature in sequence
    ]
    num_numerical = len(numerical_features)
    
    # Initialize scaler with known parameters
    # We'll use identity scaling initially (mean=0, std=1)
    # Clients can optionally do local normalization
    scaler = StandardScaler()
    scaler.mean_ = np.zeros(num_numerical)
    scaler.scale_ = np.ones(num_numerical)
    scaler.var_ = np.ones(num_numerical)
    scaler.n_samples_seen_ = 0
    
    # Pre-define categorical vocabularies based on domain knowledge
    # These are the possible values we expect to see in the data
    categorical_schemas = {
        # NR (5G) connection status
        'nrStatus': ['CONNECTED', 'NOT_RESTRICTED', 'RESTRICTED', 'UNAVAILABLE', 
                     'NONE', 'UNKNOWN'],
        
        # Mobility mode (stationary, walking, driving, etc.)
        'mobility_mode': ['stationary', 'walking', 'driving', 'UNKNOWN'],
        
        # Trajectory direction (CW=clockwise, ACW=anti-clockwise)
        'trajectory_direction': ['CW', 'ACW', 'UNKNOWN'],
    }
    
    print("\nCategorical Feature Schemas:")
    label_encoders = {}
    num_categorical = 0
    
    for feature, classes in categorical_schemas.items():
        le = LabelEncoder()
        le.classes_ = np.array(classes)
        label_encoders[feature] = le
        num_categorical += len(classes)
        print(f"  {feature}: {len(classes)} classes")
    
    total_dim = num_numerical + num_categorical
    
    print(f"\n{'=' * 70}")
    print(f"Feature Engineering Summary:")
    print(f"  Numerical features: {num_numerical}")
    print(f"  Categorical features (one-hot): {num_categorical}")
    print(f"  Total input dimension: {total_dim}")
    print(f"{'=' * 70}")
    
    # Save preprocessors
    os.makedirs(output_dir, exist_ok=True)
    
    scaler_path = os.path.join(output_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"\nSaved scaler to: {scaler_path}")
    
    encoders_path = os.path.join(output_dir, 'label_encoders.pkl')
    with open(encoders_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"Saved label encoders to: {encoders_path}")
    
    # Save config
    config_path = os.path.join(output_dir, 'feature_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"Input Dimension: {total_dim}\n")
        f.write(f"Numerical Features: {num_numerical}\n")
        f.write(f"Categorical Features (one-hot): {num_categorical}\n")
        f.write(f"Schema-based (pre-defined vocabulary)\n")
        f.write(f"\nNumerical Features:\n")
        for feat in numerical_features:
            f.write(f"  - {feat}\n")
        f.write(f"\nCategorical Features:\n")
        for feat, le in label_encoders.items():
            f.write(f"  - {feat}: {len(le.classes_)} classes\n")
    print(f"Saved feature configuration to: {config_path}")
    
    print(f"\n✓ Schema-based preprocessors created!")
    print(f"\nNOTE: This uses pre-defined vocabularies.")
    print(f"If clients encounter unknown categorical values, they will be")
    print(f"mapped to 'unknown' category. Make sure your data.py handles this.")
    
    return total_dim


def main():
    parser = argparse.ArgumentParser(
        description='Create preprocessors from pre-defined schema (no data needed)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python create_schema_based_preprocessors.py --output_dir federated_data
  
This creates preprocessors without needing access to any training data.
Suitable for real-world federated learning scenarios.
        """
    )
    
    parser.add_argument('--output_dir', type=str, default='federated_data',
                       help='Directory to save preprocessor files (default: federated_data)')
    
    args = parser.parse_args()
    
    input_dim = create_preprocessors_from_schema(args.output_dir)
    
    print(f"\nNext steps:")
    print(f"  1. Distribute data to clients (or they collect it locally)")
    print(f"  2. Run: python job.py --n_clients N --num_rounds R --input_dim {input_dim}")
    
    return 0


if __name__ == '__main__':
    exit(main())

