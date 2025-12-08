#!/usr/bin/env python3
"""
Quick test to verify preprocessors work with actual data.
"""

import pandas as pd
import pickle
import sys
import os

def test_preprocessors(data_file, preprocessor_dir='federated_data'):
    """Test if preprocessors match the data schema"""
    
    print(f"Testing preprocessors with data file: {data_file}")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(data_file)
    print(f"\nData columns: {list(df.columns)}")
    
    # Load preprocessors
    scaler_path = os.path.join(preprocessor_dir, 'scaler.pkl')
    encoders_path = os.path.join(preprocessor_dir, 'label_encoders.pkl')
    
    if not os.path.exists(scaler_path):
        print(f"\nERROR: Scaler not found at {scaler_path}")
        return False
    
    if not os.path.exists(encoders_path):
        print(f"\nERROR: Label encoders not found at {encoders_path}")
        return False
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(encoders_path, 'rb') as f:
        label_encoders = pickle.load(f)
    
    print(f"\nPreprocessors loaded successfully!")
    print(f"  Scaler features: {len(scaler.mean_)}")
    print(f"  Label encoders: {list(label_encoders.keys())}")
    
    # Check for mismatches
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    # Expected categorical features in data
    expected_categorical = ['nrStatus', 'mobility_mode', 'trajectory_direction']
    
    issues = []
    for cat_feature in expected_categorical:
        if cat_feature not in df.columns:
            issues.append(f"❌ Data missing column: {cat_feature}")
        elif cat_feature not in label_encoders:
            issues.append(f"❌ Preprocessor missing encoder for: {cat_feature}")
        else:
            # Check if data values are in encoder vocab
            unique_vals = df[cat_feature].unique()
            le = label_encoders[cat_feature]
            unseen = [v for v in unique_vals if str(v) not in le.classes_ and pd.notna(v)]
            if unseen:
                issues.append(f"⚠️  Column '{cat_feature}' has unseen values: {unseen[:5]}")
            else:
                print(f"✅ {cat_feature}: OK")
    
    if issues:
        print("\nISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("\nRECOMMENDATION:")
        print("  Run: python create_schema_based_preprocessors.py --output_dir federated_data")
        return False
    else:
        print("\n✅ All checks passed! Preprocessors match data schema.")
        return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_preprocessors.py <data_file> [preprocessor_dir]")
        print("\nExample:")
        print("  python test_preprocessors.py federated_data/site-1.csv")
        sys.exit(1)
    
    data_file = sys.argv[1]
    preprocessor_dir = sys.argv[2] if len(sys.argv) > 2 else 'federated_data'
    
    success = test_preprocessors(data_file, preprocessor_dir)
    sys.exit(0 if success else 1)

