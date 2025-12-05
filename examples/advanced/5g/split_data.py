#!/usr/bin/env python3
"""
Utility script to split Lumos5G data into training and validation sets
based on run_num (entire trajectories).

This ensures clean separation for proper evaluation:
- Training set: Used for model training
- Validation set: Held-out test data for final evaluation
"""

import pandas as pd
import numpy as np
import argparse
import os


def split_data_by_runs(data_path, output_dir, train_split=0.8, seed=42, prefix=''):
    """
    Split data into train and validation sets based on run_num.
    
    Args:
        data_path: Path to the input CSV file
        output_dir: Directory to save the split files
        train_split: Ratio of runs to use for training
        seed: Random seed for reproducibility
        prefix: Optional prefix for output files
    
    Returns:
        train_path, val_path: Paths to the saved files
    """
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Get unique runs
    run_nums = df['run_num'].unique()
    print(f"Found {len(run_nums)} unique runs")
    
    # Shuffle runs with seed
    np.random.seed(seed)
    shuffled_runs = run_nums.copy()
    np.random.shuffle(shuffled_runs)
    
    # Split runs
    num_train_runs = int(len(shuffled_runs) * train_split)
    train_runs = set(shuffled_runs[:num_train_runs])
    val_runs = set(shuffled_runs[num_train_runs:])
    
    print(f"\nSplit configuration:")
    print(f"  Train runs: {len(train_runs)} ({len(train_runs)/len(run_nums)*100:.1f}%)")
    print(f"  Validation runs: {len(val_runs)} ({len(val_runs)/len(run_nums)*100:.1f}%)")
    print(f"  Random seed: {seed}")
    
    # Split dataframe
    train_df = df[df['run_num'].isin(train_runs)].copy()
    val_df = df[df['run_num'].isin(val_runs)].copy()
    
    print(f"\nData distribution:")
    print(f"  Train rows: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation rows: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save files
    train_filename = f"{prefix}train.csv" if prefix else "train.csv"
    val_filename = f"{prefix}val.csv" if prefix else "val.csv"
    
    train_path = os.path.join(output_dir, train_filename)
    val_path = os.path.join(output_dir, val_filename)
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    
    print(f"\n✓ Files saved:")
    print(f"  Train: {train_path}")
    print(f"  Validation: {val_path}")
    
    # Save run splits for reference
    splits_file = os.path.join(output_dir, f"{prefix}run_splits.txt" if prefix else "run_splits.txt")
    with open(splits_file, 'w') as f:
        f.write("TRAIN RUNS\n")
        f.write("=" * 60 + "\n")
        for run in sorted(train_runs):
            f.write(f"{run}\n")
        f.write("\n")
        f.write("VALIDATION RUNS\n")
        f.write("=" * 60 + "\n")
        for run in sorted(val_runs):
            f.write(f"{run}\n")
    
    print(f"  Run splits: {splits_file}")
    
    return train_path, val_path


def main():
    parser = argparse.ArgumentParser(
        description='Split Lumos5G data into train and validation sets by runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default 80/20 split
  python split_data.py --data_path Lumos5G-v1.0/Lumos5G-v1.0.csv
  
  # Custom split ratio
  python split_data.py --data_path Lumos5G-v1.0/Lumos5G-v1.0.csv --train_split 0.7
  
  # Specify output directory
  python split_data.py --data_path Lumos5G-v1.0/Lumos5G-v1.0.csv --output_dir data_splits
  
  # Use different random seed
  python split_data.py --data_path Lumos5G-v1.0/Lumos5G-v1.0.csv --seed 123

After splitting, use the files:
  python train.py --data_path train.csv
  python inference.py --checkpoint outputs/best_model.pth --data_path val.csv
        """
    )
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the input CSV file')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save split files (default: current directory)')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Ratio of runs for training (default: 0.8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--prefix', type=str, default='',
                       help='Optional prefix for output files')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return 1
    
    if not 0 < args.train_split < 1:
        print(f"Error: train_split must be between 0 and 1, got {args.train_split}")
        return 1
    
    print("=" * 70)
    print("Lumos5G Data Splitter")
    print("=" * 70)
    
    # Split the data
    split_data_by_runs(
        data_path=args.data_path,
        output_dir=args.output_dir,
        train_split=args.train_split,
        seed=args.seed,
        prefix=args.prefix
    )
    
    print("\n" + "=" * 70)
    print("✓ Split complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Train on the training set:")
    print(f"     python train.py --data_path {os.path.join(args.output_dir, args.prefix + 'train.csv')}")
    print("\n  2. Run inference on validation set (test data):")
    print(f"     python inference.py --checkpoint outputs/best_model.pth \\")
    print(f"         --data_path {os.path.join(args.output_dir, args.prefix + 'val.csv')} \\")
    print(f"         --output_dir inference_val --plot")
    print("\n  3. (Optional) Run inference on full data for comparison:")
    print(f"     python inference.py --checkpoint outputs/best_model.pth \\")
    print(f"         --data_path {args.data_path} \\")
    print(f"         --output_dir inference_all --plot")
    
    return 0


if __name__ == '__main__':
    exit(main())

