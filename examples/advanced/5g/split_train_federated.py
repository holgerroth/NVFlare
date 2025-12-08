#!/usr/bin/env python3
"""
Utility script to split training data into N parts for federated learning.
Each part contains a unique, non-overlapping set of runs (trajectories).

This is useful for simulating federated learning scenarios where different
clients/sites have different subsets of the data.
"""

import pandas as pd
import numpy as np
import argparse
import os


def split_train_federated(data_path, output_dir, num_clients, seed=42, prefix='site'):
    """
    Split training data into N parts by runs for federated learning.
    
    Args:
        data_path: Path to the training CSV file
        output_dir: Directory to save the split files
        num_clients: Number of federated clients/sites
        seed: Random seed for reproducibility
        prefix: Prefix for output files (e.g., 'site', 'client')
    
    Returns:
        List of paths to the saved files
    """
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")
    
    # Get unique runs
    run_nums = df['run_num'].unique()
    print(f"Found {len(run_nums)} unique runs")
    
    if num_clients > len(run_nums):
        print(f"\n⚠️  WARNING: num_clients ({num_clients}) > number of runs ({len(run_nums)})")
        print(f"   Some clients will have no data!")
        print(f"   Consider reducing num_clients to {len(run_nums)} or fewer.\n")
    
    # Shuffle runs with seed
    np.random.seed(seed)
    shuffled_runs = run_nums.copy()
    np.random.shuffle(shuffled_runs)
    
    # Split runs as evenly as possible across clients
    runs_per_client = []
    base_size = len(shuffled_runs) // num_clients
    remainder = len(shuffled_runs) % num_clients
    
    start_idx = 0
    for i in range(num_clients):
        # First 'remainder' clients get one extra run
        size = base_size + (1 if i < remainder else 0)
        client_runs = shuffled_runs[start_idx:start_idx + size]
        runs_per_client.append(client_runs)
        start_idx += size
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    output_files = []
    print(f"\nSplit configuration:")
    print(f"  Total runs: {len(run_nums)}")
    print(f"  Number of clients: {num_clients}")
    print(f"  Base runs per client: {base_size}")
    print(f"  Clients with extra run: {remainder}")
    print(f"  Random seed: {seed}")
    print(f"\n{'Client':<15} {'Runs':<10} {'Rows':<10} {'% of Total':<12}")
    print("=" * 60)
    
    for i, client_runs in enumerate(runs_per_client):
        client_name = f"{prefix}-{i+1}"
        client_df = df[df['run_num'].isin(client_runs)].copy()
        
        # Save file
        output_file = os.path.join(output_dir, f"{client_name}.csv")
        client_df.to_csv(output_file, index=False)
        output_files.append(output_file)
        
        # Print stats
        pct = (len(client_df) / len(df)) * 100 if len(df) > 0 else 0
        print(f"{client_name:<15} {len(client_runs):<10} {len(client_df):<10} {pct:>6.2f}%")
    
    print("=" * 60)
    
    # Save run assignments for reference
    assignments_file = os.path.join(output_dir, f"{prefix}_run_assignments.txt")
    with open(assignments_file, 'w') as f:
        f.write("FEDERATED RUN ASSIGNMENTS\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total runs: {len(run_nums)}\n")
        f.write(f"Number of clients: {num_clients}\n")
        f.write(f"Random seed: {seed}\n")
        f.write("=" * 60 + "\n\n")
        
        for i, client_runs in enumerate(runs_per_client):
            client_name = f"{prefix}_{i+1}"
            f.write(f"{client_name.upper()} ({len(client_runs)} runs)\n")
            f.write("-" * 60 + "\n")
            for run in sorted(client_runs):
                f.write(f"{run}\n")
            f.write("\n")
    
    print(f"\n✓ Files saved to: {output_dir}")
    for output_file in output_files:
        print(f"  - {os.path.basename(output_file)}")
    print(f"  - {os.path.basename(assignments_file)}")
    
    return output_files


def validate_splits(output_files):
    """
    Validate that splits have no overlapping runs.
    """
    print("\n" + "=" * 60)
    print("VALIDATING SPLITS")
    print("=" * 60)
    
    all_runs = []
    for file_path in output_files:
        df = pd.read_csv(file_path)
        runs = set(df['run_num'].unique())
        all_runs.append(runs)
    
    # Check for overlaps
    overlap_found = False
    for i in range(len(all_runs)):
        for j in range(i + 1, len(all_runs)):
            overlap = all_runs[i] & all_runs[j]
            if overlap:
                overlap_found = True
                print(f"⚠️  Overlap between {os.path.basename(output_files[i])} and "
                      f"{os.path.basename(output_files[j])}: {overlap}")
    
    if not overlap_found:
        print("✓ No overlaps detected - all runs are unique across clients")
        total_runs = sum(len(runs) for runs in all_runs)
        print(f"✓ Total unique runs across all clients: {total_runs}")
    
    return not overlap_found


def main():
    parser = argparse.ArgumentParser(
        description='Split training data into N parts for federated learning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split train.csv into 4 clients
  python split_train_federated.py --data_path train.csv --num_clients 4
  
  # Split into 3 sites with custom output directory
  python split_train_federated.py --data_path train.csv --num_clients 3 --output_dir federated_data
  
  # Custom prefix for output files
  python split_train_federated.py --data_path train.csv --num_clients 5 --prefix client
  
  # Use different random seed
  python split_train_federated.py --data_path train.csv --num_clients 3 --seed 123

Output:
  - site-1.csv, site-2.csv, ..., site-N.csv
  - site-run_assignments.txt (documentation)

After splitting, train each client:
  python train.py --data_path federated_data/site-1.csv --output_dir outputs/site-1
  python train.py --data_path federated_data/site-2.csv --output_dir outputs/site-2
  ...
        """
    )
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the training CSV file (e.g., train.csv)')
    parser.add_argument('--num_clients', type=int, required=True,
                       help='Number of federated clients/sites to split into')
    parser.add_argument('--output_dir', type=str, default='federated_data',
                       help='Directory to save split files (default: federated_data)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--prefix', type=str, default='site',
                       help='Prefix for output files (default: site)')
    parser.add_argument('--validate', action='store_true',
                       help='Validate splits have no overlapping runs')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        return 1
    
    if args.num_clients < 1:
        print(f"Error: num_clients must be >= 1, got {args.num_clients}")
        return 1
    
    print("=" * 70)
    print("Lumos5G Federated Data Splitter")
    print("=" * 70)
    
    # Split the data
    output_files = split_train_federated(
        data_path=args.data_path,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        seed=args.seed,
        prefix=args.prefix
    )
    
    # Validate if requested
    if args.validate:
        is_valid = validate_splits(output_files)
        if not is_valid:
            print("\n⚠️  WARNING: Overlaps detected in splits!")
            return 1
    
    print("\n" + "=" * 70)
    print("✓ Split complete!")
    print("=" * 70)
    print(f"\nCreated {len(output_files)} client datasets in: {args.output_dir}")
    print("\nNext steps (Federated Learning):")
    print("  1. Train each client on their local data:")
    for i, output_file in enumerate(output_files, 1):
        print(f"     python train.py --data_path {output_file} --output_dir outputs/{args.prefix}_{i}")
    print("\n  2. Aggregate models using federated averaging (FedAvg)")
    print("  3. Evaluate aggregated model on test set (val.csv)")
    print("\nNext steps (Centralized Comparison):")
    print("  1. Train on full train.csv for comparison:")
    print(f"     python train.py --data_path {args.data_path} --output_dir outputs/centralized")
    print("  2. Compare centralized vs federated performance")
    
    return 0


if __name__ == '__main__':
    exit(main())


