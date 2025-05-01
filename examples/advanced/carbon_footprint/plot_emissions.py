import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_emissions(base_dir="/tmp/nvflare/carbon_footprint"):
    """
    Plot carbon emissions data from all clients.
    
    Args:
        base_dir: Base directory containing client emissions data
    """
    # Set style
    sns.set_palette("husl")
    
    # Get all client directories
    client_dirs = [d for d in os.listdir(base_dir) if d.startswith('site-')]
    
    # Read first file to get column names
    first_file = os.path.join(base_dir, client_dirs[0], 'emissions.csv')
    df = pd.read_csv(first_file)
    timestamp_col = df.columns[0]  # First column is timestamp
    metric_cols = df.columns[1:]   # All other columns are metrics to plot
    
    # Create a figure for each metric
    for metric in metric_cols:
        plt.figure(figsize=(12, 6))
        
        # Plot data for each client
        for client_dir in sorted(client_dirs):
            emissions_file = os.path.join(base_dir, client_dir, 'emissions.csv')
            if os.path.exists(emissions_file):
                # Read emissions data
                df = pd.read_csv(emissions_file)
                
                # Convert timestamp to datetime
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                
                # Plot the metric
                plt.plot(df[timestamp_col], df[metric], 
                        label=f'Client {client_dir.split("-")[1]}',
                        marker='o')
        
        # Customize plot
        plt.title(f'{metric} Over Time by Client')
        plt.xlabel('Time')
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(f'{metric.lower().replace(" ", "_")}_plot.png')
        print(f"Plot saved as '{metric.lower().replace(' ', '_')}_plot.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot carbon emissions data from FL clients')
    parser.add_argument('--base_dir', type=str, default="/tmp/nvflare/carbon_footprint",
                      help='Base directory containing client emissions data')
    args = parser.parse_args()
    plot_emissions(args.base_dir) 