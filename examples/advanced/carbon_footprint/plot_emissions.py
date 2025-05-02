import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_emissions(emissions_csv_file="client_emissions.csv"):
    """
    Plot carbon emissions data from all clients.
    
    Args:
        base_dir: Base directory containing client emissions data
    """
    # Set style
    sns.set_palette("husl")
    
    # Read the emissions data
    if not os.path.exists(emissions_csv_file):
        raise FileNotFoundError(f"Emissions file not found at {emissions_csv_file}")
    
    df = pd.read_csv(emissions_csv_file)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Columns to plot (excluding round, timestamp, and client)
    metric_cols = ['emissions', 'cpu_energy', 'gpu_energy', 'ram_energy', 'energy_consumed']
    
    # Create a figure for each metric
    for metric in metric_cols:
        units = "kWh" if "energy" in metric else "kg CO₂"  # TODO: double check this the units used by CodeCarbon
        plt.figure(figsize=(12, 6))
        
        # Create the plot using seaborn
        sns.lineplot(data=df, x='round', y=metric, hue='client', 
                    marker='o', style='client')
        
        # Customize plot
        plt.title(f'{metric.replace("_", " ").title()} Over Time by Client')
        plt.xlabel('Round')
        plt.ylabel(metric.replace("_", " ").title() + f" [{units}]")
        plt.legend(title='Client')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        os.makedirs('figs', exist_ok=True)
        output_prefix = f'figs/{metric.lower().replace(" ", "_")}_plot'
        plt.savefig(output_prefix+".png")
        plt.savefig(output_prefix+".svg")
        print(f"Plot saved as '{output_prefix}*'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot carbon emissions data from FL clients')
    parser.add_argument('--emissions_csv_file', type=str, default="client_emissions.csv",
                      help='Path to the emissions CSV file')
    args = parser.parse_args()
    plot_emissions(args.emissions_csv_file) 