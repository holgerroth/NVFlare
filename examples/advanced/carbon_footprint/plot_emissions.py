import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse


def plot_emissions(emissions_csv_file="client_emissions.csv"):
    """
    Plot carbon/energy/communication data from all clients.

    Args:
        emissions_csv_file: Path to CSV file with emission data
    """
    sns.set_palette("husl")

    if not os.path.exists(emissions_csv_file):
        raise FileNotFoundError(f"Emissions file not found at {emissions_csv_file}")

    df = pd.read_csv(emissions_csv_file)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df = df.sort_values(by=["round", "client"], kind="mergesort")

    metric_cols = [
        "emissions",           # training emissions (kg CO2e)
        "cpu_energy",          # kWh
        "gpu_energy",          # kWh
        "ram_energy",          # kWh
        "energy_consumed",     # kWh
        "comm_data_kb",        # kB (legacy)
        "comm_data_gb",        # GB (new)
        "comm_energy_kwh",     # kWh (new)
        "comm_emissions_kg",   # kg CO2e (new)
        "inet_kwh_per_gb",     # kWh/GB (new)
        "grid_kg_per_kwh",     # kg CO2e/kWh (new)
    ]

    unit_map = {
        "emissions": "kg CO2e",
        "cpu_energy": "kWh",
        "gpu_energy": "kWh",
        "ram_energy": "kWh",
        "energy_consumed": "kWh",
        "comm_data_kb": "kB",
        "comm_data_gb": "GB",
        "comm_energy_kwh": "kWh",
        "comm_emissions_kg": "kg CO2e",
        "inet_kwh_per_gb": "kWh/GB",
        "grid_kg_per_kwh": "kg CO2e/kWh",
    }

    # One plot per metric that exists
    for metric in metric_cols:
        if metric not in df.columns:
            print(f"Skipping '{metric}': not found in CSV.")
            continue

        units = unit_map.get(metric, "")
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df,
            x="round",
            y=metric,
            hue="client",
            marker="o",
            style="client"
        )

        plt.title(f'{metric.replace("_", " ").title()} Over Time by Client')
        ylabel = metric.replace("_", " ").title()
        if units:
            ylabel += f" [{units}]"
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        plt.legend(title="Client")
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        os.makedirs("figs", exist_ok=True)
        output_prefix = f"figs/{metric.lower().replace(' ', '_')}_plot"
        plt.savefig(output_prefix + ".png", dpi=200)
        plt.savefig(output_prefix + ".svg")
        print(f"Plot saved as '{output_prefix}.png' and '.svg'")

    # Optional totals across clients per round
    if "round" in df.columns:
        total_cols = [
            c for c in ["comm_data_gb", "comm_energy_kwh", "comm_emissions_kg",
                        "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed"]
            if c in df.columns
        ]
        if total_cols:
            totals = df.groupby("round", as_index=False)[total_cols].sum()

            for metric in total_cols:
                units = unit_map.get(metric, "")
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=totals, x="round", y=metric, marker="o")

                plt.title(f"Total {metric.replace('_', ' ').title()} Over Time (All Clients)")
                ylabel = f"Total {metric.replace('_', ' ').title()}"
                if units:
                    ylabel += f" [{units}]"
                plt.xlabel("Round")
                plt.ylabel(ylabel)
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()

                os.makedirs("figs", exist_ok=True)
                output_prefix = f"figs/total_{metric.lower().replace(' ', '_')}_plot"
                plt.savefig(output_prefix + ".png", dpi=200)
                plt.savefig(output_prefix + ".svg")
                print(f"Total plot saved as '{output_prefix}.png' and '.svg'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot carbon emissions data from FL clients")
    parser.add_argument(
        "--emissions_csv_file",
        type=str,
        default="client_emissions.csv",
        help="Path to the emissions CSV file",
    )
    args = parser.parse_args()
    plot_emissions(args.emissions_csv_file)
