#!/usr/bin/env python3
import os
import argparse
import pickle
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def _get(obj, name, alt=None):
    try:
        return getattr(obj, name)
    except Exception:
        try:
            return obj.get(name, alt)
        except Exception:
            return alt


def _extract_idle_from_pkl(pkl_path: str) -> Optional[pd.DataFrame]:
    """Return idle metrics per (client, round) from client_emissions.pkl if present.

    Produces columns:
      client, round, idle_timestamp, idle_emissions, idle_cpu_energy,
      idle_gpu_energy, idle_ram_energy, idle_energy_consumed, idle_duration_sec
    """
    if not pkl_path or not os.path.exists(pkl_path):
        return None

    try:
        with open(pkl_path, "rb") as f:
            client_emissions = pickle.load(f)
    except Exception as e:
        print(f"Warning: failed to load '{pkl_path}': {e}")
        return None

    rows = []
    # Expected structure: dict[client] -> list[record], where record contains:
    #   'current_round', 'idle' (object/dict with CodeCarbon metrics)
    for client, records in client_emissions.items():
        for rec in records:
            rnd = _get(rec, "current_round")
            idle = _get(rec, "idle")
            if idle is None:
                continue

            ts = _get(idle, "timestamp")
            emissions = _get(idle, "emissions")
            cpu_e = _get(idle, "cpu_energy")
            gpu_e = _get(idle, "gpu_energy")
            ram_e = _get(idle, "ram_energy")
            tot_e = _get(idle, "energy_consumed")
            duration = _get(idle, "duration")
            if duration is None:
                duration = _get(idle, "duration_sec", _get(idle, "duration_seconds"))

            rows.append(
                {
                    "client": client,
                    "round": pd.to_numeric(rnd, errors="coerce"),
                    "idle_timestamp": ts,
                    "idle_emissions": emissions,
                    "idle_cpu_energy": cpu_e,
                    "idle_gpu_energy": gpu_e,
                    "idle_ram_energy": ram_e,
                    "idle_energy_consumed": tot_e,
                    "idle_duration_sec": duration,
                }
            )
    if not rows:
        return None

    idle_df = pd.DataFrame(rows)
    if "idle_timestamp" in idle_df.columns:
        idle_df["idle_timestamp"] = pd.to_datetime(
            idle_df["idle_timestamp"], errors="coerce"
        )
    return idle_df


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric (errors->NaN) to avoid plotting skips."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add totals and ratios that include idle and comm contributions."""
    default0 = lambda name: df.get(name, pd.Series(index=df.index, dtype=float)).fillna(0).astype(float)

    train_energy = default0("energy_consumed")
    comm_energy = default0("comm_energy_kwh")
    idle_energy = default0("idle_energy_consumed")

    train_emiss = default0("emissions")
    comm_emiss = default0("comm_emissions_kg")
    idle_emiss = default0("idle_emissions")

    df["total_energy_kwh"] = train_energy + comm_energy + idle_energy
    df["total_emissions_kg"] = train_emiss + comm_emiss + idle_emiss

    df["idle_to_train_energy_ratio"] = np.where(
        train_energy > 0, idle_energy / train_energy, np.nan
    )
    df["idle_share_of_total_energy"] = np.where(
        df["total_energy_kwh"] > 0, idle_energy / df["total_energy_kwh"], np.nan
    )
    return df


def _ensure_round_client_types(df: pd.DataFrame) -> pd.DataFrame:
    if "round" in df.columns:
        df["round"] = pd.to_numeric(df["round"], errors="coerce")
    if "client" in df.columns:
        df["client"] = df["client"].astype(str)
    return df.sort_values(by=[c for c in ["round", "client"] if c in df.columns], kind="mergesort")


def _save_fig(prefix: str):
    os.makedirs("figs", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"figs/{prefix}.png", dpi=200)
    plt.savefig(f"figs/{prefix}.svg")
    plt.close()
    print(f"Plot saved as 'figs/{prefix}.png' and '.svg'")


def plot_emissions(emissions_csv_file: str, emissions_pkl_file: Optional[str] = None):
    """
    Plot carbon/energy/communication (GB-only) and idle data from FL clients.

    Parameters
    ----------
    emissions_csv_file : str
        Path to the CSV produced by the server controller (now includes idle_*).
    emissions_pkl_file : Optional[str]
        Optional path to client_emissions.pkl for idle merge (if CSV lacks idle).
    """
    sns.set_palette("husl")

    if not os.path.exists(emissions_csv_file):
        raise FileNotFoundError(f"Emissions file not found at {emissions_csv_file}")

    df = pd.read_csv(emissions_csv_file)
    print("Loaded CSV:", emissions_csv_file, "| columns:", list(df.columns))

    idle_df = _extract_idle_from_pkl(emissions_pkl_file) if emissions_pkl_file else None
    if idle_df is not None:
        df = df.merge(idle_df, on=["round", "client"], how="left")

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = _ensure_round_client_types(df)

    numeric_cols = [
        # training
        "emissions", "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed",
        # comm (GB-only)
        "comm_data_gb", "comm_energy_kwh", "comm_emissions_kg",
        "inet_kwh_per_gb", "grid_kg_per_kwh",
        # idle
        "idle_emissions", "idle_cpu_energy", "idle_gpu_energy",
        "idle_ram_energy", "idle_energy_consumed", "idle_duration_sec",
    ]
    df = _coerce_numeric(df, numeric_cols)

    # Derived totals/ratios
    df = _add_derived_columns(df)

    metric_cols: List[str] = [
        # Training metrics
        "emissions", "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed",
        # Communication metrics (GB-only)
        "comm_data_gb", "comm_energy_kwh", "comm_emissions_kg",
        "inet_kwh_per_gb", "grid_kg_per_kwh",
        # Idle-time metrics
        "idle_emissions", "idle_cpu_energy", "idle_gpu_energy",
        "idle_ram_energy", "idle_energy_consumed", "idle_duration_sec",
        # Derived
        "total_energy_kwh", "total_emissions_kg",
        "idle_to_train_energy_ratio", "idle_share_of_total_energy",
    ]

    unit_map: Dict[str, str] = {
        # train
        "emissions": "kg CO2e",
        "cpu_energy": "kWh",
        "gpu_energy": "kWh",
        "ram_energy": "kWh",
        "energy_consumed": "kWh",
        # comm
        "comm_data_gb": "GB",
        "comm_energy_kwh": "kWh",
        "comm_emissions_kg": "kg CO2e",
        "inet_kwh_per_gb": "kWh/GB",
        "grid_kg_per_kwh": "kg CO2e/kWh",
        # idle
        "idle_emissions": "kg CO2e",
        "idle_cpu_energy": "kWh",
        "idle_gpu_energy": "kWh",
        "idle_ram_energy": "kWh",
        "idle_energy_consumed": "kWh",
        "idle_duration_sec": "sec",
        # derived
        "total_energy_kwh": "kWh",
        "total_emissions_kg": "kg CO2e",
        "idle_to_train_energy_ratio": "",
        "idle_share_of_total_energy": "",
    }

    for metric in metric_cols:
        if metric not in df.columns:
            print(f"Skipping '{metric}': not found in data.")
            continue

        # Skip if column is entirely NaN
        if df[metric].dropna().empty:
            print(f"Skipping '{metric}': no non-NaN values to plot.")
            continue

        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df,
            x="round",
            y=metric,
            hue="client",
            marker="o",
            style="client",
            linewidth=1.5,
        )
        title = f"{metric.replace('_', ' ').title()} Over Time by Client"
        ylabel = metric.replace("_", " ").title()
        units = unit_map.get(metric, "")
        if units:
            ylabel += f" [{units}]"

        plt.title(title)
        plt.xlabel("Round")
        plt.ylabel(ylabel)
        plt.legend(title="Client", ncol=2, fontsize="small")
        plt.grid(True, axis="y", linestyle="--", alpha=0.4)
        _save_fig(metric.lower().replace(" ", "_") + "_plot")

    if "round" in df.columns:
        total_cols = [
            c
            for c in [
                "comm_data_gb", "comm_energy_kwh", "comm_emissions_kg",
                "cpu_energy", "gpu_energy", "ram_energy", "energy_consumed",
                "idle_emissions", "idle_cpu_energy", "idle_gpu_energy",
                "idle_ram_energy", "idle_energy_consumed", "idle_duration_sec",
                "total_energy_kwh", "total_emissions_kg",
            ]
            if c in df.columns
        ]
        if total_cols:
            totals = df.groupby("round", as_index=False)[total_cols].sum(numeric_only=True)
            for metric in total_cols:
                if totals[metric].dropna().empty:
                    continue
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=totals, x="round", y=metric, marker="o", linewidth=1.8)
                title = f"Total {metric.replace('_', ' ').title()} Over Time (All Clients)"
                ylabel = "Total " + metric.replace("_", " ").title()
                units = unit_map.get(metric, "")
                if units:
                    ylabel += f" [{units}]"
                plt.title(title)
                plt.xlabel("Round")
                plt.ylabel(ylabel)
                plt.grid(True, axis="y", linestyle="--", alpha=0.4)
                _save_fig("total_" + metric.lower().replace(" ", "_") + "_plot")

    try:
        need = {"energy_consumed", "comm_energy_kwh", "idle_energy_consumed", "round"}
        if need.issubset(df.columns):
            st = df.groupby("round", as_index=False)[
                ["energy_consumed", "comm_energy_kwh", "idle_energy_consumed"]
            ].sum(numeric_only=True)
            plt.figure(figsize=(12, 6))
            plt.stackplot(
                st["round"],
                st["energy_consumed"].fillna(0),
                st["comm_energy_kwh"].fillna(0),
                st["idle_energy_consumed"].fillna(0),
                labels=["Train Energy (kWh)", "Comm Energy (kWh)", "Idle Energy (kWh)"],
            )
            plt.legend(loc="upper left")
            plt.title("Energy Breakdown Over Time (All Clients)")
            plt.xlabel("Round")
            plt.ylabel("kWh")
            plt.grid(True, axis="y", linestyle="--", alpha=0.5)
            _save_fig("total_energy_breakdown_stacked")
    except Exception as e:
        print(f"Skipping stacked breakdown: {e}")



def main():
    parser = argparse.ArgumentParser(description="Plot carbon/energy/idle data from FL clients (GB-only comms).")
    parser.add_argument(
        "--emissions_csv_file",
        type=str,
        default="client_emissions.csv",
        help="Path to the emissions CSV file (from server).",
    )
    parser.add_argument(
        "--emissions_pkl_file",
        type=str,
        default="client_emissions.pkl",
        help="Optional path to client_emissions.pkl (used to merge idle metrics if CSV lacks them).",
    )
    args = parser.parse_args()
    plot_emissions(args.emissions_csv_file, args.emissions_pkl_file)


if __name__ == "__main__":
    main()
