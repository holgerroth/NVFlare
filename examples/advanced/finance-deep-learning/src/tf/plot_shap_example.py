#!/usr/bin/env python3
"""
Example script demonstrating how to use the factored-out SHAP plotting functions.

This script shows how to load pre-computed SHAP metrics and generate plots
without needing to recompute the SHAP values.
"""

from utils import load_shap_metrics, plot_all_shap_plots


def main():
    """
    Main function demonstrating the use of factored-out plotting functions.
    """
    print("SHAP Plotting Example")
    print("=" * 50)

    # Example 1: Load pre-computed metrics and generate all plots
    print("\n1. Loading pre-computed SHAP metrics and generating all plots...")

    # Or you can load the global metrics from the ShapCollectionFilter
    all_shap_metrics = load_shap_metrics("/tmp/nvflare/simulation/hello-tf-mlflow/server/simulate_job/app_server/shap_values.npy")
    print(f"All SHAP metrics: {all_shap_metrics.keys()}")

    shap_metrics = all_shap_metrics["round0"]["site-1"]
    print(f"SHAP metrics for round 0 and site-1: {shap_metrics.keys()}")

    # Generate all plots
    plot_all_shap_plots(shap_metrics, plot_prefix="figs/example", save_fig=True)


if __name__ == "__main__":
    main()
