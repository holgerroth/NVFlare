#!/usr/bin/env python3
"""
Example script demonstrating how to use the factored-out SHAP plotting functions.

This script shows how to load pre-computed SHAP metrics and generate plots
without needing to recompute the SHAP values.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import plot_shap_summary, plot_shap_feature_importance, plot_all_shap_plots


def load_shap_metrics(file_path):
    """
    Load SHAP metrics from a saved .npy file.
    
    Args:
        file_path: Path to the saved SHAP metrics file
        
    Returns:
        dict: Loaded SHAP metrics
    """
    try:
        return np.load(file_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading SHAP metrics from {file_path}: {e}")
        return None


def main():
    """
    Main function demonstrating the use of factored-out plotting functions.
    """
    print("SHAP Plotting Example")
    print("=" * 50)
    
    # Example 1: Load pre-computed metrics and generate all plots
    print("\n1. Loading pre-computed SHAP metrics and generating all plots...")
    
    # You would typically load from a saved file like this:
    # shap_metrics = load_shap_metrics("path/to/your_shap_metrics.npy")

    # Or you can load the global metrics from the ShapCollectionFilter
    all_shap_metrics = load_shap_metrics("/tmp/nvflare/simulation/hello-tf-mlflow/server/shap_values.npy")
    print(f"All SHAP metrics: {all_shap_metrics.keys()}")

    shap_metrics = all_shap_metrics[f"round0"]["site-1"]
    print(f"SHAP metrics for round 0 and site-1: {shap_metrics.keys()}")   
    
    # Generate all plots
    plot_all_shap_plots(shap_metrics, "example")


if __name__ == "__main__":
    main()
