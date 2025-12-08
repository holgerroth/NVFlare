#!/usr/bin/env python3
"""
Compare and visualize results from centralized, federated, and local-only training.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_training_history(log_file):
    """Extract training metrics from log file"""
    if not os.path.exists(log_file):
        return None
    
    epochs = []
    train_losses = []
    val_losses = []
    val_maes = []
    val_rmses = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Epoch" in line and "Train Loss:" in line:
                try:
                    # Parse: Epoch X/Y
                    epoch = int(line.split("Epoch")[1].split("/")[0].strip())
                    epochs.append(epoch)
                except:
                    pass
            
            if "Train Loss:" in line:
                try:
                    train_loss = float(line.split("Train Loss:")[1].split()[0])
                    train_losses.append(train_loss)
                except:
                    pass
            
            if "Val Loss:" in line:
                try:
                    parts = line.split("Val Loss:")[1]
                    val_loss = float(parts.split(",")[0].strip())
                    val_losses.append(val_loss)
                except:
                    pass
            
            if "Val MAE:" in line:
                try:
                    mae = float(line.split("Val MAE:")[1].split(",")[0].strip())
                    val_maes.append(mae)
                except:
                    pass
            
            if "Val RMSE:" in line:
                try:
                    rmse = float(line.split("Val RMSE:")[1].split()[0].strip())
                    val_rmses.append(rmse)
                except:
                    pass
    
    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_mae': val_maes,
        'val_rmse': val_rmses
    }


def load_evaluation_metrics(json_path):
    """Load metrics from evaluation JSON file"""
    if not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            metrics = json.load(f)
        return {
            'val_loss': metrics.get('val_loss'),
            'val_mae': metrics.get('val_mae'),
            'val_rmse': metrics.get('val_rmse'),
        }
    except Exception as e:
        print(f"    Warning: Could not load {json_path}: {e}")
        return None


def load_model_checkpoint(model_path):
    """Load model checkpoint and extract metrics"""
    if not os.path.exists(model_path):
        return None
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Debug: print available keys
        print(f"    Checkpoint keys: {list(checkpoint.keys())[:10]}...")  # Show first 10 keys
        
        # Try different possible key names
        metrics = {
            'val_loss': (checkpoint.get('eval_loss') or 
                        checkpoint.get('val_loss') or 
                        checkpoint.get('loss')),
            'val_mae': (checkpoint.get('eval_mae') or 
                       checkpoint.get('val_mae') or 
                       checkpoint.get('mae')),
            'val_rmse': (checkpoint.get('eval_rmse') or 
                        checkpoint.get('val_rmse') or 
                        checkpoint.get('rmse')),
        }
        
        # Only return if we have at least one valid metric
        if any(v is not None for v in metrics.values()):
            return metrics
        
        # If no metrics found, return None
        print(f"    Warning: No validation metrics found in checkpoint")
        return None
    except Exception as e:
        print(f"    Error loading {model_path}: {e}")
        return None


def compare_results(exp_dir):
    """Compare results from all three scenarios"""
    
    print(f"\n{'='*70}")
    print("COMPARING RESULTS")
    print(f"{'='*70}\n")
    
    results = {
        'centralized': {},
        'federated': {},
        'local': {}
    }
    
    # 1. Centralized Training
    print("Loading centralized results...")
    
    # Prefer evaluation metrics (on val.csv) over training metrics
    centralized_eval = f"{exp_dir}/models/centralized/val_metrics.json"
    centralized_model = f"{exp_dir}/models/centralized/best_model.pth"
    
    if os.path.exists(centralized_eval):
        print(f"  Loading evaluation metrics: {centralized_eval}")
        results['centralized']['final'] = load_evaluation_metrics(centralized_eval)
    elif os.path.exists(centralized_model):
        print(f"  Loading from checkpoint: {centralized_model}")
        results['centralized']['final'] = load_model_checkpoint(centralized_model)
    
    if results['centralized'].get('final'):
        m = results['centralized']['final']
        print(f"  ✓ Centralized: Loss={m.get('val_loss', 'N/A')}, "
              f"MAE={m.get('val_mae', 'N/A')}, RMSE={m.get('val_rmse', 'N/A')}")
    
    # 2. Federated Learning
    print("Loading federated results...")
    
    # Prefer evaluation metrics (on val.csv)
    federated_eval = f"{exp_dir}/models/federated/val_metrics.json"
    federated_model = f"{exp_dir}/models/federated/FL_global_model.pt"
    
    if os.path.exists(federated_eval):
        print(f"  Loading evaluation metrics: {federated_eval}")
        results['federated']['final'] = load_evaluation_metrics(federated_eval)
    elif os.path.exists(federated_model):
        print(f"  Trying to load from checkpoint: {federated_model}")
        metrics = load_model_checkpoint(federated_model)
        if metrics and any(v is not None for v in metrics.values()):
            results['federated']['final'] = metrics
        else:
            # Fall back to client outputs
            print(f"  Falling back to client outputs...")
            client_metrics = []
            for i in range(1, 20):
                client_name = f"site-{i}"
                client_output_dir = f"outputs/{client_name}"
                
                if not os.path.exists(client_output_dir):
                    break
                
                checkpoints = list(Path(client_output_dir).glob("model_round_*.pth"))
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
                    metrics = load_model_checkpoint(str(latest))
                    if metrics and any(v is not None for v in metrics.values()):
                        client_metrics.append(metrics)
            
            if client_metrics:
                results['federated']['final'] = {
                    'val_loss': np.mean([m['val_loss'] for m in client_metrics if m.get('val_loss')]),
                    'val_mae': np.mean([m['val_mae'] for m in client_metrics if m.get('val_mae')]),
                    'val_rmse': np.mean([m['val_rmse'] for m in client_metrics if m.get('val_rmse')]),
                }
                print(f"  ✓ Loaded from {len(client_metrics)} clients (averaged)")
    
    if results['federated'].get('final'):
        m = results['federated']['final']
        print(f"  ✓ Federated: Loss={m.get('val_loss', 'N/A')}, "
              f"MAE={m.get('val_mae', 'N/A')}, RMSE={m.get('val_rmse', 'N/A')}")
    
    # 3. Local-Only Training
    print("Loading local-only results...")
    local_clients = []
    for i in range(1, 20):  # Support up to 20 clients
        client_name = f"site-{i}"
        
        # Prefer evaluation metrics
        local_eval = f"{exp_dir}/models/local_{client_name}/val_metrics.json"
        local_model = f"{exp_dir}/models/local_{client_name}/best_model.pth"
        
        if not os.path.exists(local_eval) and not os.path.exists(local_model):
            break
        
        client_data = {'client': client_name}
        
        if os.path.exists(local_eval):
            metrics = load_evaluation_metrics(local_eval)
            if metrics:
                client_data.update(metrics)
        elif os.path.exists(local_model):
            metrics = load_model_checkpoint(local_model)
            if metrics:
                client_data.update(metrics)
        
        if 'val_loss' in client_data:
            local_clients.append(client_data)
            print(f"  ✓ {client_name}: Loss={client_data.get('val_loss', 'N/A')}")
    
    if local_clients:
        results['local']['clients'] = local_clients
        # Average across clients
        valid_clients = [c for c in local_clients if 'val_loss' in c and c['val_loss']]
        if valid_clients:
            results['local']['avg'] = {
                'val_loss': np.mean([c['val_loss'] for c in valid_clients]),
                'val_mae': np.mean([c['val_mae'] for c in valid_clients]),
                'val_rmse': np.mean([c['val_rmse'] for c in valid_clients]),
            }
            print(f"  ✓ Local-only average: Loss={results['local']['avg']['val_loss']:.4f}")
    
    return results


def create_comparison_plots(results, exp_dir):
    """Create visualization comparing all scenarios"""
    
    plots_dir = f"{exp_dir}/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Final Performance Comparison (Bar Chart)
    print("\nCreating final performance comparison...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['val_loss', 'val_mae', 'val_rmse']
    titles = ['Validation Loss (MSE)', 'Mean Absolute Error', 'Root Mean Squared Error']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        scenarios = []
        values = []
        colors = []
        
        # Centralized
        if 'final' in results['centralized'] and metric in results['centralized']['final']:
            val = results['centralized']['final'][metric]
            if val is not None:
                scenarios.append('Centralized')
                values.append(val)
                colors.append('#2ecc71')
        
        # Federated
        if 'final' in results['federated'] and metric in results['federated']['final']:
            val = results['federated']['final'][metric]
            if val is not None:
                scenarios.append('Federated')
                values.append(val)
                colors.append('#3498db')
        
        # Local-only (average)
        if 'avg' in results['local'] and metric in results['local']['avg']:
            val = results['local']['avg'][metric]
            if val is not None:
                scenarios.append('Local-Only\n(Average)')
                values.append(val)
                colors.append('#e74c3c')
        
        if scenarios and values:
            bars = ax.bar(scenarios, values, color=colors, alpha=0.7, edgecolor='black')
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'{title}\n(Lower is Better)', fontsize=13, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.4f}', ha='center', va='bottom', fontsize=10)
            
            # Highlight best performer
            if values:
                best_idx = np.argmin(values)
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
        else:
            # No data for this metric
            ax.text(0.5, 0.5, f'No data for {title}', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/final_performance_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {plots_dir}/final_performance_comparison.png")
    plt.close()
    
    # Plot 2: Per-Client Performance (if local has multiple clients)
    if 'clients' in results['local']:
        print("\nCreating per-client comparison...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[idx]
            
            # Local-only clients
            if 'clients' in results['local']:
                local_values = [c[metric] for c in results['local']['clients'] if metric in c and c[metric]]
                local_clients = [c['client'] for c in results['local']['clients'] if metric in c and c[metric]]
                if local_values:
                    x_pos = np.arange(len(local_clients))
                    ax.bar(x_pos, local_values, width=0.6, label='Local-Only',
                          color='#e74c3c', alpha=0.7, edgecolor='black')
            
            # Federated global model (horizontal line)
            if 'final' in results['federated'] and metric in results['federated']['final']:
                ax.axhline(y=results['federated']['final'][metric], 
                          color='#3498db', linestyle='--', linewidth=2,
                          label='Federated (Global)', alpha=0.8)
            
            # Centralized baseline (horizontal line)
            if 'final' in results['centralized'] and metric in results['centralized']['final']:
                ax.axhline(y=results['centralized']['final'][metric], 
                          color='#2ecc71', linestyle='--', linewidth=2,
                          label='Centralized', alpha=0.8)
            
            ax.set_ylabel(title, fontsize=11)
            ax.set_xlabel('Client', fontsize=11)
            ax.set_title(f'{title} per Client', fontsize=12, fontweight='bold')
            if local_values:
                ax.set_xticks(x_pos)
                ax.set_xticklabels([f'Site-{i+1}' for i in range(len(x_pos))])
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{plots_dir}/per_client_comparison.png", dpi=300, bbox_inches='tight')
        print(f"  Saved: {plots_dir}/per_client_comparison.png")
        plt.close()
    
    # Plot 3: Summary Table
    print("\nCreating summary table...")
    create_summary_table(results, plots_dir)
    
    print(f"\n✓ All plots saved to: {plots_dir}/")


def create_summary_table(results, plots_dir):
    """Create a summary table comparing all scenarios"""
    
    summary_data = []
    
    # Centralized
    if 'final' in results['centralized']:
        summary_data.append({
            'Scenario': 'Centralized',
            'Description': 'Single model on full dataset',
            'Val Loss': f"{results['centralized']['final'].get('val_loss', 'N/A'):.4f}" if results['centralized']['final'].get('val_loss') else 'N/A',
            'Val MAE': f"{results['centralized']['final'].get('val_mae', 'N/A'):.4f}" if results['centralized']['final'].get('val_mae') else 'N/A',
            'Val RMSE': f"{results['centralized']['final'].get('val_rmse', 'N/A'):.4f}" if results['centralized']['final'].get('val_rmse') else 'N/A',
        })
    
    # Federated
    if 'final' in results['federated']:
        summary_data.append({
            'Scenario': 'Federated',
            'Description': 'Collaborative (global model)',
            'Val Loss': f"{results['federated']['final'].get('val_loss', 'N/A'):.4f}" if results['federated']['final'].get('val_loss') else 'N/A',
            'Val MAE': f"{results['federated']['final'].get('val_mae', 'N/A'):.4f}" if results['federated']['final'].get('val_mae') else 'N/A',
            'Val RMSE': f"{results['federated']['final'].get('val_rmse', 'N/A'):.4f}" if results['federated']['final'].get('val_rmse') else 'N/A',
        })
    
    # Local-only
    if 'avg' in results['local']:
        n_clients = len(results['local'].get('clients', []))
        summary_data.append({
            'Scenario': 'Local-Only',
            'Description': f'Independent ({n_clients} clients avg)',
            'Val Loss': f"{results['local']['avg'].get('val_loss', 'N/A'):.4f}" if results['local']['avg'].get('val_loss') else 'N/A',
            'Val MAE': f"{results['local']['avg'].get('val_mae', 'N/A'):.4f}" if results['local']['avg'].get('val_mae') else 'N/A',
            'Val RMSE': f"{results['local']['avg'].get('val_rmse', 'N/A'):.4f}" if results['local']['avg'].get('val_rmse') else 'N/A',
        })
    
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        df.to_csv(f"{plots_dir}/summary.csv", index=False)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colWidths=[0.15, 0.35, 0.15, 0.15, 0.15])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Style rows
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#ecf0f1')
        
        plt.savefig(f"{plots_dir}/summary_table.png", dpi=300, bbox_inches='tight')
        print(f"  Saved: {plots_dir}/summary_table.png")
        print(f"  Saved: {plots_dir}/summary.csv")
        plt.close()
        
        # Print to console
        print(f"\n{'='*70}")
        print("SUMMARY RESULTS")
        print(f"{'='*70}")
        print(df.to_string(index=False))
        print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Compare training scenarios')
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='Experiment directory containing results')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.exp_dir):
        print(f"Error: Experiment directory not found: {args.exp_dir}")
        return 1
    
    # Load and compare results
    results = compare_results(args.exp_dir)
    
    # Create visualizations
    create_comparison_plots(results, args.exp_dir)
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"\nView results in: {args.exp_dir}/plots/")
    print(f"  - final_performance_comparison.png")
    print(f"  - per_client_comparison.png")
    print(f"  - summary_table.png")
    print(f"  - summary.csv")
    print(f"{'='*70}\n")
    
    return 0


if __name__ == "__main__":
    exit(main())

