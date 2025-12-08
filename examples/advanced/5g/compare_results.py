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
    
    # Plot 4: Time Series Comparison
    create_timeseries_comparison(results, exp_dir, plots_dir)
    
    print(f"\n✓ All plots saved to: {plots_dir}/")


def create_timeseries_comparison(results, exp_dir, plots_dir, max_samples=500):
    """
    Create a time series comparison plot showing predictions from all three models
    on the same validation data.
    
    Note: Each model may have been trained with different preprocessors/feature sets,
    so we need to load the appropriate dataset for each model.
    """
    print("\nCreating time series comparison plot...")
    
    from model import TransformerTimeSeriesRegressor
    from data import Lumos5GTimeSeriesDataset
    from torch.utils.data import DataLoader
    import pickle
    
    # Check if val.csv exists
    val_csv = Path(exp_dir).parent.parent / 'val.csv'
    if not val_csv.exists():
        val_csv = Path('val.csv')
    
    if not val_csv.exists():
        print(f"  ⚠️  val.csv not found, skipping timeseries comparison")
        return
    
    print(f"  Loading validation data from {val_csv}...")
    
    # Load shared preprocessors (for federated and local models)
    config_dir = Path('federated_data')
    if not config_dir.exists():
        config_dir = Path(exp_dir) / 'federated_data'
    
    scaler_path = config_dir / 'scaler.pkl'
    encoders_path = config_dir / 'label_encoders.pkl'
    
    if not (scaler_path.exists() and encoders_path.exists()):
        print(f"  ⚠️  Preprocessors not found, skipping timeseries comparison")
        return
    
    with open(scaler_path, 'rb') as f:
        shared_scaler = pickle.load(f)
    with open(encoders_path, 'rb') as f:
        shared_label_encoders = pickle.load(f)
    
    # Function to get predictions from a model
    def get_predictions(model_path, checkpoint_type='regular'):
        if not Path(model_path).exists():
            print(f"    Model not found: {model_path}")
            return None, None
        
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # Determine which preprocessors to use
            # Try to use model's own preprocessors first, fallback to shared
            scaler = checkpoint.get('scaler', shared_scaler)
            label_encoders = checkpoint.get('label_encoders', shared_label_encoders)
            
            # If model doesn't have preprocessors and it's federated, must use shared
            if checkpoint_type == 'federated':
                scaler = shared_scaler
                label_encoders = shared_label_encoders
            
            # Create dataset with appropriate preprocessors
            dataset = Lumos5GTimeSeriesDataset(
                val_csv,
                scaler=scaler,
                label_encoders=label_encoders,
                fit_transform=False,
                sequence_length=10,
                prediction_horizon=1
            )
            
            # Get actual values (same across all models with same preprocessors)
            actuals = []
            for i in range(len(dataset)):
                _, y = dataset[i]
                actuals.append(y.item())
            
            # Extract model config and state dict
            if checkpoint_type == 'federated':
                model_state_dict = checkpoint['model']
                # Infer input_dim from the state dict
                if 'input_embedding.weight' in model_state_dict:
                    input_dim = model_state_dict['input_embedding.weight'].shape[1]
                else:
                    input_dim = 26
                model_config = {
                    'd_model': 128,
                    'nhead': 8,
                    'num_layers': 3,
                    'dim_feedforward': 512,
                    'dropout': 0.1
                }
            else:
                model_config = checkpoint.get('model_config', {})
                input_dim = checkpoint.get('input_dim')
                
                # If input_dim not in checkpoint, try to infer from state dict
                if input_dim is None:
                    state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
                    if 'input_embedding.weight' in state_dict:
                        input_dim = state_dict['input_embedding.weight'].shape[1]
                    else:
                        input_dim = 26
                
                model_state_dict = checkpoint.get('model_state_dict', checkpoint.get('model'))
            
            print(f"    Model input_dim: {input_dim}, Dataset input_dim: {dataset[0][0].shape[-1]}")
            
            # Create model
            model = TransformerTimeSeriesRegressor(
                input_dim=input_dim,
                d_model=model_config.get('d_model', 128),
                nhead=model_config.get('nhead', 8),
                num_layers=model_config.get('num_layers', 3),
                dim_feedforward=model_config.get('dim_feedforward', 512),
                dropout=model_config.get('dropout', 0.1)
            )
            model.load_state_dict(model_state_dict)
            model.to(device)
            model.eval()
            
            # Get predictions
            predictions = []
            with torch.no_grad():
                for i in range(len(dataset)):
                    X, _ = dataset[i]
                    X = X.unsqueeze(0).to(device)
                    pred = model(X)
                    predictions.append(pred.item())
            
            return predictions, actuals
            
        except Exception as e:
            print(f"    Error loading model {model_path}: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    # Get predictions from all models
    models_dir = Path(exp_dir) / 'models'
    
    print("  Getting predictions from centralized model...")
    centralized_preds, actuals_cent = get_predictions(models_dir / 'centralized' / 'best_model.pth', 'regular')
    
    print("  Getting predictions from federated model...")
    federated_preds, actuals_fed = get_predictions(models_dir / 'federated' / 'FL_global_model.pt', 'federated')
    
    # For local, use the first client as an example
    print("  Getting predictions from local model...")
    local_preds = None
    actuals_local = None
    local_dirs = sorted((models_dir).glob('local_site-*'))
    if local_dirs:
        local_preds, actuals_local = get_predictions(local_dirs[0] / 'best_model.pth', 'local')
    
    # Use whichever actuals are available (they should all be the same)
    actuals = actuals_cent or actuals_fed or actuals_local
    if actuals is None:
        print("  ⚠️  No predictions generated, skipping timeseries comparison")
        return
    
    # Limit samples for plotting
    if len(actuals) > max_samples:
        sample_indices = np.linspace(0, len(actuals)-1, max_samples, dtype=int)
    else:
        sample_indices = np.arange(len(actuals))
    
    actuals_plot = [actuals[i] for i in sample_indices]
    
    # Sample predictions
    if centralized_preds:
        centralized_preds = [centralized_preds[i] for i in sample_indices]
    if federated_preds:
        federated_preds = [federated_preds[i] for i in sample_indices]
    if local_preds:
        local_preds = [local_preds[i] for i in sample_indices]
    
    # Create the plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    time_idx = np.arange(len(actuals_plot))
    
    # Top plot: All predictions vs actual
    axes[0].plot(time_idx, actuals_plot, label='Actual', 
                linewidth=2, alpha=0.9, color='black', linestyle='-')
    
    if centralized_preds:
        axes[0].plot(time_idx, centralized_preds, label='Centralized', 
                    linewidth=1.5, alpha=0.7, color='#2ecc71')
    
    if federated_preds:
        axes[0].plot(time_idx, federated_preds, label='Federated', 
                    linewidth=1.5, alpha=0.7, color='#3498db')
    
    if local_preds:
        axes[0].plot(time_idx, local_preds, label='Local-Only (Site-1)', 
                    linewidth=1.5, alpha=0.7, color='#e74c3c', linestyle='--')
    
    axes[0].set_xlabel('Time Step', fontsize=12)
    axes[0].set_ylabel('Throughput (Mbps)', fontsize=12)
    axes[0].set_title('Time Series Comparison: Actual vs Model Predictions', 
                     fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Bottom plot: Prediction errors over time
    if centralized_preds:
        error_cent = np.array(centralized_preds) - np.array(actuals_plot)
        axes[1].plot(time_idx, error_cent, label='Centralized', 
                    linewidth=1.5, alpha=0.7, color='#2ecc71')
        mae_cent = np.mean(np.abs(error_cent))
        rmse_cent = np.sqrt(np.mean(error_cent**2))
    
    if federated_preds:
        error_fed = np.array(federated_preds) - np.array(actuals_plot)
        axes[1].plot(time_idx, error_fed, label='Federated', 
                    linewidth=1.5, alpha=0.7, color='#3498db')
        mae_fed = np.mean(np.abs(error_fed))
        rmse_fed = np.sqrt(np.mean(error_fed**2))
    
    if local_preds:
        error_local = np.array(local_preds) - np.array(actuals_plot)
        axes[1].plot(time_idx, error_local, label='Local-Only', 
                    linewidth=1.5, alpha=0.7, color='#e74c3c', linestyle='--')
        mae_local = np.mean(np.abs(error_local))
        rmse_local = np.sqrt(np.mean(error_local**2))
    
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    axes[1].set_xlabel('Time Step', fontsize=12)
    axes[1].set_ylabel('Prediction Error (Mbps)', fontsize=12)
    axes[1].set_title('Prediction Errors Over Time', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11, loc='best')
    axes[1].grid(True, alpha=0.3)
    
    # Add statistics as text
    stats_text = ""
    if centralized_preds:
        stats_text += f"Centralized: MAE={mae_cent:.2f}, RMSE={rmse_cent:.2f}\n"
    if federated_preds:
        stats_text += f"Federated: MAE={mae_fed:.2f}, RMSE={rmse_fed:.2f}\n"
    if local_preds:
        stats_text += f"Local-Only: MAE={mae_local:.2f}, RMSE={rmse_local:.2f}"
    
    if stats_text:
        axes[1].text(0.02, 0.98, stats_text.strip(),
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=10, family='monospace')
    
    plt.tight_layout()
    save_path = f"{plots_dir}/timeseries_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")
    
    # Create zoomed-in version (time steps 200-300)
    print("  Creating zoomed-in view (steps 200-300)...")
    if len(time_idx) > 300:
        zoom_start = 200
        zoom_end = 300
        
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Top plot: All predictions vs actual (zoomed)
        axes[0].plot(time_idx[zoom_start:zoom_end], actuals_plot[zoom_start:zoom_end], 
                    label='Actual', linewidth=2.5, alpha=0.9, color='black', 
                    linestyle='-', marker='o', markersize=4)
        
        if centralized_preds:
            axes[0].plot(time_idx[zoom_start:zoom_end], centralized_preds[zoom_start:zoom_end], 
                        label='Centralized', linewidth=2, alpha=0.8, color='#2ecc71',
                        marker='s', markersize=3)
        
        if federated_preds:
            axes[0].plot(time_idx[zoom_start:zoom_end], federated_preds[zoom_start:zoom_end], 
                        label='Federated', linewidth=2, alpha=0.8, color='#3498db',
                        marker='^', markersize=3)
        
        if local_preds:
            axes[0].plot(time_idx[zoom_start:zoom_end], local_preds[zoom_start:zoom_end], 
                        label='Local-Only (Site-1)', linewidth=2, alpha=0.8, 
                        color='#e74c3c', linestyle='--', marker='d', markersize=3)
        
        axes[0].set_xlabel('Time Step', fontsize=12)
        axes[0].set_ylabel('Throughput (Mbps)', fontsize=12)
        axes[0].set_title(f'Time Series Comparison (Zoomed: Steps {zoom_start}-{zoom_end})', 
                         fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11, loc='best')
        axes[0].grid(True, alpha=0.3)
        
        # Bottom plot: Prediction errors over time (zoomed)
        if centralized_preds:
            error_cent_zoom = np.array(centralized_preds[zoom_start:zoom_end]) - np.array(actuals_plot[zoom_start:zoom_end])
            axes[1].plot(time_idx[zoom_start:zoom_end], error_cent_zoom, 
                        label='Centralized', linewidth=2, alpha=0.8, color='#2ecc71',
                        marker='s', markersize=3)
        
        if federated_preds:
            error_fed_zoom = np.array(federated_preds[zoom_start:zoom_end]) - np.array(actuals_plot[zoom_start:zoom_end])
            axes[1].plot(time_idx[zoom_start:zoom_end], error_fed_zoom, 
                        label='Federated', linewidth=2, alpha=0.8, color='#3498db',
                        marker='^', markersize=3)
        
        if local_preds:
            error_local_zoom = np.array(local_preds[zoom_start:zoom_end]) - np.array(actuals_plot[zoom_start:zoom_end])
            axes[1].plot(time_idx[zoom_start:zoom_end], error_local_zoom, 
                        label='Local-Only', linewidth=2, alpha=0.8, 
                        color='#e74c3c', linestyle='--', marker='d', markersize=3)
        
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        axes[1].set_xlabel('Time Step', fontsize=12)
        axes[1].set_ylabel('Prediction Error (Mbps)', fontsize=12)
        axes[1].set_title('Prediction Errors (Zoomed)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11, loc='best')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics for zoomed region
        stats_text_zoom = ""
        if centralized_preds:
            mae_cent_zoom = np.mean(np.abs(error_cent_zoom))
            rmse_cent_zoom = np.sqrt(np.mean(error_cent_zoom**2))
            stats_text_zoom += f"Centralized: MAE={mae_cent_zoom:.2f}, RMSE={rmse_cent_zoom:.2f}\n"
        if federated_preds:
            mae_fed_zoom = np.mean(np.abs(error_fed_zoom))
            rmse_fed_zoom = np.sqrt(np.mean(error_fed_zoom**2))
            stats_text_zoom += f"Federated: MAE={mae_fed_zoom:.2f}, RMSE={rmse_fed_zoom:.2f}\n"
        if local_preds:
            mae_local_zoom = np.mean(np.abs(error_local_zoom))
            rmse_local_zoom = np.sqrt(np.mean(error_local_zoom**2))
            stats_text_zoom += f"Local-Only: MAE={mae_local_zoom:.2f}, RMSE={rmse_local_zoom:.2f}"
        
        if stats_text_zoom:
            axes[1].text(0.02, 0.98, stats_text_zoom.strip(),
                        transform=axes[1].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=10, family='monospace')
        
        plt.tight_layout()
        zoom_save_path = f"{plots_dir}/timeseries_comparison_zoomed.png"
        plt.savefig(zoom_save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {zoom_save_path}")
    else:
        print(f"  Not enough data points for zoom (need > 300, have {len(time_idx)})")


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

