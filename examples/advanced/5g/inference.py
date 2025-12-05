import torch
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import TransformerTimeSeriesRegressor
from data import preprocess_timeseries_data


def run_inference(model, data, device, batch_size=256):
    """
    Run inference on data
    
    Args:
        model: Trained model
        data: Preprocessed numpy array
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns:
        Predictions array
    """
    model.eval()
    predictions = []
    
    # Convert to tensor
    data_tensor = torch.from_numpy(data)
    
    # Process in batches
    num_samples = len(data_tensor)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Running inference"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            batch = data_tensor[start_idx:end_idx].to(device)
            outputs = model(batch)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)


def calculate_metrics(predictions, actuals):
    """Calculate evaluation metrics"""
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    
    # R-squared
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    # MAPE (Mean Absolute Percentage Error)
    # Avoid division by zero
    mask = actuals != 0
    mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask])) * 100 if mask.sum() > 0 else 0
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE': mape
    }


def plot_predictions_comparison(predictions, actuals, save_path):
    """Plot predictions vs actual values with various visualizations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Scatter plot
    axes[0, 0].scatter(actuals, predictions, alpha=0.3, s=1)
    axes[0, 0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Throughput (Mbps)')
    axes[0, 0].set_ylabel('Predicted Throughput (Mbps)')
    axes[0, 0].set_title('Predictions vs Actual Values')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals plot
    residuals = predictions - actuals
    axes[0, 1].scatter(actuals, residuals, alpha=0.3, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Actual Throughput (Mbps)')
    axes[0, 1].set_ylabel('Residuals (Predicted - Actual)')
    axes[0, 1].set_title('Residual Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Distribution comparison
    axes[1, 0].hist(actuals, bins=50, alpha=0.5, label='Actual', density=True)
    axes[1, 0].hist(predictions, bins=50, alpha=0.5, label='Predicted', density=True)
    axes[1, 0].set_xlabel('Throughput (Mbps)')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution
    axes[1, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_xlabel('Prediction Error (Mbps)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison plots saved to {save_path}")


def plot_timeseries(df, save_path, max_samples=500):
    """Plot time series of actual vs predicted throughput
    
    Args:
        df: Dataframe with 'Throughput' and 'Predicted_Throughput' columns
        save_path: Path to save the plot
        max_samples: Maximum number of samples to plot (to avoid overcrowding)
    """
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Get valid predictions
    valid_mask = df['Predicted_Throughput'].notna()
    valid_df = df[valid_mask].copy()
    
    # If we have run_num, plot one run as an example
    if 'run_num' in valid_df.columns:
        # Select a run with good coverage
        run_lengths = valid_df.groupby('run_num').size()
        selected_run = run_lengths.idxmax()  # Run with most predictions
        plot_df = valid_df[valid_df['run_num'] == selected_run].copy()
        
        # Limit samples if needed
        if len(plot_df) > max_samples:
            plot_df = plot_df.iloc[:max_samples]
        
        # Create time index
        plot_df['time_idx'] = range(len(plot_df))
        
        # Top plot: Actual vs Predicted
        axes[0].plot(plot_df['time_idx'], plot_df['Throughput'], 
                    label='Actual', linewidth=1.5, alpha=0.8, color='blue')
        axes[0].plot(plot_df['time_idx'], plot_df['Predicted_Throughput'], 
                    label='Predicted', linewidth=1.5, alpha=0.8, color='red')
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Throughput (Mbps)')
        axes[0].set_title(f'Time Series: Actual vs Predicted Throughput (Run {selected_run})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Bottom plot: Prediction error over time
        plot_df['error'] = plot_df['Predicted_Throughput'] - plot_df['Throughput']
        axes[1].plot(plot_df['time_idx'], plot_df['error'], 
                    linewidth=1, alpha=0.7, color='green')
        axes[1].axhline(y=0, color='black', linestyle='--', lw=1)
        axes[1].fill_between(plot_df['time_idx'], 0, plot_df['error'], 
                            alpha=0.3, color='green')
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Prediction Error (Mbps)')
        axes[1].set_title('Prediction Error Over Time')
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics as text
        mae = np.mean(np.abs(plot_df['error']))
        rmse = np.sqrt(np.mean(plot_df['error']**2))
        axes[1].text(0.02, 0.98, f'MAE: {mae:.2f} Mbps\nRMSE: {rmse:.2f} Mbps',
                    transform=axes[1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        # If no run_num, just plot first max_samples
        plot_df = valid_df.iloc[:max_samples].copy()
        plot_df['time_idx'] = range(len(plot_df))
        
        axes[0].plot(plot_df['time_idx'], plot_df['Throughput'], 
                    label='Actual', linewidth=1.5, alpha=0.8)
        axes[0].plot(plot_df['time_idx'], plot_df['Predicted_Throughput'], 
                    label='Predicted', linewidth=1.5, alpha=0.8)
        axes[0].set_xlabel('Time Step')
        axes[0].set_ylabel('Throughput (Mbps)')
        axes[0].set_title('Time Series: Actual vs Predicted Throughput')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Error plot
        plot_df['error'] = plot_df['Predicted_Throughput'] - plot_df['Throughput']
        axes[1].plot(plot_df['time_idx'], plot_df['error'], linewidth=1, alpha=0.7)
        axes[1].axhline(y=0, color='black', linestyle='--', lw=1)
        axes[1].set_xlabel('Time Step')
        axes[1].set_ylabel('Prediction Error (Mbps)')
        axes[1].set_title('Prediction Error Over Time')
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Time series plot saved to {save_path}")


def main(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, weights_only=False)
    
    # Load scaler and label encoders
    scaler = checkpoint['scaler']
    label_encoders = checkpoint['label_encoders']
    
    # Load model
    print("Loading model...")
    sequence_length = checkpoint.get('sequence_length', 10)  # Default to 10 if not found
    prediction_horizon = checkpoint.get('prediction_horizon', 1)
    
    model = TransformerTimeSeriesRegressor(
        input_dim=checkpoint['input_dim'],
        **checkpoint['model_config']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Sequence length: {sequence_length}, Prediction horizon: {prediction_horizon}")
    print(f"Model configuration: {checkpoint['model_config']}")
    
    # Load data
    print(f"\nLoading data from {args.data_path}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} samples")
    
    # Check if we have ground truth
    has_ground_truth = 'Throughput' in df.columns
    if has_ground_truth:
        actuals = df['Throughput'].values
        print("Ground truth available - will compute metrics")
    else:
        print("No ground truth available - will only generate predictions")
    
    # Preprocess data
    print("\nPreprocessing data...")
    X, sequence_indices = preprocess_timeseries_data(df.copy(), scaler, label_encoders, sequence_length)
    print(f"Created {len(X)} sequences with shape: {X.shape}")
    
    # Run inference
    print("\nRunning inference...")
    predictions = run_inference(model, X, device, batch_size=args.batch_size)
    print(f"Generated {len(predictions)} predictions")
    
    # Map predictions back to dataframe
    # Create a new column initialized with NaN
    df['Predicted_Throughput'] = np.nan
    
    # Assign predictions to the corresponding indices
    for i, idx in enumerate(sequence_indices):
        if idx < len(df):
            df.loc[idx, 'Predicted_Throughput'] = predictions[i]
    
    # For evaluation, only use rows where we have predictions
    if has_ground_truth:
        valid_mask = df['Predicted_Throughput'].notna()
        valid_df = df[valid_mask]
        actuals = valid_df['Throughput'].values
        valid_predictions = valid_df['Predicted_Throughput'].values
        
        print(f"\nValid predictions: {len(valid_predictions)} out of {len(df)} total rows")
    
    # Save predictions
    output_file = os.path.join(args.output_dir, 'predictions.csv')
    df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    
    # Calculate and display metrics if ground truth is available
    if has_ground_truth:
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        metrics = calculate_metrics(valid_predictions, actuals)
        
        print(f"MAE (Mean Absolute Error):       {metrics['MAE']:.4f} Mbps")
        print(f"MSE (Mean Squared Error):        {metrics['MSE']:.4f}")
        print(f"RMSE (Root Mean Squared Error):  {metrics['RMSE']:.4f} Mbps")
        print(f"R² Score:                        {metrics['R2']:.4f}")
        print(f"MAPE (Mean Absolute % Error):    {metrics['MAPE']:.2f}%")
        
        # Save metrics to file
        metrics_file = os.path.join(args.output_dir, 'metrics.txt')
        with open(metrics_file, 'w') as f:
            f.write("EVALUATION METRICS\n")
            f.write("="*60 + "\n")
            f.write(f"Valid predictions: {len(valid_predictions)} out of {len(df)} rows\n")
            f.write(f"Sequence length: {sequence_length}, Prediction horizon: {prediction_horizon}\n")
            f.write("="*60 + "\n")
            f.write(f"MAE (Mean Absolute Error):       {metrics['MAE']:.4f} Mbps\n")
            f.write(f"MSE (Mean Squared Error):        {metrics['MSE']:.4f}\n")
            f.write(f"RMSE (Root Mean Squared Error):  {metrics['RMSE']:.4f} Mbps\n")
            f.write(f"R² Score:                        {metrics['R2']:.4f}\n")
            f.write(f"MAPE (Mean Absolute % Error):    {metrics['MAPE']:.2f}%\n")
        print(f"\nMetrics saved to {metrics_file}")
        
        # Generate comparison plots
        if args.plot:
            print("\nGenerating comparison plots...")
            plot_file = os.path.join(args.output_dir, 'inference_comparison.png')
            plot_predictions_comparison(valid_predictions, actuals, plot_file)
            
            # Generate time series plot
            print("Generating time series plot...")
            timeseries_file = os.path.join(args.output_dir, 'timeseries_comparison.png')
            plot_timeseries(df, timeseries_file)
    
    # Display prediction statistics
    print("\n" + "="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    valid_preds = df['Predicted_Throughput'].dropna()
    print(f"Mean:      {np.mean(valid_preds):.4f} Mbps")
    print(f"Median:    {np.median(valid_preds):.4f} Mbps")
    print(f"Std Dev:   {np.std(valid_preds):.4f} Mbps")
    print(f"Min:       {np.min(valid_preds):.4f} Mbps")
    print(f"Max:       {np.max(valid_preds):.4f} Mbps")
    print(f"Q1 (25%):  {np.percentile(valid_preds, 25):.4f} Mbps")
    print(f"Q3 (75%):  {np.percentile(valid_preds, 75):.4f} Mbps")
    
    if has_ground_truth:
        print("\n" + "="*60)
        print("ACTUAL VALUES STATISTICS")
        print("="*60)
        print(f"Mean:      {np.mean(actuals):.4f} Mbps")
        print(f"Median:    {np.median(actuals):.4f} Mbps")
        print(f"Std Dev:   {np.std(actuals):.4f} Mbps")
        print(f"Min:       {np.min(actuals):.4f} Mbps")
        print(f"Max:       {np.max(actuals):.4f} Mbps")
        print(f"Q1 (25%):  {np.percentile(actuals, 25):.4f} Mbps")
        print(f"Q3 (75%):  {np.percentile(actuals, 75):.4f} Mbps")
    
    print("\n" + "="*60)
    print("Inference complete!")
    print("="*60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with trained Transformer model')
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV file for inference')
    
    # Optional arguments
    parser.add_argument('--output_dir', type=str, default='inference_outputs',
                       help='Directory to save outputs (default: inference_outputs)')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size for inference (default: 256)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots (only if ground truth available)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)

