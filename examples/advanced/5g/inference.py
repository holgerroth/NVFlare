import torch
import pandas as pd
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from model import TransformerRegressor
from data import preprocess_data


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
    model = TransformerRegressor(
        input_dim=checkpoint['input_dim'],
        **checkpoint['model_config']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
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
    X = preprocess_data(df.copy(), scaler, label_encoders)
    print(f"Preprocessed data shape: {X.shape}")
    
    # Run inference
    print("\nRunning inference...")
    predictions = run_inference(model, X, device, batch_size=args.batch_size)
    print(f"Generated {len(predictions)} predictions")
    
    # Add predictions to dataframe
    df['Predicted_Throughput'] = predictions
    
    # Save predictions
    output_file = os.path.join(args.output_dir, 'predictions.csv')
    df.to_csv(output_file, index=False)
    print(f"\nPredictions saved to {output_file}")
    
    # Calculate and display metrics if ground truth is available
    if has_ground_truth:
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        
        metrics = calculate_metrics(predictions, actuals)
        
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
            plot_predictions_comparison(predictions, actuals, plot_file)
    
    # Display prediction statistics
    print("\n" + "="*60)
    print("PREDICTION STATISTICS")
    print("="*60)
    print(f"Mean:      {np.mean(predictions):.4f} Mbps")
    print(f"Median:    {np.median(predictions):.4f} Mbps")
    print(f"Std Dev:   {np.std(predictions):.4f} Mbps")
    print(f"Min:       {np.min(predictions):.4f} Mbps")
    print(f"Max:       {np.max(predictions):.4f} Mbps")
    print(f"Q1 (25%):  {np.percentile(predictions, 25):.4f} Mbps")
    print(f"Q3 (75%):  {np.percentile(predictions, 75):.4f} Mbps")
    
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

