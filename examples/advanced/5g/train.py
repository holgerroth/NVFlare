import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import argparse
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import TransformerRegressor
from data import Lumos5GDataset


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for features, targets in tqdm(dataloader, desc="Training"):
        features = features.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0.0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for features, targets in tqdm(dataloader, desc="Evaluating"):
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(targets.cpu().numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate additional metrics
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
    
    return total_loss / len(dataloader), mae, rmse, predictions, actuals


def plot_results(train_losses, val_losses, save_path):
    """Plot training and validation losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss plot saved to {save_path}")


def plot_predictions(predictions, actuals, save_path):
    """Plot predictions vs actual values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(actuals, predictions, alpha=0.5, s=1)
    plt.plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
    plt.xlabel('Actual Throughput')
    plt.ylabel('Predicted Throughput')
    plt.title('Predictions vs Actual Values')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Predictions plot saved to {save_path}")


def main(args):
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    full_dataset = Lumos5GDataset(args.data_path, fit_transform=True)
    
    # Get scaler and label encoders for validation set
    scaler = full_dataset.get_scaler()
    label_encoders = full_dataset.get_label_encoders()
    
    # Split dataset
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Get input dimension
    input_dim = full_dataset.features.shape[1]
    print(f"Input dimension: {input_dim}")
    
    # Create model
    model = TransformerRegressor(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        val_loss, val_mae, val_rmse, _, _ = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'scaler': scaler,
                'label_encoders': label_encoders,
                'input_dim': input_dim,
                'model_config': {
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_layers': args.num_layers,
                    'dim_feedforward': args.dim_feedforward,
                    'dropout': args.dropout
                }
            }, os.path.join(args.output_dir, 'best_model.pth'))
            print(f"Saved best model with validation loss: {val_loss:.4f}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\nFinal evaluation on validation set:")
    val_loss, val_mae, val_rmse, predictions, actuals = evaluate(
        model, val_loader, criterion, device
    )
    print(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}, Val RMSE: {val_rmse:.4f}")
    
    # Plot results
    plot_results(train_losses, val_losses, os.path.join(args.output_dir, 'loss_plot.png'))
    plot_predictions(predictions, actuals, os.path.join(args.output_dir, 'predictions_plot.png'))
    
    print(f"\nTraining complete! Best model saved to {args.output_dir}/best_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Transformer model on Lumos5G dataset')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, 
                       default='Lumos5G-v1.0/Lumos5G-v1.0.csv',
                       help='Path to the CSV file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save outputs')
    parser.add_argument('--train_split', type=float, default=0.8,
                       help='Train/validation split ratio')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=128,
                       help='Dimension of the model')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='Number of transformer encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                       help='Dimension of feedforward network')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)

