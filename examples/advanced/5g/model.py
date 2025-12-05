"""
Model architecture for Lumos5G throughput prediction
"""
import torch
import torch.nn as nn


class TransformerRegressor(nn.Module):
    """Transformer-based model for throughput prediction"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1):
        """
        Args:
            input_dim: Number of input features
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerRegressor, self).__init__()
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers
        self.fc1 = nn.Linear(d_model, dim_feedforward // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_feedforward // 2, 1)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        
        # Embed input
        x = self.input_embedding(x)  # (batch_size, d_model)
        
        # Add batch dimension for sequence (treating each sample as a sequence of length 1)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Add positional encoding
        x = x + self.pos_embedding
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, 1, d_model)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch_size, d_model)
        
        # Pass through output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)  # (batch_size,)

