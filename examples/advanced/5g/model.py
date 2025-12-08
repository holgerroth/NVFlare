"""
Model architecture for Lumos5G throughput prediction
"""
import torch
import torch.nn as nn


class TransformerTimeSeriesRegressor(nn.Module):
    """Transformer-based model for time series throughput prediction"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=3, 
                 dim_feedforward=512, dropout=0.1):
        """
        Args:
            input_dim: Number of input features per timestep
            d_model: Dimension of the model
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
        """
        super(TransformerTimeSeriesRegressor, self).__init__()
        
        # Store initialization parameters as member variables for NVFlare persistence
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout
        
        # Input embedding layer
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model))  # Max sequence length 100
        
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
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Embed input
        x = self.input_embedding(x)  # (batch_size, sequence_length, d_model)
        
        # Add positional encoding
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        
        # Pass through transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, sequence_length, d_model)
        
        # Use the last timestep's representation for prediction
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Pass through output layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x.squeeze(-1)  # (batch_size,)

