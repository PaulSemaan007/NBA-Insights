"""
Transformer Model for NBA Player Performance Prediction
Uses attention mechanisms to process game sequences and contextual features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    Adds position information to the input embeddings
    """

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)

        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class PlayerTransformer(nn.Module):
    """
    Transformer model for NBA player performance prediction

    Uses self-attention to identify relationships between:
    - Recent game performances
    - Contextual factors (opponent, rest, home/away)
    - Historical patterns
    """

    def __init__(
        self,
        input_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        output_size=4
    ):
        """
        Args:
            input_size: Number of input features
            d_model: Dimension of model (must be divisible by nhead)
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            output_size: Number of output predictions
        """
        super(PlayerTransformer, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size

        # Input projection layer
        self.input_projection = nn.Linear(input_size, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(d_model // 4, output_size)

        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.bn2 = nn.BatchNorm1d(d_model // 4)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # Project input to d_model dimensions
        # x shape: (batch_size, seq_len, d_model)
        x = self.input_projection(x)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Pass through transformer encoder
        # encoded shape: (batch_size, seq_len, d_model)
        encoded = self.transformer_encoder(x)

        # Use the last time step or average pooling
        # Here we'll use mean pooling across sequence
        # pooled shape: (batch_size, d_model)
        pooled = encoded.mean(dim=1)

        # Fully connected layers
        out = self.fc1(pooled)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        # Output layer
        out = self.fc3(out)

        return out

    def predict(self, x):
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()


class PlayerTransformerClassToken(nn.Module):
    """
    Transformer with CLS token (similar to BERT)
    Uses a special classification token for aggregating sequence information
    """

    def __init__(
        self,
        input_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        dim_feedforward=512,
        dropout=0.1,
        output_size=4
    ):
        super(PlayerTransformerClassToken, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size

        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)

        # CLS token (learnable)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )

        # Output layers
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(d_model // 4, output_size)

        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.bn2 = nn.BatchNorm1d(d_model // 4)

    def forward(self, x):
        """
        Forward pass with CLS token

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Predictions of shape (batch_size, output_size)
        """
        batch_size = x.size(0)

        # Project input
        x = self.input_projection(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        encoded = self.transformer_encoder(x)

        # Extract CLS token representation
        cls_output = encoded[:, 0, :]

        # Fully connected layers
        out = self.fc1(cls_output)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        out = self.fc3(out)

        return out

    def predict(self, x):
        """Make predictions"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()


def get_model_config():
    """
    Get default transformer model configuration

    Returns:
        Dictionary with model hyperparameters
    """
    return {
        'input_size': None,  # Will be set based on features
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'output_size': 4,  # PTS, REB, AST, FANTASY_PTS
        'learning_rate': 0.0001,
        'batch_size': 64,
        'num_epochs': 100,
        'sequence_length': 10,
        'early_stopping_patience': 10
    }


if __name__ == "__main__":
    # Test the model
    print("Testing Transformer model architecture...")

    # Create dummy input
    batch_size = 32
    seq_length = 10
    input_size = 50

    dummy_input = torch.randn(batch_size, seq_length, input_size)

    # Test basic Transformer
    print("\n1. Testing PlayerTransformer...")
    model = PlayerTransformer(
        input_size=input_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3
    )
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test Transformer with CLS token
    print("\n2. Testing PlayerTransformerClassToken...")
    model_cls = PlayerTransformerClassToken(
        input_size=input_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3
    )
    output_cls = model_cls(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output_cls.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")

    print("\nModel test successful!")
