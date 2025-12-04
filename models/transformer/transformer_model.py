"""
Transformer Model for NBA Player Prediction
Uses attention to figure out which games in the sequence matter most
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    Adds position info to the input since transformers don't have
    any built-in sense of order (unlike LSTMs)
    """

    def __init__(self, d_model, max_len=100):
        super(PositionalEncoding, self).__init__()

        # Standard sinusoidal encoding from "Attention is All You Need"
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class PlayerTransformer(nn.Module):
    """
    Basic transformer for player prediction

    Uses self-attention to look at relationships between games
    in the sequence. Mean pooling at the end to aggregate.
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
        input_size: number of features per game
        d_model: transformer hidden dimension
        nhead: number of attention heads
        num_encoder_layers: how many transformer layers to stack
        """
        super(PlayerTransformer, self).__init__()

        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size

        # Project input features to d_model dimensions
        self.input_projection = nn.Linear(input_size, d_model)

        self.pos_encoder = PositionalEncoding(d_model)

        # PyTorch's built-in transformer encoder
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

        # Output head
        self.fc1 = nn.Linear(d_model, d_model // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_model // 2, d_model // 4)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(d_model // 4, output_size)

        self.bn1 = nn.BatchNorm1d(d_model // 2)
        self.bn2 = nn.BatchNorm1d(d_model // 4)

    def forward(self, x):
        """
        x shape: (batch, seq_len, input_size)
        returns: (batch, output_size)
        """
        # Project to d_model
        x = self.input_projection(x)

        # Add position encoding
        x = self.pos_encoder(x)

        # Run through transformer
        encoded = self.transformer_encoder(x)

        # Average pool across sequence
        pooled = encoded.mean(dim=1)

        # Dense layers
        out = self.fc1(pooled)
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
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()


class PlayerTransformerClassToken(nn.Module):
    """
    Transformer with CLS token - this is what we actually use

    Similar to BERT, we prepend a special "classification" token
    to the sequence and use its output as the representation.
    This tends to work better than mean pooling.
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

        self.input_projection = nn.Linear(input_size, d_model)

        # Learnable CLS token (randomly initialized)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_encoder = PositionalEncoding(d_model)

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

        We stick the CLS token at the front, let attention do its thing,
        then use the CLS token's output as our sequence representation
        """
        batch_size = x.size(0)

        # Project input
        x = self.input_projection(x)

        # Prepend CLS token to each sequence in the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Position encoding
        x = self.pos_encoder(x)

        # Transformer
        encoded = self.transformer_encoder(x)

        # Grab the CLS token output (first position)
        cls_output = encoded[:, 0, :]

        # Dense layers
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
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()


def get_model_config():
    """Default hyperparameters"""
    return {
        'input_size': None,
        'd_model': 128,
        'nhead': 8,
        'num_encoder_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1,
        'output_size': 4,
        'learning_rate': 0.0001,  # transformers like smaller lr
        'batch_size': 64,
        'num_epochs': 100,
        'sequence_length': 10,
        'early_stopping_patience': 10
    }


if __name__ == "__main__":
    # Quick test
    print("Testing Transformer models...")

    batch_size = 32
    seq_length = 10
    input_size = 50

    dummy_input = torch.randn(batch_size, seq_length, input_size)

    print("\n1. Basic Transformer:")
    model = PlayerTransformer(
        input_size=input_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3
    )
    output = model(dummy_input)
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n2. Transformer with CLS token:")
    model_cls = PlayerTransformerClassToken(
        input_size=input_size,
        d_model=128,
        nhead=8,
        num_encoder_layers=3
    )
    output_cls = model_cls(dummy_input)
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output_cls.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_cls.parameters()):,}")

    print("\nAll good!")
