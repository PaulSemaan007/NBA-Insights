"""
LSTM Model for NBA Player Performance Prediction
Processes sequential game data to predict future performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlayerLSTM(nn.Module):
    """
    LSTM model for predicting NBA player performance

    Architecture:
    - Input: Sequential game features (last N games)
    - LSTM layers: Process temporal patterns
    - Fully connected layers: Generate predictions
    - Output: Predicted statistics (PTS, REB, AST, FANTASY_PTS)
    """

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        output_size=4
    ):
        """
        Args:
            input_size: Number of features per timestep
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            output_size: Number of output predictions (default: 4 for PTS, REB, AST, FANTASY_PTS)
        """
        super(PlayerLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # h_n shape: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last hidden state from the last layer
        # h_n[-1] shape: (batch_size, hidden_size)
        last_hidden = h_n[-1]

        # Fully connected layers with ReLU and dropout
        out = self.fc1(last_hidden)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        # Final output layer (no activation for regression)
        out = self.fc3(out)

        return out

    def predict(self, x):
        """
        Make predictions (wrapper for inference)

        Args:
            x: Input tensor

        Returns:
            Predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()


class PlayerLSTMWithAttention(nn.Module):
    """
    Enhanced LSTM with attention mechanism
    Allows model to focus on most relevant games in sequence
    """

    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        output_size=4
    ):
        super(PlayerLSTMWithAttention, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)

        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)

    def forward(self, x):
        """
        Forward pass with attention

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Predictions of shape (batch_size, output_size)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size)

        # Calculate attention weights
        # attention_scores shape: (batch_size, seq_len, 1)
        attention_scores = self.attention(lstm_out)

        # Softmax across sequence dimension
        # attention_weights shape: (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Apply attention weights to LSTM outputs
        # context_vector shape: (batch_size, hidden_size)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Fully connected layers
        out = self.fc1(context_vector)
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
    Get default model configuration

    Returns:
        Dictionary with model hyperparameters
    """
    return {
        'input_size': None,  # Will be set based on features
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'output_size': 4,  # PTS, REB, AST, FANTASY_PTS
        'learning_rate': 0.001,
        'batch_size': 64,
        'num_epochs': 100,
        'sequence_length': 10,  # Last 10 games
        'early_stopping_patience': 10
    }


if __name__ == "__main__":
    # Test the model
    print("Testing LSTM model architecture...")

    # Create dummy input
    batch_size = 32
    seq_length = 10
    input_size = 50  # Number of features

    dummy_input = torch.randn(batch_size, seq_length, input_size)

    # Test basic LSTM
    print("\n1. Testing PlayerLSTM...")
    model = PlayerLSTM(input_size=input_size, hidden_size=128, num_layers=2)
    output = model(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test LSTM with attention
    print("\n2. Testing PlayerLSTMWithAttention...")
    model_attn = PlayerLSTMWithAttention(input_size=input_size, hidden_size=128, num_layers=2)
    output_attn = model_attn(dummy_input)
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Output shape: {output_attn.shape}")
    print(f"   Model parameters: {sum(p.numel() for p in model_attn.parameters()):,}")

    print("\nModel test successful!")
