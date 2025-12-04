"""
LSTM Model for NBA Player Prediction
We use this to look at a player's recent games and predict their next performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlayerLSTM(nn.Module):
    """
    Basic LSTM for predicting player stats

    Takes in the last N games worth of features and outputs predicted
    points, rebounds, assists, and fantasy points
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
        input_size: how many features we have per game
        hidden_size: LSTM hidden dimension (128 worked well for us)
        num_layers: stacking 2 LSTM layers
        dropout: 0.3 to prevent overfitting
        output_size: 4 outputs (PTS, REB, AST, FANTASY_PTS)
        """
        super(PlayerLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # The actual LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dense layers after the LSTM
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)

        # Batch norm helps training stability
        self.bn1 = nn.BatchNorm1d(hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 4)

    def forward(self, x):
        """
        x shape: (batch_size, sequence_length, input_size)
        returns: (batch_size, output_size)
        """
        # Run through LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Grab the final hidden state
        last_hidden = h_n[-1]

        # Pass through dense layers
        out = self.fc1(last_hidden)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout2(out)

        # No activation on output since this is regression
        out = self.fc3(out)

        return out

    def predict(self, x):
        """Convenience method for inference"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()


class PlayerLSTMWithAttention(nn.Module):
    """
    LSTM with attention - this is the one we actually use

    The attention mechanism lets the model figure out which games
    in the sequence are most important for the prediction
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

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Simple attention - just a linear layer that scores each timestep
        self.attention = nn.Linear(hidden_size, 1)

        # Same dense layers as before
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

        Instead of just using the last hidden state, we compute
        a weighted average of all hidden states based on attention scores
        """
        # LSTM outputs hidden states for every timestep
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Score each timestep
        attention_scores = self.attention(lstm_out)

        # Softmax to get weights that sum to 1
        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)

        # Dense layers
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
        """Inference mode"""
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()


def get_model_config():
    """Default hyperparameters that worked for us"""
    return {
        'input_size': None,  # depends on feature count
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'output_size': 4,
        'learning_rate': 0.001,
        'batch_size': 64,
        'num_epochs': 100,
        'sequence_length': 10,  # look at last 10 games
        'early_stopping_patience': 10
    }


if __name__ == "__main__":
    # Quick test to make sure everything works
    print("Testing LSTM models...")

    batch_size = 32
    seq_length = 10
    input_size = 50

    dummy_input = torch.randn(batch_size, seq_length, input_size)

    print("\n1. Basic LSTM:")
    model = PlayerLSTM(input_size=input_size, hidden_size=128, num_layers=2)
    output = model(dummy_input)
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n2. LSTM with Attention:")
    model_attn = PlayerLSTMWithAttention(input_size=input_size, hidden_size=128, num_layers=2)
    output_attn = model_attn(dummy_input)
    print(f"   Input: {dummy_input.shape}")
    print(f"   Output: {output_attn.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model_attn.parameters()):,}")

    print("\nAll good!")
