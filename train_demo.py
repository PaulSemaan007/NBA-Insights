"""
Training Script
Trains both LSTM and Transformer models on our NBA data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add paths so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'lstm'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models', 'transformer'))

from training.data_loader import prepare_data_loaders, save_scaler
from lstm_model import PlayerLSTMWithAttention
from transformer_model import PlayerTransformerClassToken

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Run one epoch of training"""
    model.train()
    total_loss = 0
    num_batches = 0

    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        outputs = model(sequences)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, data_loader, criterion, device):
    """Evaluate on validation or test set"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in data_loader:
            sequences = sequences.to(device)
            targets = targets.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            num_batches += 1

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = total_loss / num_batches
    predictions = np.vstack(all_predictions)
    targets_arr = np.vstack(all_targets)

    # Calculate MAE and R2 for each target
    mae_per_target = np.mean(np.abs(predictions - targets_arr), axis=0)
    r2_per_target = []
    for i in range(targets_arr.shape[1]):
        ss_res = np.sum((targets_arr[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((targets_arr[:, i] - np.mean(targets_arr[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_per_target.append(r2)

    return avg_loss, predictions, targets_arr, mae_per_target, np.array(r2_per_target)


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, model_name):
    """Train a model with early stopping based on validation loss"""
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()}")
    print(f"{'='*60}\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_r2': []}

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _, _, val_mae, val_r2 = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae.tolist())
        history['val_r2'].append(val_r2.tolist())

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"  Val MAE: PTS={val_mae[0]:.2f}, REB={val_mae[1]:.2f}, AST={val_mae[2]:.2f}, FP={val_mae[3]:.2f}")
        print(f"  Val R2:  PTS={val_r2[0]:.3f}, REB={val_r2[1]:.3f}, AST={val_r2[2]:.3f}, FP={val_r2[3]:.3f}")

        # Save if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_r2': val_r2
            }, f'models/saved/{model_name}_best.pth')
            print(f"  ** New best model saved! **")

        print()

    return history, best_val_loss


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("NBA INSIGHTS - MODEL TRAINING")
    print("=" * 60)

    # Settings
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    SEQUENCE_LENGTH = 10

    # Load data
    print("\nPreparing data...")
    train_loader, val_loader, test_loader, scaler, feature_cols, target_cols = prepare_data_loaders(
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE
    )

    save_scaler(scaler, 'demo_scaler.pkl')

    input_size = len(feature_cols)
    output_size = len(target_cols)

    print(f"\nModel Configuration:")
    print(f"  Input size: {input_size} features")
    print(f"  Output size: {output_size} targets")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Targets: {target_cols}")

    os.makedirs('models/saved', exist_ok=True)

    # Train LSTM
    lstm_model = PlayerLSTMWithAttention(
        input_size=input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        output_size=output_size
    ).to(device)

    print(f"\nLSTM Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    lstm_history, lstm_best_loss = train_model(
        lstm_model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, 'lstm'
    )

    # Train Transformer
    transformer_model = PlayerTransformerClassToken(
        input_size=input_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dropout=0.3,
        output_size=output_size
    ).to(device)

    print(f"\nTransformer Parameters: {sum(p.numel() for p in transformer_model.parameters()):,}")

    transformer_history, transformer_best_loss = train_model(
        transformer_model, train_loader, val_loader, NUM_EPOCHS, LEARNING_RATE, 'transformer'
    )

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)

    criterion = nn.MSELoss()

    # Load best LSTM
    lstm_model.load_state_dict(torch.load('models/saved/lstm_best.pth', weights_only=False)['model_state_dict'])
    lstm_test_loss, lstm_preds, lstm_targets, lstm_mae, lstm_r2 = evaluate(
        lstm_model, test_loader, criterion, device
    )

    # Load best Transformer
    transformer_model.load_state_dict(torch.load('models/saved/transformer_best.pth', weights_only=False)['model_state_dict'])
    transformer_test_loss, transformer_preds, transformer_targets, transformer_mae, transformer_r2 = evaluate(
        transformer_model, test_loader, criterion, device
    )

    print("\n" + "-" * 40)
    print("LSTM Test Results:")
    print("-" * 40)
    print(f"  MSE Loss: {lstm_test_loss:.4f}")
    print(f"  MAE: PTS={lstm_mae[0]:.2f}, REB={lstm_mae[1]:.2f}, AST={lstm_mae[2]:.2f}, FP={lstm_mae[3]:.2f}")
    print(f"  R2:  PTS={lstm_r2[0]:.3f}, REB={lstm_r2[1]:.3f}, AST={lstm_r2[2]:.3f}, FP={lstm_r2[3]:.3f}")

    print("\n" + "-" * 40)
    print("Transformer Test Results:")
    print("-" * 40)
    print(f"  MSE Loss: {transformer_test_loss:.4f}")
    print(f"  MAE: PTS={transformer_mae[0]:.2f}, REB={transformer_mae[1]:.2f}, AST={transformer_mae[2]:.2f}, FP={transformer_mae[3]:.2f}")
    print(f"  R2:  PTS={transformer_r2[0]:.3f}, REB={transformer_r2[1]:.3f}, AST={transformer_r2[2]:.3f}, FP={transformer_r2[3]:.3f}")

    # Save results
    results = {
        'lstm': {
            'test_loss': float(lstm_test_loss),
            'test_mae': lstm_mae.tolist(),
            'test_r2': lstm_r2.tolist(),
            'history': lstm_history
        },
        'transformer': {
            'test_loss': float(transformer_test_loss),
            'test_mae': transformer_mae.tolist(),
            'test_r2': transformer_r2.tolist(),
            'history': transformer_history
        },
        'config': {
            'num_epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'sequence_length': SEQUENCE_LENGTH,
            'input_size': input_size,
            'output_size': output_size,
            'feature_columns': feature_cols,
            'target_columns': target_cols
        }
    }

    with open('models/saved/training_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModels saved to: models/saved/")
    print(f"  - lstm_best.pth")
    print(f"  - transformer_best.pth")
    print(f"  - demo_scaler.pkl")
    print(f"  - training_results.json")
    print("\nNext step: Run Streamlit demo")
    print("  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
