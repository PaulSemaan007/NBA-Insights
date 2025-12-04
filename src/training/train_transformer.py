"""
Training Script for Transformer Model
Trains PlayerTransformer model on NBA player performance data
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.data_loader import prepare_data_loaders, save_scaler
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'transformer'))
from transformer_model import PlayerTransformer, PlayerTransformerClassToken, get_model_config

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for sequences, targets in train_loader:
        sequences = sequences.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on validation/test set"""
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
    targets = np.vstack(all_targets)

    # Calculate MAE for each target
    mae_per_target = np.mean(np.abs(predictions - targets), axis=0)

    return avg_loss, predictions, targets, mae_per_target


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs,
    learning_rate,
    early_stopping_patience,
    model_save_path,
    log_dir
):
    """
    Train the model with early stopping

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Maximum number of epochs
        learning_rate: Learning rate
        early_stopping_patience: Patience for early stopping
        model_save_path: Path to save best model
        log_dir: Directory for tensorboard logs

    Returns:
        Training history
    """
    print(f"\n{'='*60}")
    print("TRAINING TRANSFORMER MODEL")
    print(f"{'='*60}\n")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Learning rate scheduler with warmup
    warmup_epochs = 5
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 ** ((epoch - warmup_epochs) // 10)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Tensorboard writer
    writer = SummaryWriter(log_dir)

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': []
    }

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_preds, val_targets, val_mae = evaluate(
            model, val_loader, criterion, device
        )

        # Update learning rate
        scheduler.step()

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae.tolist())

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        for i, mae in enumerate(val_mae):
            writer.add_scalar(f'MAE/target_{i}', mae, epoch)

        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val MAE: {val_mae}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae
            }, model_save_path)

            print(f"  ** New best model saved! **")
        else:
            patience_counter += 1
            print(f"  Early stopping patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

        print()

    writer.close()

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")

    return history


def main():
    """Main training pipeline"""
    # Configuration
    config = get_model_config()
    config['num_epochs'] = 100
    config['early_stopping_patience'] = 15
    config['learning_rate'] = 0.0001
    config['batch_size'] = 64
    config['sequence_length'] = 10

    # Prepare data
    train_loader, val_loader, test_loader, scaler, feature_cols, target_cols = prepare_data_loaders(
        sequence_length=config['sequence_length'],
        batch_size=config['batch_size']
    )

    # Save scaler
    save_scaler(scaler, 'transformer_scaler.pkl')

    # Model configuration
    input_size = len(feature_cols)
    output_size = len(target_cols)

    print(f"\nModel configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Output size: {output_size}")
    print(f"  d_model: {config['d_model']}")
    print(f"  nhead: {config['nhead']}")
    print(f"  num_encoder_layers: {config['num_encoder_layers']}")
    print(f"  dim_feedforward: {config['dim_feedforward']}")
    print(f"  dropout: {config['dropout']}")

    # Create model (using CLS token version)
    model = PlayerTransformerClassToken(
        input_size=input_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        output_size=output_size
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training paths
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join('models', 'saved', f'transformer_model_{timestamp}.pth')
    log_dir = os.path.join('models', 'saved', 'logs', f'transformer_{timestamp}')

    # Ensure directories exist
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Train model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        learning_rate=config['learning_rate'],
        early_stopping_patience=config['early_stopping_patience'],
        model_save_path=model_save_path,
        log_dir=log_dir
    )

    # Save config and history
    config_path = os.path.join('models', 'saved', f'transformer_config_{timestamp}.json')
    with open(config_path, 'w') as f:
        config['input_size'] = input_size
        config['output_size'] = output_size
        config['feature_columns'] = feature_cols
        config['target_columns'] = target_cols
        json.dump(config, f, indent=2)

    history_path = os.path.join('models', 'saved', f'transformer_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_save_path)['model_state_dict'])
    test_loss, test_preds, test_targets, test_mae = evaluate(
        model, test_loader, nn.MSELoss(), device
    )

    print(f"\nTest Results:")
    print(f"  Test Loss (MSE): {test_loss:.4f}")
    print(f"  Test MAE per target: {test_mae}")
    print(f"\nTarget columns: {target_cols}")

    print(f"\nModel saved to: {model_save_path}")
    print(f"Config saved to: {config_path}")
    print(f"Training logs: {log_dir}")


if __name__ == "__main__":
    main()
