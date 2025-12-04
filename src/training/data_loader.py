"""
Data Loader for NBA Player Performance Models
Prepares sequential data for LSTM and Transformer models
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

PROCESSED_DATA_DIR = os.path.join('data', 'processed')

class NBASequenceDataset(Dataset):
    """
    PyTorch Dataset for sequential NBA player data

    Creates sequences of last N games for each prediction
    """

    def __init__(self, sequences, features, targets):
        """
        Args:
            sequences: numpy array of shape (n_samples, seq_len, n_features)
            features: numpy array of features for current game (optional, can be None)
            targets: numpy array of target values
        """
        self.sequences = torch.FloatTensor(sequences)
        self.features = torch.FloatTensor(features) if features is not None else None
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_feature_data():
    """Load engineered feature data"""
    print("Loading feature data...")

    df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'player_features_final.csv'),
        parse_dates=['GAME_DATE']
    )

    # Load feature and target column lists
    with open(os.path.join(PROCESSED_DATA_DIR, 'feature_columns.txt'), 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]

    with open(os.path.join(PROCESSED_DATA_DIR, 'target_columns.txt'), 'r') as f:
        target_columns = [line.strip() for line in f.readlines()]

    print(f"Loaded {len(df)} samples")
    print(f"Features: {len(feature_columns)}")
    print(f"Targets: {len(target_columns)}")

    return df, feature_columns, target_columns


def create_sequences(df, feature_columns, target_columns, sequence_length=10):
    """
    Create sequences of games for each player

    Args:
        df: Dataframe with features and targets
        feature_columns: List of feature column names
        target_columns: List of target column names
        sequence_length: Number of previous games to include

    Returns:
        sequences, targets, player_ids, game_dates
    """
    print(f"\nCreating sequences (sequence_length={sequence_length})...")

    # Sort by player and date
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    sequences = []
    targets = []
    player_ids = []
    game_dates = []

    # Group by player
    for player_id, player_df in df.groupby('PLAYER_ID'):
        player_df = player_df.reset_index(drop=True)

        # Need at least sequence_length games
        if len(player_df) < sequence_length:
            continue

        # Create sequences
        for i in range(sequence_length, len(player_df)):
            # Get sequence of previous games
            seq = player_df.iloc[i-sequence_length:i][feature_columns].values.astype(np.float64)

            # Get target for next game
            target = player_df.iloc[i][target_columns].values.astype(np.float64)

            # Check for NaN values
            if not np.isnan(seq).any() and not np.isnan(target).any():
                sequences.append(seq)
                targets.append(target)
                player_ids.append(player_id)
                game_dates.append(player_df.iloc[i]['GAME_DATE'])

    sequences = np.array(sequences)
    targets = np.array(targets)

    print(f"Created {len(sequences)} sequences")
    print(f"Sequence shape: {sequences.shape}")
    print(f"Target shape: {targets.shape}")

    return sequences, targets, player_ids, game_dates


def normalize_data(sequences_train, sequences_val, sequences_test):
    """
    Normalize sequences using StandardScaler

    Args:
        sequences_train: Training sequences
        sequences_val: Validation sequences
        sequences_test: Test sequences

    Returns:
        Normalized sequences and fitted scaler
    """
    print("\nNormalizing data...")

    # Reshape for scaling (combine batch and sequence dimensions)
    n_train, seq_len, n_features = sequences_train.shape

    # Fit scaler on training data
    scaler = StandardScaler()
    sequences_train_flat = sequences_train.reshape(-1, n_features)
    scaler.fit(sequences_train_flat)

    # Transform all datasets
    sequences_train_norm = scaler.transform(sequences_train_flat).reshape(n_train, seq_len, n_features)

    if sequences_val is not None:
        n_val = sequences_val.shape[0]
        sequences_val_flat = sequences_val.reshape(-1, n_features)
        sequences_val_norm = scaler.transform(sequences_val_flat).reshape(n_val, seq_len, n_features)
    else:
        sequences_val_norm = None

    if sequences_test is not None:
        n_test = sequences_test.shape[0]
        sequences_test_flat = sequences_test.reshape(-1, n_features)
        sequences_test_norm = scaler.transform(sequences_test_flat).reshape(n_test, seq_len, n_features)
    else:
        sequences_test_norm = None

    print("Data normalized")

    return sequences_train_norm, sequences_val_norm, sequences_test_norm, scaler


def prepare_data_loaders(sequence_length=10, batch_size=64, val_split=0.15, test_split=0.15):
    """
    Prepare train, validation, and test data loaders

    Args:
        sequence_length: Number of previous games in sequence
        batch_size: Batch size for data loaders
        val_split: Validation split ratio
        test_split: Test split ratio

    Returns:
        train_loader, val_loader, test_loader, scaler, feature_columns, target_columns
    """
    print(f"\n{'='*60}")
    print("PREPARING DATA LOADERS")
    print(f"{'='*60}")

    # Load data
    df, feature_columns, target_columns = load_feature_data()

    # Create sequences
    sequences, targets, player_ids, game_dates = create_sequences(
        df, feature_columns, target_columns, sequence_length
    )

    # Split data (stratified by player to avoid data leakage)
    # First split: train + val vs test
    train_val_idx, test_idx = train_test_split(
        range(len(sequences)),
        test_size=test_split,
        random_state=42
    )

    # Second split: train vs val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_split / (1 - test_split),
        random_state=42
    )

    # Create splits
    sequences_train = sequences[train_idx]
    targets_train = targets[train_idx]

    sequences_val = sequences[val_idx]
    targets_val = targets[val_idx]

    sequences_test = sequences[test_idx]
    targets_test = targets[test_idx]

    print(f"\nData splits:")
    print(f"  Training: {len(sequences_train)} samples")
    print(f"  Validation: {len(sequences_val)} samples")
    print(f"  Test: {len(sequences_test)} samples")

    # Normalize data
    sequences_train_norm, sequences_val_norm, sequences_test_norm, scaler = normalize_data(
        sequences_train, sequences_val, sequences_test
    )

    # Create datasets
    train_dataset = NBASequenceDataset(sequences_train_norm, None, targets_train)
    val_dataset = NBASequenceDataset(sequences_val_norm, None, targets_val)
    test_dataset = NBASequenceDataset(sequences_test_norm, None, targets_test)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nData loaders created (batch_size={batch_size})")
    print(f"{'='*60}\n")

    return train_loader, val_loader, test_loader, scaler, feature_columns, target_columns


def save_scaler(scaler, filename='scaler.pkl'):
    """Save fitted scaler to disk"""
    scaler_path = os.path.join('models', 'saved', filename)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Scaler saved to {scaler_path}")


def load_scaler(filename='scaler.pkl'):
    """Load fitted scaler from disk"""
    scaler_path = os.path.join('models', 'saved', filename)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"Scaler loaded from {scaler_path}")
    return scaler


if __name__ == "__main__":
    # Test data loader
    print("Testing data loader...")

    train_loader, val_loader, test_loader, scaler, feature_cols, target_cols = prepare_data_loaders(
        sequence_length=10,
        batch_size=32
    )

    # Test batch
    sequences_batch, targets_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Sequences: {sequences_batch.shape}")
    print(f"  Targets: {targets_batch.shape}")

    print("\nFeatures:", len(feature_cols))
    print("Targets:", target_cols)

    print("\nData loader test successful!")
