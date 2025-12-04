"""
Data Loader
Turns our player data into sequences that PyTorch can use
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
    PyTorch Dataset that gives us sequences of games

    Each sample is (last N games, target stats for next game)
    """

    def __init__(self, sequences, features, targets):
        """
        sequences: (n_samples, seq_len, n_features) array
        targets: what we're trying to predict
        """
        self.sequences = torch.FloatTensor(sequences)
        self.features = torch.FloatTensor(features) if features is not None else None
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def load_feature_data():
    """Load the engineered features"""
    print("Loading feature data...")

    df = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'player_features_final.csv'),
        parse_dates=['GAME_DATE']
    )

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
    Turn the dataframe into sequences

    For each player, we take sliding windows of their game history
    to create (input sequence, target) pairs
    """
    print(f"\nCreating sequences (sequence_length={sequence_length})...")

    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    sequences = []
    targets = []
    player_ids = []
    game_dates = []

    for player_id, player_df in df.groupby('PLAYER_ID'):
        player_df = player_df.reset_index(drop=True)

        # Skip players with not enough games
        if len(player_df) < sequence_length:
            continue

        # Slide through their games
        for i in range(sequence_length, len(player_df)):
            seq = player_df.iloc[i-sequence_length:i][feature_columns].values.astype(np.float64)
            target = player_df.iloc[i][target_columns].values.astype(np.float64)

            # Skip if there are NaN values
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
    Normalize the data using StandardScaler

    Fit on training data only to avoid data leakage
    """
    print("\nNormalizing data...")

    n_train, seq_len, n_features = sequences_train.shape

    # Fit on training data
    scaler = StandardScaler()
    sequences_train_flat = sequences_train.reshape(-1, n_features)
    scaler.fit(sequences_train_flat)

    # Transform everything
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
    Set up the train/val/test data loaders

    Does 70/15/15 split by default
    """
    print(f"\n{'='*60}")
    print("PREPARING DATA LOADERS")
    print(f"{'='*60}")

    df, feature_columns, target_columns = load_feature_data()

    sequences, targets, player_ids, game_dates = create_sequences(
        df, feature_columns, target_columns, sequence_length
    )

    # Split into train+val and test
    train_val_idx, test_idx = train_test_split(
        range(len(sequences)),
        test_size=test_split,
        random_state=42
    )

    # Split train and val
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_split / (1 - test_split),
        random_state=42
    )

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

    sequences_train_norm, sequences_val_norm, sequences_test_norm, scaler = normalize_data(
        sequences_train, sequences_val, sequences_test
    )

    train_dataset = NBASequenceDataset(sequences_train_norm, None, targets_train)
    val_dataset = NBASequenceDataset(sequences_val_norm, None, targets_val)
    test_dataset = NBASequenceDataset(sequences_test_norm, None, targets_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nData loaders created (batch_size={batch_size})")
    print(f"{'='*60}\n")

    return train_loader, val_loader, test_loader, scaler, feature_columns, target_columns


def save_scaler(scaler, filename='scaler.pkl'):
    """Save the scaler so we can use it for inference"""
    scaler_path = os.path.join('models', 'saved', filename)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Scaler saved to {scaler_path}")


def load_scaler(filename='scaler.pkl'):
    """Load a saved scaler"""
    scaler_path = os.path.join('models', 'saved', filename)

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    print(f"Scaler loaded from {scaler_path}")
    return scaler


if __name__ == "__main__":
    # Quick test
    print("Testing data loader...")

    train_loader, val_loader, test_loader, scaler, feature_cols, target_cols = prepare_data_loaders(
        sequence_length=10,
        batch_size=32
    )

    sequences_batch, targets_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Sequences: {sequences_batch.shape}")
    print(f"  Targets: {targets_batch.shape}")

    print("\nFeatures:", len(feature_cols))
    print("Targets:", target_cols)

    print("\nData loader test successful!")
