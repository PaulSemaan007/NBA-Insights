"""
Data Preprocessing Pipeline
Cleans raw NBA data and prepares it for feature engineering
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
RAW_DATA_DIR = os.path.join('data', 'raw')
PROCESSED_DATA_DIR = os.path.join('data', 'processed')

def ensure_directory_exists(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_raw_data():
    """Load raw data from CSV files"""
    print("Loading raw data...")

    player_logs = pd.read_csv(os.path.join(RAW_DATA_DIR, 'player_game_logs.csv'))
    team_stats = pd.read_csv(os.path.join(RAW_DATA_DIR, 'team_stats.csv'))
    games = pd.read_csv(os.path.join(RAW_DATA_DIR, 'all_games.csv'))

    print(f"Player game logs: {len(player_logs)} records")
    print(f"Team stats: {len(team_stats)} records")
    print(f"Games: {len(games)} records")

    return player_logs, team_stats, games

def clean_player_game_logs(df):
    """Clean and standardize player game log data"""
    print("\nCleaning player game logs...")

    # Create a copy
    df = df.copy()

    # Standardize column names (NBA API uses 'Player_ID' vs our synthetic 'PLAYER_ID')
    column_mapping = {
        'Player_ID': 'PLAYER_ID',
        'Game_ID': 'GAME_ID',
    }
    df = df.rename(columns=column_mapping)

    # Convert date to datetime
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Sort by player and date
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    # Extract home/away from MATCHUP (e.g., "LAL vs. GSW" or "LAL @ GSW")
    df['HOME_AWAY'] = df['MATCHUP'].apply(lambda x: 'home' if ' vs. ' in x else 'away')

    # Extract opponent team
    df['OPPONENT'] = df['MATCHUP'].apply(lambda x: x.split()[-1])

    # Extract team abbreviation from matchup (e.g., "LAL vs. GSW" -> "LAL")
    df['TEAM_ABBREVIATION'] = df['MATCHUP'].apply(lambda x: x.split()[0])

    # Handle missing values in key statistics
    stat_columns = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'MIN']

    for col in stat_columns:
        if col in df.columns:
            # Fill missing percentages with 0
            if 'PCT' in col:
                df[col] = df[col].fillna(0)
            # Fill missing counting stats with 0
            else:
                df[col] = df[col].fillna(0)

    # Convert MIN from string format (e.g., "35:24") to float (minutes)
    if df['MIN'].dtype == 'object':
        df['MIN'] = df['MIN'].apply(convert_minutes_to_float)

    # Remove games with 0 minutes played (DNP)
    df = df[df['MIN'] > 0]

    # Add season year (extract from SEASON string like "2023-24")
    df['SEASON_YEAR'] = df['SEASON'].apply(lambda x: int(x.split('-')[0]))

    print(f"Cleaned data: {len(df)} records")
    print(f"Players: {df['PLAYER_ID'].nunique()}")
    print(f"Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")

    return df

def convert_minutes_to_float(min_str):
    """Convert minutes from MM:SS format to float"""
    if pd.isna(min_str) or min_str == '' or min_str == 0:
        return 0.0

    if isinstance(min_str, (int, float)):
        return float(min_str)

    try:
        parts = str(min_str).split(':')
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes + (seconds / 60.0)
        else:
            return float(min_str)
    except:
        return 0.0

def clean_team_stats(df):
    """Clean team statistics data"""
    print("\nCleaning team stats...")

    df = df.copy()

    # Extract season year
    df['SEASON_YEAR'] = df['SEASON'].apply(lambda x: int(x.split('-')[0]))

    # Calculate offensive and defensive ratings if not present
    # OFF_RATING = (Points / Possessions) * 100
    # Possessions ≈ FGA + 0.44*FTA - OREB + TOV
    if 'OFF_RATING' not in df.columns:
        possessions = df['FGA'] + 0.44 * df['FTA'] - df['OREB'] + df['TOV']
        df['OFF_RATING'] = (df['PTS'] / possessions) * 100
        # DEF_RATING would need opponent data, so we'll approximate
        # Higher PLUS_MINUS indicates better defense relative to offense
        df['DEF_RATING'] = df['OFF_RATING'] - (df['PLUS_MINUS'] / df['GP'])
        df['NET_RATING'] = df['PLUS_MINUS'] / df['GP']

    # Add team abbreviation mapping from NBA team names
    team_abbrev_map = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BKN',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
        'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL',
        'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA', 'Milwaukee Bucks': 'MIL',
        'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
        'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }

    if 'TEAM_ABBREVIATION' not in df.columns:
        df['TEAM_ABBREVIATION'] = df['TEAM_NAME'].map(team_abbrev_map)

    # Select relevant defensive and offensive stats
    important_cols = [
        'TEAM_ID', 'TEAM_NAME', 'TEAM_ABBREVIATION', 'SEASON', 'SEASON_YEAR', 'GP', 'W', 'L',
        'W_PCT', 'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PTS'
    ]

    # Check which columns exist
    available_cols = [col for col in important_cols if col in df.columns]

    df_clean = df[available_cols].copy()

    print(f"Cleaned team stats: {len(df_clean)} records")

    return df_clean

def clean_games_data(df):
    """Clean games data for schedule and matchup analysis"""
    print("\nCleaning games data...")

    df = df.copy()

    # Convert date
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

    # Sort by date
    df = df.sort_values('GAME_DATE')

    # Calculate days rest for each team
    # This will be done in feature engineering

    print(f"Cleaned games: {len(df)} records")

    return df

def add_basic_features(df):
    """Add basic derived features to player game logs"""
    print("\nAdding basic features...")

    df = df.copy()

    # Calculate efficiency metrics
    # Player Efficiency Rating (simplified version)
    df['EFFICIENCY'] = (
        df['PTS'] + df['REB'] + df['AST'] + df['STL'] + df['BLK']
        - (df['FGA'] - df['FGM']) - (df['FTA'] - df['FTM']) - df['TOV']
    )

    # True Shooting Percentage
    df['TS_PCT'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
    df['TS_PCT'] = df['TS_PCT'].fillna(0)

    # Usage rate approximation (simplified)
    # Actual usage rate requires team totals, this is a simplified version

    # Points per minute
    df['PTS_PER_MIN'] = df['PTS'] / df['MIN'].replace(0, np.nan)
    df['PTS_PER_MIN'] = df['PTS_PER_MIN'].fillna(0)

    # Fantasy points (standard scoring)
    df['FANTASY_PTS'] = (
        df['PTS'] * 1.0 +
        df['REB'] * 1.2 +
        df['AST'] * 1.5 +
        df['STL'] * 3.0 +
        df['BLK'] * 3.0 -
        df['TOV'] * 1.0
    )

    print(f"Added efficiency and fantasy metrics")

    return df

def save_processed_data(player_logs, team_stats, games):
    """Save processed data to CSV files"""
    print("\nSaving processed data...")

    ensure_directory_exists(PROCESSED_DATA_DIR)

    # Save player logs
    player_output = os.path.join(PROCESSED_DATA_DIR, 'player_game_logs_cleaned.csv')
    player_logs.to_csv(player_output, index=False)
    print(f"Saved: {player_output}")

    # Save team stats
    team_output = os.path.join(PROCESSED_DATA_DIR, 'team_stats_cleaned.csv')
    team_stats.to_csv(team_output, index=False)
    print(f"Saved: {team_output}")

    # Save games
    games_output = os.path.join(PROCESSED_DATA_DIR, 'games_cleaned.csv')
    games.to_csv(games_output, index=False)
    print(f"Saved: {games_output}")

def print_data_summary(player_logs):
    """Print summary statistics of the processed data"""
    print(f"\n{'='*60}")
    print("DATA SUMMARY")
    print(f"{'='*60}")

    print(f"\nTotal games: {len(player_logs)}")
    print(f"Total players: {player_logs['PLAYER_ID'].nunique()}")
    print(f"Date range: {player_logs['GAME_DATE'].min()} to {player_logs['GAME_DATE'].max()}")

    print("\nAverage statistics per game:")
    print(f"  Points: {player_logs['PTS'].mean():.2f}")
    print(f"  Rebounds: {player_logs['REB'].mean():.2f}")
    print(f"  Assists: {player_logs['AST'].mean():.2f}")
    print(f"  Minutes: {player_logs['MIN'].mean():.2f}")

    print("\nMissing values:")
    missing = player_logs.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing)
    else:
        print("  No missing values!")

    print(f"{'='*60}\n")

def main():
    """Main preprocessing pipeline"""
    print(f"\n{'#'*60}")
    print("NBA DATA PREPROCESSING PIPELINE")
    print(f"{'#'*60}\n")

    # Load raw data
    player_logs, team_stats, games = load_raw_data()

    # Clean each dataset
    player_logs_clean = clean_player_game_logs(player_logs)
    team_stats_clean = clean_team_stats(team_stats)
    games_clean = clean_games_data(games)

    # Add basic features
    player_logs_final = add_basic_features(player_logs_clean)

    # Save processed data
    save_processed_data(player_logs_final, team_stats_clean, games_clean)

    # Print summary
    print_data_summary(player_logs_final)

    print("PREPROCESSING COMPLETE!")
    print("Next step: Run feature engineering")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()
