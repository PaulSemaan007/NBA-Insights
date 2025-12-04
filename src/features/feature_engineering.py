"""
Feature Engineering
Creates all the features we use for prediction from the cleaned data
"""

import os
import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

PROCESSED_DATA_DIR = os.path.join('data', 'processed')

def load_processed_data():
    """Load the cleaned data files"""
    print("Loading processed data...")

    player_logs = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'player_game_logs_cleaned.csv'),
        parse_dates=['GAME_DATE']
    )
    team_stats = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'team_stats_cleaned.csv')
    )
    games = pd.read_csv(
        os.path.join(PROCESSED_DATA_DIR, 'games_cleaned.csv'),
        parse_dates=['GAME_DATE']
    )

    print(f"Loaded {len(player_logs)} player game logs")

    return player_logs, team_stats, games

def calculate_rolling_averages(df, windows=[5, 10, 20]):
    """
    Calculate rolling averages for different window sizes
    This helps capture recent trends vs longer-term performance
    """
    print(f"\nCalculating rolling averages (windows: {windows})...")

    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    # Stats we want rolling averages for
    stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
             'MIN', 'EFFICIENCY', 'FANTASY_PTS']

    for window in windows:
        print(f"  Processing {window}-game rolling average...")

        for stat in stats:
            if stat in df.columns:
                col_name = f'{stat}_ROLL_{window}'
                df[col_name] = df.groupby('PLAYER_ID')[stat].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

    print(f"Added rolling averages for {len(stats)} stats across {len(windows)} windows")

    return df

def calculate_rest_days(df):
    """Calculate how many days off a player had before each game"""
    print("\nCalculating rest days...")

    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    # Days since last game
    df['PREV_GAME_DATE'] = df.groupby('PLAYER_ID')['GAME_DATE'].shift(1)
    df['DAYS_REST'] = (df['GAME_DATE'] - df['PREV_GAME_DATE']).dt.days

    # Default to 3 for first game of season
    df['DAYS_REST'] = df['DAYS_REST'].fillna(3)

    # Cap at 10 (anything more is basically the same - well rested)
    df['DAYS_REST'] = df['DAYS_REST'].clip(upper=10)

    # Binary features for common scenarios
    df['BACK_TO_BACK'] = (df['DAYS_REST'] <= 1).astype(int)
    df['WELL_RESTED'] = (df['DAYS_REST'] >= 3).astype(int)

    print("Added rest day features")

    return df

def calculate_recent_form(df):
    """Figure out if a player is hot or cold lately"""
    print("\nCalculating recent form indicators...")

    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    # Last 3 games average
    df['PTS_L3'] = df.groupby('PLAYER_ID')['PTS'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    # Season average so far
    df['PTS_SEASON_AVG'] = df.groupby('PLAYER_ID')['PTS'].transform(lambda x: x.expanding().mean())

    # Hot = scoring 10% above average, cold = 10% below
    df['HOT_STREAK'] = (df['PTS_L3'] > df['PTS_SEASON_AVG'] * 1.1).astype(int)
    df['COLD_STREAK'] = (df['PTS_L3'] < df['PTS_SEASON_AVG'] * 0.9).astype(int)

    print("Added form indicators")

    return df

def add_opponent_defense_rating(df, team_stats):
    """Add how good the opponent's defense is"""
    print("\nAdding opponent defensive ratings...")

    df = df.copy()

    if 'DEF_RATING' in team_stats.columns:
        # Create lookup by team and season
        if 'TEAM_ABBREVIATION' in team_stats.columns:
            team_def_map = team_stats.set_index(['TEAM_ABBREVIATION', 'SEASON'])['DEF_RATING'].to_dict()
        else:
            team_def_map = team_stats.set_index(['TEAM_NAME', 'SEASON'])['DEF_RATING'].to_dict()

        def get_opp_def_rating(row):
            key = (row['OPPONENT'], row['SEASON'])
            return team_def_map.get(key, 110)  # 110 is roughly league average

        df['OPP_DEF_RATING'] = df.apply(get_opp_def_rating, axis=1)

        # Normalize it (lower is better defense, which is harder to score on)
        df['OPP_DEF_RATING_NORM'] = (df['OPP_DEF_RATING'] - df['OPP_DEF_RATING'].mean()) / df['OPP_DEF_RATING'].std()

        print("Added opponent defensive ratings")
    else:
        print("DEF_RATING not available in team stats")
        df['OPP_DEF_RATING'] = 110
        df['OPP_DEF_RATING_NORM'] = 0

    return df

def calculate_home_away_splits(df):
    """Some players perform better at home vs on the road"""
    print("\nCalculating home/away performance splits...")

    df = df.copy()

    # Running averages for home and away games separately
    df['HOME_PTS_AVG'] = df[df['HOME_AWAY'] == 'home'].groupby('PLAYER_ID')['PTS'].transform(lambda x: x.expanding().mean())
    df['AWAY_PTS_AVG'] = df[df['HOME_AWAY'] == 'away'].groupby('PLAYER_ID')['PTS'].transform(lambda x: x.expanding().mean())

    # Fill missing with overall average
    overall_avg = df.groupby('PLAYER_ID')['PTS'].transform(lambda x: x.expanding().mean())
    df['HOME_PTS_AVG'] = df['HOME_PTS_AVG'].fillna(overall_avg)
    df['AWAY_PTS_AVG'] = df['AWAY_PTS_AVG'].fillna(overall_avg)

    # Simple binary
    df['IS_HOME'] = (df['HOME_AWAY'] == 'home').astype(int)

    print("Added home/away features")

    return df

def calculate_matchup_history(df):
    """How does this player usually do against this specific team?"""
    print("\nCalculating matchup history...")

    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    # Average points vs this opponent
    df['PTS_VS_OPP_AVG'] = df.groupby(['PLAYER_ID', 'OPPONENT'])['PTS'].transform(lambda x: x.expanding().mean())

    # How many times they've played this team
    df['GAMES_VS_OPP'] = df.groupby(['PLAYER_ID', 'OPPONENT']).cumcount()

    print("Added matchup history features")

    return df

def create_target_variables(df):
    """Create the target (what we're trying to predict - next game stats)"""
    print("\nCreating target variables...")

    df = df.copy()
    df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

    # Shift to get NEXT game's stats as the target
    target_stats = ['PTS', 'REB', 'AST', 'FANTASY_PTS']

    for stat in target_stats:
        if stat in df.columns:
            df[f'TARGET_{stat}'] = df.groupby('PLAYER_ID')[stat].shift(-1)

    # Remove rows where we don't have a target (last game for each player)
    df_with_targets = df.dropna(subset=[f'TARGET_{target_stats[0]}'])

    print(f"Created targets for {len(target_stats)} statistics")
    print(f"Rows with targets: {len(df_with_targets)} (removed {len(df) - len(df_with_targets)} last games)")

    return df_with_targets

def select_features_for_modeling(df):
    """Pick which features to actually use in the model"""
    print("\nSelecting features for modeling...")

    # Group features by type
    rolling_features = [col for col in df.columns if 'ROLL_' in col]
    rest_features = ['DAYS_REST', 'BACK_TO_BACK', 'WELL_RESTED']
    form_features = ['HOT_STREAK', 'COLD_STREAK', 'PTS_SEASON_AVG']
    opponent_features = ['OPP_DEF_RATING_NORM', 'IS_HOME']
    matchup_features = ['PTS_VS_OPP_AVG', 'GAMES_VS_OPP']
    base_stats = ['MIN', 'EFFICIENCY', 'TS_PCT', 'PTS_PER_MIN']

    # Combine all
    feature_columns = (
        rolling_features +
        rest_features +
        form_features +
        opponent_features +
        matchup_features +
        base_stats
    )

    # Only keep columns that actually exist
    feature_columns = [col for col in feature_columns if col in df.columns]

    target_columns = [col for col in df.columns if col.startswith('TARGET_')]
    metadata_columns = ['PLAYER_ID', 'PLAYER_NAME', 'GAME_DATE', 'SEASON', 'OPPONENT', 'HOME_AWAY']

    print(f"\nFeature summary:")
    print(f"  Rolling features: {len(rolling_features)}")
    print(f"  Rest features: {len(rest_features)}")
    print(f"  Form features: {len(form_features)}")
    print(f"  Opponent features: {len(opponent_features)}")
    print(f"  Matchup features: {len(matchup_features)}")
    print(f"  Base stats: {len(base_stats)}")
    print(f"  Total features: {len(feature_columns)}")
    print(f"  Target variables: {len(target_columns)}")

    all_columns = metadata_columns + feature_columns + target_columns
    all_columns = [col for col in all_columns if col in df.columns]

    df_final = df[all_columns].copy()

    return df_final, feature_columns, target_columns

def save_engineered_data(df, feature_columns, target_columns):
    """Save the final dataset and feature lists"""
    print("\nSaving engineered data...")

    output_file = os.path.join(PROCESSED_DATA_DIR, 'player_features_final.csv')
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")

    # Save feature list so we know what columns to use later
    feature_list_file = os.path.join(PROCESSED_DATA_DIR, 'feature_columns.txt')
    with open(feature_list_file, 'w') as f:
        f.write('\n'.join(feature_columns))
    print(f"Saved: {feature_list_file}")

    target_list_file = os.path.join(PROCESSED_DATA_DIR, 'target_columns.txt')
    with open(target_list_file, 'w') as f:
        f.write('\n'.join(target_columns))
    print(f"Saved: {target_list_file}")

def main():
    """Run the feature engineering pipeline"""
    print(f"\n{'#'*60}")
    print("NBA FEATURE ENGINEERING PIPELINE")
    print(f"{'#'*60}\n")

    player_logs, team_stats, games = load_processed_data()

    # Build up features one by one
    df = calculate_rolling_averages(player_logs, windows=[5, 10, 20])
    df = calculate_rest_days(df)
    df = calculate_recent_form(df)
    df = add_opponent_defense_rating(df, team_stats)
    df = calculate_home_away_splits(df)
    df = calculate_matchup_history(df)
    df = create_target_variables(df)

    df_final, feature_columns, target_columns = select_features_for_modeling(df)
    save_engineered_data(df_final, feature_columns, target_columns)

    print(f"\n{'='*60}")
    print("FEATURE ENGINEERING COMPLETE!")
    print(f"{'='*60}")
    print(f"Final dataset: {len(df_final)} samples")
    print(f"Features: {len(feature_columns)}")
    print(f"Targets: {len(target_columns)}")
    print(f"\nNext step: Train models")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
