"""
Sample Data Generator
Creates fake but realistic NBA data for demo purposes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# So we get the same "random" data every time
np.random.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.join('data', 'raw')

# All 30 NBA teams
NBA_TEAMS = [
    {'id': 1610612737, 'abbrev': 'ATL', 'name': 'Atlanta Hawks'},
    {'id': 1610612738, 'abbrev': 'BOS', 'name': 'Boston Celtics'},
    {'id': 1610612751, 'abbrev': 'BKN', 'name': 'Brooklyn Nets'},
    {'id': 1610612766, 'abbrev': 'CHA', 'name': 'Charlotte Hornets'},
    {'id': 1610612741, 'abbrev': 'CHI', 'name': 'Chicago Bulls'},
    {'id': 1610612739, 'abbrev': 'CLE', 'name': 'Cleveland Cavaliers'},
    {'id': 1610612742, 'abbrev': 'DAL', 'name': 'Dallas Mavericks'},
    {'id': 1610612743, 'abbrev': 'DEN', 'name': 'Denver Nuggets'},
    {'id': 1610612765, 'abbrev': 'DET', 'name': 'Detroit Pistons'},
    {'id': 1610612744, 'abbrev': 'GSW', 'name': 'Golden State Warriors'},
    {'id': 1610612745, 'abbrev': 'HOU', 'name': 'Houston Rockets'},
    {'id': 1610612754, 'abbrev': 'IND', 'name': 'Indiana Pacers'},
    {'id': 1610612746, 'abbrev': 'LAC', 'name': 'LA Clippers'},
    {'id': 1610612747, 'abbrev': 'LAL', 'name': 'Los Angeles Lakers'},
    {'id': 1610612763, 'abbrev': 'MEM', 'name': 'Memphis Grizzlies'},
    {'id': 1610612748, 'abbrev': 'MIA', 'name': 'Miami Heat'},
    {'id': 1610612749, 'abbrev': 'MIL', 'name': 'Milwaukee Bucks'},
    {'id': 1610612750, 'abbrev': 'MIN', 'name': 'Minnesota Timberwolves'},
    {'id': 1610612740, 'abbrev': 'NOP', 'name': 'New Orleans Pelicans'},
    {'id': 1610612752, 'abbrev': 'NYK', 'name': 'New York Knicks'},
    {'id': 1610612760, 'abbrev': 'OKC', 'name': 'Oklahoma City Thunder'},
    {'id': 1610612753, 'abbrev': 'ORL', 'name': 'Orlando Magic'},
    {'id': 1610612755, 'abbrev': 'PHI', 'name': 'Philadelphia 76ers'},
    {'id': 1610612756, 'abbrev': 'PHX', 'name': 'Phoenix Suns'},
    {'id': 1610612757, 'abbrev': 'POR', 'name': 'Portland Trail Blazers'},
    {'id': 1610612758, 'abbrev': 'SAC', 'name': 'Sacramento Kings'},
    {'id': 1610612759, 'abbrev': 'SAS', 'name': 'San Antonio Spurs'},
    {'id': 1610612761, 'abbrev': 'TOR', 'name': 'Toronto Raptors'},
    {'id': 1610612762, 'abbrev': 'UTA', 'name': 'Utah Jazz'},
    {'id': 1610612764, 'abbrev': 'WAS', 'name': 'Washington Wizards'},
]

# Made up players across different skill tiers
SAMPLE_PLAYERS = [
    # Stars - the big scorers
    {'id': 201566, 'name': 'Marcus Johnson', 'team': 'LAL', 'position': 'SF', 'tier': 'star'},
    {'id': 201567, 'name': 'DeShawn Williams', 'team': 'BOS', 'position': 'SG', 'tier': 'star'},
    {'id': 201568, 'name': 'Tyler Anderson', 'team': 'MIL', 'position': 'PF', 'tier': 'star'},
    {'id': 201569, 'name': 'Kevin Thompson', 'team': 'PHX', 'position': 'PG', 'tier': 'star'},
    {'id': 201570, 'name': 'Jordan Mitchell', 'team': 'DEN', 'position': 'C', 'tier': 'star'},

    # Solid starters
    {'id': 201571, 'name': 'Chris Davis', 'team': 'GSW', 'position': 'SG', 'tier': 'starter'},
    {'id': 201572, 'name': 'Marcus Brown', 'team': 'MIA', 'position': 'SF', 'tier': 'starter'},
    {'id': 201573, 'name': 'Anthony Harris', 'team': 'DAL', 'position': 'PG', 'tier': 'starter'},
    {'id': 201574, 'name': 'Brandon Clark', 'team': 'PHI', 'position': 'PF', 'tier': 'starter'},
    {'id': 201575, 'name': 'Jason Lee', 'team': 'BKN', 'position': 'C', 'tier': 'starter'},
    {'id': 201576, 'name': 'Michael Scott', 'team': 'ATL', 'position': 'SG', 'tier': 'starter'},
    {'id': 201577, 'name': 'David Wilson', 'team': 'CHI', 'position': 'SF', 'tier': 'starter'},
    {'id': 201578, 'name': 'Robert Taylor', 'team': 'CLE', 'position': 'PG', 'tier': 'starter'},
    {'id': 201579, 'name': 'James Moore', 'team': 'TOR', 'position': 'PF', 'tier': 'starter'},
    {'id': 201580, 'name': 'William Jackson', 'team': 'MIN', 'position': 'C', 'tier': 'starter'},

    # Role players
    {'id': 201581, 'name': 'Daniel White', 'team': 'OKC', 'position': 'SG', 'tier': 'rotation'},
    {'id': 201582, 'name': 'Matthew Martin', 'team': 'SAC', 'position': 'SF', 'tier': 'rotation'},
    {'id': 201583, 'name': 'Andrew Garcia', 'team': 'POR', 'position': 'PG', 'tier': 'rotation'},
    {'id': 201584, 'name': 'Joshua Martinez', 'team': 'IND', 'position': 'PF', 'tier': 'rotation'},
    {'id': 201585, 'name': 'Ryan Robinson', 'team': 'NOP', 'position': 'C', 'tier': 'rotation'},
    {'id': 201586, 'name': 'Nicholas Clark', 'team': 'HOU', 'position': 'SG', 'tier': 'rotation'},
    {'id': 201587, 'name': 'Tyler Lewis', 'team': 'ORL', 'position': 'SF', 'tier': 'rotation'},
    {'id': 201588, 'name': 'Austin Walker', 'team': 'DET', 'position': 'PG', 'tier': 'rotation'},
    {'id': 201589, 'name': 'Brandon Hall', 'team': 'CHA', 'position': 'PF', 'tier': 'rotation'},
    {'id': 201590, 'name': 'Justin Allen', 'team': 'WAS', 'position': 'C', 'tier': 'rotation'},

    # Bench guys
    {'id': 201591, 'name': 'Eric Young', 'team': 'UTA', 'position': 'SG', 'tier': 'bench'},
    {'id': 201592, 'name': 'Steven King', 'team': 'MEM', 'position': 'SF', 'tier': 'bench'},
    {'id': 201593, 'name': 'Patrick Wright', 'team': 'SAS', 'position': 'PG', 'tier': 'bench'},
    {'id': 201594, 'name': 'Alexander Lopez', 'team': 'NYK', 'position': 'PF', 'tier': 'bench'},
    {'id': 201595, 'name': 'Samuel Hill', 'team': 'LAC', 'position': 'C', 'tier': 'bench'},
    {'id': 201596, 'name': 'Benjamin Scott', 'team': 'LAL', 'position': 'SG', 'tier': 'bench'},
    {'id': 201597, 'name': 'Henry Adams', 'team': 'BOS', 'position': 'SF', 'tier': 'bench'},
    {'id': 201598, 'name': 'Jack Nelson', 'team': 'MIL', 'position': 'PG', 'tier': 'bench'},
    {'id': 201599, 'name': 'Luke Carter', 'team': 'PHX', 'position': 'PF', 'tier': 'bench'},
    {'id': 201600, 'name': 'Owen Mitchell', 'team': 'DEN', 'position': 'C', 'tier': 'bench'},
]

# Average stats by player tier - (mean, std_dev)
TIER_STATS = {
    'star': {
        'PTS': (26, 4), 'REB': (7, 2.5), 'AST': (6, 2), 'STL': (1.3, 0.5),
        'BLK': (0.8, 0.5), 'TOV': (3, 1), 'MIN': (34, 3),
        'FG_PCT': (0.48, 0.05), 'FG3_PCT': (0.37, 0.06), 'FT_PCT': (0.85, 0.05)
    },
    'starter': {
        'PTS': (18, 4), 'REB': (5.5, 2), 'AST': (4, 1.5), 'STL': (1.1, 0.4),
        'BLK': (0.6, 0.4), 'TOV': (2.2, 0.8), 'MIN': (30, 4),
        'FG_PCT': (0.46, 0.05), 'FG3_PCT': (0.36, 0.06), 'FT_PCT': (0.80, 0.07)
    },
    'rotation': {
        'PTS': (13, 3), 'REB': (4, 1.5), 'AST': (2.5, 1), 'STL': (0.9, 0.3),
        'BLK': (0.4, 0.3), 'TOV': (1.5, 0.6), 'MIN': (24, 5),
        'FG_PCT': (0.44, 0.05), 'FG3_PCT': (0.35, 0.07), 'FT_PCT': (0.77, 0.08)
    },
    'bench': {
        'PTS': (8, 3), 'REB': (2.5, 1), 'AST': (1.5, 0.8), 'STL': (0.6, 0.3),
        'BLK': (0.3, 0.2), 'TOV': (1, 0.5), 'MIN': (18, 6),
        'FG_PCT': (0.42, 0.06), 'FG3_PCT': (0.33, 0.08), 'FT_PCT': (0.74, 0.10)
    }
}


def generate_game_stats(player, game_date, opponent, is_home, days_rest, season):
    """Generate a single game's worth of stats for a player"""
    tier = player['tier']
    stats = TIER_STATS[tier]

    # Minutes played (capped at 42)
    min_played = max(5, np.random.normal(stats['MIN'][0], stats['MIN'][1]))
    min_played = min(42, min_played)

    # Adjustments for rest and home/away
    rest_factor = 1.0
    if days_rest == 0:  # back to back - tired
        rest_factor = 0.92
    elif days_rest >= 3:  # well rested
        rest_factor = 1.03

    home_factor = 1.02 if is_home else 0.98

    # Generate the counting stats
    pts = max(0, np.random.normal(stats['PTS'][0], stats['PTS'][1]) * rest_factor * home_factor)
    reb = max(0, np.random.normal(stats['REB'][0], stats['REB'][1]) * rest_factor)
    ast = max(0, np.random.normal(stats['AST'][0], stats['AST'][1]) * rest_factor)
    stl = max(0, np.random.normal(stats['STL'][0], stats['STL'][1]))
    blk = max(0, np.random.normal(stats['BLK'][0], stats['BLK'][1]))
    tov = max(0, np.random.normal(stats['TOV'][0], stats['TOV'][1]))

    # Shooting percentages
    fg_pct = np.clip(np.random.normal(stats['FG_PCT'][0], stats['FG_PCT'][1]), 0.25, 0.70)
    fg3_pct = np.clip(np.random.normal(stats['FG3_PCT'][0], stats['FG3_PCT'][1]), 0.15, 0.55)
    ft_pct = np.clip(np.random.normal(stats['FT_PCT'][0], stats['FT_PCT'][1]), 0.50, 0.95)

    # Work backwards from points to get shot attempts
    fta = int(pts * 0.15 / ft_pct) if ft_pct > 0 else 0
    ftm = int(fta * ft_pct)

    fg3a = int(pts * 0.15 / (3 * fg3_pct)) if fg3_pct > 0 else 0
    fg3m = int(fg3a * fg3_pct)

    remaining_pts = pts - ftm - (fg3m * 3)
    fg2a = int(remaining_pts / (2 * fg_pct)) if fg_pct > 0 else 0
    fg2m = int(fg2a * fg_pct)

    fga = fg2a + fg3a
    fgm = fg2m + fg3m

    actual_fg_pct = fgm / fga if fga > 0 else 0
    actual_fg3_pct = fg3m / fg3a if fg3a > 0 else 0
    actual_ft_pct = ftm / fta if fta > 0 else 0

    actual_pts = (fg2m * 2) + (fg3m * 3) + ftm

    # Plus/minus (somewhat correlated with how well they played)
    plus_minus = int(np.random.normal((actual_pts - stats['PTS'][0]) * 0.5, 10))

    # Build the matchup string
    team_abbrev = player['team']
    if is_home:
        matchup = f"{team_abbrev} vs. {opponent}"
    else:
        matchup = f"{team_abbrev} @ {opponent}"

    # Format minutes as MM:SS
    min_int = int(min_played)
    min_sec = int((min_played - min_int) * 60)
    min_str = f"{min_int}:{min_sec:02d}"

    return {
        'PLAYER_ID': player['id'],
        'PLAYER_NAME': player['name'],
        'TEAM_ID': [t['id'] for t in NBA_TEAMS if t['abbrev'] == player['team']][0],
        'TEAM_ABBREVIATION': player['team'],
        'GAME_DATE': game_date.strftime('%Y-%m-%d'),
        'MATCHUP': matchup,
        'WL': random.choice(['W', 'L']),
        'MIN': min_str,
        'FGM': int(fgm),
        'FGA': int(fga),
        'FG_PCT': round(actual_fg_pct, 3),
        'FG3M': int(fg3m),
        'FG3A': int(fg3a),
        'FG3_PCT': round(actual_fg3_pct, 3),
        'FTM': int(ftm),
        'FTA': int(fta),
        'FT_PCT': round(actual_ft_pct, 3),
        'OREB': int(reb * 0.25),
        'DREB': int(reb * 0.75),
        'REB': int(reb),
        'AST': int(ast),
        'STL': int(stl),
        'BLK': int(blk),
        'TOV': int(tov),
        'PF': random.randint(1, 5),
        'PTS': int(actual_pts),
        'PLUS_MINUS': plus_minus,
        'SEASON': season
    }


def generate_player_game_logs():
    """Generate game logs for all players across multiple seasons"""
    print("Generating player game logs...")

    all_games = []
    seasons = ['2023-24', '2024-25']

    for season in seasons:
        print(f"  Generating season {season}...")

        if season == '2023-24':
            start_date = datetime(2023, 10, 24)
            end_date = datetime(2024, 4, 14)
        else:
            start_date = datetime(2024, 10, 22)
            end_date = datetime(2025, 4, 13)

        games_per_player = 70  # roughly a full season

        for player in SAMPLE_PLAYERS:
            current_date = start_date
            prev_game_date = None
            games_count = 0

            while current_date < end_date and games_count < games_per_player:
                # Skip some games (rest days, minor injuries)
                if random.random() < 0.15:
                    current_date += timedelta(days=1)
                    continue

                if prev_game_date:
                    days_rest = (current_date - prev_game_date).days - 1
                else:
                    days_rest = 3

                # Pick a random opponent
                opponents = [t['abbrev'] for t in NBA_TEAMS if t['abbrev'] != player['team']]
                opponent = random.choice(opponents)

                is_home = random.random() < 0.5

                game = generate_game_stats(player, current_date, opponent, is_home, days_rest, season)
                all_games.append(game)

                prev_game_date = current_date
                games_count += 1

                # Next game in 1-4 days
                current_date += timedelta(days=random.randint(1, 4))

    df = pd.DataFrame(all_games)
    print(f"  Generated {len(df)} player game records")

    return df


def generate_team_stats():
    """Generate team-level stats for each season"""
    print("Generating team stats...")

    all_stats = []
    seasons = ['2023-24', '2024-25']

    for season in seasons:
        for team in NBA_TEAMS:
            wins = random.randint(20, 60)
            losses = 82 - wins

            stats = {
                'TEAM_ID': team['id'],
                'TEAM_NAME': team['name'],
                'TEAM_ABBREVIATION': team['abbrev'],
                'SEASON': season,
                'GP': 82,
                'W': wins,
                'L': losses,
                'W_PCT': round(wins / 82, 3),
                'OFF_RATING': round(np.random.normal(112, 4), 1),
                'DEF_RATING': round(np.random.normal(112, 4), 1),
                'NET_RATING': round(np.random.normal(0, 5), 1),
                'PACE': round(np.random.normal(100, 3), 1),
                'PIE': round(np.random.normal(0.5, 0.05), 3)
            }
            all_stats.append(stats)

    df = pd.DataFrame(all_stats)
    print(f"  Generated {len(df)} team stat records")

    return df


def generate_games_schedule():
    """Generate a schedule of games"""
    print("Generating games schedule...")

    all_games = []
    game_id = 22300001

    seasons = ['2023-24', '2024-25']

    for season in seasons:
        if season == '2023-24':
            start_date = datetime(2023, 10, 24)
            end_date = datetime(2024, 4, 14)
        else:
            start_date = datetime(2024, 10, 22)
            end_date = datetime(2025, 4, 13)

        current_date = start_date

        while current_date < end_date:
            games_today = random.randint(5, 15)

            teams_copy = NBA_TEAMS.copy()
            random.shuffle(teams_copy)

            for i in range(0, min(games_today * 2, len(teams_copy)), 2):
                if i + 1 >= len(teams_copy):
                    break

                home_team = teams_copy[i]
                away_team = teams_copy[i + 1]

                game = {
                    'GAME_ID': game_id,
                    'GAME_DATE': current_date.strftime('%Y-%m-%d'),
                    'HOME_TEAM_ID': home_team['id'],
                    'HOME_TEAM_ABBREVIATION': home_team['abbrev'],
                    'AWAY_TEAM_ID': away_team['id'],
                    'AWAY_TEAM_ABBREVIATION': away_team['abbrev'],
                    'SEASON': season
                }
                all_games.append(game)
                game_id += 1

            current_date += timedelta(days=1)

    df = pd.DataFrame(all_games)
    print(f"  Generated {len(df)} game records")

    return df


def save_data(player_logs, team_stats, games):
    """Save all the generated data to CSV files"""
    print("\nSaving data...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    player_logs.to_csv(os.path.join(OUTPUT_DIR, 'player_game_logs.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/player_game_logs.csv")

    team_stats.to_csv(os.path.join(OUTPUT_DIR, 'team_stats.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/team_stats.csv")

    games.to_csv(os.path.join(OUTPUT_DIR, 'all_games.csv'), index=False)
    print(f"  Saved: {OUTPUT_DIR}/all_games.csv")


def main():
    """Generate all the sample data"""
    print("=" * 60)
    print("NBA SAMPLE DATA GENERATOR")
    print("=" * 60)
    print()

    player_logs = generate_player_game_logs()
    team_stats = generate_team_stats()
    games = generate_games_schedule()

    save_data(player_logs, team_stats, games)

    print()
    print("=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Player Game Logs: {len(player_logs)} records")
    print(f"  - Players: {player_logs['PLAYER_ID'].nunique()}")
    print(f"  - Seasons: {player_logs['SEASON'].nunique()}")
    print(f"Team Stats: {len(team_stats)} records")
    print(f"Games Schedule: {len(games)} records")
    print()
    print("Next step: Run preprocessing pipeline")
    print("  python src/data_pipeline/preprocess.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
