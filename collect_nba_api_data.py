#!/usr/bin/env python3
"""
NBA API Data Collection Script
Fetches all necessary data from NBA API and stores it in a structured format.
This ensures we use accurate, live data instead of potentially outdated CSV files.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
import sys
import time

# Add NBA API to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nba_api', 'src'))

try:
    from nba_api.stats.endpoints import (
        leaguegamefinder,
        teamgamelog,
        teamgamelogs,
        playergamelogs,
        leaguedashteamstats,
        scoreboardv3
    )
    from nba_api.stats.static import teams
    from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
    NBA_API_AVAILABLE = True
except ImportError:
    print("Error: NBA API not available. Please ensure nba_api is installed.")
    print("Install with: pip install nba_api")
    sys.exit(1)

# Configuration
DATA_FOLDER = "api_collected_data"
RAW_FOLDER = os.path.join(DATA_FOLDER, "raw")
CLEANED_FOLDER = os.path.join(DATA_FOLDER, "cleaned")

def get_current_season():
    """Determine current NBA season"""
    today = datetime.now()
    if today.month >= 10:
        season = f"{today.year}-{str(today.year + 1)[-2:]}"
    else:
        season = f"{today.year - 1}-{str(today.year)[-2:]}"
    return season

def parse_api_date(date_str):
    """Parse NBA API date format"""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        date_str = str(date_str).strip()
        if ',' in date_str and len(date_str.split()) == 3:
            try:
                return datetime.strptime(date_str, '%b %d, %Y')
            except:
                try:
                    return datetime.strptime(date_str, '%B %d, %Y')
                except:
                    pass
        elif '-' in date_str and len(date_str.split('-')) == 3:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except:
                pass
        elif '/' in date_str and len(date_str.split('/')) == 3:
            try:
                return datetime.strptime(date_str, '%m/%d/%Y')
            except:
                pass
        from dateutil import parser
        return parser.parse(date_str)
    except:
        return None

def create_folders():
    """Create necessary folders for data storage"""
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(RAW_FOLDER, exist_ok=True)
    os.makedirs(CLEANED_FOLDER, exist_ok=True)
    print(f"✓ Created data folders: {DATA_FOLDER}")

def fetch_all_games(season=None, date_to=None):
    """
    Fetch all NBA games using LeagueGameFinder.
    This gets comprehensive game data for all teams.
    """
    print("\n" + "="*60)
    print("Fetching All NBA Games")
    print("="*60)
    
    if season is None:
        season = get_current_season()
    
    if date_to is None:
        date_to = datetime.now()
    else:
        date_to = datetime.strptime(date_to, '%Y-%m-%d') if isinstance(date_to, str) else date_to
    
    date_to_str = date_to.strftime('%Y-%m-%d')
    
    print(f"Season: {season}")
    print(f"Date to: {date_to_str}")
    
    try:
        # Fetch all games for the season up to today
        print("Fetching games from NBA API...")
        games = leaguegamefinder.LeagueGameFinder(
            player_or_team_abbreviation='T',  # 'T' for team
            season_nullable=season,
            date_to_nullable=date_to_str,
            get_request=True
        )
        
        df = games.league_game_finder_results.get_data_frame()
        
        if len(df) > 0:
            # Parse dates
            df['parsed_date'] = df['GAME_DATE'].apply(parse_api_date)
            df = df[df['parsed_date'].notna()].copy()
            df = df[df['parsed_date'] <= date_to].copy()
            df = df.sort_values('parsed_date', ascending=False)
            
            print(f"✓ Retrieved {len(df)} games")
            print(f"  Date range: {df['parsed_date'].min()} to {df['parsed_date'].max()}")
            
            # Save raw data
            raw_file = os.path.join(RAW_FOLDER, f"all_games_{season}.csv")
            df.to_csv(raw_file, index=False)
            print(f"  Saved to: {raw_file}")
            
            return df
        else:
            print("⚠ No games found")
            return None
            
    except Exception as e:
        print(f"✗ Error fetching games: {e}")
        import traceback
        traceback.print_exc()
        return None

def fetch_team_game_logs(season=None, date_to=None):
    """
    Fetch team game logs for all teams.
    This provides team-level statistics for each game.
    """
    print("\n" + "="*60)
    print("Fetching Team Game Logs")
    print("="*60)
    
    if season is None:
        season = get_current_season()
    
    if date_to is None:
        date_to = datetime.now()
    else:
        date_to = datetime.strptime(date_to, '%Y-%m-%d') if isinstance(date_to, str) else date_to
    
    date_to_str = date_to.strftime('%m/%d/%Y')
    
    try:
        print("Fetching team game logs from NBA API...")
        team_logs = teamgamelogs.TeamGameLogs(
            season_nullable=season,
            date_to_nullable=date_to_str,
            get_request=True
        )
        
        df = team_logs.team_game_logs.get_data_frame()
        
        if len(df) > 0:
            # Parse dates
            df['parsed_date'] = df['GAME_DATE'].apply(parse_api_date)
            df = df[df['parsed_date'].notna()].copy()
            df = df[df['parsed_date'] <= date_to].copy()
            df = df.sort_values('parsed_date', ascending=False)
            
            print(f"✓ Retrieved {len(df)} team game logs")
            print(f"  Teams: {df['TEAM_ABBREVIATION'].nunique()}")
            print(f"  Date range: {df['parsed_date'].min()} to {df['parsed_date'].max()}")
            
            # Save raw data
            raw_file = os.path.join(RAW_FOLDER, f"team_game_logs_{season}.csv")
            df.to_csv(raw_file, index=False)
            print(f"  Saved to: {raw_file}")
            
            return df
        else:
            print("⚠ No team game logs found")
            return None
            
    except Exception as e:
        print(f"✗ Error fetching team game logs: {e}")
        import traceback
        traceback.print_exc()
        return None

def fetch_player_game_logs(season=None, date_to=None):
    """
    Fetch player game logs for all players.
    This provides player-level statistics for each game.
    """
    print("\n" + "="*60)
    print("Fetching Player Game Logs")
    print("="*60)
    
    if season is None:
        season = get_current_season()
    
    if date_to is None:
        date_to = datetime.now()
    else:
        date_to = datetime.strptime(date_to, '%Y-%m-%d') if isinstance(date_to, str) else date_to
    
    date_to_str = date_to.strftime('%m/%d/%Y')
    
    try:
        print("Fetching player game logs from NBA API...")
        print("  (This may take a few minutes due to large dataset)...")
        
        player_logs = playergamelogs.PlayerGameLogs(
            season_nullable=season,
            date_to_nullable=date_to_str,
            get_request=True
        )
        
        df = player_logs.player_game_logs.get_data_frame()
        
        if len(df) > 0:
            # Parse dates
            df['parsed_date'] = df['GAME_DATE'].apply(parse_api_date)
            df = df[df['parsed_date'].notna()].copy()
            df = df[df['parsed_date'] <= date_to].copy()
            df = df.sort_values('parsed_date', ascending=False)
            
            print(f"✓ Retrieved {len(df)} player game logs")
            print(f"  Players: {df['PLAYER_NAME'].nunique()}")
            print(f"  Teams: {df['TEAM_ABBREVIATION'].nunique()}")
            print(f"  Date range: {df['parsed_date'].min()} to {df['parsed_date'].max()}")
            
            # Save raw data
            raw_file = os.path.join(RAW_FOLDER, f"player_game_logs_{season}.csv")
            df.to_csv(raw_file, index=False)
            print(f"  Saved to: {raw_file}")
            
            return df
        else:
            print("⚠ No player game logs found")
            return None
            
    except Exception as e:
        print(f"✗ Error fetching player game logs: {e}")
        import traceback
        traceback.print_exc()
        return None

def fetch_team_stats(season=None):
    """Fetch team season statistics"""
    print("\n" + "="*60)
    print("Fetching Team Season Statistics")
    print("="*60)
    
    if season is None:
        season = get_current_season()
    
    try:
        print("Fetching team stats from NBA API...")
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed='PerGame',
            get_request=True
        )
        
        df = team_stats.league_dash_team_stats.get_data_frame()
        
        if len(df) > 0:
            print(f"✓ Retrieved stats for {len(df)} teams")
            
            # Save raw data
            raw_file = os.path.join(RAW_FOLDER, f"team_stats_{season}.csv")
            df.to_csv(raw_file, index=False)
            print(f"  Saved to: {raw_file}")
            
            return df
        else:
            print("⚠ No team stats found")
            return None
            
    except Exception as e:
        print(f"✗ Error fetching team stats: {e}")
        return None

def clean_and_process_data(all_games_df, team_logs_df, player_logs_df, team_stats_df):
    """
    Clean and process all collected data into a unified format.
    Creates game-by-game features similar to the original dataset structure.
    """
    print("\n" + "="*60)
    print("Cleaning and Processing Data")
    print("="*60)
    
    if all_games_df is None or len(all_games_df) == 0:
        print("✗ No game data to process")
        return None
    
    print("Processing game data...")
    
    # Start with all games
    games_processed = []
    
    # Group by game to create matchups
    unique_games = all_games_df.groupby(['GAME_ID', 'GAME_DATE']).first().reset_index()
    
    print(f"Processing {len(unique_games)} unique games...")
    
    for idx, game_info in unique_games.iterrows():
        game_id = game_info['GAME_ID']
        game_date = game_info['GAME_DATE']
        
        # Get both teams' data for this game
        game_teams = all_games_df[all_games_df['GAME_ID'] == game_id]
        
        if len(game_teams) != 2:
            continue  # Skip if we don't have both teams
        
        team1_data = game_teams.iloc[0]
        team2_data = game_teams.iloc[1]
        
        # Get team game log data if available
        team1_log = None
        team2_log = None
        if team_logs_df is not None and len(team_logs_df) > 0:
            team1_logs = team_logs_df[
                (team_logs_df['GAME_ID'] == game_id) & 
                (team_logs_df['TEAM_ABBREVIATION'] == team1_data['TEAM_ABBREVIATION'])
            ]
            team2_logs = team_logs_df[
                (team_logs_df['GAME_ID'] == game_id) & 
                (team_logs_df['TEAM_ABBREVIATION'] == team2_data['TEAM_ABBREVIATION'])
            ]
            
            if len(team1_logs) > 0:
                team1_log = team1_logs.iloc[0]
            if len(team2_logs) > 0:
                team2_log = team2_logs.iloc[0]
        
        # Determine winner
        team1_wins = 1 if team1_data['WL'] == 'W' else 0
        
        # Create feature set
        features = {
            'game_id': game_id,
            'date': game_date,
            'season': team1_data.get('SEASON_ID', ''),
            'team1': team1_data['TEAM_ABBREVIATION'],
            'team2': team2_data['TEAM_ABBREVIATION'],
            
            # Team 1 stats (from all_games_df)
            'team1_pts': team1_data['PTS'],
            'team1_fg_pct': team1_data.get('FG_PCT', 0),
            'team1_3p_pct': team1_data.get('FG3_PCT', 0),
            'team1_ft_pct': team1_data.get('FT_PCT', 0),
            'team1_reb': team1_data.get('REB', 0),
            'team1_ast': team1_data.get('AST', 0),
            'team1_stl': team1_data.get('STL', 0),
            'team1_blk': team1_data.get('BLK', 0),
            'team1_tov': team1_data.get('TOV', 0),
            'team1_pf': team1_data.get('PF', 0),
            'team1_plus_minus': team1_data.get('PLUS_MINUS', 0),
            
            # Team 2 stats
            'team2_pts': team2_data['PTS'],
            'team2_fg_pct': team2_data.get('FG_PCT', 0),
            'team2_3p_pct': team2_data.get('FG3_PCT', 0),
            'team2_ft_pct': team2_data.get('FT_PCT', 0),
            'team2_reb': team2_data.get('REB', 0),
            'team2_ast': team2_data.get('AST', 0),
            'team2_stl': team2_data.get('STL', 0),
            'team2_blk': team2_data.get('BLK', 0),
            'team2_tov': team2_data.get('TOV', 0),
            'team2_pf': team2_data.get('PF', 0),
            'team2_plus_minus': team2_data.get('PLUS_MINUS', 0),
            
            # Derived features
            'pts_diff': team1_data['PTS'] - team2_data['PTS'],
            'fg_pct_diff': team1_data.get('FG_PCT', 0) - team2_data.get('FG_PCT', 0),
            'reb_diff': team1_data.get('REB', 0) - team2_data.get('REB', 0),
            'ast_diff': team1_data.get('AST', 0) - team2_data.get('AST', 0),
            'stl_diff': team1_data.get('STL', 0) - team2_data.get('STL', 0),
            'blk_diff': team1_data.get('BLK', 0) - team2_data.get('BLK', 0),
            'tov_diff': team1_data.get('TOV', 0) - team2_data.get('TOV', 0),
            'plus_minus_diff': team1_data.get('PLUS_MINUS', 0) - team2_data.get('PLUS_MINUS', 0),
            
            # Target
            'team1_wins': team1_wins
        }
        
        # Add team log data if available
        if team1_log is not None:
            features['team1_fgm'] = team1_log.get('FGM', 0)
            features['team1_fga'] = team1_log.get('FGA', 0)
            features['team1_fg3m'] = team1_log.get('FG3M', 0)
            features['team1_fg3a'] = team1_log.get('FG3A', 0)
            features['team1_ftm'] = team1_log.get('FTM', 0)
            features['team1_fta'] = team1_log.get('FTA', 0)
            features['team1_oreb'] = team1_log.get('OREB', 0)
            features['team1_dreb'] = team1_log.get('DREB', 0)
        
        if team2_log is not None:
            features['team2_fgm'] = team2_log.get('FGM', 0)
            features['team2_fga'] = team2_log.get('FGA', 0)
            features['team2_fg3m'] = team2_log.get('FG3M', 0)
            features['team2_fg3a'] = team2_log.get('FG3A', 0)
            features['team2_ftm'] = team2_log.get('FTM', 0)
            features['team2_fta'] = team2_log.get('FTA', 0)
            features['team2_oreb'] = team2_log.get('OREB', 0)
            features['team2_dreb'] = team2_log.get('DREB', 0)
        
        # Calculate efficiency
        if team1_log is not None:
            fga = team1_log.get('FGA', 0)
            fta = team1_log.get('FTA', 0)
            tov = team1_log.get('TOV', 0)
            pts = team1_data['PTS']
            if (fga + 0.44 * fta + tov) > 0:
                features['team1_efficiency'] = pts / (fga + 0.44 * fta + tov)
            else:
                features['team1_efficiency'] = 0
        
        if team2_log is not None:
            fga = team2_log.get('FGA', 0)
            fta = team2_log.get('FTA', 0)
            tov = team2_log.get('TOV', 0)
            pts = team2_data['PTS']
            if (fga + 0.44 * fta + tov) > 0:
                features['team2_efficiency'] = pts / (fga + 0.44 * fta + tov)
            else:
                features['team2_efficiency'] = 0
        
        if 'team1_efficiency' in features and 'team2_efficiency' in features:
            features['efficiency_diff'] = features['team1_efficiency'] - features['team2_efficiency']
        else:
            features['efficiency_diff'] = 0
        
        games_processed.append(features)
    
    # Create DataFrame
    game_features_df = pd.DataFrame(games_processed)
    
    if len(game_features_df) > 0:
        # Parse dates
        game_features_df['date'] = pd.to_datetime(game_features_df['date'])
        game_features_df = game_features_df.sort_values('date')
        
        # Fill NaN values
        numeric_cols = game_features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'team1_wins':
                game_features_df[col] = game_features_df[col].fillna(0)
        
        print(f"✓ Processed {len(game_features_df)} games")
        print(f"  Date range: {game_features_df['date'].min()} to {game_features_df['date'].max()}")
        print(f"  Teams: {game_features_df['team1'].nunique()}")
        
        # Save cleaned data
        cleaned_file = os.path.join(CLEANED_FOLDER, "game_features_cleaned.csv")
        game_features_df.to_csv(cleaned_file, index=False)
        print(f"  Saved to: {cleaned_file}")
        
        return game_features_df
    else:
        print("✗ No games processed")
        return None

def add_recent_form_features(game_features_df, window=5):
    """Add recent form features to the dataset"""
    print("\nAdding recent form features...")
    
    if game_features_df is None or len(game_features_df) == 0:
        return game_features_df
    
    game_features_df = game_features_df.sort_values('date')
    
    # Initialize columns
    game_features_df['team1_recent_win_rate'] = 0.5
    game_features_df['team2_recent_win_rate'] = 0.5
    
    for team in game_features_df['team1'].unique():
        team_games = game_features_df[
            (game_features_df['team1'] == team) | 
            (game_features_df['team2'] == team)
        ].copy()
        team_games = team_games.sort_values('date')
        
        for idx, row in team_games.iterrows():
            recent_games = team_games[team_games['date'] < row['date']].tail(window)
            
            if len(recent_games) > 0:
                wins = 0
                for _, recent_game in recent_games.iterrows():
                    if recent_game['team1'] == team:
                        if recent_game['team1_wins'] == 1:
                            wins += 1
                    elif recent_game['team2'] == team:
                        if recent_game['team1_wins'] == 0:
                            wins += 1
                
                win_rate = wins / len(recent_games)
                
                if row['team1'] == team:
                    game_features_df.loc[idx, 'team1_recent_win_rate'] = win_rate
                else:
                    game_features_df.loc[idx, 'team2_recent_win_rate'] = win_rate
    
    game_features_df['recent_win_rate_diff'] = (
        game_features_df['team1_recent_win_rate'] - 
        game_features_df['team2_recent_win_rate']
    )
    
    print(f"✓ Added recent form features (window={window})")
    return game_features_df

def create_summary_report(all_games_df, team_logs_df, player_logs_df, game_features_df):
    """Create a summary report of collected data"""
    print("\n" + "="*60)
    print("Creating Summary Report")
    print("="*60)
    
    report = {
        'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'season': get_current_season(),
        'data_sources': {
            'all_games': {
                'collected': all_games_df is not None,
                'rows': len(all_games_df) if all_games_df is not None else 0,
                'date_range': None
            },
            'team_logs': {
                'collected': team_logs_df is not None,
                'rows': len(team_logs_df) if team_logs_df is not None else 0,
            },
            'player_logs': {
                'collected': player_logs_df is not None,
                'rows': len(player_logs_df) if player_logs_df is not None else 0,
            },
            'game_features': {
                'collected': game_features_df is not None,
                'rows': len(game_features_df) if game_features_df is not None else 0,
            }
        }
    }
    
    if all_games_df is not None and len(all_games_df) > 0:
        all_games_df['parsed_date'] = all_games_df['GAME_DATE'].apply(parse_api_date)
        dates = all_games_df[all_games_df['parsed_date'].notna()]['parsed_date']
        if len(dates) > 0:
            report['data_sources']['all_games']['date_range'] = {
                'min': dates.min().strftime('%Y-%m-%d'),
                'max': dates.max().strftime('%Y-%m-%d')
            }
    
    if game_features_df is not None and len(game_features_df) > 0:
        report['data_sources']['game_features']['date_range'] = {
            'min': game_features_df['date'].min().strftime('%Y-%m-%d'),
            'max': game_features_df['date'].max().strftime('%Y-%m-%d')
        }
        report['data_sources']['game_features']['teams'] = game_features_df['team1'].nunique()
        report['data_sources']['game_features']['win_distribution'] = {
            'wins': int(game_features_df['team1_wins'].sum()),
            'losses': int((game_features_df['team1_wins'] == 0).sum())
        }
    
    # Save report
    report_file = os.path.join(DATA_FOLDER, "collection_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✓ Summary report created")
    print(f"  Saved to: {report_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("COLLECTION SUMMARY")
    print("="*60)
    print(f"Collection Date: {report['collection_date']}")
    print(f"Season: {report['season']}")
    print(f"\nData Collected:")
    print(f"  All Games: {report['data_sources']['all_games']['rows']} rows")
    print(f"  Team Logs: {report['data_sources']['team_logs']['rows']} rows")
    print(f"  Player Logs: {report['data_sources']['player_logs']['rows']} rows")
    print(f"  Game Features: {report['data_sources']['game_features']['rows']} rows")
    
    if game_features_df is not None and len(game_features_df) > 0:
        print(f"\nGame Features Details:")
        print(f"  Teams: {report['data_sources']['game_features']['teams']}")
        print(f"  Date Range: {report['data_sources']['game_features']['date_range']['min']} to {report['data_sources']['game_features']['date_range']['max']}")
        print(f"  Win/Loss Distribution: {report['data_sources']['game_features']['win_distribution']}")
    
    return report

def main():
    """Main data collection function"""
    print("="*60)
    print("NBA API Data Collection Script")
    print("Fetching Live Data from Official NBA API")
    print("="*60)
    
    # Create folders
    create_folders()
    
    # Get current date and season
    today = datetime.now()
    current_season = get_current_season()
    
    print(f"\nCurrent Date: {today.strftime('%Y-%m-%d')}")
    print(f"Current Season: {current_season}")
    
    # Fetch all data
    print("\nStarting data collection...")
    print("(This may take several minutes depending on data size)")
    
    # 1. Fetch all games
    all_games_df = fetch_all_games(season=current_season, date_to=today)
    time.sleep(1)  # Rate limiting
    
    # 2. Fetch team game logs
    team_logs_df = fetch_team_game_logs(season=current_season, date_to=today)
    time.sleep(1)
    
    # 3. Fetch player game logs (optional, can be slow)
    print("\nNote: Player game logs are large. Fetching...")
    player_logs_df = fetch_player_game_logs(season=current_season, date_to=today)
    time.sleep(1)
    
    # 4. Fetch team stats
    team_stats_df = fetch_team_stats(season=current_season)
    time.sleep(1)
    
    # 5. Clean and process data
    game_features_df = clean_and_process_data(
        all_games_df, team_logs_df, player_logs_df, team_stats_df
    )
    
    # 6. Add recent form features
    if game_features_df is not None:
        game_features_df = add_recent_form_features(game_features_df, window=5)
        
        # Save final cleaned dataset
        final_file = os.path.join(CLEANED_FOLDER, "final_game_features.csv")
        game_features_df.to_csv(final_file, index=False)
        print(f"\n✓ Final dataset saved to: {final_file}")
        print(f"  This is the file to use for predictions!")
    
    # 7. Create summary report
    create_summary_report(all_games_df, team_logs_df, player_logs_df, game_features_df)
    
    print("\n" + "="*60)
    print("Data Collection Complete!")
    print("="*60)
    print(f"\nData stored in: {DATA_FOLDER}/")
    print(f"  Raw data: {RAW_FOLDER}/")
    print(f"  Cleaned data: {CLEANED_FOLDER}/")
    print(f"\nUse this file for predictions:")
    print(f"  {CLEANED_FOLDER}/final_game_features.csv")

if __name__ == "__main__":
    main()
