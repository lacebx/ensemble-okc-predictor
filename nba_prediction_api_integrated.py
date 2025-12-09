#!/usr/bin/env python3
"""
NBA Game Prediction Model for OKC Thunder - API Integrated Version
Fixes: Win counting, dates, better visuals
Integrates NBA API for live/accurate data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone
from dateutil import parser
import sys
import os

# Add NBA API to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nba_api', 'src'))

try:
    from nba_api.stats.endpoints import teamgamelog
    from nba_api.stats.static import teams
    from nba_api.live.nba.endpoints import scoreboard as live_scoreboard
    from nba_api.live.nba.endpoints import boxscore as live_boxscore
    NBA_API_AVAILABLE = True
except ImportError:
    print("Warning: NBA API not available. Install with: pip install nba_api")
    NBA_API_AVAILABLE = False

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Team ID mapping (OKC Thunder = 1610612760)
TEAM_IDS = {
    'OKC': 1610612760,
    'PHO': 1610612756,  # Phoenix Suns
    'LAC': 1610612746,  # LA Clippers
    'MIN': 1610612750,  # Minnesota Timberwolves
    'MEM': 1610612763,  # Memphis Grizzlies
    'SAS': 1610612759,  # San Antonio Spurs
}

def get_current_season():
    """Determine current NBA season based on today's date"""
    today = datetime.now()
    # NBA season typically starts in October
    # If we're in Oct-Dec, we're in the new season (e.g., Dec 2025 = 2025-26 season)
    # If we're in Jan-Sep, we're in the previous season (e.g., Jan 2025 = 2024-25 season)
    if today.month >= 10:
        # October-December: current year to next year
        season = f"{today.year}-{str(today.year + 1)[-2:]}"
    else:
        # January-September: previous year to current year
        season = f"{today.year - 1}-{str(today.year)[-2:]}"
    return season

def parse_api_date(date_str):
    """Parse NBA API date format (e.g., 'APR 13, 2025' or '2025-04-13')"""
    if pd.isna(date_str) or date_str == '':
        return None
    try:
        date_str = str(date_str).strip()
        # Try parsing as 'MON DD, YYYY' format (e.g., 'APR 13, 2025')
        if ',' in date_str and len(date_str.split()) == 3:
            try:
                return datetime.strptime(date_str, '%b %d, %Y')
            except:
                try:
                    return datetime.strptime(date_str, '%B %d, %Y')  # Full month name
                except:
                    pass
        # Try parsing as 'YYYY-MM-DD' format
        elif '-' in date_str and len(date_str.split('-')) == 3:
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except:
                pass
        # Try parsing as 'MM/DD/YYYY' format
        elif '/' in date_str and len(date_str.split('/')) == 3:
            try:
                return datetime.strptime(date_str, '%m/%d/%Y')
            except:
                pass
        # Fallback to dateutil parser
        return parser.parse(date_str)
    except Exception as e:
        print(f"Warning: Could not parse date '{date_str}': {e}")
        return None

def test_nba_api():
    """
    Test NBA API functionality and return sample data for user verification.
    Fetches LAST 5 GAMES FROM TODAY (not old season data).
    """
    if not NBA_API_AVAILABLE:
        print("NBA API is not available. Please install it first.")
        return False
    
    print("="*60)
    print("NBA API TEST - Please Verify the Following Information")
    print("="*60)
    
    try:
        # Get current date
        today = datetime.now()
        current_season = get_current_season()
        print(f"\nCurrent Date: {today.strftime('%Y-%m-%d')}")
        print(f"Current Season: {current_season}")
        
        # Test 1: Get OKC team info
        print("\n1. Testing Team Information Retrieval...")
        okc_team = teams.find_team_by_abbreviation('OKC')
        if okc_team:
            print(f"   ✓ Found OKC Thunder")
            print(f"     Team ID: {okc_team['id']}")
            print(f"     Full Name: {okc_team['full_name']}")
            print(f"     City: {okc_team['city']}")
        else:
            print("   ✗ Could not find OKC Thunder")
            return False
        
        # Test 2: Get recent game log for OKC - FIXED to get current season and filter to today
        print("\n2. Testing Game Log Retrieval (Last 5 Games from Today)...")
        
        # Try current season first
        game_log = teamgamelog.TeamGameLog(
            team_id=okc_team['id'],
            season=current_season,
            date_to_nullable=today.strftime('%m/%d/%Y'),  # Filter to games up to today
            get_request=True
        )
        
        df = game_log.get_data_frames()[0]
        
        # If no games in current season, try previous season
        if len(df) == 0:
            # Try previous season
            prev_year = today.year - 1 if today.month >= 10 else today.year - 2
            prev_season = f"{prev_year}-{str(prev_year + 1)[-2:]}"
            print(f"   Trying previous season: {prev_season}")
            game_log = teamgamelog.TeamGameLog(
                team_id=okc_team['id'],
                season=prev_season,
                date_to_nullable=today.strftime('%m/%d/%Y'),
                get_request=True
            )
            df = game_log.get_data_frames()[0]
        
        if len(df) > 0:
            # Parse dates and filter to games up to today
            df['parsed_date'] = df['GAME_DATE'].apply(parse_api_date)
            
            # Remove rows where date parsing failed
            df = df[df['parsed_date'].notna()].copy()
            
            if len(df) > 0:
                # Filter to games up to today
                df_filtered = df[df['parsed_date'] <= today].copy()
                
                if len(df_filtered) == 0:
                    # If no games up to today, show all games but warn
                    print(f"   ⚠ WARNING: No games found up to today ({today.strftime('%Y-%m-%d')})")
                    print(f"   Showing all {len(df)} games from season (may be old data)")
                    df_filtered = df.copy()
                
                df_filtered = df_filtered.sort_values('parsed_date', ascending=False)  # Most recent first
                
                print(f"   ✓ Retrieved {len(df)} total games, {len(df_filtered)} games up to today")
                print(f"     Most recent game: {df_filtered.iloc[0]['GAME_DATE']}")
                most_recent_parsed = df_filtered.iloc[0]['parsed_date']
                if pd.notna(most_recent_parsed):
                    print(f"     Game date parsed: {most_recent_parsed.strftime('%Y-%m-%d')}")
                    days_ago = (today - most_recent_parsed).days
                    print(f"     Days ago: {days_ago}")
                print(f"     Result: {df_filtered.iloc[0]['MATCHUP']} - {df_filtered.iloc[0]['WL']}")
                print(f"     Score: {df_filtered.iloc[0]['PTS']} points")
                
                # Show LAST 5 games from today (most recent first)
                recent_5 = df_filtered.head(5)
                print("\n   Last 5 Games (Most Recent First):")
                for idx, row in recent_5.iterrows():
                    game_date = row['parsed_date'].strftime('%Y-%m-%d') if pd.notna(row['parsed_date']) else row['GAME_DATE']
                    print(f"     {game_date}: {row['MATCHUP']} - {row['WL']} ({row['PTS']} pts)")
                
                # Verify dates are recent (within last 30 days)
                if pd.notna(most_recent_parsed):
                    if days_ago > 30:
                        print(f"\n   ⚠ WARNING: Most recent game is {days_ago} days old!")
                        print(f"   Expected: Games within last few days (today is {today.strftime('%Y-%m-%d')})")
                        print(f"   This might indicate the API is returning old season data.")
                    elif days_ago < 0:
                        print(f"\n   ⚠ WARNING: Most recent game is in the future!")
                        print(f"   This indicates a date parsing issue.")
                
                # Update df to filtered version for record calculation
                df = df_filtered
            else:
                print("   ✗ No games with valid dates found")
                return False
        else:
            print("   ✗ No games found")
            return False
        
        # Test 3: Get current season record (only games up to today)
        print("\n3. Testing Season Statistics (Games up to Today)...")
        wins = df[df['WL'] == 'W'].shape[0]
        losses = df[df['WL'] == 'L'].shape[0]
        total = wins + losses
        if total > 0:
            print(f"   ✓ OKC Thunder Record: {wins}-{losses} (out of {total} games)")
            print(f"     Win Percentage: {wins/total*100:.1f}%")
        else:
            print("   ⚠ No completed games found")
        
        # Test 4: Live scoreboard (if available)
        print("\n4. Testing Live Scoreboard...")
        try:
            live_board = live_scoreboard.ScoreBoard()
            print(f"   ✓ Live scoreboard available")
            print(f"     Date: {live_board.score_board_date}")
            print(f"     Games today: {len(live_board.games.get_dict())}")
        except Exception as e:
            print(f"   ⚠ Live scoreboard not available: {e}")
        
        print("\n" + "="*60)
        print("API TEST SUMMARY")
        print("="*60)
        print(f"✓ Team Information: Working")
        print(f"✓ Game Logs: Working ({len(df)} games retrieved, filtered to today)")
        if total > 0:
            print(f"✓ Season Stats: Working (Record: {wins}-{losses})")
        print(f"✓ API Status: OPERATIONAL")
        print(f"\n⚠ IMPORTANT: Please verify the dates are recent (within last few days)")
        print("⚠ The most recent game should be close to today's date (Dec 8, 2025)")
        print("\nDoes the OKC record and recent games look accurate? (y/n): ", end='')
        
        user_input = input().strip().lower()
        if user_input == 'y':
            print("\n✓ API test confirmed by user. Proceeding with predictions...")
            return True
        else:
            print("\n✗ API test not confirmed. Please check the data and try again.")
            print("   The API might be returning old data. We'll use local dataset instead.")
            return False
            
    except Exception as e:
        print(f"\n✗ API Test Failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your internet connection and NBA API installation.")
        return False

def get_team_data_from_api(team_abbrev, season=None, date_to=None):
    """
    Get team game log data from NBA API.
    Fetches games up to the specified date (or today if not specified).
    """
    if not NBA_API_AVAILABLE:
        return None
    
    try:
        team = teams.find_team_by_abbreviation(team_abbrev)
        if not team:
            return None
        
        # Use current season if not specified
        if season is None:
            season = get_current_season()
        
        # Use today if date not specified
        if date_to is None:
            date_to = datetime.now().strftime('%m/%d/%Y')
        
        game_log = teamgamelog.TeamGameLog(
            team_id=team['id'],
            season=season,
            date_to_nullable=date_to,
            get_request=True
        )
        
        df = game_log.get_data_frames()[0]
        
        # Parse dates and filter to games up to date_to
        if len(df) > 0:
            df['parsed_date'] = df['GAME_DATE'].apply(parse_api_date)
            target_date = datetime.strptime(date_to, '%m/%d/%Y') if isinstance(date_to, str) else date_to
            df = df[df['parsed_date'] <= target_date].copy()
            df = df.sort_values('parsed_date', ascending=False)  # Most recent first
        
        return df
    except Exception as e:
        print(f"Error fetching API data for {team_abbrev}: {e}")
        return None

def load_data(use_api=False):
    """Load datasets - can use API or local files"""
    print("Loading datasets...")
    
    if use_api and NBA_API_AVAILABLE:
        print("Using NBA API for live data...")
        # Get OKC data from API
        okc_api_data = get_team_data_from_api('OKC')
        if okc_api_data is not None:
            print(f"✓ Retrieved {len(okc_api_data)} games from API")
            return okc_api_data, None, None
        else:
            print("⚠ API data retrieval failed, falling back to local files...")
    
    # Fallback to local files
    try:
        player_stats = pd.read_csv('NBA - Player Stats - Season 24-25/database_24_25.csv')
        team_stats = pd.read_csv('dataset/Team Stats Per Game.csv')
        team_summaries = pd.read_csv('dataset/Team Summaries.csv')
        
        print(f"Player Stats shape: {player_stats.shape}")
        print(f"Team Stats shape: {team_stats.shape}")
        print(f"Team Summaries shape: {team_summaries.shape}")
        
        return player_stats, team_stats, team_summaries
    except FileNotFoundError as e:
        print(f"Error loading local files: {e}")
        return None, None, None

def aggregate_player_stats_by_game(player_stats):
    """Aggregate player statistics by team and game"""
    print("\nAggregating player statistics by game...")
    
    game_stats = player_stats.groupby(['Tm', 'Opp', 'Data', 'Res']).agg({
        'PTS': 'sum',
        'FG': 'sum', 'FGA': 'sum',
        '3P': 'sum', '3PA': 'sum',
        'FT': 'sum', 'FTA': 'sum',
        'TRB': 'sum',
        'AST': 'sum',
        'STL': 'sum',
        'BLK': 'sum',
        'TOV': 'sum',
        'PF': 'sum',
        'MP': 'sum',
        'GmSc': 'mean'
    }).reset_index()
    
    game_stats['FG%'] = game_stats['FG'] / game_stats['FGA'].replace(0, np.nan)
    game_stats['3P%'] = game_stats['3P'] / game_stats['3PA'].replace(0, np.nan)
    game_stats['FT%'] = game_stats['FT'] / game_stats['FTA'].replace(0, np.nan)
    
    print(f"Aggregated game stats shape: {game_stats.shape}")
    return game_stats

def create_game_features(game_stats):
    """Create features for each game matchup"""
    print("\nCreating game features from player statistics...")
    
    games = []
    unique_games = game_stats[['Tm', 'Opp', 'Data']].drop_duplicates()
    
    for idx, game in unique_games.iterrows():
        team1 = game['Tm']
        team2 = game['Opp']
        date = game['Data']
        
        team1_stats = game_stats[(game_stats['Tm'] == team1) & 
                                 (game_stats['Opp'] == team2) & 
                                 (game_stats['Data'] == date)]
        team2_stats = game_stats[(game_stats['Tm'] == team2) & 
                                 (game_stats['Opp'] == team1) & 
                                 (game_stats['Data'] == date)]
        
        if len(team1_stats) > 0 and len(team2_stats) > 0:
            t1 = team1_stats.iloc[0]
            t2 = team2_stats.iloc[0]
            
            result = 1 if t1['Res'] == 'W' else 0
            
            features = {
                'date': date,
                'team1': team1,
                'team2': team2,
                'team1_pts': t1['PTS'],
                'team1_fg_pct': t1['FG%'] if pd.notna(t1['FG%']) else 0,
                'team1_3p_pct': t1['3P%'] if pd.notna(t1['3P%']) else 0,
                'team1_ft_pct': t1['FT%'] if pd.notna(t1['FT%']) else 0,
                'team1_reb': t1['TRB'],
                'team1_ast': t1['AST'],
                'team1_stl': t1['STL'],
                'team1_blk': t1['BLK'],
                'team1_tov': t1['TOV'],
                'team1_pf': t1['PF'],
                'team1_gmsc_avg': t1['GmSc'],
                'team2_pts': t2['PTS'],
                'team2_fg_pct': t2['FG%'] if pd.notna(t2['FG%']) else 0,
                'team2_3p_pct': t2['3P%'] if pd.notna(t2['3P%']) else 0,
                'team2_ft_pct': t2['FT%'] if pd.notna(t2['FT%']) else 0,
                'team2_reb': t2['TRB'],
                'team2_ast': t2['AST'],
                'team2_stl': t2['STL'],
                'team2_blk': t2['BLK'],
                'team2_tov': t2['TOV'],
                'team2_pf': t2['PF'],
                'team2_gmsc_avg': t2['GmSc'],
                'pts_diff': t1['PTS'] - t2['PTS'],
                'fg_pct_diff': (t1['FG%'] if pd.notna(t1['FG%']) else 0) - (t2['FG%'] if pd.notna(t2['FG%']) else 0),
                'reb_diff': t1['TRB'] - t2['TRB'],
                'ast_diff': t1['AST'] - t2['AST'],
                'stl_diff': t1['STL'] - t2['STL'],
                'blk_diff': t1['BLK'] - t2['BLK'],
                'tov_diff': t1['TOV'] - t2['TOV'],
                'gmsc_diff': t1['GmSc'] - t2['GmSc'],
                'team1_efficiency': t1['PTS'] / (t1['FGA'] + 0.44 * t1['FTA'] + t1['TOV']) if (t1['FGA'] + 0.44 * t1['FTA'] + t1['TOV']) > 0 else 0,
                'team2_efficiency': t2['PTS'] / (t2['FGA'] + 0.44 * t2['FTA'] + t2['TOV']) if (t2['FGA'] + 0.44 * t2['FTA'] + t2['TOV']) > 0 else 0,
                'efficiency_diff': (t1['PTS'] / (t1['FGA'] + 0.44 * t1['FTA'] + t1['TOV']) if (t1['FGA'] + 0.44 * t1['FTA'] + t1['TOV']) > 0 else 0) - \
                                   (t2['PTS'] / (t2['FGA'] + 0.44 * t2['FTA'] + t2['TOV']) if (t2['FGA'] + 0.44 * t2['FTA'] + t2['TOV']) > 0 else 0),
                'team1_wins': result
            }
            
            games.append(features)
    
    game_features_df = pd.DataFrame(games)
    print(f"Created {len(game_features_df)} game features")
    return game_features_df

def add_recent_form_features(game_features_df, window=5):
    """Add recent form features"""
    print(f"\nAdding recent form features (last {window} games)...")
    
    game_features_df['date'] = pd.to_datetime(game_features_df['date'])
    game_features_df = game_features_df.sort_values('date')
    
    for team in game_features_df['team1'].unique():
        team_games = game_features_df[(game_features_df['team1'] == team) | (game_features_df['team2'] == team)].copy()
        team_games = team_games.sort_values('date')
        
        for idx, row in team_games.iterrows():
            if row['team1'] == team:
                recent_games = team_games[team_games['date'] < row['date']].tail(window)
                if len(recent_games) > 0:
                    win_rate = recent_games[recent_games['team1'] == team]['team1_wins'].sum() / len(recent_games)
                    game_features_df.loc[idx, 'team1_recent_win_rate'] = win_rate
                else:
                    game_features_df.loc[idx, 'team1_recent_win_rate'] = 0.5
            else:
                recent_games = team_games[team_games['date'] < row['date']].tail(window)
                if len(recent_games) > 0:
                    win_rate = 1 - (recent_games[recent_games['team2'] == team]['team1_wins'].sum() / len(recent_games))
                    game_features_df.loc[idx, 'team2_recent_win_rate'] = win_rate
                else:
                    game_features_df.loc[idx, 'team2_recent_win_rate'] = 0.5
    
    game_features_df['team1_recent_win_rate'] = game_features_df['team1_recent_win_rate'].fillna(0.5)
    game_features_df['team2_recent_win_rate'] = game_features_df['team2_recent_win_rate'].fillna(0.5)
    game_features_df['recent_win_rate_diff'] = game_features_df['team1_recent_win_rate'] - game_features_df['team2_recent_win_rate']
    
    return game_features_df

def preprocess_data(game_features_df):
    """Preprocess the game data"""
    print("\nPreprocessing data...")
    
    game_features_df = game_features_df.dropna(subset=['team1_pts', 'team2_pts', 'team1_wins'])
    
    numeric_cols = game_features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'team1_wins':
            game_features_df[col] = game_features_df[col].fillna(game_features_df[col].median() if pd.notna(game_features_df[col].median()) else 0)
    
    print(f"Dataset after cleaning: {game_features_df.shape}")
    print(f"\nClass distribution:")
    print(game_features_df['team1_wins'].value_counts())
    
    return game_features_df

def get_okc_recent_record(game_features_df, use_api=False):
    """
    Get OKC's recent record - FIXED to correctly count wins.
    Returns wins, losses, and game results for last 10 games.
    """
    okc_games = game_features_df[(game_features_df['team1'] == 'OKC') | (game_features_df['team2'] == 'OKC')].copy()
    
    if len(okc_games) == 0:
        return 0, 0, []
    
    # Sort by date (most recent last)
    okc_games['date'] = pd.to_datetime(okc_games['date'])
    okc_games = okc_games.sort_values('date')
    
    # Get last 10 games (most recent)
    recent_games = okc_games.tail(10)
    
    wins = 0
    losses = 0
    game_results = []
    
    for _, game in recent_games.iterrows():
        if game['team1'] == 'OKC':
            # OKC is team1, so team1_wins == 1 means OKC won
            if game['team1_wins'] == 1:
                wins += 1
                game_results.append('W')
            else:
                losses += 1
                game_results.append('L')
        elif game['team2'] == 'OKC':
            # OKC is team2, so team1_wins == 0 means OKC won (team1 lost)
            if game['team1_wins'] == 0:
                wins += 1
                game_results.append('W')
            else:
                losses += 1
                game_results.append('L')
    
    return wins, losses, game_results

def train_models(X_train, X_test, y_train, y_test):
    """Train ensemble models"""
    print("\n" + "="*60)
    print("Training Ensemble Models")
    print("="*60)
    
    models = {}
    results = {}
    
    # Random Forest
    print("\n1. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'accuracy': rf_accuracy,
        'cv_mean': rf_cv_scores.mean(),
        'cv_std': rf_cv_scores.std(),
        'predictions': rf_pred,
        'probabilities': rf_pred_proba
    }
    print(f"Accuracy: {rf_accuracy:.4f}, CV Mean: {rf_cv_scores.mean():.4f} (+/- {rf_cv_scores.std()*2:.4f})")
    
    # Gradient Boosting
    print("\n2. Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_pred_proba = gb_model.predict_proba(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5, scoring='accuracy')
    
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = {
        'accuracy': gb_accuracy,
        'cv_mean': gb_cv_scores.mean(),
        'cv_std': gb_cv_scores.std(),
        'predictions': gb_pred,
        'probabilities': gb_pred_proba
    }
    print(f"Accuracy: {gb_accuracy:.4f}, CV Mean: {gb_cv_scores.mean():.4f} (+/- {gb_cv_scores.std()*2:.4f})")
    
    return models, results

def calculate_error_margins(results, X_test, y_test):
    """Calculate error margins"""
    print("\nCalculating error margins and uncertainty...")
    
    error_margins = {}
    
    for model_name, result in results.items():
        probabilities = result['probabilities'][:, 1]
        confidence = np.abs(probabilities - 0.5) * 2
        
        cv_std = result['cv_std']
        base_error = cv_std * 100
        
        adjusted_error = base_error + (1 - confidence.mean()) * 10
        
        error_margins[model_name] = {
            'base_error': base_error,
            'confidence_mean': confidence.mean(),
            'adjusted_error': adjusted_error,
            'min_confidence': confidence.min(),
            'max_confidence': confidence.max()
        }
        
        print(f"{model_name}:")
        print(f"  Base Error Margin: ±{base_error:.2f}%")
        print(f"  Average Confidence: {confidence.mean():.3f}")
        print(f"  Adjusted Error Margin: ±{adjusted_error:.2f}%")
    
    return error_margins

def create_prediction_features_from_recent_games(okc_recent_games, opp_recent_games, feature_cols):
    """Create feature vector for prediction"""
    features = {}
    
    okc_team1_games = okc_recent_games[okc_recent_games['team1'] == 'OKC']
    okc_team2_games = okc_recent_games[okc_recent_games['team2'] == 'OKC']
    
    for col in feature_cols:
        if 'team1_' in col:
            okc_vals = okc_team1_games[col].values if len(okc_team1_games) > 0 else []
            if len(okc_team2_games) > 0:
                team2_col = col.replace('team1_', 'team2_')
                if team2_col in okc_team2_games.columns:
                    okc_vals = np.concatenate([okc_vals, okc_team2_games[team2_col].values])
            features[col] = np.mean(okc_vals) if len(okc_vals) > 0 else 0
        elif 'team2_' in col:
            opp_vals = []
            if len(opp_recent_games) > 0:
                opp_team1 = opp_recent_games.iloc[0]['team1'] if 'team1' in opp_recent_games.columns else None
                opp_team1_games = opp_recent_games[opp_recent_games['team1'] == opp_team1] if opp_team1 else pd.DataFrame()
                if len(opp_team1_games) > 0:
                    team1_col = col.replace('team2_', 'team1_')
                    if team1_col in opp_team1_games.columns:
                        opp_vals.extend(opp_team1_games[team1_col].values)
                opp_team2_games = opp_recent_games[opp_recent_games['team2'] == opp_team1] if opp_team1 else pd.DataFrame()
                if len(opp_team2_games) > 0:
                    opp_vals.extend(opp_team2_games[col].values)
            features[col] = np.mean(opp_vals) if len(opp_vals) > 0 else 0
        elif 'recent_win_rate_diff' in col:
            okc_wins = sum([1 for _, g in okc_recent_games.iterrows() 
                           if (g['team1'] == 'OKC' and g['team1_wins'] == 1) or 
                              (g['team2'] == 'OKC' and g['team1_wins'] == 0)])
            okc_win_rate = okc_wins / len(okc_recent_games) if len(okc_recent_games) > 0 else 0.5
            
            opp_team = opp_recent_games.iloc[0]['team1'] if len(opp_recent_games) > 0 and 'team1' in opp_recent_games.columns else None
            if opp_team:
                opp_wins = sum([1 for _, g in opp_recent_games.iterrows()
                               if (g['team1'] == opp_team and g['team1_wins'] == 1) or
                                  (g['team2'] == opp_team and g['team1_wins'] == 0)])
                opp_win_rate = opp_wins / len(opp_recent_games) if len(opp_recent_games) > 0 else 0.5
            else:
                opp_win_rate = 0.5
            
            features[col] = okc_win_rate - opp_win_rate
        else:
            features[col] = 0
    
    # Calculate derived features
    if 'pts_diff' in feature_cols:
        features['pts_diff'] = features.get('team1_pts', 0) - features.get('team2_pts', 0)
    if 'fg_pct_diff' in feature_cols:
        features['fg_pct_diff'] = features.get('team1_fg_pct', 0) - features.get('team2_fg_pct', 0)
    if 'reb_diff' in feature_cols:
        features['reb_diff'] = features.get('team1_reb', 0) - features.get('team2_reb', 0)
    if 'ast_diff' in feature_cols:
        features['ast_diff'] = features.get('team1_ast', 0) - features.get('team2_ast', 0)
    if 'stl_diff' in feature_cols:
        features['stl_diff'] = features.get('team1_stl', 0) - features.get('team2_stl', 0)
    if 'blk_diff' in feature_cols:
        features['blk_diff'] = features.get('team1_blk', 0) - features.get('team2_blk', 0)
    if 'tov_diff' in feature_cols:
        features['tov_diff'] = features.get('team1_tov', 0) - features.get('team2_tov', 0)
    if 'gmsc_diff' in feature_cols:
        features['gmsc_diff'] = features.get('team1_gmsc_avg', 0) - features.get('team2_gmsc_avg', 0)
    if 'efficiency_diff' in feature_cols:
        features['efficiency_diff'] = features.get('team1_efficiency', 0) - features.get('team2_efficiency', 0)
    
    feature_vector = np.array([features.get(col, 0) for col in feature_cols]).reshape(1, -1)
    return feature_vector

def predict_okc_games(models, results, error_margins, game_features_df):
    """Predict OKC Thunder's next 5 games - FIXED dates to 2025"""
    print("\n" + "="*60)
    print("Predicting OKC Thunder Next 5 Games")
    print("="*60)
    
    # Actual schedule - FIXED: dates are in 2025
    schedule = [
        {'date': '2025-12-10', 'opponent': 'PHO', 'location': 'home', 'opp_name': 'Phoenix Suns'},
        {'date': '2025-12-17', 'opponent': 'LAC', 'location': 'home', 'opp_name': 'LA Clippers'},
        {'date': '2025-12-19', 'opponent': 'MIN', 'location': 'away', 'opp_name': 'Minnesota Timberwolves'},
        {'date': '2025-12-22', 'opponent': 'MEM', 'location': 'home', 'opp_name': 'Memphis Grizzlies'},
        {'date': '2025-12-23', 'opponent': 'SAS', 'location': 'away', 'opp_name': 'San Antonio Spurs'}
    ]
    
    # Get OKC's recent performance - FIXED win counting
    wins, losses, game_results = get_okc_recent_record(game_features_df)
    
    print(f"\nOKC Thunder Recent Performance:")
    print(f"  Last 10 games: {wins}-{losses}")
    print(f"  Win rate: {wins/10:.1%}")
    print(f"  Recent results: {' '.join(game_results)}")
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    best_model = models[best_model_name]
    best_error = error_margins[best_model_name]['adjusted_error']
    
    print(f"\nUsing {best_model_name} (Best CV Score: {results[best_model_name]['cv_mean']:.4f})")
    print(f"Error Margin: ±{best_error:.2f}%")
    
    feature_cols = [col for col in game_features_df.columns 
                    if col not in ['date', 'team1', 'team2', 'team1_wins']]
    
    predictions = []
    okc_games = game_features_df[(game_features_df['team1'] == 'OKC') | (game_features_df['team2'] == 'OKC')].copy()
    okc_games = okc_games.sort_values('date')
    
    for game_info in schedule:
        opp = game_info['opponent']
        location = game_info['location']
        
        okc_recent = okc_games.tail(5) if len(okc_games) >= 5 else okc_games
        opp_games = game_features_df[(game_features_df['team1'] == opp) | (game_features_df['team2'] == opp)].copy()
        opp_games = opp_games.sort_values('date').tail(5) if len(opp_games) >= 5 else opp_games
        
        if len(okc_recent) > 0 and len(opp_games) > 0:
            try:
                pred_features = create_prediction_features_from_recent_games(okc_recent, opp_games, feature_cols)
                win_prob = best_model.predict_proba(pred_features)[0][1]
                prediction = best_model.predict(pred_features)[0]
                
                win_prob_lower = max(0, win_prob - best_error/100)
                win_prob_upper = min(1, win_prob + best_error/100)
                
                predictions.append({
                    'Date': game_info['date'],
                    'Opponent': game_info['opp_name'],
                    'Location': location,
                    'Prediction': 'WIN' if prediction == 1 else 'LOSS',
                    'Win Probability': f"{win_prob*100:.1f}%",
                    'Confidence Range': f"{win_prob_lower*100:.1f}% - {win_prob_upper*100:.1f}%",
                    'Error Margin': f"±{best_error:.1f}%"
                })
            except Exception as e:
                print(f"Warning: Error predicting {opp} game: {e}")
                predictions.append({
                    'Date': game_info['date'],
                    'Opponent': game_info['opp_name'],
                    'Location': location,
                    'Prediction': 'ERROR',
                    'Win Probability': 'N/A',
                    'Confidence Range': 'N/A',
                    'Error Margin': 'N/A'
                })
        else:
            predictions.append({
                'Date': game_info['date'],
                'Opponent': game_info['opp_name'],
                'Location': location,
                'Prediction': 'INSUFFICIENT DATA',
                'Win Probability': 'N/A',
                'Confidence Range': 'N/A',
                'Error Margin': 'N/A'
            })
    
    predictions_df = pd.DataFrame(predictions)
    print("\n\nPredictions for OKC Thunder Next 5 Games:")
    print("=" * 80)
    print(predictions_df.to_string(index=False))
    
    wins_pred = sum([1 for p in predictions if p['Prediction'] == 'WIN'])
    losses_pred = sum([1 for p in predictions if p['Prediction'] == 'LOSS'])
    print(f"\n\nPredicted Record: {wins_pred}-{losses_pred}")
    print(f"Win Rate: {wins_pred/5*100:.1f}%")
    print(f"\nNote: All predictions include error margin of ±{best_error:.1f}%")
    
    return predictions_df

def create_enhanced_visualizations(results, error_margins, predictions_df):
    """Create enhanced visualizations with better styling"""
    print("\nCreating enhanced visualizations...")
    
    # Model comparison
    comparison_df = pd.DataFrame({
        name: {
            'accuracy': res['accuracy'],
            'cv_mean': res['cv_mean'],
            'cv_std': res['cv_std']
        }
        for name, res in results.items()
    }).T
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    bars1 = axes[0].bar(comparison_df.index, comparison_df['accuracy'], 
                color=['#3498db', '#2ecc71'], alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].set_title('Model Accuracy Comparison', fontsize=16, fontweight='bold', pad=20)
    axes[0].set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_ylim([0.5, 1.0])
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0].set_facecolor('#f8f9fa')
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=11)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=12, fontweight='bold')
    
    bars2 = axes[1].bar(comparison_df.index, comparison_df['cv_mean'], 
                yerr=comparison_df['cv_std'],
                color=['#3498db', '#2ecc71'], alpha=0.8, capsize=8, 
                edgecolor='black', linewidth=1.5, error_kw={'elinewidth': 2, 'capthick': 2})
    axes[1].set_title('Cross-Validation Mean Scores', fontsize=16, fontweight='bold', pad=20)
    axes[1].set_ylabel('CV Mean Score', fontsize=13, fontweight='bold')
    axes[1].set_ylim([0.5, 1.0])
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[1].set_facecolor('#f8f9fa')
    for bar, err in zip(bars2, comparison_df['cv_std']):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + err,
                    f'{height:.3f}\n±{err:.3f}',
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison_api.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: model_comparison_api.png")
    
    # Enhanced predictions visualization
    if len(predictions_df) > 0 and 'Win Probability' in predictions_df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        win_probs = predictions_df['Win Probability'].str.rstrip('%').astype(float)
        colors = ['#2ecc71' if p['Prediction'] == 'WIN' else '#e74c3c' for _, p in predictions_df.iterrows()]
        
        confidence_ranges = []
        for _, row in predictions_df.iterrows():
            if 'Confidence Range' in row and row['Confidence Range'] != 'N/A':
                lower, upper = row['Confidence Range'].split(' - ')
                lower = float(lower.rstrip('%'))
                upper = float(upper.rstrip('%'))
                confidence_ranges.append((lower, upper))
            else:
                prob = float(row['Win Probability'].rstrip('%'))
                confidence_ranges.append((prob, prob))
        
        bars = ax.bar(range(len(predictions_df)), win_probs, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=2)
        
        # Add error bars
        for i, (bar, (lower, upper)) in enumerate(zip(bars, confidence_ranges)):
            center = bar.get_height()
            ax.errorbar(i, center, yerr=[[center - lower], [upper - center]], 
                       fmt='none', color='black', capsize=8, capthick=2, elinewidth=2)
        
        # Customize
        ax.set_title('OKC Thunder Next 5 Games Predictions (2025 Season)\nWith Error Margins', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Win Probability (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Game', fontsize=14, fontweight='bold')
        
        # Create custom labels with dates and opponents
        labels = []
        for _, row in predictions_df.iterrows():
            date_parts = row['Date'].split('-')
            month_day = f"{date_parts[1]}/{date_parts[2]}"
            opp_short = row['Opponent'].split()[-1]  # Get last word (Suns, Clippers, etc.)
            labels.append(f"{month_day}\nvs {opp_short}\n({row['Location']})")
        
        ax.set_xticks(range(len(predictions_df)))
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.axhline(y=50, color='black', linestyle='--', alpha=0.5, linewidth=2, label='50% threshold')
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, win_probs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{prob:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))
        
        plt.tight_layout()
        plt.savefig('okc_predictions_api.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("Saved: okc_predictions_api.png")

def main():
    """Main execution function"""
    print("="*60)
    print("NBA Game Prediction Model - OKC Thunder (API Integrated)")
    print("Using Player Statistics, NBA API, and Actual Schedule")
    print("="*60)
    
    # Step 1: Test NBA API
    if NBA_API_AVAILABLE:
        print("\nStep 1: Testing NBA API...")
        api_verified = test_nba_api()
        if not api_verified:
            print("\n⚠ API test not verified. Continuing with local dataset...")
            use_api = False
        else:
            use_api = True
    else:
        print("\n⚠ NBA API not available. Using local dataset...")
        use_api = False
    
    # Step 2: Load data
    print("\nStep 2: Loading data...")
    data = load_data(use_api=use_api)
    
    if data[0] is None:
        print("Error: Could not load data. Exiting.")
        return
    
    # If using API, we need to convert API data to our format
    if use_api and isinstance(data[0], pd.DataFrame):
        # API returns game log DataFrame, need to process it
        print("Processing API data...")
        # For now, fall back to local player stats for training
        # API data will be used for recent form and predictions
        try:
            player_stats = pd.read_csv('NBA - Player Stats - Season 24-25/database_24_25.csv')
        except:
            print("⚠ Local player stats not found, using API data only...")
            player_stats = None
    else:
        player_stats, team_stats, team_summaries = data
    
    # Step 3: Process data
    print("\nStep 3: Processing data...")
    if player_stats is not None:
        game_stats = aggregate_player_stats_by_game(player_stats)
        game_features_df = create_game_features(game_stats)
        game_features_df = add_recent_form_features(game_features_df, window=5)
        game_features_df = preprocess_data(game_features_df)
    else:
        print("⚠ No local player stats available. Cannot create training data.")
        print("Please ensure the player stats CSV file is available.")
        return
    
    # Step 4: Prepare features
    feature_cols = [col for col in game_features_df.columns 
                    if col not in ['date', 'team1', 'team2', 'team1_wins']]
    X = game_features_df[feature_cols]
    y = game_features_df['team1_wins']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    # Step 5: Train models
    print("\nStep 4: Training models...")
    models, results = train_models(X_train, X_test, y_train, y_test)
    
    # Step 6: Calculate error margins
    error_margins = calculate_error_margins(results, X_test, y_test)
    
    # Step 7: Make predictions
    print("\nStep 5: Making predictions...")
    predictions_df = predict_okc_games(models, results, error_margins, game_features_df)
    
    # Step 8: Create visualizations
    print("\nStep 6: Creating visualizations...")
    create_enhanced_visualizations(results, error_margins, predictions_df)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - model_comparison_api.png")
    print("  - okc_predictions_api.png")

if __name__ == "__main__":
    main()
