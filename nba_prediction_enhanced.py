#!/usr/bin/env python3
"""
NBA Game Prediction Model for OKC Thunder - Enhanced Version
Using Player Statistics and Actual Schedule
Implements Random Forest and Gradient Boosting with Error Margins
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load all required datasets"""
    print("Loading datasets...")
    
    # Load player game-by-game stats
    player_stats = pd.read_csv('NBA - Player Stats - Season 24-25/database_24_25.csv')
    
    # Load team statistics
    team_stats = pd.read_csv('dataset/Team Stats Per Game.csv')
    team_summaries = pd.read_csv('dataset/Team Summaries.csv')
    
    print(f"Player Stats shape: {player_stats.shape}")
    print(f"Team Stats shape: {team_stats.shape}")
    print(f"Team Summaries shape: {team_summaries.shape}")
    
    return player_stats, team_stats, team_summaries

def aggregate_player_stats_by_game(player_stats):
    """
    Aggregate player statistics by team and game to create team-level game features.
    This provides more granular data than season averages.
    """
    print("\nAggregating player statistics by game...")
    
    # Group by team, opponent, and date to get team stats per game
    game_stats = player_stats.groupby(['Tm', 'Opp', 'Data', 'Res']).agg({
        'PTS': 'sum',  # Total team points
        'FG': 'sum', 'FGA': 'sum',
        '3P': 'sum', '3PA': 'sum',
        'FT': 'sum', 'FTA': 'sum',
        'TRB': 'sum',  # Total rebounds
        'AST': 'sum',  # Total assists
        'STL': 'sum',  # Total steals
        'BLK': 'sum',  # Total blocks
        'TOV': 'sum',  # Total turnovers
        'PF': 'sum',   # Total personal fouls
        'MP': 'sum',   # Total minutes played
        'GmSc': 'mean'  # Average game score (indicator of player performance quality)
    }).reset_index()
    
    # Calculate percentages
    game_stats['FG%'] = game_stats['FG'] / game_stats['FGA'].replace(0, np.nan)
    game_stats['3P%'] = game_stats['3P'] / game_stats['3PA'].replace(0, np.nan)
    game_stats['FT%'] = game_stats['FT'] / game_stats['FTA'].replace(0, np.nan)
    
    # Calculate per-game averages (normalize by minutes if needed)
    game_stats['PTS_per_48'] = (game_stats['PTS'] / game_stats['MP']) * 48 if game_stats['MP'].sum() > 0 else 0
    game_stats['AST_per_48'] = (game_stats['AST'] / game_stats['MP']) * 48 if game_stats['MP'].sum() > 0 else 0
    game_stats['REB_per_48'] = (game_stats['TRB'] / game_stats['MP']) * 48 if game_stats['MP'].sum() > 0 else 0
    
    print(f"Aggregated game stats shape: {game_stats.shape}")
    return game_stats

def create_game_features(game_stats):
    """
    Create features for each game matchup.
    For each game, we create features comparing team1 vs team2.
    """
    print("\nCreating game features from player statistics...")
    
    games = []
    
    # Get unique games
    unique_games = game_stats[['Tm', 'Opp', 'Data']].drop_duplicates()
    
    for idx, game in unique_games.iterrows():
        team1 = game['Tm']
        team2 = game['Opp']
        date = game['Data']
        
        # Get stats for both teams in this game
        team1_stats = game_stats[(game_stats['Tm'] == team1) & 
                                 (game_stats['Opp'] == team2) & 
                                 (game_stats['Data'] == date)]
        team2_stats = game_stats[(game_stats['Tm'] == team2) & 
                                 (game_stats['Opp'] == team1) & 
                                 (game_stats['Data'] == date)]
        
        if len(team1_stats) > 0 and len(team2_stats) > 0:
            t1 = team1_stats.iloc[0]
            t2 = team2_stats.iloc[0]
            
            # Determine winner (1 if team1 wins, 0 if team2 wins)
            result = 1 if t1['Res'] == 'W' else 0
            
            # Create comprehensive feature set
            features = {
                'date': date,
                'team1': team1,
                'team2': team2,
                
                # Team 1 offensive stats
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
                
                # Team 2 offensive stats
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
                
                # Derived features (differences)
                'pts_diff': t1['PTS'] - t2['PTS'],
                'fg_pct_diff': (t1['FG%'] if pd.notna(t1['FG%']) else 0) - (t2['FG%'] if pd.notna(t2['FG%']) else 0),
                'reb_diff': t1['TRB'] - t2['TRB'],
                'ast_diff': t1['AST'] - t2['AST'],
                'stl_diff': t1['STL'] - t2['STL'],
                'blk_diff': t1['BLK'] - t2['BLK'],
                'tov_diff': t1['TOV'] - t2['TOV'],
                'gmsc_diff': t1['GmSc'] - t2['GmSc'],
                
                # Efficiency metrics
                'team1_efficiency': t1['PTS'] / (t1['FGA'] + 0.44 * t1['FTA'] + t1['TOV']) if (t1['FGA'] + 0.44 * t1['FTA'] + t1['TOV']) > 0 else 0,
                'team2_efficiency': t2['PTS'] / (t2['FGA'] + 0.44 * t2['FTA'] + t2['TOV']) if (t2['FGA'] + 0.44 * t2['FTA'] + t2['TOV']) > 0 else 0,
                'efficiency_diff': (t1['PTS'] / (t1['FGA'] + 0.44 * t1['FTA'] + t1['TOV']) if (t1['FGA'] + 0.44 * t1['FTA'] + t1['TOV']) > 0 else 0) - \
                                   (t2['PTS'] / (t2['FGA'] + 0.44 * t2['FTA'] + t2['TOV']) if (t2['FGA'] + 0.44 * t2['FTA'] + t2['TOV']) > 0 else 0),
                
                # Target variable
                'team1_wins': result
            }
            
            games.append(features)
    
    game_features_df = pd.DataFrame(games)
    print(f"Created {len(game_features_df)} game features")
    return game_features_df

def add_recent_form_features(game_features_df, window=5):
    """
    Add recent form features - how teams have been performing in recent games.
    This captures momentum and current form, which is crucial for predictions.
    """
    print(f"\nAdding recent form features (last {window} games)...")
    
    # Sort by date
    game_features_df['date'] = pd.to_datetime(game_features_df['date'])
    game_features_df = game_features_df.sort_values('date')
    
    # Calculate rolling averages for each team
    for team in game_features_df['team1'].unique():
        team_games = game_features_df[(game_features_df['team1'] == team) | (game_features_df['team2'] == team)].copy()
        team_games = team_games.sort_values('date')
        
        # Calculate win rate in last N games
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
    
    # Fill NaN values
    game_features_df['team1_recent_win_rate'] = game_features_df['team1_recent_win_rate'].fillna(0.5)
    game_features_df['team2_recent_win_rate'] = game_features_df['team2_recent_win_rate'].fillna(0.5)
    game_features_df['recent_win_rate_diff'] = game_features_df['team1_recent_win_rate'] - game_features_df['team2_recent_win_rate']
    
    return game_features_df

def preprocess_data(game_features_df):
    """Preprocess the game data"""
    print("\nPreprocessing data...")
    
    # Remove rows with missing critical values
    game_features_df = game_features_df.dropna(subset=['team1_pts', 'team2_pts', 'team1_wins'])
    
    # Fill remaining NaN values with 0 or median
    numeric_cols = game_features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'team1_wins':
            game_features_df[col] = game_features_df[col].fillna(game_features_df[col].median() if game_features_df[col].median() != np.nan else 0)
    
    print(f"Dataset after cleaning: {game_features_df.shape}")
    print(f"\nClass distribution:")
    print(game_features_df['team1_wins'].value_counts())
    
    return game_features_df

def train_models(X_train, X_test, y_train, y_test):
    """
    Train two ensemble methods: Random Forest and Gradient Boosting.
    These are selected as they typically perform best for this type of problem.
    """
    print("\n" + "="*60)
    print("Training Ensemble Models")
    print("="*60)
    
    models = {}
    results = {}
    
    # Random Forest - Excellent for handling non-linear relationships and feature interactions
    print("\n1. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,  # More trees for better performance
        max_depth=15,      # Deeper trees to capture complex patterns
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',  # Reduces overfitting
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle any class imbalance
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
    
    # Gradient Boosting - Excellent for sequential learning and capturing complex patterns
    print("\n2. Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,      # More estimators for better performance
        learning_rate=0.05,    # Lower learning rate for more stable learning
        max_depth=6,           # Moderate depth to prevent overfitting
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,         # Stochastic gradient boosting
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
    """
    Calculate error margins and uncertainty estimates.
    Accounts for model uncertainty and prediction confidence.
    """
    print("\nCalculating error margins and uncertainty...")
    
    error_margins = {}
    
    for model_name, result in results.items():
        # Calculate prediction confidence (distance from 0.5 probability)
        probabilities = result['probabilities'][:, 1]  # Probability of win
        confidence = np.abs(probabilities - 0.5) * 2  # Scale to 0-1
        
        # Error margin based on cross-validation standard deviation
        cv_std = result['cv_std']
        base_error = cv_std * 100  # Convert to percentage
        
        # Adjust error margin based on confidence
        # Lower confidence = higher error margin
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
    """
    Create feature vector for prediction by averaging recent games for both teams.
    Properly handles cases where OKC was team1 or team2.
    """
    # Initialize feature dictionary
    features = {}
    
    # Process OKC stats
    okc_team1_games = okc_recent_games[okc_recent_games['team1'] == 'OKC']
    okc_team2_games = okc_recent_games[okc_recent_games['team2'] == 'OKC']
    
    # Process opponent stats
    opp_team1_games = opp_recent_games[opp_recent_games['team1'] == opp_recent_games.iloc[0]['team1'] if len(opp_recent_games) > 0 else pd.DataFrame()]
    opp_team2_games = opp_recent_games[opp_recent_games['team2'] == opp_recent_games.iloc[0]['team2'] if len(opp_recent_games) > 0 else pd.DataFrame()]
    
    # Calculate averages for each feature
    for col in feature_cols:
        if 'team1_' in col:
            # OKC as team1 stats
            okc_vals = okc_team1_games[col].values if len(okc_team1_games) > 0 else []
            # OKC as team2 stats (need to use team2 columns)
            if len(okc_team2_games) > 0:
                team2_col = col.replace('team1_', 'team2_')
                if team2_col in okc_team2_games.columns:
                    okc_vals = np.concatenate([okc_vals, okc_team2_games[team2_col].values])
            
            features[col] = np.mean(okc_vals) if len(okc_vals) > 0 else 0
            
        elif 'team2_' in col:
            # Opponent stats (they're team2 when OKC is team1, team1 when OKC is team2)
            opp_vals = []
            if len(opp_team1_games) > 0:
                team1_col = col.replace('team2_', 'team1_')
                if team1_col in opp_team1_games.columns:
                    opp_vals.extend(opp_team1_games[team1_col].values)
            if len(opp_team2_games) > 0:
                opp_vals.extend(opp_team2_games[col].values)
            
            features[col] = np.mean(opp_vals) if len(opp_vals) > 0 else 0
            
        elif 'recent' in col or 'diff' in col:
            # Derived features - calculate from base features
            if 'recent_win_rate_diff' in col:
                # Calculate recent win rates
                okc_wins = 0
                okc_total = len(okc_recent_games)
                for _, game in okc_recent_games.iterrows():
                    if (game['team1'] == 'OKC' and game['team1_wins'] == 1) or \
                       (game['team2'] == 'OKC' and game['team1_wins'] == 0):
                        okc_wins += 1
                okc_win_rate = okc_wins / okc_total if okc_total > 0 else 0.5
                
                opp_wins = 0
                opp_total = len(opp_recent_games)
                for _, game in opp_recent_games.iterrows():
                    opp_team = game['team1'] if game['team1'] != 'OKC' else game['team2']
                    if (game['team1'] == opp_team and game['team1_wins'] == 1) or \
                       (game['team2'] == opp_team and game['team1_wins'] == 0):
                        opp_wins += 1
                opp_win_rate = opp_wins / opp_total if opp_total > 0 else 0.5
                
                features[col] = okc_win_rate - opp_win_rate
            else:
                # For other derived features, calculate from averages
                features[col] = 0  # Will be calculated from base features
    
    # Calculate derived features from averages
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
    
    # Create feature vector in correct order
    feature_vector = np.array([features.get(col, 0) for col in feature_cols]).reshape(1, -1)
    return feature_vector

def predict_okc_games(models, results, error_margins, game_features_df, team_stats, team_summaries):
    """
    Predict OKC Thunder's next 5 games using actual schedule.
    Games:
    1. Dec 10 vs Phoenix Suns (home)
    2. Dec 17 vs LA Clippers (home)
    3. Dec 19 @ Minnesota Timberwolves (away)
    4. Dec 22 vs Memphis Grizzlies (home)
    5. Dec 23 @ San Antonio Spurs (away)
    """
    print("\n" + "="*60)
    print("Predicting OKC Thunder Next 5 Games")
    print("="*60)
    
    # Actual schedule
    schedule = [
        {'date': '2024-12-10', 'opponent': 'PHO', 'location': 'home', 'opp_name': 'Phoenix Suns'},
        {'date': '2024-12-17', 'opponent': 'LAC', 'location': 'home', 'opp_name': 'LA Clippers'},
        {'date': '2024-12-19', 'opponent': 'MIN', 'location': 'away', 'opp_name': 'Minnesota Timberwolves'},
        {'date': '2024-12-22', 'opponent': 'MEM', 'location': 'home', 'opp_name': 'Memphis Grizzlies'},
        {'date': '2024-12-23', 'opponent': 'SAS', 'location': 'away', 'opp_name': 'San Antonio Spurs'}
    ]
    
    # Get OKC's recent performance
    okc_games = game_features_df[(game_features_df['team1'] == 'OKC') | (game_features_df['team2'] == 'OKC')].copy()
    okc_games = okc_games.sort_values('date')
    
    if len(okc_games) > 0:
        recent_okc = okc_games.tail(10)  # Last 10 games
        okc_wins = 0
        for _, game in recent_okc.iterrows():
            if game['team1'] == 'OKC' and game['team1_wins'] == 1:
                okc_wins += 1
            elif game['team2'] == 'OKC' and game['team1_wins'] == 0:
                okc_wins += 1
        
        print(f"\nOKC Thunder Recent Performance:")
        print(f"  Last 10 games: {okc_wins}-{10-okc_wins}")
        print(f"  Win rate: {okc_wins/10:.1%}")
    
    # Select best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    best_model = models[best_model_name]
    best_error = error_margins[best_model_name]['adjusted_error']
    
    print(f"\nUsing {best_model_name} (Best CV Score: {results[best_model_name]['cv_mean']:.4f})")
    print(f"Error Margin: ±{best_error:.2f}%")
    
    # Get feature columns (excluding non-feature columns)
    feature_cols = [col for col in game_features_df.columns 
                    if col not in ['date', 'team1', 'team2', 'team1_wins']]
    
    predictions = []
    
    for game_info in schedule:
        opp = game_info['opponent']
        location = game_info['location']
        
        # Get recent games for OKC (last 5)
        okc_recent = okc_games.tail(5) if len(okc_games) >= 5 else okc_games
        
        # Get opponent's recent games
        opp_games = game_features_df[(game_features_df['team1'] == opp) | (game_features_df['team2'] == opp)].copy()
        opp_games = opp_games.sort_values('date').tail(5) if len(opp_games) >= 5 else opp_games
        
        # Create feature vector for prediction
        if len(okc_recent) > 0 and len(opp_games) > 0:
            try:
                pred_features = create_prediction_features_from_recent_games(okc_recent, opp_games, feature_cols)
                
                # Make prediction
                win_prob = best_model.predict_proba(pred_features)[0][1]
                prediction = best_model.predict(pred_features)[0]
                
                # Apply error margin
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
            # Fallback if not enough data
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
    
    wins = sum([1 for p in predictions if p['Prediction'] == 'WIN'])
    losses = sum([1 for p in predictions if p['Prediction'] == 'LOSS'])
    print(f"\n\nPredicted Record: {wins}-{losses}")
    print(f"Win Rate: {wins/5*100:.1f}%")
    print(f"\nNote: All predictions include error margin of ±{best_error:.1f}%")
    print("Actual outcomes may vary due to factors like injuries, rest, and game flow.")
    
    return predictions_df

def create_visualizations(results, error_margins, predictions_df):
    """Create visualization plots"""
    print("\nCreating visualizations...")
    
    # Model comparison
    comparison_df = pd.DataFrame({
        name: {
            'accuracy': res['accuracy'],
            'cv_mean': res['cv_mean'],
            'cv_std': res['cv_std']
        }
        for name, res in results.items()
    }).T
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(comparison_df.index, comparison_df['accuracy'], 
                color=['#3498db', '#2ecc71'])
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_ylim([0.5, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    axes[1].bar(comparison_df.index, comparison_df['cv_mean'], 
                yerr=comparison_df['cv_std'],
                color=['#3498db', '#2ecc71'], capsize=5)
    axes[1].set_title('Cross-Validation Mean Scores', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('CV Mean Score', fontsize=12)
    axes[1].set_ylim([0.5, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('model_comparison_enhanced.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison_enhanced.png")
    
    # Predictions with error bars
    if len(predictions_df) > 0 and 'Win Probability' in predictions_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        win_probs = predictions_df['Win Probability'].str.rstrip('%').astype(float)
        colors = ['green' if p['Prediction'] == 'WIN' else 'red' for _, p in predictions_df.iterrows()]
        
        # Extract confidence ranges
        confidence_ranges = []
        for _, row in predictions_df.iterrows():
            if 'Confidence Range' in row and row['Confidence Range'] != 'N/A':
                lower, upper = row['Confidence Range'].split(' - ')
                lower = float(lower.rstrip('%'))
                upper = float(upper.rstrip('%'))
                confidence_ranges.append((lower, upper))
            else:
                confidence_ranges.append((win_probs.iloc[len(confidence_ranges)], win_probs.iloc[len(confidence_ranges)]))
        
        bars = ax.bar(range(len(predictions_df)), win_probs, color=colors, alpha=0.7)
        
        # Add error bars
        for i, (bar, (lower, upper)) in enumerate(zip(bars, confidence_ranges)):
            center = bar.get_height()
            ax.errorbar(i, center, yerr=[[center - lower], [upper - center]], 
                       fmt='none', color='black', capsize=5, capthick=2)
        
        ax.set_title('OKC Thunder Next 5 Games Predictions (with Error Margins)', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel('Win Probability (%)', fontsize=12)
        ax.set_xlabel('Game', fontsize=12)
        ax.set_xticks(range(len(predictions_df)))
        ax.set_xticklabels([f"{row['Date']}\nvs {row['Opponent']}" for _, row in predictions_df.iterrows()], 
                          rotation=45, ha='right')
        ax.set_ylim([0, 100])
        ax.axhline(y=50, color='black', linestyle='--', alpha=0.3, label='50% threshold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, prob) in enumerate(zip(bars, win_probs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{prob:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('okc_predictions_enhanced.png', dpi=300, bbox_inches='tight')
        print("Saved: okc_predictions_enhanced.png")

def main():
    """Main execution function"""
    print("="*60)
    print("NBA Game Prediction Model - OKC Thunder (Enhanced)")
    print("Using Player Statistics and Actual Schedule")
    print("="*60)
    
    # Load data
    player_stats, team_stats, team_summaries = load_data()
    
    # Aggregate player stats by game
    game_stats = aggregate_player_stats_by_game(player_stats)
    
    # Create game features
    game_features_df = create_game_features(game_stats)
    
    # Add recent form features
    game_features_df = add_recent_form_features(game_features_df, window=5)
    
    # Preprocess
    game_features_df = preprocess_data(game_features_df)
    
    # Prepare features and target
    feature_cols = [col for col in game_features_df.columns 
                    if col not in ['date', 'team1', 'team2', 'team1_wins']]
    X = game_features_df[feature_cols]
    y = game_features_df['team1_wins']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    # Train models
    models, results = train_models(X_train, X_test, y_train, y_test)
    
    # Calculate error margins
    error_margins = calculate_error_margins(results, X_test, y_test)
    
    # Create visualizations
    create_visualizations(results, error_margins, pd.DataFrame())
    
    # Make predictions
    predictions_df = predict_okc_games(models, results, error_margins, 
                                      game_features_df, team_stats, team_summaries)
    
    # Update visualizations with predictions
    create_visualizations(results, error_margins, predictions_df)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - model_comparison_enhanced.png")
    print("  - okc_predictions_enhanced.png")
    print("\nSee DESIGN_DECISIONS.md for detailed explanations of methodology.")

if __name__ == "__main__":
    main()
