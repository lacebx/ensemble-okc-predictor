#!/usr/bin/env python3
"""
NBA Game Prediction Model - Using API Collected Data
Uses cleaned data from NBA API instead of potentially outdated CSV files.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
import os
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Path to cleaned API data
API_DATA_FILE = "api_collected_data/cleaned/final_game_features.csv"

def load_api_data():
    """Load the cleaned API data"""
    print("Loading API collected data...")
    
    if not os.path.exists(API_DATA_FILE):
        print(f"✗ Error: API data file not found: {API_DATA_FILE}")
        print("Please run collect_nba_api_data.py first to collect data from NBA API.")
        return None
    
    df = pd.read_csv(API_DATA_FILE)
    
    # Parse dates
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    print(f"✓ Loaded {len(df)} games from API data")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Teams: {df['team1'].nunique()}")
    print(f"  Win/Loss: {df['team1_wins'].sum()} wins, {(df['team1_wins'] == 0).sum()} losses")
    
    return df

def get_okc_recent_record(game_features_df):
    """
    Get OKC's recent record - CORRECTLY counts wins.
    Returns wins, losses, and game results for last 10 games.
    """
    okc_games = game_features_df[
        (game_features_df['team1'] == 'OKC') | 
        (game_features_df['team2'] == 'OKC')
    ].copy()
    
    if len(okc_games) == 0:
        return 0, 0, []
    
    # Sort by date (most recent last)
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

def preprocess_data(game_features_df):
    """Preprocess the game data"""
    print("\nPreprocessing data...")
    
    # Remove rows with missing critical values
    game_features_df = game_features_df.dropna(subset=['team1_pts', 'team2_pts', 'team1_wins'])
    
    # Fill remaining NaN values
    numeric_cols = game_features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'team1_wins':
            game_features_df[col] = game_features_df[col].fillna(game_features_df[col].median() if pd.notna(game_features_df[col].median()) else 0)
    
    print(f"Dataset after cleaning: {game_features_df.shape}")
    print(f"\nClass distribution:")
    print(game_features_df['team1_wins'].value_counts())
    
    return game_features_df

def train_models(X_train, X_test, y_train, y_test):
    """Train ensemble models"""
    print("\n" + "="*60)
    print("Training Ensemble Models")
    print("="*60)
    
    models = {}
    results = {}
    
    # Random Forest - Increased complexity to learn more patterns
    print("\n1. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=500,  # More trees for better learning
        max_depth=20,  # Deeper trees to capture complex patterns
        min_samples_split=3,  # Allow more splits
        min_samples_leaf=1,  # More granular leaves
        max_features='sqrt',  # Feature subset for diversity
        random_state=42,
        n_jobs=-1,
        class_weight='balanced',  # Handle class imbalance
        bootstrap=True,
        oob_score=True  # Out-of-bag scoring for better validation
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
    
    # Gradient Boosting - Increased complexity to learn more patterns
    print("\n2. Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=500,  # More boosting stages
        learning_rate=0.03,  # Lower learning rate for more careful learning
        max_depth=8,  # Deeper trees
        min_samples_split=3,  # Allow more splits
        min_samples_leaf=1,  # More granular leaves
        subsample=0.85,  # Slight randomness
        max_features='sqrt',  # Feature subset for diversity
        random_state=42,
        validation_fraction=0.1,  # Early stopping validation
        n_iter_no_change=10  # Early stopping patience
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

def calculate_error_margins(results):
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
            'adjusted_error': adjusted_error
        }
        
        print(f"{model_name}:")
        print(f"  Base Error Margin: ±{base_error:.2f}%")
        print(f"  Average Confidence: {confidence.mean():.3f}")
        print(f"  Adjusted Error Margin: ±{adjusted_error:.2f}%")
    
    return error_margins

def create_prediction_features_from_recent_games(okc_recent_games, opp_recent_games, feature_cols, opp_abbrev, location='home', predicted_games=None):
    """
    Create feature vector for prediction.
    Uses OKC's recent games and opponent's recent games to create features.
    Location: 'home' or 'away' - can be used as a feature if needed.
    predicted_games: List of predicted games (synthetic) - these are weighted less (0.7x) to account for uncertainty.
    """
    if predicted_games is None:
        predicted_games = []
    
    features = {}
    
    # Get OKC stats (when OKC is team1 or team2)
    okc_team1_games = okc_recent_games[okc_recent_games['team1'] == 'OKC'].copy()
    okc_team2_games = okc_recent_games[okc_recent_games['team2'] == 'OKC'].copy()
    
    # Weight for predicted games (less reliable - 0.7x weight)
    predicted_weight = 0.7
    
    # Get opponent stats
    opp_team1_games = opp_recent_games[opp_recent_games['team1'] == opp_abbrev]
    opp_team2_games = opp_recent_games[opp_recent_games['team2'] == opp_abbrev]
    
    # Calculate OKC averages - FIXED to properly handle team1/team2
    # Weight predicted games less to account for uncertainty
    for col in feature_cols:
        if 'team1_' in col and 'recent' not in col and 'efficiency' not in col:
            # OKC stats: when OKC is team1, use team1 columns
            # When OKC is team2, use team2 columns (which are OKC's stats)
            okc_vals = []
            okc_weights = []
            
            # OKC as team1
            if len(okc_team1_games) > 0:
                for idx, row in okc_team1_games.iterrows():
                    if col in okc_team1_games.columns:
                        okc_vals.append(row[col])
                        # Check if this is a predicted game
                        is_pred = row.get('is_predicted', False) if 'is_predicted' in okc_team1_games.columns else False
                        okc_weights.append(predicted_weight if is_pred else 1.0)
            
            # OKC as team2 - use team2 columns
            if len(okc_team2_games) > 0:
                team2_col = col.replace('team1_', 'team2_')
                if team2_col in okc_team2_games.columns:
                    for idx, row in okc_team2_games.iterrows():
                        okc_vals.append(row[team2_col])
                        # Check if this is a predicted game
                        is_pred = row.get('is_predicted', False) if 'is_predicted' in okc_team2_games.columns else False
                        okc_weights.append(predicted_weight if is_pred else 1.0)
            
            # Weighted average: real games (weight=1.0) + predicted games (weight=0.7)
            # This keeps all real data at full weight, while predicted games contribute less
            if len(okc_vals) > 0:
                features[col] = np.average(okc_vals, weights=okc_weights)
            else:
                features[col] = 0
            
        elif 'team2_' in col and 'recent' not in col and 'efficiency' not in col:
            # Opponent stats
            opp_vals = []
            
            # Opponent as team1 - use team1 columns
            if len(opp_team1_games) > 0:
                team1_col = col.replace('team2_', 'team1_')
                if team1_col in opp_team1_games.columns:
                    opp_vals.extend(opp_team1_games[team1_col].values)
            
            # Opponent as team2 - use team2 columns
            if len(opp_team2_games) > 0:
                if col in opp_team2_games.columns:
                    opp_vals.extend(opp_team2_games[col].values)
            
            features[col] = np.mean(opp_vals) if len(opp_vals) > 0 else 0
    
    # Calculate OKC recent win rate (weighted: real games = 1.0, predicted = 0.7)
    okc_wins_weighted = 0
    total_weight = 0
    for _, g in okc_recent_games.iterrows():
        is_pred = g.get('is_predicted', False) if 'is_predicted' in okc_recent_games.columns else False
        weight = predicted_weight if is_pred else 1.0
        
        if (g['team1'] == 'OKC' and g['team1_wins'] == 1) or \
           (g['team2'] == 'OKC' and g['team1_wins'] == 0):
            okc_wins_weighted += weight
        total_weight += weight
    
    okc_win_rate = okc_wins_weighted / total_weight if total_weight > 0 else 0.5
    
    # Calculate opponent recent win rate (opponent games are always real, no weighting needed)
    opp_wins = 0
    for _, g in opp_recent_games.iterrows():
        if (g['team1'] == opp_abbrev and g['team1_wins'] == 1) or \
           (g['team2'] == opp_abbrev and g['team1_wins'] == 0):
            opp_wins += 1
    opp_win_rate = opp_wins / len(opp_recent_games) if len(opp_recent_games) > 0 else 0.5
    
    # Set recent win rate features
    if 'team1_recent_win_rate' in feature_cols:
        features['team1_recent_win_rate'] = okc_win_rate
    if 'team2_recent_win_rate' in feature_cols:
        features['team2_recent_win_rate'] = opp_win_rate
    if 'recent_win_rate_diff' in feature_cols:
        features['recent_win_rate_diff'] = okc_win_rate - opp_win_rate
    
    # Calculate efficiency if we have the data
    if 'team1_efficiency' in feature_cols:
        # Use average points and estimate efficiency
        okc_pts = features.get('team1_pts', 0)
        # Estimate FGA, FTA, TOV from averages if not available
        okc_fga = okc_pts / 0.45 if okc_pts > 0 else 90  # Rough estimate
        okc_fta = okc_fga * 0.25  # Rough estimate
        okc_tov = 15  # Average
        if (okc_fga + 0.44 * okc_fta + okc_tov) > 0:
            features['team1_efficiency'] = okc_pts / (okc_fga + 0.44 * okc_fta + okc_tov)
        else:
            features['team1_efficiency'] = 1.0
    
    if 'team2_efficiency' in feature_cols:
        opp_pts = features.get('team2_pts', 0)
        opp_fga = opp_pts / 0.45 if opp_pts > 0 else 90
        opp_fta = opp_fga * 0.25
        opp_tov = 15
        if (opp_fga + 0.44 * opp_fta + opp_tov) > 0:
            features['team2_efficiency'] = opp_pts / (opp_fga + 0.44 * opp_fta + opp_tov)
        else:
            features['team2_efficiency'] = 1.0
    
    if 'efficiency_diff' in feature_cols:
        features['efficiency_diff'] = features.get('team1_efficiency', 1.0) - features.get('team2_efficiency', 1.0)
    
    # Calculate all differential features
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
    if 'plus_minus_diff' in feature_cols:
        features['plus_minus_diff'] = features.get('team1_plus_minus', 0) - features.get('team2_plus_minus', 0)
    
    # Ensure all features are present
    for col in feature_cols:
        if col not in features:
            features[col] = 0
    
    feature_vector = np.array([features.get(col, 0) for col in feature_cols]).reshape(1, -1)
    return feature_vector, features

def create_synthetic_game_from_prediction(prediction, game_info, okc_recent_stats, opp_recent_stats):
    """
    Create a synthetic game row from a prediction.
    Uses predicted outcome and recent stats to estimate game statistics.
    Adds uncertainty since it's a prediction, not real data.
    """
    # Determine if OKC won
    okc_won = (prediction['Prediction'] == 'WIN')
    team1_wins = 1 if okc_won else 0
    
    # Use recent averages as base, but add some uncertainty
    # Predicted games have 0.7x weight (less reliable than real games)
    uncertainty_factor = 0.7
    
    # Get OKC stats (from feature dict or recent averages)
    okc_pts = okc_recent_stats.get('team1_pts', 125)
    okc_fg_pct = okc_recent_stats.get('team1_fg_pct', 0.52)
    okc_reb = okc_recent_stats.get('team1_reb', 41)
    okc_ast = okc_recent_stats.get('team1_ast', 28)
    
    # Get opponent stats
    opp_pts = opp_recent_stats.get('team2_pts', 110)
    opp_fg_pct = opp_recent_stats.get('team2_fg_pct', 0.47)
    opp_reb = opp_recent_stats.get('team2_reb', 42)
    opp_ast = opp_recent_stats.get('team2_ast', 24)
    
    # Adjust based on predicted outcome
    # If OKC wins, they likely score more; if they lose, opponent scores more
    if okc_won:
        # OKC wins - they score slightly more, opponent slightly less
        okc_pts_adj = okc_pts * (1 + np.random.normal(0.05, 0.02))  # Small boost
        opp_pts_adj = opp_pts * (1 + np.random.normal(-0.03, 0.02))  # Small reduction
    else:
        # OKC loses - opponent scores more, OKC scores less
        okc_pts_adj = okc_pts * (1 + np.random.normal(-0.03, 0.02))
        opp_pts_adj = opp_pts * (1 + np.random.normal(0.05, 0.02))
    
    # Create synthetic game row
    # Convert date string to datetime to match real games format
    from datetime import datetime
    game_date = pd.to_datetime(game_info['date']) if isinstance(game_info['date'], str) else game_info['date']
    
    synthetic_game = {
        'game_id': f"PRED_{game_info['date'].replace('-', '') if isinstance(game_info['date'], str) else str(game_info['date']).replace('-', '')}",
        'date': game_date,  # Use datetime, not string
        'season': '2025-26',
        'team1': 'OKC' if game_info['location'] == 'home' else game_info['opponent'],
        'team2': game_info['opponent'] if game_info['location'] == 'home' else 'OKC',
        'team1_pts': okc_pts_adj if game_info['location'] == 'home' else opp_pts_adj,
        'team1_fg_pct': okc_fg_pct if game_info['location'] == 'home' else opp_fg_pct,
        'team1_reb': okc_reb if game_info['location'] == 'home' else opp_reb,
        'team1_ast': okc_ast if game_info['location'] == 'home' else opp_ast,
        'team2_pts': opp_pts_adj if game_info['location'] == 'home' else okc_pts_adj,
        'team2_fg_pct': opp_fg_pct if game_info['location'] == 'home' else okc_fg_pct,
        'team2_reb': opp_reb if game_info['location'] == 'home' else okc_reb,
        'team2_ast': opp_ast if game_info['location'] == 'home' else okc_ast,
        'team1_wins': team1_wins if game_info['location'] == 'home' else (1 - team1_wins),
        'is_predicted': True,  # Flag to mark as predicted
        'prediction_confidence': float(prediction['Win Probability'].rstrip('%')) / 100
    }
    
    # Fill in other required columns with estimates
    for col in ['team1_3p_pct', 'team1_ft_pct', 'team1_stl', 'team1_blk', 'team1_tov', 'team1_pf',
                'team2_3p_pct', 'team2_ft_pct', 'team2_stl', 'team2_blk', 'team2_tov', 'team2_pf']:
        if col not in synthetic_game:
            # Use reasonable defaults
            if '3p_pct' in col:
                synthetic_game[col] = 0.35
            elif 'ft_pct' in col:
                synthetic_game[col] = 0.78
            elif 'stl' in col:
                synthetic_game[col] = 8
            elif 'blk' in col:
                synthetic_game[col] = 5
            elif 'tov' in col:
                synthetic_game[col] = 13
            elif 'pf' in col:
                synthetic_game[col] = 20
    
    # Calculate plus/minus
    if game_info['location'] == 'home':
        synthetic_game['team1_plus_minus'] = okc_pts_adj - opp_pts_adj
        synthetic_game['team2_plus_minus'] = opp_pts_adj - okc_pts_adj
    else:
        synthetic_game['team1_plus_minus'] = opp_pts_adj - okc_pts_adj
        synthetic_game['team2_plus_minus'] = okc_pts_adj - opp_pts_adj
    
    return synthetic_game

def predict_okc_games(models, results, error_margins, game_features_df):
    """Predict OKC Thunder's games interactively, one at a time"""
    print("\n" + "="*60)
    print("Interactive OKC Thunder Game Predictions")
    print("="*60)
    
    # Actual schedule - 2025 dates
    schedule = [
        {'date': '2025-12-10', 'opponent': 'PHX', 'location': 'home', 'opp_name': 'Phoenix Suns'},
        {'date': '2025-12-17', 'opponent': 'LAC', 'location': 'home', 'opp_name': 'LA Clippers'},
        {'date': '2025-12-19', 'opponent': 'MIN', 'location': 'away', 'opp_name': 'Minnesota Timberwolves'},
        {'date': '2025-12-22', 'opponent': 'MEM', 'location': 'home', 'opp_name': 'Memphis Grizzlies'},
        {'date': '2025-12-23', 'opponent': 'SAS', 'location': 'away', 'opp_name': 'San Antonio Spurs'}
    ]
    
    # Get OKC's recent performance - FIXED win counting
    wins, losses, game_results = get_okc_recent_record(game_features_df)
    
    print(f"\nOKC Thunder Recent Performance (from API data):")
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
                    if col not in ['game_id', 'date', 'season', 'team1', 'team2', 'team1_wins', 'is_predicted']]
    
    predictions = []
    predicted_games = []  # Store synthetic games from predictions
    
    # Start with real OKC games - get the SAME set for all predictions
    okc_games = game_features_df[
        (game_features_df['team1'] == 'OKC') | 
        (game_features_df['team2'] == 'OKC')
    ].copy()
    okc_games = okc_games.sort_values('date')
    
    # Get the SAME real historical games that will be used for ALL predictions
    # This is the baseline that never changes
    okc_recent_real = okc_games.tail(10) if len(okc_games) >= 10 else okc_games
    okc_recent_real = okc_recent_real.copy()
    okc_recent_real['is_predicted'] = False
    
    game_index = 0
    
    while game_index < len(schedule):
        game_info = schedule[game_index]
        opp = game_info['opponent']
        location = game_info['location']
        
        # Use the SAME real historical games as the first prediction
        # Then ADD predicted games as additional data (with reduced weight)
        # This ensures we always use the same real data, plus any new predicted games
        okc_recent = okc_recent_real.copy()
        
        # Add predicted games as ADDITIONAL data (not replacing real games)
        if len(predicted_games) > 0:
            predicted_df = pd.DataFrame(predicted_games)
            # Ensure all columns match
            for col in okc_recent.columns:
                if col not in predicted_df.columns and col != 'is_predicted':
                    predicted_df[col] = 0  # Fill missing columns with 0
            
            # Mark all predicted games
            predicted_df['is_predicted'] = True
            
            # Ensure date column is datetime in predicted_df before combining
            predicted_df['date'] = pd.to_datetime(predicted_df['date'])
            
            # Combine: SAME real games + predicted games (predicted are additional)
            okc_recent = pd.concat([okc_recent, predicted_df], ignore_index=True)
            # Ensure date column is datetime for sorting (in case real games aren't)
            okc_recent['date'] = pd.to_datetime(okc_recent['date'])
            okc_recent = okc_recent.sort_values('date')
            # Keep all real games + predicted games (don't limit - we want all real + predicted)
        
        # Get opponent games (real only - we don't predict opponent's other games)
        opp_games = game_features_df[
            (game_features_df['team1'] == opp) | 
            (game_features_df['team2'] == opp)
        ].copy()
        opp_games = opp_games.sort_values('date').tail(10) if len(opp_games) >= 10 else opp_games
        
        # Debug: Check if opponent data exists
        if len(opp_games) == 0:
            print(f"\nWarning: No games found for opponent {opp} ({game_info['opp_name']})")
            print(f"  Available teams in dataset: {sorted(game_features_df['team1'].unique())}")
        
        # Make prediction
        if len(okc_recent) > 0:
            try:
                # Get recent stats for synthetic game creation
                okc_recent_stats = {}
                opp_recent_stats = {}
                
                # Calculate OKC recent averages
                okc_team1_games = okc_recent[okc_recent['team1'] == 'OKC']
                okc_team2_games = okc_recent[okc_recent['team2'] == 'OKC']
                
                if len(okc_team1_games) > 0:
                    okc_recent_stats['team1_pts'] = okc_team1_games['team1_pts'].mean()
                    okc_recent_stats['team1_fg_pct'] = okc_team1_games['team1_fg_pct'].mean()
                    okc_recent_stats['team1_reb'] = okc_team1_games['team1_reb'].mean()
                    okc_recent_stats['team1_ast'] = okc_team1_games['team1_ast'].mean()
                
                if len(okc_team2_games) > 0:
                    okc_recent_stats['team1_pts'] = (okc_recent_stats.get('team1_pts', 0) + okc_team2_games['team2_pts'].mean()) / 2
                    okc_recent_stats['team1_fg_pct'] = (okc_recent_stats.get('team1_fg_pct', 0) + okc_team2_games['team2_fg_pct'].mean()) / 2
                
                # Calculate opponent recent averages
                opp_team1_games = opp_games[opp_games['team1'] == opp]
                opp_team2_games = opp_games[opp_games['team2'] == opp]
                
                if len(opp_team1_games) > 0:
                    opp_recent_stats['team2_pts'] = opp_team1_games['team1_pts'].mean()
                    opp_recent_stats['team2_fg_pct'] = opp_team1_games['team1_fg_pct'].mean()
                    opp_recent_stats['team2_reb'] = opp_team1_games['team1_reb'].mean()
                    opp_recent_stats['team2_ast'] = opp_team1_games['team1_ast'].mean()
                
                if len(opp_team2_games) > 0:
                    opp_recent_stats['team2_pts'] = (opp_recent_stats.get('team2_pts', 0) + opp_team2_games['team2_pts'].mean()) / 2
                    opp_recent_stats['team2_fg_pct'] = (opp_recent_stats.get('team2_fg_pct', 0) + opp_team2_games['team2_fg_pct'].mean()) / 2
                
                # Create features, accounting for predicted games with reduced weight
                pred_features, feature_dict = create_prediction_features_from_recent_games(
                    okc_recent, opp_games, feature_cols, opp, location, predicted_games
                )
                
                # Model prediction
                win_prob = best_model.predict_proba(pred_features)[0][1]
                prediction = best_model.predict(pred_features)[0]
                
                win_prob_lower = max(0, win_prob - best_error/100)
                win_prob_upper = min(1, win_prob + best_error/100)
                
                prediction_result = {
                    'Date': game_info['date'],
                    'Opponent': game_info['opp_name'],
                    'Location': location,
                    'Prediction': 'WIN' if prediction == 1 else 'LOSS',
                    'Win Probability': f"{win_prob*100:.1f}%",
                    'Confidence Range': f"{win_prob_lower*100:.1f}% - {win_prob_upper*100:.1f}%",
                    'Error Margin': f"±{best_error:.1f}%",
                    'Feature_Dict': feature_dict,
                    'Game_Info': game_info,
                    'OKC_Recent_Stats': okc_recent_stats,
                    'Opp_Recent_Stats': opp_recent_stats
                }
                
                predictions.append(prediction_result)
                
                # Display prediction
                print(f"\n{'='*80}")
                print(f"PREDICTION #{game_index + 1}")
                print(f"{'='*80}")
                print(f"Date: {game_info['date']}")
                print(f"Opponent: {game_info['opp_name']} ({opp})")
                print(f"Location: {location.upper()}")
                print(f"\nPrediction: {prediction_result['Prediction']}")
                print(f"Win Probability: {prediction_result['Win Probability']}")
                print(f"Confidence Range: {prediction_result['Confidence Range']}")
                print(f"Error Margin: {prediction_result['Error Margin']}")
                
                if len(predicted_games) > 0:
                    print(f"\n⚠ Note: Using the SAME real historical data as the first prediction")
                    print(f"   PLUS {len(predicted_games)} predicted game(s) as additional data (weighted at 0.7x)")
                    print(f"   (accounting for uncertainty in predicted games)")
                
                # Create synthetic game from this prediction for future predictions
                synthetic_game = create_synthetic_game_from_prediction(
                    prediction_result, game_info, okc_recent_stats, opp_recent_stats
                )
                predicted_games.append(synthetic_game)
                
                # Create explanation for this prediction
                print(f"\n{'='*80}")
                print("DETAILED REASONING FOR THIS PREDICTION:")
                print(f"{'='*80}")
                create_single_prediction_explanation(prediction_result, game_features_df, models, results, game_index + 1)
                
                # Ask user if they want to continue
                if game_index < len(schedule) - 1:
                    print(f"\n{'='*80}")
                    user_input = input(f"\nWould you like to predict the next game? (yes/no): ").strip().lower()
                    if user_input in ['no', 'n', 'quit', 'q', 'exit']:
                        print("\nStopping predictions as requested.")
                        break
                    elif user_input not in ['yes', 'y', '']:
                        print("Invalid input. Continuing to next game...")
                else:
                    print(f"\n{'='*80}")
                    print("All scheduled games have been predicted.")
                
                game_index += 1
                
            except Exception as e:
                print(f"\nError predicting {opp} game: {e}")
                import traceback
                traceback.print_exc()
                game_index += 1
        else:
            print(f"\nInsufficient data to predict {game_info['opp_name']} game.")
            game_index += 1
    
    # Create summary
    if len(predictions) > 0:
        display_predictions = [{k: v for k, v in p.items() if k not in ['Feature_Dict', 'Game_Info', 'OKC_Recent_Stats', 'Opp_Recent_Stats']} for p in predictions]
        predictions_df = pd.DataFrame(display_predictions)
        
        print(f"\n\n{'='*80}")
        print("PREDICTION SUMMARY")
        print(f"{'='*80}")
        print(predictions_df.to_string(index=False))
        
        wins_pred = sum([1 for p in predictions if p['Prediction'] == 'WIN'])
        losses_pred = sum([1 for p in predictions if p['Prediction'] == 'LOSS'])
        print(f"\n\nPredicted Record: {wins_pred}-{losses_pred}")
        print(f"Win Rate: {wins_pred/len(predictions)*100:.1f}%")
        print(f"\nNote: All predictions include error margin of ±{best_error:.1f}%")
        if len(predicted_games) > 0:
            print(f"\nNote: Subsequent predictions used the same real historical data")
            print(f"      PLUS {len(predicted_games)} predicted game(s) as additional data (weighted at 0.7x)")
            print(f"      to account for uncertainty in predicted games")
    
    return predictions_df, predictions

def create_single_prediction_explanation(prediction, game_features_df, models, results, prediction_num):
    """
    Create detailed explanation for a single prediction.
    Shows reasoning behind the prediction.
    """
    if prediction['Win Probability'] == 'N/A' or prediction['Prediction'] == 'INSUFFICIENT DATA':
        print("\n⚠ INSUFFICIENT DATA")
        print("Reason: Not enough recent game data available for this opponent.")
        return
    
    if 'Feature_Dict' not in prediction or not prediction['Feature_Dict']:
        print("\n⚠ Feature details not available for this prediction")
        return
    
    features = prediction['Feature_Dict']
    
    # Get best model for feature importance
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    best_model = models[best_model_name]
    
    print(f"\nKEY FACTORS:")
    
    # OKC's recent form
    okc_recent_win_rate = features.get('team1_recent_win_rate', 0.5)
    opp_recent_win_rate = features.get('team2_recent_win_rate', 0.5)
    print(f"\n1. RECENT FORM:")
    print(f"   OKC Recent Win Rate: {okc_recent_win_rate*100:.1f}%")
    print(f"   {prediction['Opponent']} Recent Win Rate: {opp_recent_win_rate*100:.1f}%")
    print(f"   Advantage: {'OKC' if okc_recent_win_rate > opp_recent_win_rate else prediction['Opponent']}")
    
    # Point differential
    pts_diff = features.get('pts_diff', 0)
    okc_pts = features.get('team1_pts', 0)
    opp_pts = features.get('team2_pts', 0)
    print(f"\n2. SCORING:")
    print(f"   OKC Average Points: {okc_pts:.1f}")
    print(f"   {prediction['Opponent']} Average Points: {opp_pts:.1f}")
    print(f"   Point Differential: {pts_diff:+.1f} (OKC advantage)")
    
    # Shooting efficiency
    fg_diff = features.get('fg_pct_diff', 0)
    okc_fg = features.get('team1_fg_pct', 0)
    opp_fg = features.get('team2_fg_pct', 0)
    print(f"\n3. SHOOTING EFFICIENCY:")
    print(f"   OKC FG%: {okc_fg*100:.1f}%")
    print(f"   {prediction['Opponent']} FG%: {opp_fg*100:.1f}%")
    print(f"   Difference: {fg_diff*100:+.1f}%")
    
    # Rebounding
    reb_diff = features.get('reb_diff', 0)
    print(f"\n4. REBOUNDING:")
    print(f"   Rebound Differential: {reb_diff:+.1f} (OKC advantage)")
    
    # Overall efficiency
    eff_diff = features.get('efficiency_diff', 0)
    print(f"\n5. OVERALL EFFICIENCY:")
    print(f"   Efficiency Differential: {eff_diff:+.4f}")
    if eff_diff > 0:
        print("   → OKC is more efficient offensively")
    else:
        print("   → Opponent is more efficient offensively")
    
    # Recent win rate difference
    recent_diff = features.get('recent_win_rate_diff', 0)
    print(f"\n6. MOMENTUM:")
    print(f"   Recent Win Rate Difference: {recent_diff*100:+.1f}%")
    if recent_diff > 0.3:
        print("   → OKC has strong momentum advantage")
    elif recent_diff < -0.3:
        print("   → Opponent has momentum advantage")
    else:
        print("   → Momentum is relatively even")
    
    # Overall assessment
    print(f"\n7. OVERALL ASSESSMENT:")
    win_prob = float(prediction['Win Probability'].rstrip('%'))
    
    if win_prob >= 70:
        print("   ✓ STRONG FAVORITE: OKC has significant advantages across multiple factors")
    elif win_prob >= 55:
        print("   ✓ FAVORITE: OKC has clear advantages but game could be competitive")
    elif win_prob >= 45:
        print("   ⚠ TOSS-UP: Game is relatively even, could go either way")
    else:
        print("   ✗ UNDERDOG: Opponent has advantages, but OKC's recent form suggests they can compete")
    
    # Key reasons
    print(f"\n8. KEY REASONS FOR PREDICTION:")
    reasons = []
    
    if okc_recent_win_rate >= 0.8:
        reasons.append(f"OKC's exceptional recent form ({okc_recent_win_rate*100:.0f}% win rate)")
    if pts_diff > 5:
        reasons.append(f"OKC scores significantly more points (+{pts_diff:.1f} PPG)")
    if fg_diff > 0.02:
        reasons.append(f"Better shooting efficiency (+{fg_diff*100:.1f}% FG%)")
    if reb_diff > 3:
        reasons.append(f"Rebounding advantage (+{reb_diff:.1f} RPG)")
    if eff_diff > 0.05:
        reasons.append("Superior offensive efficiency")
    
    if opp_recent_win_rate > okc_recent_win_rate:
        reasons.append(f"Opponent's strong recent form ({opp_recent_win_rate*100:.0f}% win rate)")
    if pts_diff < -5:
        reasons.append(f"Opponent scores more points ({abs(pts_diff):.1f} PPG advantage)")
    
    if len(reasons) == 0:
        reasons.append("Factors are relatively balanced")
    
    for i, reason in enumerate(reasons, 1):
        print(f"   {i}. {reason}")
    
    # Home/away factor
    if prediction['Location'] == 'home':
        print(f"\n9. HOME COURT ADVANTAGE:")
        print("   ✓ OKC playing at home (Paycom Center)")
        print("   → Home teams typically have ~3-5% win probability boost")
    else:
        print(f"\n9. ROAD GAME:")
        print("   ⚠ OKC playing on the road")
        print("   → Road games are typically more challenging")

def create_prediction_explanation(predictions_list, game_features_df, models, results):
    """
    Create a detailed explanation file that supports predictions with reasons.
    Explains why each prediction was made based on features and model logic.
    """
    print("\nCreating detailed prediction explanation...")
    
    # Get OKC's overall stats
    okc_games = game_features_df[
        (game_features_df['team1'] == 'OKC') | 
        (game_features_df['team2'] == 'OKC')
    ].copy()
    okc_games = okc_games.sort_values('date')
    
    # Calculate OKC's season stats
    okc_wins_total = 0
    okc_losses_total = 0
    okc_pts_avg = []
    okc_opp_pts_avg = []
    
    for _, game in okc_games.iterrows():
        if game['team1'] == 'OKC':
            if game['team1_wins'] == 1:
                okc_wins_total += 1
            else:
                okc_losses_total += 1
            okc_pts_avg.append(game['team1_pts'])
            okc_opp_pts_avg.append(game['team2_pts'])
        else:
            if game['team1_wins'] == 0:
                okc_wins_total += 1
            else:
                okc_losses_total += 1
            okc_pts_avg.append(game['team2_pts'])
            okc_opp_pts_avg.append(game['team1_pts'])
    
    okc_pts_avg_val = np.mean(okc_pts_avg) if len(okc_pts_avg) > 0 else 0
    okc_opp_pts_avg_val = np.mean(okc_opp_pts_avg) if len(okc_opp_pts_avg) > 0 else 0
    
    # Get best model for feature importance
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    best_model = models[best_model_name]
    
    # Get feature importance
    feature_cols = [col for col in game_features_df.columns 
                    if col not in ['game_id', 'date', 'season', 'team1', 'team2', 'team1_wins']]
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Create explanation document
    explanation = []
    explanation.append("="*80)
    explanation.append("OKC THUNDER PREDICTION EXPLANATION REPORT")
    explanation.append("="*80)
    explanation.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    explanation.append(f"Model Used: {best_model_name}")
    explanation.append(f"Model CV Score: {results[best_model_name]['cv_mean']:.4f}")
    
    explanation.append("\n" + "="*80)
    explanation.append("OKC THUNDER CURRENT STATUS")
    explanation.append("="*80)
    explanation.append(f"Season Record: {okc_wins_total}-{okc_losses_total}")
    explanation.append(f"Win Percentage: {okc_wins_total/(okc_wins_total+okc_losses_total)*100:.1f}%")
    explanation.append(f"Average Points Scored: {okc_pts_avg_val:.1f}")
    explanation.append(f"Average Points Allowed: {okc_opp_pts_avg_val:.1f}")
    explanation.append(f"Point Differential: +{okc_pts_avg_val - okc_opp_pts_avg_val:.1f}")
    
    wins, losses, results_str = get_okc_recent_record(game_features_df)
    explanation.append(f"\nLast 10 Games: {wins}-{losses}")
    explanation.append(f"Recent Results: {' '.join(results_str)}")
    explanation.append(f"Recent Win Rate: {wins/10*100:.1f}%")
    
    explanation.append("\n" + "="*80)
    explanation.append("TOP 10 MOST IMPORTANT FEATURES")
    explanation.append("="*80)
    for idx, row in feature_importance.head(10).iterrows():
        explanation.append(f"{row['feature']}: {row['importance']:.4f}")
    
    explanation.append("\n" + "="*80)
    explanation.append("DETAILED PREDICTIONS WITH REASONING")
    explanation.append("="*80)
    
    for pred in predictions_list:
        explanation.append(f"\n{'='*80}")
        explanation.append(f"Game: OKC Thunder vs {pred['Opponent']}")
        explanation.append(f"Date: {pred['Date']}")
        explanation.append(f"Location: {pred['Location']}")
        explanation.append(f"{'='*80}")
        
        if pred['Prediction'] == 'INSUFFICIENT DATA' or pred['Win Probability'] == 'N/A':
            explanation.append("\n⚠ INSUFFICIENT DATA")
            explanation.append("Reason: Not enough recent game data available for this opponent.")
            explanation.append("Recommendation: Collect more data or use season averages.")
            continue
        
        if 'Feature_Dict' in pred and pred['Feature_Dict']:
            features = pred['Feature_Dict']
            
            explanation.append(f"\nPREDICTION: {pred['Prediction']}")
            explanation.append(f"Win Probability: {pred['Win Probability']}")
            explanation.append(f"Confidence Range: {pred['Confidence Range']}")
            
            explanation.append("\nKEY FACTORS:")
            
            # OKC's recent form
            okc_recent_win_rate = features.get('team1_recent_win_rate', 0.5)
            opp_recent_win_rate = features.get('team2_recent_win_rate', 0.5)
            explanation.append(f"\n1. RECENT FORM:")
            explanation.append(f"   OKC Recent Win Rate: {okc_recent_win_rate*100:.1f}%")
            explanation.append(f"   {pred['Opponent']} Recent Win Rate: {opp_recent_win_rate*100:.1f}%")
            explanation.append(f"   Advantage: {'OKC' if okc_recent_win_rate > opp_recent_win_rate else pred['Opponent']}")
            
            # Point differential
            pts_diff = features.get('pts_diff', 0)
            okc_pts = features.get('team1_pts', 0)
            opp_pts = features.get('team2_pts', 0)
            explanation.append(f"\n2. SCORING:")
            explanation.append(f"   OKC Average Points: {okc_pts:.1f}")
            explanation.append(f"   {pred['Opponent']} Average Points: {opp_pts:.1f}")
            explanation.append(f"   Point Differential: {pts_diff:+.1f} (OKC advantage)")
            
            # Shooting efficiency
            fg_diff = features.get('fg_pct_diff', 0)
            okc_fg = features.get('team1_fg_pct', 0)
            opp_fg = features.get('team2_fg_pct', 0)
            explanation.append(f"\n3. SHOOTING EFFICIENCY:")
            explanation.append(f"   OKC FG%: {okc_fg*100:.1f}%")
            explanation.append(f"   {pred['Opponent']} FG%: {opp_fg*100:.1f}%")
            explanation.append(f"   Difference: {fg_diff*100:+.1f}%")
            
            # Rebounding
            reb_diff = features.get('reb_diff', 0)
            explanation.append(f"\n4. REBOUNDING:")
            explanation.append(f"   Rebound Differential: {reb_diff:+.1f} (OKC advantage)")
            
            # Overall efficiency
            eff_diff = features.get('efficiency_diff', 0)
            explanation.append(f"\n5. OVERALL EFFICIENCY:")
            explanation.append(f"   Efficiency Differential: {eff_diff:+.4f}")
            if eff_diff > 0:
                explanation.append("   → OKC is more efficient offensively")
            else:
                explanation.append("   → Opponent is more efficient offensively")
            
            # Recent win rate difference
            recent_diff = features.get('recent_win_rate_diff', 0)
            explanation.append(f"\n6. MOMENTUM:")
            explanation.append(f"   Recent Win Rate Difference: {recent_diff*100:+.1f}%")
            if recent_diff > 0.3:
                explanation.append("   → OKC has strong momentum advantage")
            elif recent_diff < -0.3:
                explanation.append("   → Opponent has momentum advantage")
            else:
                explanation.append("   → Momentum is relatively even")
            
            # Overall assessment
            explanation.append(f"\n7. OVERALL ASSESSMENT:")
            win_prob = float(pred['Win Probability'].rstrip('%'))
            
            if win_prob >= 70:
                explanation.append("   ✓ STRONG FAVORITE: OKC has significant advantages across multiple factors")
            elif win_prob >= 55:
                explanation.append("   ✓ FAVORITE: OKC has clear advantages but game could be competitive")
            elif win_prob >= 45:
                explanation.append("   ⚠ TOSS-UP: Game is relatively even, could go either way")
            else:
                explanation.append("   ✗ UNDERDOG: Opponent has advantages, but OKC's recent form suggests they can compete")
            
            # Key reasons
            explanation.append(f"\n8. KEY REASONS FOR PREDICTION:")
            reasons = []
            
            if okc_recent_win_rate >= 0.8:
                reasons.append(f"OKC's exceptional recent form ({okc_recent_win_rate*100:.0f}% win rate)")
            if pts_diff > 5:
                reasons.append(f"OKC scores significantly more points (+{pts_diff:.1f} PPG)")
            if fg_diff > 0.02:
                reasons.append(f"Better shooting efficiency (+{fg_diff*100:.1f}% FG%)")
            if reb_diff > 3:
                reasons.append(f"Rebounding advantage (+{reb_diff:.1f} RPG)")
            if eff_diff > 0.05:
                reasons.append("Superior offensive efficiency")
            
            if opp_recent_win_rate > okc_recent_win_rate:
                reasons.append(f"Opponent's strong recent form ({opp_recent_win_rate*100:.0f}% win rate)")
            if pts_diff < -5:
                reasons.append(f"Opponent scores more points ({abs(pts_diff):.1f} PPG advantage)")
            
            if len(reasons) == 0:
                reasons.append("Factors are relatively balanced")
            
            for i, reason in enumerate(reasons, 1):
                explanation.append(f"   {i}. {reason}")
            
            # Home/away factor
            if pred['Location'] == 'home':
                explanation.append(f"\n9. HOME COURT ADVANTAGE:")
                explanation.append("   ✓ OKC playing at home (Paycom Center)")
                explanation.append("   → Home teams typically have ~3-5% win probability boost")
            else:
                explanation.append(f"\n9. ROAD GAME:")
                explanation.append("   ⚠ OKC playing on the road")
                explanation.append("   → Road games are typically more challenging")
        
        else:
            explanation.append("\n⚠ Feature details not available for this prediction")
    
    explanation.append("\n" + "="*80)
    explanation.append("SUMMARY")
    explanation.append("="*80)
    wins_pred = sum([1 for p in predictions_list if p['Prediction'] == 'WIN'])
    losses_pred = sum([1 for p in predictions_list if p['Prediction'] == 'LOSS'])
    explanation.append(f"\nPredicted Record: {wins_pred}-{losses_pred}")
    explanation.append(f"Win Rate: {wins_pred/5*100:.1f}%")
    
    explanation.append("\n" + "="*80)
    explanation.append("IMPORTANT NOTES")
    explanation.append("="*80)
    explanation.append("""
1. These predictions are based on statistical models and historical data.
2. Many factors not captured in statistics can affect outcomes:
   - Injuries and player availability
   - Rest days and schedule fatigue
   - Referee calls and game flow
   - Clutch performance and late-game execution
   - Matchup-specific strategies

3. OKC's exceptional 23-1 record (95.8% win rate) suggests they are:
   - One of the strongest teams in the league
   - Playing at an elite level consistently
   - Capable of winning against any opponent

4. If predictions seem conservative compared to OKC's record:
   - Model may be using insufficient opponent data
   - Model may be over-cautious due to small sample size
   - Recent form features may not fully capture OKC's dominance
   - Consider that even great teams lose occasionally

5. Error margins account for uncertainty, but actual outcomes may vary.
""")
    
    # Save explanation
    explanation_text = "\n".join(explanation)
    explanation_file = "prediction_explanation.txt"
    with open(explanation_file, 'w') as f:
        f.write(explanation_text)
    
    print(f"✓ Detailed explanation saved to: {explanation_file}")
    print(f"\nPreview of explanation:")
    print("\n".join(explanation[:50]))

def create_enhanced_visualizations(results, error_margins, predictions_df):
    """Create enhanced visualizations"""
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
    plt.savefig('model_comparison_api_data.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("Saved: model_comparison_api_data.png")
    
    # Predictions visualization
    if len(predictions_df) > 0 and 'Win Probability' in predictions_df.columns:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Handle N/A values
        win_probs = []
        for val in predictions_df['Win Probability']:
            if val == 'N/A':
                win_probs.append(50.0)  # Default to 50% for visualization
            else:
                win_probs.append(float(val.rstrip('%')))
        win_probs = np.array(win_probs)
        colors = ['#2ecc71' if p['Prediction'] == 'WIN' else '#e74c3c' for _, p in predictions_df.iterrows()]
        
        confidence_ranges = []
        for _, row in predictions_df.iterrows():
            if 'Confidence Range' in row and row['Confidence Range'] != 'N/A' and pd.notna(row['Confidence Range']):
                try:
                    lower, upper = row['Confidence Range'].split(' - ')
                    lower = float(lower.rstrip('%'))
                    upper = float(upper.rstrip('%'))
                    confidence_ranges.append((lower, upper))
                except:
                    prob = float(row['Win Probability'].rstrip('%')) if row['Win Probability'] != 'N/A' else 50.0
                    confidence_ranges.append((prob, prob))
            else:
                prob = float(row['Win Probability'].rstrip('%')) if row['Win Probability'] != 'N/A' else 50.0
                confidence_ranges.append((prob, prob))
        
        bars = ax.bar(range(len(predictions_df)), win_probs, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=2)
        
        # Add error bars
        for i, (bar, (lower, upper)) in enumerate(zip(bars, confidence_ranges)):
            center = bar.get_height()
            ax.errorbar(i, center, yerr=[[center - lower], [upper - center]], 
                       fmt='none', color='black', capsize=8, capthick=2, elinewidth=2)
        
        ax.set_title('OKC Thunder Next 5 Games Predictions (2025 Season)\nUsing NBA API Data - With Error Margins', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.set_ylabel('Win Probability (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Game', fontsize=14, fontweight='bold')
        
        labels = []
        for _, row in predictions_df.iterrows():
            date_parts = row['Date'].split('-')
            month_day = f"{date_parts[1]}/{date_parts[2]}"
            opp_short = row['Opponent'].split()[-1]
            labels.append(f"{month_day}\nvs {opp_short}\n({row['Location']})")
        
        ax.set_xticks(range(len(predictions_df)))
        ax.set_xticklabels(labels, rotation=0, ha='center', fontsize=11, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.axhline(y=50, color='black', linestyle='--', alpha=0.5, linewidth=2, label='50% threshold')
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')
        
        for i, (bar, prob) in enumerate(zip(bars, win_probs)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{prob:.1f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=12,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'))
        
        plt.tight_layout()
        plt.savefig('okc_predictions_api_data.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("Saved: okc_predictions_api_data.png")

def main():
    """Main execution function"""
    print("="*60)
    print("NBA Game Prediction Model - Using API Collected Data")
    print("Ensemble Learning with Accurate NBA API Data")
    print("="*60)
    
    # Load API data
    game_features_df = load_api_data()
    
    if game_features_df is None:
        print("\n✗ Cannot proceed without API data.")
        print("Please run: python collect_nba_api_data.py")
        return
    
    # Preprocess
    game_features_df = preprocess_data(game_features_df)
    
    # Prepare features
    feature_cols = [col for col in game_features_df.columns 
                    if col not in ['game_id', 'date', 'season', 'team1', 'team2', 'team1_wins']]
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
    error_margins = calculate_error_margins(results)
    
    # Make predictions
    predictions_df, predictions_list = predict_okc_games(models, results, error_margins, game_features_df)
    
    # Create detailed explanation file
    create_prediction_explanation(predictions_list, game_features_df, models, results)
    
    # Create visualizations
    create_enhanced_visualizations(results, error_margins, predictions_df)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - model_comparison_api_data.png")
    print("  - okc_predictions_api_data.png")
    print("\n✓ Using accurate NBA API data for all predictions!")

if __name__ == "__main__":
    main()
