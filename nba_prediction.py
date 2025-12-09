#!/usr/bin/env python3
"""
NBA Game Prediction Model for OKC Thunder
Using Ensemble Learning Techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load all required datasets"""
    print("Loading datasets...")
    team_stats = pd.read_csv('dataset/Team Stats Per Game.csv')
    team_summaries = pd.read_csv('dataset/Team Summaries.csv')
    opponent_stats = pd.read_csv('dataset/Opponent Stats Per Game.csv')
    
    print(f"Team Stats shape: {team_stats.shape}")
    print(f"Team Summaries shape: {team_summaries.shape}")
    print(f"Opponent Stats shape: {opponent_stats.shape}")
    
    return team_stats, team_summaries, opponent_stats

def create_game_dataset(team_stats, team_summaries, opponent_stats):
    """
    Create a dataset where each row represents a potential game matchup.
    Since we don't have actual game-by-game data, we'll create matchups
    based on team statistics from the same season.
    """
    print("\nCreating game dataset...")
    
    # Filter for recent seasons (last 5 years for better relevance)
    recent_seasons = team_stats[team_stats['season'] >= 2021].copy()
    
    games = []
    
    for season in recent_seasons['season'].unique():
        season_teams = recent_seasons[recent_seasons['season'] == season].copy()
        season_summaries = team_summaries[team_summaries['season'] == season].copy()
        season_opp = opponent_stats[opponent_stats['season'] == season].copy()
        
        # Get team abbreviations
        teams = season_teams['abbreviation'].unique()
        teams = [t for t in teams if t != 'NA']  # Remove league average
        
        # Create matchups (each team plays every other team)
        for i, team1 in enumerate(teams):
            for team2 in teams[i+1:]:
                team1_data = season_teams[season_teams['abbreviation'] == team1]
                team2_data = season_teams[season_teams['abbreviation'] == team2]
                
                if len(team1_data) > 0 and len(team2_data) > 0:
                    team1_data = team1_data.iloc[0]
                    team2_data = team2_data.iloc[0]
                    
                    team1_summary = season_summaries[season_summaries['abbreviation'] == team1]
                    team2_summary = season_summaries[season_summaries['abbreviation'] == team2]
                    
                    if len(team1_summary) > 0 and len(team2_summary) > 0:
                        team1_summary = team1_summary.iloc[0]
                        team2_summary = team2_summary.iloc[0]
                        
                        # Determine winner based on win percentage
                        team1_wpct = team1_summary['w'] / (team1_summary['w'] + team1_summary['l']) if (team1_summary['w'] + team1_summary['l']) > 0 else 0.5
                        team2_wpct = team2_summary['w'] / (team2_summary['w'] + team2_summary['l']) if (team2_summary['w'] + team2_summary['l']) > 0 else 0.5
                        
                        # Create features for team1 perspective
                        game_features = {
                            'season': season,
                            # Team 1 offensive stats
                            'team1_pts_per_game': team1_data['pts_per_game'],
                            'team1_fg_percent': team1_data['fg_percent'],
                            'team1_3p_percent': team1_data['x3p_percent'],
                            'team1_ft_percent': team1_data['ft_percent'],
                            'team1_reb_per_game': team1_data['trb_per_game'],
                            'team1_ast_per_game': team1_data['ast_per_game'],
                            'team1_stl_per_game': team1_data['stl_per_game'],
                            'team1_blk_per_game': team1_data['blk_per_game'],
                            'team1_tov_per_game': team1_data['tov_per_game'],
                            
                            # Team 1 advanced stats
                            'team1_o_rtg': team1_summary['o_rtg'],
                            'team1_d_rtg': team1_summary['d_rtg'],
                            'team1_n_rtg': team1_summary['n_rtg'],
                            'team1_srs': team1_summary['srs'],
                            'team1_mov': team1_summary['mov'],
                            
                            # Team 2 (opponent) offensive stats
                            'team2_pts_per_game': team2_data['pts_per_game'],
                            'team2_fg_percent': team2_data['fg_percent'],
                            'team2_3p_percent': team2_data['x3p_percent'],
                            'team2_ft_percent': team2_data['ft_percent'],
                            'team2_reb_per_game': team2_data['trb_per_game'],
                            'team2_ast_per_game': team2_data['ast_per_game'],
                            'team2_stl_per_game': team2_data['stl_per_game'],
                            'team2_blk_per_game': team2_data['blk_per_game'],
                            'team2_tov_per_game': team2_data['tov_per_game'],
                            
                            # Team 2 advanced stats
                            'team2_o_rtg': team2_summary['o_rtg'],
                            'team2_d_rtg': team2_summary['d_rtg'],
                            'team2_n_rtg': team2_summary['n_rtg'],
                            'team2_srs': team2_summary['srs'],
                            'team2_mov': team2_summary['mov'],
                            
                            # Derived features (differences)
                            'pts_diff': team1_data['pts_per_game'] - team2_data['pts_per_game'],
                            'rtg_diff': team1_summary['n_rtg'] - team2_summary['n_rtg'],
                            'srs_diff': team1_summary['srs'] - team2_summary['srs'],
                            'off_def_diff': (team1_summary['o_rtg'] - team2_summary['d_rtg']),
                            
                            # Target: 1 if team1 wins, 0 if team2 wins
                            'team1_wins': 1 if team1_wpct > team2_wpct else 0
                        }
                        
                        games.append(game_features)
    
    game_df = pd.DataFrame(games)
    print(f"Game dataset created: {game_df.shape}")
    return game_df

def preprocess_data(game_data):
    """Preprocess the game data"""
    print("\nPreprocessing data...")
    
    # Check for missing values
    missing = game_data.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        game_data = game_data.dropna()
    
    print(f"Dataset after cleaning: {game_data.shape}")
    print(f"\nClass distribution:")
    print(game_data['team1_wins'].value_counts())
    
    return game_data

def train_models(X_train, X_test, y_train, y_test):
    """Train all ensemble models"""
    print("\n" + "="*60)
    print("Training Ensemble Models")
    print("="*60)
    
    models = {}
    results = {}
    
    # Random Forest
    print("\n1. Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
    
    models['Random Forest'] = rf_model
    results['Random Forest'] = {
        'accuracy': rf_accuracy,
        'cv_mean': rf_cv_scores.mean(),
        'cv_std': rf_cv_scores.std(),
        'predictions': rf_pred
    }
    print(f"Accuracy: {rf_accuracy:.4f}, CV Mean: {rf_cv_scores.mean():.4f}")
    
    # Gradient Boosting
    print("\n2. Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_accuracy = accuracy_score(y_test, gb_pred)
    gb_cv_scores = cross_val_score(gb_model, X_train, y_train, cv=5)
    
    models['Gradient Boosting'] = gb_model
    results['Gradient Boosting'] = {
        'accuracy': gb_accuracy,
        'cv_mean': gb_cv_scores.mean(),
        'cv_std': gb_cv_scores.std(),
        'predictions': gb_pred
    }
    print(f"Accuracy: {gb_accuracy:.4f}, CV Mean: {gb_cv_scores.mean():.4f}")
    
    # Bagging
    print("\n3. Training Bagging...")
    bagging_model = BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    )
    bagging_model.fit(X_train, y_train)
    bagging_pred = bagging_model.predict(X_test)
    bagging_accuracy = accuracy_score(y_test, bagging_pred)
    bagging_cv_scores = cross_val_score(bagging_model, X_train, y_train, cv=5)
    
    models['Bagging'] = bagging_model
    results['Bagging'] = {
        'accuracy': bagging_accuracy,
        'cv_mean': bagging_cv_scores.mean(),
        'cv_std': bagging_cv_scores.std(),
        'predictions': bagging_pred
    }
    print(f"Accuracy: {bagging_accuracy:.4f}, CV Mean: {bagging_cv_scores.mean():.4f}")
    
    # Voting Classifier
    print("\n4. Training Voting Classifier...")
    voting_model = VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42)),
            ('bag', BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=10), n_estimators=30, random_state=42, n_jobs=-1))
        ],
        voting='hard'
    )
    voting_model.fit(X_train, y_train)
    voting_pred = voting_model.predict(X_test)
    voting_accuracy = accuracy_score(y_test, voting_pred)
    voting_cv_scores = cross_val_score(voting_model, X_train, y_train, cv=5)
    
    models['Voting Classifier'] = voting_model
    results['Voting Classifier'] = {
        'accuracy': voting_accuracy,
        'cv_mean': voting_cv_scores.mean(),
        'cv_std': voting_cv_scores.std(),
        'predictions': voting_pred
    }
    print(f"Accuracy: {voting_accuracy:.4f}, CV Mean: {voting_cv_scores.mean():.4f}")
    
    return models, results

def create_visualizations(results, models, X_test, y_test, feature_cols):
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
                color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'])
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].set_ylim([0.5, 1.0])
    axes[0].grid(axis='y', alpha=0.3)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    axes[1].bar(comparison_df.index, comparison_df['cv_mean'], 
                yerr=comparison_df['cv_std'],
                color=['#3498db', '#2ecc71', '#e74c3c', '#f39c12'], capsize=5)
    axes[1].set_title('Cross-Validation Mean Scores', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('CV Mean Score', fontsize=12)
    axes[1].set_ylim([0.5, 1.0])
    axes[1].grid(axis='y', alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: model_comparison.png")
    
    # Feature importance
    rf_model = models['Random Forest']
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance', fontsize=12)
    plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("Saved: feature_importance.png")
    
    # Confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    model_names = ['Random Forest', 'Gradient Boosting', 'Bagging', 'Voting Classifier']
    colors = ['Blues', 'Greens', 'Reds', 'Oranges']
    
    for idx, (name, color) in enumerate(zip(model_names, colors)):
        row = idx // 2
        col = idx % 2
        cm = confusion_matrix(y_test, results[name]['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap=color, ax=axes[row, col])
        axes[row, col].set_title(f'{name} Confusion Matrix', fontweight='bold')
        axes[row, col].set_ylabel('Actual')
        axes[row, col].set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("Saved: confusion_matrices.png")
    
    return comparison_df, feature_importance

def predict_okc_games(models, team_stats, team_summaries, feature_cols):
    """Predict OKC Thunder's next 5 games"""
    print("\n" + "="*60)
    print("Predicting OKC Thunder Next 5 Games")
    print("="*60)
    
    # Get OKC Thunder current stats (2026 season)
    okc_stats = team_stats[(team_stats['abbreviation'] == 'OKC') & (team_stats['season'] == 2026)]
    okc_summary = team_summaries[(team_summaries['abbreviation'] == 'OKC') & (team_summaries['season'] == 2026)]
    
    if len(okc_stats) == 0 or len(okc_summary) == 0:
        print("ERROR: OKC Thunder data not found for 2026 season")
        return None
    
    okc_stats = okc_stats.iloc[0]
    okc_summary = okc_summary.iloc[0]
    
    print(f"\nOKC Thunder Current Stats (2026):")
    print(f"  Record: {okc_summary['w']}-{okc_summary['l']}")
    print(f"  Net Rating: {okc_summary['n_rtg']:.2f}")
    print(f"  Points Per Game: {okc_stats['pts_per_game']:.1f}")
    
    # Select 5 opponents (mix of strong and weaker teams)
    opponent_abbrevs = ['DEN', 'LAL', 'HOU', 'NYK', 'SAC']
    
    def create_prediction_features(okc_stats, okc_summary, opponent_stats, opponent_summary):
        """Create features for a single game prediction"""
        return pd.DataFrame([{
            'team1_pts_per_game': okc_stats['pts_per_game'],
            'team1_fg_percent': okc_stats['fg_percent'],
            'team1_3p_percent': okc_stats['x3p_percent'],
            'team1_ft_percent': okc_stats['ft_percent'],
            'team1_reb_per_game': okc_stats['trb_per_game'],
            'team1_ast_per_game': okc_stats['ast_per_game'],
            'team1_stl_per_game': okc_stats['stl_per_game'],
            'team1_blk_per_game': okc_stats['blk_per_game'],
            'team1_tov_per_game': okc_stats['tov_per_game'],
            'team1_o_rtg': okc_summary['o_rtg'],
            'team1_d_rtg': okc_summary['d_rtg'],
            'team1_n_rtg': okc_summary['n_rtg'],
            'team1_srs': okc_summary['srs'],
            'team1_mov': okc_summary['mov'],
            'team2_pts_per_game': opponent_stats['pts_per_game'],
            'team2_fg_percent': opponent_stats['fg_percent'],
            'team2_3p_percent': opponent_stats['x3p_percent'],
            'team2_ft_percent': opponent_stats['ft_percent'],
            'team2_reb_per_game': opponent_stats['trb_per_game'],
            'team2_ast_per_game': opponent_stats['ast_per_game'],
            'team2_stl_per_game': opponent_stats['stl_per_game'],
            'team2_blk_per_game': opponent_stats['blk_per_game'],
            'team2_tov_per_game': opponent_stats['tov_per_game'],
            'team2_o_rtg': opponent_summary['o_rtg'],
            'team2_d_rtg': opponent_summary['d_rtg'],
            'team2_n_rtg': opponent_summary['n_rtg'],
            'team2_srs': opponent_summary['srs'],
            'team2_mov': opponent_summary['mov'],
            'pts_diff': okc_stats['pts_per_game'] - opponent_stats['pts_per_game'],
            'rtg_diff': okc_summary['n_rtg'] - opponent_summary['n_rtg'],
            'srs_diff': okc_summary['srs'] - opponent_summary['srs'],
            'off_def_diff': okc_summary['o_rtg'] - opponent_summary['d_rtg'],
        }])
    
    # Make predictions
    predictions = []
    best_model = models['Random Forest']  # Use best model
    
    for opp_abbrev in opponent_abbrevs:
        opp_stats = team_stats[(team_stats['abbreviation'] == opp_abbrev) & (team_stats['season'] == 2026)]
        opp_summary = team_summaries[(team_summaries['abbreviation'] == opp_abbrev) & (team_summaries['season'] == 2026)]
        
        if len(opp_stats) > 0 and len(opp_summary) > 0:
            game_features = create_prediction_features(
                okc_stats, okc_summary, 
                opp_stats.iloc[0], opp_summary.iloc[0]
            )
            
            # Ensure feature order matches training data
            game_features = game_features[feature_cols]
            
            win_prob = best_model.predict_proba(game_features)[0][1]
            prediction = best_model.predict(game_features)[0]
            
            predictions.append({
                'Opponent': opp_abbrev,
                'Opponent Record': f"{opp_summary.iloc[0]['w']}-{opp_summary.iloc[0]['l']}",
                'Prediction': 'WIN' if prediction == 1 else 'LOSS',
                'Win Probability': f"{win_prob*100:.1f}%"
            })
    
    predictions_df = pd.DataFrame(predictions)
    print("\n\nPredictions for OKC Thunder Next 5 Games:")
    print("=" * 60)
    print(predictions_df.to_string(index=False))
    
    wins = sum([1 for p in predictions if p['Prediction'] == 'WIN'])
    losses = 5 - wins
    print(f"\n\nPredicted Record: {wins}-{losses}")
    print(f"Win Rate: {wins/5*100:.1f}%")
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if p == 'WIN' else 'red' for p in predictions_df['Prediction']]
    win_probs = predictions_df['Win Probability'].str.rstrip('%').astype(float)
    bars = ax.bar(predictions_df['Opponent'], win_probs, color=colors, alpha=0.7)
    ax.set_title('OKC Thunder Next 5 Games Predictions', fontsize=14, fontweight='bold')
    ax.set_ylabel('Win Probability (%)', fontsize=12)
    ax.set_xlabel('Opponent', fontsize=12)
    ax.set_ylim([0, 100])
    ax.axhline(y=50, color='black', linestyle='--', alpha=0.3, label='50% threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    for bar, prob in zip(bars, win_probs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{prob:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('okc_predictions.png', dpi=300, bbox_inches='tight')
    print("\nSaved: okc_predictions.png")
    
    return predictions_df

def main():
    """Main execution function"""
    print("="*60)
    print("NBA Game Prediction Model - OKC Thunder")
    print("Ensemble Learning Project")
    print("="*60)
    
    # Load data
    team_stats, team_summaries, opponent_stats = load_data()
    
    # Create game dataset
    game_data = create_game_dataset(team_stats, team_summaries, opponent_stats)
    
    # Preprocess
    game_data = preprocess_data(game_data)
    
    # Prepare features and target
    feature_cols = [col for col in game_data.columns if col not in ['season', 'team1_wins']]
    X = game_data[feature_cols]
    y = game_data['team1_wins']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Testing set: {X_test.shape}")
    
    # Train models
    models, results = train_models(X_train, X_test, y_train, y_test)
    
    # Create visualizations
    comparison_df, feature_importance = create_visualizations(
        results, models, X_test, y_test, feature_cols
    )
    
    # Print comparison
    print("\n" + "="*60)
    print("Model Comparison Summary")
    print("="*60)
    print(comparison_df)
    
    best_model_name = comparison_df['cv_mean'].idxmax()
    print(f"\nBest Model: {best_model_name}")
    print(f"Best CV Score: {comparison_df.loc[best_model_name, 'cv_mean']:.4f}")
    
    # Make predictions
    predictions_df = predict_okc_games(models, team_stats, team_summaries, feature_cols)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - model_comparison.png")
    print("  - feature_importance.png")
    print("  - confusion_matrices.png")
    print("  - okc_predictions.png")

if __name__ == "__main__":
    main()
