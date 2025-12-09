"""
NBA Game Prediction Model - Streamlit Web Interface
For showcasing in OpenKora Connect
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page config
st.set_page_config(
    page_title="NBA Game Prediction - OKC Thunder",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .win-prediction {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .loss-prediction {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏀 NBA Game Prediction Model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; color: #666; margin-bottom: 2rem;">OKC Thunder Next 5 Games</h2>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Model Information")
    
    st.markdown("""
    ### Model Type
    **Ensemble Learning:**
    - Random Forest (500 trees)
    - Gradient Boosting (500 stages)
    
    ### Features
    - 52 features including:
      - Team statistics
      - Recent form (last 10 games)
      - Efficiency metrics
      - Point differentials
      - Shooting percentages
    
    ### Data Source
    - Official NBA API
    - Real-time game data
    - Season 2025-26
    """)
    
    st.markdown("---")
    st.markdown("""
    ### How It Works
    1. Collects data from NBA API
    2. Trains ensemble models
    3. Makes predictions with uncertainty
    4. Provides detailed reasoning
    
    ### Interactive Predictions
    - Predicts one game at a time
    - Uses predicted games for future predictions
    - Accounts for uncertainty (0.7x weight)
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Model Performance", "📝 About"])

with tab1:
    st.header("Generate Game Predictions")
    
    # Check if data file exists
    data_file = "api_collected_data/cleaned/final_game_features.csv"
    if not os.path.exists(data_file):
        st.warning("⚠️ Data file not found. Please run `collect_nba_api_data.py` first.")
        st.info("""
        **To collect data:**
        1. Run: `python collect_nba_api_data.py`
        2. Wait for data collection to complete
        3. Refresh this page
        """)
        st.stop()
    
    if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
        with st.spinner("Loading data and training models... This may take a minute."):
            try:
                # Import here to avoid errors if modules not available
                from nba_prediction_from_api_data import (
                    load_api_data,
                    preprocess_data,
                    train_models,
                    calculate_error_margins,
                    predict_okc_games,
                    get_okc_recent_record
                )
                
                # Load data
                game_features_df = load_api_data()
                
                # Preprocess
                game_features_df = preprocess_data(game_features_df)
                
                # Split data
                from sklearn.model_selection import train_test_split
                feature_cols = [col for col in game_features_df.columns 
                              if col not in ['game_id', 'date', 'season', 'team1', 'team2', 'team1_wins', 'is_predicted']]
                X = game_features_df[feature_cols]
                y = game_features_df['team1_wins']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Train models
                models, results = train_models(X_train, X_test, y_train, y_test)
                
                # Calculate error margins
                error_margins = calculate_error_margins(results)
                
                # Get OKC recent record
                wins, losses, game_results = get_okc_recent_record(game_features_df)
                
                # Display recent performance
                st.subheader("📊 OKC Thunder Recent Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Last 10 Games", f"{wins}-{losses}")
                with col2:
                    st.metric("Win Rate", f"{wins/10*100:.1f}%")
                with col3:
                    st.metric("Recent Form", " ".join(game_results))
                
                st.markdown("---")
                
                # Make predictions (non-interactive version for web)
                # We'll make all 5 predictions at once for the web demo
                schedule = [
                    {'date': '2025-12-10', 'opponent': 'PHX', 'location': 'home', 'opp_name': 'Phoenix Suns'},
                    {'date': '2025-12-17', 'opponent': 'LAC', 'location': 'home', 'opp_name': 'LA Clippers'},
                    {'date': '2025-12-19', 'opponent': 'MIN', 'location': 'away', 'opp_name': 'Minnesota Timberwolves'},
                    {'date': '2025-12-22', 'opponent': 'MEM', 'location': 'home', 'opp_name': 'Memphis Grizzlies'},
                    {'date': '2025-12-23', 'opponent': 'SAS', 'location': 'away', 'opp_name': 'San Antonio Spurs'}
                ]
                
                # Get best model
                best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
                best_model = models[best_model_name]
                best_error = error_margins[best_model_name]['adjusted_error']
                
                # Get OKC games
                okc_games = game_features_df[
                    (game_features_df['team1'] == 'OKC') | 
                    (game_features_df['team2'] == 'OKC')
                ].copy()
                okc_games = okc_games.sort_values('date')
                okc_recent = okc_games.tail(10) if len(okc_games) >= 10 else okc_games
                
                predictions_list = []
                
                for game_info in schedule:
                    opp = game_info['opponent']
                    location = game_info['location']
                    
                    opp_games = game_features_df[
                        (game_features_df['team1'] == opp) | 
                        (game_features_df['team2'] == opp)
                    ].copy()
                    opp_games = opp_games.sort_values('date').tail(10) if len(opp_games) >= 10 else opp_games
                    
                    if len(okc_recent) > 0:
                        from nba_prediction_from_api_data import create_prediction_features_from_recent_games
                        
                        pred_features, feature_dict = create_prediction_features_from_recent_games(
                            okc_recent, opp_games, feature_cols, opp, location, []
                        )
                        
                        win_prob = best_model.predict_proba(pred_features)[0][1]
                        prediction = best_model.predict(pred_features)[0]
                        
                        win_prob_lower = max(0, win_prob - best_error/100)
                        win_prob_upper = min(1, win_prob + best_error/100)
                        
                        predictions_list.append({
                            'Date': game_info['date'],
                            'Opponent': game_info['opp_name'],
                            'Location': location,
                            'Prediction': 'WIN' if prediction == 1 else 'LOSS',
                            'Win Probability': f"{win_prob*100:.1f}%",
                            'Confidence Range': f"{win_prob_lower*100:.1f}% - {win_prob_upper*100:.1f}%",
                            'Error Margin': f"±{best_error:.1f}%",
                            'Feature_Dict': feature_dict
                        })
                
                # Display predictions
                st.subheader("🎯 Predicted Outcomes")
                
                for i, pred in enumerate(predictions_list, 1):
                    win_prob = float(pred['Win Probability'].rstrip('%'))
                    is_win = pred['Prediction'] == 'WIN'
                    
                    card_class = "win-prediction" if is_win else "loss-prediction"
                    icon = "✅" if is_win else "❌"
                    
                    st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <h3>{icon} Game {i}: vs {pred['Opponent']}</h3>
                        <p><strong>Date:</strong> {pred['Date']}</p>
                        <p><strong>Location:</strong> {pred['Location'].upper()}</p>
                        <p><strong>Prediction:</strong> {pred['Prediction']}</p>
                        <p><strong>Win Probability:</strong> {pred['Win Probability']}</p>
                        <p><strong>Confidence Range:</strong> {pred['Confidence Range']}</p>
                        <p><strong>Error Margin:</strong> {pred['Error Margin']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Summary
                wins_pred = sum([1 for p in predictions_list if p['Prediction'] == 'WIN'])
                losses_pred = 5 - wins_pred
                
                st.success(f"**Predicted Record:** {wins_pred}-{losses_pred} ({wins_pred/5*100:.1f}% win rate)")
                
                # Store in session state
                st.session_state['predictions'] = predictions_list
                st.session_state['models'] = models
                st.session_state['results'] = results
                st.session_state['error_margins'] = error_margins
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.exception(e)
    
    # Show cached predictions if available
    if 'predictions' in st.session_state:
        st.info("💡 Predictions are cached. Click 'Generate Predictions' to refresh with latest data.")

with tab2:
    st.header("Model Performance Metrics")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        error_margins = st.session_state.get('error_margins', {})
        
        # Model comparison
        st.subheader("Model Comparison")
        
        comparison_data = {
            'Model': list(results.keys()),
            'Accuracy': [f"{r['accuracy']:.4f}" for r in results.values()],
            'CV Mean': [f"{r['cv_mean']:.4f}" for r in results.values()],
            'CV Std': [f"±{r['cv_std']:.4f}" for r in results.values()],
            'Error Margin': [f"±{error_margins.get(m, {}).get('adjusted_error', 0):.2f}%" for m in results.keys()]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(df.set_index('Model')['Accuracy'].str.rstrip('%').astype(float))
        
        with col2:
            st.bar_chart(df.set_index('Model')['CV Mean'].astype(float))
        
    else:
        st.info("👆 Generate predictions first to see model performance metrics.")

with tab3:
    st.header("About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    
    This is a machine learning model that predicts NBA game outcomes for the OKC Thunder using ensemble learning techniques.
    
    ### 🔬 Methodology
    
    **Ensemble Learning:**
    - **Random Forest:** 500 trees, max depth 20
    - **Gradient Boosting:** 500 stages, max depth 8
    
    **Features (52 total):**
    - Team statistics (points, rebounds, assists, etc.)
    - Shooting percentages (FG%, 3P%, FT%)
    - Recent form (last 10 games win rate)
    - Efficiency metrics
    - Point differentials
    - Plus/minus statistics
    
    ### 📊 Data Source
    
    - **Official NBA API**
    - Real-time game data
    - Season 2025-26
    - Automatically collected and cleaned
    
    ### 🎓 Key Features
    
    1. **Data-Driven:** No hardcoded rules, model learns from data
    2. **Uncertainty-Aware:** Accounts for prediction confidence
    3. **Interactive:** Predicts games one at a time
    4. **Transparent:** Provides detailed reasoning for each prediction
    
    ### 📚 Technologies
    
    - Python 3.11+
    - Scikit-learn (Ensemble Learning)
    - Pandas (Data Processing)
    - NBA API (Data Collection)
    - Streamlit (Web Interface)
    
    ### 🔗 Links
    
    - **GitHub:** [Your repo URL]
    - **Documentation:** See project files for detailed docs
    - **Report:** See `PROJECT_REPORT.md` for full analysis
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>NBA Game Prediction Model</strong></p>
        <p>Built with Ensemble Learning</p>
        <p>Data from Official NBA API</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.9rem; padding: 1rem;">
    <p>For detailed explanations, see <code>prediction_explanation.txt</code></p>
    <p>For setup instructions, see <code>WEB_DEMO_SETUP.md</code></p>
</div>
""", unsafe_allow_html=True)
