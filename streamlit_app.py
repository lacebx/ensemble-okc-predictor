"""
NBA Game Prediction Model - Streamlit Web Interface
Enhanced UI matching OpenKora design system with detailed prediction explanations
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

# OpenKora-inspired CSS - Modern, clean design
st.markdown("""
<style>
    /* OpenKora Color System - HSL based */
    :root {
        --primary: hsl(217, 91%, 60%);
        --primary-foreground: hsl(0, 0%, 100%);
        --background: hsl(0, 0%, 98%);
        --foreground: hsl(222, 47%, 11%);
        --card: hsl(0, 0%, 100%);
        --card-foreground: hsl(222, 47%, 11%);
        --muted: hsl(220, 14%, 96%);
        --muted-foreground: hsl(215, 16%, 47%);
        --border: hsl(220, 13%, 91%);
        --radius: 1rem;
    }
    
    /* Main container */
    .main {
        background: var(--background);
        color: var(--foreground);
    }
    
    /* Header styling - OpenKora style */
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: var(--foreground);
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.25rem;
        color: var(--muted-foreground);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling - matches OpenKora */
    .prediction-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: all 0.2s;
    }
    
    .prediction-card:hover {
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid var(--border);
    }
    
    .prediction-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--foreground);
        margin: 0;
    }
    
    .prediction-badge {
        padding: 0.375rem 0.75rem;
        border-radius: 0.5rem;
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    .badge-win {
        background: hsl(142, 76%, 36%);
        color: white;
    }
    
    .badge-loss {
        background: hsl(0, 84%, 60%);
        color: white;
    }
    
    /* Stats grid - OpenKora style */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .stat-card {
        background: var(--muted);
        border: 1px solid var(--border);
        border-radius: calc(var(--radius) - 2px);
        padding: 1rem;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--foreground);
        margin: 0.5rem 0;
    }
    
    .stat-label {
        font-size: 0.875rem;
        color: var(--muted-foreground);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Explanation section */
    .explanation-section {
        background: var(--muted);
        border: 1px solid var(--border);
        border-radius: calc(var(--radius) - 2px);
        padding: 1.5rem;
        margin-top: 1.5rem;
    }
    
    .explanation-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--foreground);
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .explanation-item {
        margin: 1rem 0;
        padding: 1rem;
        background: var(--card);
        border-left: 3px solid var(--primary);
        border-radius: 0.5rem;
    }
    
    .explanation-item-title {
        font-weight: 600;
        color: var(--foreground);
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    
    .explanation-item-content {
        color: var(--muted-foreground);
        font-size: 0.9375rem;
        line-height: 1.6;
    }
    
    .metric-highlight {
        color: var(--primary);
        font-weight: 600;
    }
    
    .metric-positive {
        color: hsl(142, 76%, 36%);
        font-weight: 600;
    }
    
    .metric-negative {
        color: hsl(0, 84%, 60%);
        font-weight: 600;
    }
    
    /* Button styling */
    .stButton > button {
        background: var(--primary);
        color: var(--primary-foreground);
        border: none;
        border-radius: calc(var(--radius) - 2px);
        padding: 0.625rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: hsl(217, 91%, 55%);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        border-bottom: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        border-radius: calc(var(--radius) - 2px) calc(var(--radius) - 2px) 0 0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--card);
        border-right: 1px solid var(--border);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏀 NBA Game Prediction Model</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">OKC Thunder Next 5 Games • Ensemble Learning with Real-Time NBA Data</p>', unsafe_allow_html=True)

# Sidebar - OpenKora style
with st.sidebar:
    st.markdown("### ⚙️ Model Information")
    
    st.markdown("""
    **Ensemble Learning:**
    - Random Forest (500 trees)
    - Gradient Boosting (500 stages)
    
    **Features:** 52 total
    - Team statistics
    - Recent form
    - Efficiency metrics
    - Point differentials
    
    **Data Source:**
    - Official NBA API
    - Real-time game data
    - Season 2025-26
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 How It Works")
    st.markdown("""
    1. Collects data from NBA API
    2. Trains ensemble models
    3. Makes predictions with uncertainty
    4. Provides detailed reasoning
    
    **Interactive Predictions:**
    - Uses same real data baseline
    - Adds predicted games (0.7x weight)
    - Accounts for uncertainty
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Model Performance", "📝 About"])

with tab1:
    st.markdown("### Generate Game Predictions")
    
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
                    get_okc_recent_record
                )
                from sklearn.model_selection import train_test_split
                
                # Load data
                game_features_df = load_api_data()
                
                # Preprocess
                game_features_df = preprocess_data(game_features_df)
                
                # Split data
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
                
                # Display recent performance - OpenKora style
                st.markdown("### 📊 OKC Thunder Recent Performance")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown("""
                    <div class="stat-card">
                        <div class="stat-label">Last 10 Games</div>
                        <div class="stat-value">{}-{}</div>
                    </div>
                    """.format(wins, losses), unsafe_allow_html=True)
                with col2:
                    st.markdown("""
                    <div class="stat-card">
                        <div class="stat-label">Win Rate</div>
                        <div class="stat-value">{:.1f}%</div>
                    </div>
                    """.format(wins/10*100), unsafe_allow_html=True)
                with col3:
                    st.markdown("""
                    <div class="stat-card">
                        <div class="stat-label">Current Streak</div>
                        <div class="stat-value">{}</div>
                    </div>
                    """.format("W" * wins if wins > 0 else "L"), unsafe_allow_html=True)
                with col4:
                    st.markdown("""
                    <div class="stat-card">
                        <div class="stat-label">Recent Form</div>
                        <div class="stat-value" style="font-size: 1rem;">{}</div>
                    </div>
                    """.format(" ".join(game_results)), unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Make predictions
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
                        
                        # Calculate OKC stats from recent games
                        okc_team1_games = okc_recent[okc_recent['team1'] == 'OKC']
                        okc_team2_games = okc_recent[okc_recent['team2'] == 'OKC']
                        
                        okc_pts_list = []
                        okc_fg_list = []
                        okc_reb_list = []
                        okc_ast_list = []
                        
                        if len(okc_team1_games) > 0:
                            okc_pts_list.extend(okc_team1_games['team1_pts'].values)
                            okc_fg_list.extend(okc_team1_games['team1_fg_pct'].values)
                            okc_reb_list.extend(okc_team1_games['team1_reb'].values)
                            okc_ast_list.extend(okc_team1_games['team1_ast'].values)
                        
                        if len(okc_team2_games) > 0:
                            okc_pts_list.extend(okc_team2_games['team2_pts'].values)
                            okc_fg_list.extend(okc_team2_games['team2_fg_pct'].values)
                            okc_reb_list.extend(okc_team2_games['team2_reb'].values)
                            okc_ast_list.extend(okc_team2_games['team2_ast'].values)
                        
                        okc_pts_avg = np.mean(okc_pts_list) if len(okc_pts_list) > 0 else 0
                        okc_fg_avg = np.mean(okc_fg_list) if len(okc_fg_list) > 0 else 0
                        okc_reb_avg = np.mean(okc_reb_list) if len(okc_reb_list) > 0 else 0
                        okc_ast_avg = np.mean(okc_ast_list) if len(okc_ast_list) > 0 else 0
                        
                        # Calculate opponent stats
                        opp_team1_games = opp_games[opp_games['team1'] == opp]
                        opp_team2_games = opp_games[opp_games['team2'] == opp]
                        
                        opp_pts_list = []
                        opp_fg_list = []
                        opp_reb_list = []
                        opp_ast_list = []
                        
                        if len(opp_team1_games) > 0:
                            opp_pts_list.extend(opp_team1_games['team1_pts'].values)
                            opp_fg_list.extend(opp_team1_games['team1_fg_pct'].values)
                            opp_reb_list.extend(opp_team1_games['team1_reb'].values)
                            opp_ast_list.extend(opp_team1_games['team1_ast'].values)
                        
                        if len(opp_team2_games) > 0:
                            opp_pts_list.extend(opp_team2_games['team2_pts'].values)
                            opp_fg_list.extend(opp_team2_games['team2_fg_pct'].values)
                            opp_reb_list.extend(opp_team2_games['team2_reb'].values)
                            opp_ast_list.extend(opp_team2_games['team2_ast'].values)
                        
                        opp_pts_avg = np.mean(opp_pts_list) if len(opp_pts_list) > 0 else 0
                        opp_fg_avg = np.mean(opp_fg_list) if len(opp_fg_list) > 0 else 0
                        opp_reb_avg = np.mean(opp_reb_list) if len(opp_reb_list) > 0 else 0
                        opp_ast_avg = np.mean(opp_ast_list) if len(opp_ast_list) > 0 else 0
                        
                        # Calculate win rates
                        okc_wins = sum([1 for _, g in okc_recent.iterrows() 
                                       if (g['team1'] == 'OKC' and g['team1_wins'] == 1) or 
                                          (g['team2'] == 'OKC' and g['team1_wins'] == 0)])
                        okc_win_rate = okc_wins / len(okc_recent) if len(okc_recent) > 0 else 0.5
                        
                        opp_wins = sum([1 for _, g in opp_games.iterrows()
                                       if (g['team1'] == opp and g['team1_wins'] == 1) or
                                          (g['team2'] == opp and g['team1_wins'] == 0)])
                        opp_win_rate = opp_wins / len(opp_games) if len(opp_games) > 0 else 0.5
                        
                        predictions_list.append({
                            'Date': game_info['date'],
                            'Opponent': game_info['opp_name'],
                            'Location': location,
                            'Prediction': 'WIN' if prediction == 1 else 'LOSS',
                            'Win Probability': f"{win_prob*100:.1f}%",
                            'Confidence Range': f"{win_prob_lower*100:.1f}% - {win_prob_upper*100:.1f}%",
                            'Error Margin': f"±{best_error:.1f}%",
                            'Feature_Dict': feature_dict,
                            # Detailed stats for explanation
                            'OKC_PTS': okc_pts_avg,
                            'OKC_FG': okc_fg_avg,
                            'OKC_REB': okc_reb_avg,
                            'OKC_AST': okc_ast_avg,
                            'OKC_WinRate': okc_win_rate,
                            'OKC_Games': len(okc_recent),
                            'Opp_PTS': opp_pts_avg,
                            'Opp_FG': opp_fg_avg,
                            'Opp_REB': opp_reb_avg,
                            'Opp_AST': opp_ast_avg,
                            'Opp_WinRate': opp_win_rate,
                            'Opp_Games': len(opp_games),
                            'PTS_Diff': okc_pts_avg - opp_pts_avg,
                            'FG_Diff': okc_fg_avg - opp_fg_avg,
                            'REB_Diff': okc_reb_avg - opp_reb_avg,
                            'AST_Diff': okc_ast_avg - opp_ast_avg,
                            'WinRate_Diff': okc_win_rate - opp_win_rate
                        })
                
                # Display predictions with detailed explanations
                st.markdown("### 🎯 Predicted Outcomes")
                
                for i, pred in enumerate(predictions_list, 1):
                    win_prob = float(pred['Win Probability'].rstrip('%'))
                    is_win = pred['Prediction'] == 'WIN'
                    badge_text = "WIN" if is_win else "LOSS"
                    
                    # Card container with border
                    st.markdown("---")
                    
                    # Card header
                    col_header1, col_header2 = st.columns([4, 1])
                    with col_header1:
                        st.markdown(f"#### 🏀 Game {i}: vs {pred['Opponent']}")
                    with col_header2:
                        if is_win:
                            st.success(f"### ✅ {badge_text}")
                        else:
                            st.error(f"### ❌ {badge_text}")
                    
                    # Game info in columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown("**📅 Date**")
                        st.markdown(f"`{pred['Date']}`")
                    with col2:
                        st.markdown("**📍 Location**")
                        location_icon = "🏠" if pred['Location'] == 'home' else "✈️"
                        st.markdown(f"{location_icon} {pred['Location'].upper()}")
                    with col3:
                        st.markdown("**🎯 Win Probability**")
                        st.markdown(f"### {pred['Win Probability']}")
                    with col4:
                        st.markdown("**📊 Confidence Range**")
                        st.markdown(f"`{pred['Confidence Range']}`")
                    
                    # Detailed explanation using expander - ALWAYS EXPANDED with full evidence
                    with st.expander("📊 Detailed Prediction Analysis with Evidence", expanded=True):
                        # 1. Recent Form
                        st.markdown("#### 1️⃣ Recent Form Comparison")
                        col_form1, col_form2 = st.columns(2)
                        with col_form1:
                            st.markdown(f"**OKC Thunder:**")
                            okc_wins = int(pred['OKC_WinRate']*pred['OKC_Games'])
                            okc_losses = pred['OKC_Games'] - okc_wins
                            st.metric("Win Rate", f"{pred['OKC_WinRate']*100:.1f}%", f"{okc_wins}-{okc_losses} record")
                        with col_form2:
                            st.markdown(f"**{pred['Opponent']}:**")
                            opp_wins = int(pred['Opp_WinRate']*pred['Opp_Games'])
                            opp_losses = pred['Opp_Games'] - opp_wins
                            st.metric("Win Rate", f"{pred['Opp_WinRate']*100:.1f}%", f"{opp_wins}-{opp_losses} record")
                        
                        win_rate_diff = pred['WinRate_Diff']*100
                        if win_rate_diff > 0:
                            st.success(f"✅ **Advantage: OKC has {win_rate_diff:.1f}% higher win rate** ({pred['OKC_WinRate']*100:.1f}% vs {pred['Opp_WinRate']*100:.1f}%)")
                        else:
                            st.warning(f"⚠️ **Opponent Advantage: {abs(win_rate_diff):.1f}% higher win rate** ({pred['Opp_WinRate']*100:.1f}% vs {pred['OKC_WinRate']*100:.1f}%)")
                        
                        st.markdown("---")
                        
                        # 2. Scoring Analysis
                        st.markdown("#### 2️⃣ Scoring Analysis")
                        col_score1, col_score2 = st.columns(2)
                        with col_score1:
                            st.metric("OKC Average Points", f"{pred['OKC_PTS']:.1f} PPG", 
                                    f"Based on last {pred['OKC_Games']} games")
                        with col_score2:
                            st.metric(f"{pred['Opponent']} Average Points", f"{pred['Opp_PTS']:.1f} PPG",
                                    f"Based on last {pred['Opp_Games']} games")
                        
                        pts_diff = pred['PTS_Diff']
                        if pts_diff > 0:
                            st.success(f"✅ **Point Differential: +{pts_diff:.1f} points** - OKC scores {pts_diff:.1f} more points per game on average")
                        else:
                            st.warning(f"⚠️ **Point Differential: {pts_diff:.1f} points** - Opponent scores {abs(pts_diff):.1f} more points per game")
                        
                        st.markdown("---")
                        
                        # 3. Shooting Efficiency
                        st.markdown("#### 3️⃣ Shooting Efficiency")
                        col_fg1, col_fg2 = st.columns(2)
                        with col_fg1:
                            st.metric("OKC Field Goal %", f"{pred['OKC_FG']*100:.1f}%")
                        with col_fg2:
                            st.metric(f"{pred['Opponent']} Field Goal %", f"{pred['Opp_FG']*100:.1f}%")
                        
                        fg_diff = pred['FG_Diff']*100
                        if fg_diff > 0:
                            st.success(f"✅ **Shooting Advantage: +{fg_diff:.1f}%** - OKC makes {fg_diff:.1f}% more of their shots")
                        else:
                            st.warning(f"⚠️ **Shooting Advantage: {fg_diff:.1f}%** - Opponent shoots {abs(fg_diff):.1f}% better")
                        
                        st.markdown("---")
                        
                        # 4. Rebounding & Assists
                        st.markdown("#### 4️⃣ Rebounding & Assists")
                        col_reb1, col_reb2 = st.columns(2)
                        with col_reb1:
                            st.metric("OKC Rebounds", f"{pred['OKC_REB']:.1f} RPG")
                            st.metric("OKC Assists", f"{pred['OKC_AST']:.1f} APG")
                        with col_reb2:
                            st.metric(f"{pred['Opponent']} Rebounds", f"{pred['Opp_REB']:.1f} RPG")
                            st.metric(f"{pred['Opponent']} Assists", f"{pred['Opp_AST']:.1f} APG")
                        
                        # Show differentials with metrics
                        reb_diff = pred['REB_Diff']
                        ast_diff = pred['AST_Diff']
                        
                        col_diff1, col_diff2 = st.columns(2)
                        with col_diff1:
                            if reb_diff > 0:
                                st.metric("Rebound Differential", f"+{reb_diff:.1f}", "OKC advantage", delta_color="normal")
                            else:
                                st.metric("Rebound Differential", f"{reb_diff:.1f}", "Opponent advantage", delta_color="inverse")
                        with col_diff2:
                            if ast_diff > 0:
                                st.metric("Assist Differential", f"+{ast_diff:.1f}", "OKC advantage", delta_color="normal")
                            else:
                                st.metric("Assist Differential", f"{ast_diff:.1f}", "Opponent advantage", delta_color="inverse")
                        
                        st.markdown("---")
                        
                        # 5. Key Factors with detailed numbers
                        st.markdown("#### 5️⃣ Key Factors Supporting Prediction")
                        factors = []
                        if pred['WinRate_Diff'] > 0.3:
                            factors.append(f"✅ **Strong momentum advantage:** OKC has {pred['WinRate_Diff']*100:.1f}% higher win rate ({pred['OKC_WinRate']*100:.1f}% vs {pred['Opp_WinRate']*100:.1f}%)")
                        if pred['PTS_Diff'] > 5:
                            factors.append(f"✅ **Significant scoring advantage:** OKC scores {pred['PTS_Diff']:.1f} more points per game ({pred['OKC_PTS']:.1f} vs {pred['Opp_PTS']:.1f} PPG)")
                        if pred['FG_Diff'] > 0.02:
                            factors.append(f"✅ **Better shooting efficiency:** OKC shoots {pred['FG_Diff']*100:.1f}% better ({pred['OKC_FG']*100:.1f}% vs {pred['Opp_FG']*100:.1f}%)")
                        if pred['REB_Diff'] > 3:
                            factors.append(f"✅ **Rebounding advantage:** OKC averages {pred['REB_Diff']:.1f} more rebounds ({pred['OKC_REB']:.1f} vs {pred['Opp_REB']:.1f} RPG)")
                        if pred['AST_Diff'] > 3:
                            factors.append(f"✅ **Better ball movement:** OKC averages {pred['AST_Diff']:.1f} more assists ({pred['OKC_AST']:.1f} vs {pred['Opp_AST']:.1f} APG)")
                        if pred['Location'] == 'home':
                            factors.append("✅ **Home court advantage:** Playing at Paycom Center (home teams typically have 3-5% win probability boost)")
                        if pred['WinRate_Diff'] < -0.2:
                            factors.append(f"⚠️ **Opponent's strong recent form:** {pred['Opponent']} has {pred['Opp_WinRate']*100:.1f}% win rate (vs OKC's {pred['OKC_WinRate']*100:.1f}%)")
                        if pred['PTS_Diff'] < -5:
                            factors.append(f"⚠️ **Opponent scores more:** {pred['Opponent']} averages {abs(pred['PTS_Diff']):.1f} more points per game")
                        
                        if len(factors) == 0:
                            factors.append("Factors are relatively balanced between teams")
                        
                        for factor in factors:
                            st.markdown(f"- {factor}")
                        
                        st.markdown("---")
                        
                        # 6. Model Confidence with full details
                        st.markdown("#### 6️⃣ Model Confidence & Methodology")
                        col_conf1, col_conf2, col_conf3 = st.columns(3)
                        with col_conf1:
                            st.metric("Prediction Confidence", pred['Win Probability'])
                        with col_conf2:
                            st.metric("Confidence Range", pred['Confidence Range'])
                        with col_conf3:
                            st.metric("Error Margin", pred['Error Margin'])
                        
                        st.info(f"""
                        **Model Details:**
                        - **Model Used:** {best_model_name}
                        - **Cross-Validation Score:** {results[best_model_name]['cv_mean']:.4f}
                        - **Training Data:** {len(X_train)} games
                        - **Features:** 52 features including stats, form, and efficiency metrics
                        """)
                
                # Summary
                wins_pred = sum([1 for p in predictions_list if p['Prediction'] == 'WIN'])
                losses_pred = 5 - wins_pred
                
                st.markdown("---")
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
    st.markdown("### Model Performance Metrics")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        error_margins = st.session_state.get('error_margins', {})
        
        # Model comparison
        st.markdown("#### Model Comparison")
        
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
            st.markdown("**Accuracy Comparison**")
            st.bar_chart(df.set_index('Model')['Accuracy'].str.rstrip('%').astype(float))
        
        with col2:
            st.markdown("**Cross-Validation Scores**")
            st.bar_chart(df.set_index('Model')['CV Mean'].astype(float))
        
    else:
        st.info("👆 Generate predictions first to see model performance metrics.")

with tab3:
    st.markdown("### About This Project")
    
    st.markdown("""
    #### 🎯 Project Overview
    
    This is a machine learning model that predicts NBA game outcomes for the OKC Thunder using ensemble learning techniques.
    
    #### 🔬 Methodology
    
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
    
    #### 📊 Data Source
    
    - **Official NBA API**
    - Real-time game data
    - Season 2025-26
    - Automatically collected and cleaned
    
    #### 🎓 Key Features
    
    1. **Data-Driven:** No hardcoded rules, model learns from data
    2. **Uncertainty-Aware:** Accounts for prediction confidence
    3. **Interactive:** Predicts games one at a time
    4. **Transparent:** Provides detailed reasoning for each prediction
    
    #### 📚 Technologies
    
    - Python 3.11+
    - Scikit-learn (Ensemble Learning)
    - Pandas (Data Processing)
    - NBA API (Data Collection)
    - Streamlit (Web Interface)
    """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: var(--muted-foreground); padding: 2rem;">
        <p><strong>NBA Game Prediction Model</strong></p>
        <p>Built with Ensemble Learning</p>
        <p>Data from Official NBA API</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--muted-foreground); font-size: 0.875rem; padding: 1rem;">
    <p>For detailed explanations, see <code>prediction_explanation.txt</code></p>
    <p>For setup instructions, see <code>WEB_DEMO_SETUP.md</code></p>
</div>
""", unsafe_allow_html=True)
