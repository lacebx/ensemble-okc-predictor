# NBA Prediction Project - Web Demo Setup for OpenKora

## Overview

This guide explains how to create a web interface for the NBA prediction project and showcase it in OpenKora Connect using the iframe preview feature.

## Strategy

Since the NBA prediction project is a Python/ML project, we have several options:

1. **Streamlit** (Recommended) - Easy Python web framework, perfect for ML demos
2. **Flask/FastAPI** - More control, requires more setup
3. **Gradio** - Simple ML demo interface
4. **Replit** - Can run Python web apps directly

**Best Choice: Streamlit** - It's designed for ML demos and can be deployed easily.

---

## Step 1: Create Streamlit Web Interface

### Create `streamlit_app.py` in your project directory:

```python
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add project directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your prediction functions
try:
    from nba_prediction_from_api_data import (
        load_api_data,
        preprocess_data,
        train_models,
        calculate_error_margins,
        predict_okc_games,
        get_okc_recent_record
    )
except ImportError:
    st.error("Could not import prediction modules. Make sure all dependencies are installed.")
    st.stop()

st.set_page_config(
    page_title="NBA Game Prediction - OKC Thunder",
    page_icon="🏀",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .win-prediction {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .loss-prediction {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🏀 NBA Game Prediction Model</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="text-align: center; color: #666;">OKC Thunder Next 5 Games</h2>', unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.header("⚙️ Model Controls")
    
    st.subheader("Data Collection")
    if st.button("🔄 Refresh Data", help="Re-collect data from NBA API"):
        with st.spinner("Collecting data from NBA API..."):
            import subprocess
            result = subprocess.run(
                ["python", "collect_nba_api_data.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("Data refreshed successfully!")
            else:
                st.error(f"Error: {result.stderr}")
    
    st.subheader("Model Info")
    st.info("""
    **Model Type:** Ensemble Learning
    - Random Forest (500 trees)
    - Gradient Boosting (500 stages)
    
    **Features:** 52 features including:
    - Team statistics
    - Recent form
    - Efficiency metrics
    - Point differentials
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Predictions", "📈 Model Performance", "📝 Explanation"])

with tab1:
    st.header("Game Predictions")
    
    # Load data and make predictions
    if st.button("🚀 Generate Predictions", type="primary"):
        with st.spinner("Loading data and training models..."):
            try:
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
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Last 10 Games", f"{wins}-{losses}")
                with col2:
                    st.metric("Win Rate", f"{wins/10*100:.1f}%")
                with col3:
                    st.metric("Recent Form", " ".join(game_results))
                
                # Make predictions
                predictions_df, predictions_list = predict_okc_games(models, results, error_margins, game_features_df)
                
                # Display predictions
                st.subheader("Predicted Outcomes")
                
                for i, pred in enumerate(predictions_list, 1):
                    win_prob = float(pred['Win Probability'].rstrip('%'))
                    is_win = pred['Prediction'] == 'WIN'
                    
                    card_class = "win-prediction" if is_win else "loss-prediction"
                    
                    st.markdown(f"""
                    <div class="prediction-card {card_class}">
                        <h3>Game {i}: vs {pred['Opponent']}</h3>
                        <p><strong>Date:</strong> {pred['Date']}</p>
                        <p><strong>Location:</strong> {pred['Location'].upper()}</p>
                        <p><strong>Prediction:</strong> {pred['Prediction']}</p>
                        <p><strong>Win Probability:</strong> {pred['Win Probability']}</p>
                        <p><strong>Confidence Range:</strong> {pred['Confidence Range']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Summary
                wins_pred = sum([1 for p in predictions_list if p['Prediction'] == 'WIN'])
                st.success(f"**Predicted Record:** {wins_pred}-{5-wins_pred} ({wins_pred/5*100:.1f}% win rate)")
                
                # Store in session state
                st.session_state['predictions'] = predictions_list
                st.session_state['models'] = models
                st.session_state['results'] = results
                
            except Exception as e:
                st.error(f"Error generating predictions: {str(e)}")
                st.exception(e)
    
    # Show cached predictions if available
    if 'predictions' in st.session_state:
        st.info("💡 Predictions are cached. Click 'Generate Predictions' to refresh.")

with tab2:
    st.header("Model Performance")
    
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Model comparison
        st.subheader("Model Comparison")
        
        comparison_data = {
            'Model': list(results.keys()),
            'Accuracy': [f"{r['accuracy']:.4f}" for r in results.values()],
            'CV Mean': [f"{r['cv_mean']:.4f}" for r in results.values()],
            'CV Std': [f"±{r['cv_std']:.4f}" for r in results.values()]
        }
        
        df = pd.DataFrame(comparison_data)
        st.dataframe(df, use_container_width=True)
        
        # Feature importance (if available)
        if 'models' in st.session_state:
            best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
            best_model = st.session_state['models'][best_model_name]
            
            if hasattr(best_model, 'feature_importances_'):
                st.subheader("Top 10 Most Important Features")
                feature_cols = [col for col in game_features_df.columns 
                              if col not in ['game_id', 'date', 'season', 'team1', 'team2', 'team1_wins', 'is_predicted']]
                importance_df = pd.DataFrame({
                    'Feature': feature_cols,
                    'Importance': best_model.feature_importances_
                }).sort_values('Importance', ascending=False).head(10)
                
                st.bar_chart(importance_df.set_index('Feature'))
    else:
        st.info("Generate predictions first to see model performance.")

with tab3:
    st.header("Prediction Explanation")
    
    if 'predictions' in st.session_state:
        # Read explanation file if it exists
        try:
            with open('prediction_explanation.txt', 'r') as f:
                explanation = f.read()
            st.markdown(f"```\n{explanation}\n```")
        except FileNotFoundError:
            st.info("Explanation file not found. Generate predictions first.")
    else:
        st.info("Generate predictions first to see detailed explanations.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>NBA Game Prediction Model using Ensemble Learning</p>
    <p>Built with Random Forest & Gradient Boosting</p>
    <p>Data from Official NBA API</p>
</div>
""", unsafe_allow_html=True)
```

---

## Step 2: Create Requirements File for Streamlit

Create `requirements_streamlit.txt`:

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
nba-api>=1.2.1
```

---

## Step 3: Deploy Options

### Option A: Streamlit Cloud (Easiest - Recommended)

1. **Push to GitHub:**
   ```bash
   cd /home/lace/Documents/intelligent_systems/project3
   git init
   git add .
   git commit -m "Add Streamlit web interface"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file: `streamlit_app.py`
   - Click "Deploy"
   - Get your URL: `https://your-app.streamlit.app`

3. **Add to OpenKora:**
   - Create project in OpenKora
   - Demo URL: `https://your-app.streamlit.app`
   - It will display in iframe!

### Option B: Replit (Good for Python Projects)

1. **Create Repl:**
   - Go to https://replit.com
   - Create new Python Repl
   - Upload your project files

2. **Add Replit Config:**
   Create `.replit` file:
   ```toml
   run = "streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0"
   ```

3. **Run:**
   - Click "Run"
   - Get URL: `https://your-repl.repl.co`

4. **Add to OpenKora:**
   - Demo URL: `https://your-repl.repl.co?embed=true`

### Option C: Railway/Render (More Control)

1. **Create `Procfile`:**
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Deploy:**
   - Connect GitHub repo to Railway/Render
   - Deploy
   - Get URL

3. **Add to OpenKora:**
   - Demo URL: Your deployed URL

---

## Step 4: Add Project to OpenKora

1. **Login to OpenKora:**
   - Go to your OpenKora instance
   - Login as developer

2. **Create New Project:**
   - Click "Create Project"
   - Fill in details:
     - **Title:** NBA Game Prediction Model
     - **Description:** Machine learning model using ensemble learning to predict OKC Thunder game outcomes
     - **GitHub URL:** Your GitHub repo URL (optional)
     - **Demo URL:** Your Streamlit/Replit/Railway URL
     - **Tech Stack:** Python, Machine Learning, Scikit-learn, Streamlit, NBA API
     - **Tags:** machine-learning, nba, predictions, ensemble-learning, python

3. **Upload Requirements:**
   - Upload `requirements_streamlit.txt` as requirements file

4. **Save:**
   - Click "Create Project"
   - Preview will appear in iframe!

---

## Step 5: Test the Preview

1. **Visit your project page in OpenKora**
2. **Check "Live Preview" tab**
3. **The Streamlit app should load in iframe**
4. **Users can interact with it directly!**

---

## Troubleshooting

### Iframe Not Loading:

1. **Check CORS headers:**
   - Streamlit Cloud allows iframes by default ✅
   - Replit needs `?embed=true` parameter
   - Some platforms block iframes (security)

2. **Add to Streamlit config:**
   Create `.streamlit/config.toml`:
   ```toml
   [server]
   enableCORS = false
   enableXsrfProtection = false
   ```

3. **Use "Open in New Tab" fallback:**
   - OpenKora has this built-in
   - Users can still access your demo

### API Issues:

1. **NBA API Rate Limits:**
   - Cache data locally
   - Don't refresh too often
   - Use collected data file

2. **Missing Data:**
   - Make sure `api_collected_data/cleaned/final_game_features.csv` exists
   - Run `collect_nba_api_data.py` first

---

## Quick Start Commands

```bash
# 1. Install Streamlit
pip install streamlit

# 2. Run locally to test
streamlit run streamlit_app.py

# 3. Test at http://localhost:8501

# 4. Deploy to Streamlit Cloud
# - Push to GitHub
# - Deploy on share.streamlit.io
# - Get URL

# 5. Add to OpenKora
# - Create project
# - Add demo URL
# - Preview in iframe!
```

---

## Result

Your NBA prediction project will be:
- ✅ Accessible via web browser
- ✅ Interactive (users can generate predictions)
- ✅ Embedded in OpenKora iframe
- ✅ Shareable with demo URL
- ✅ Professional presentation

**Perfect for showcasing your ML project!** 🎉
