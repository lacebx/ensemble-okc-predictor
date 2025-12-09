# 🏀 NBA Game Prediction Model - OKC Thunder

A machine learning system that predicts NBA game outcomes for the OKC Thunder using ensemble learning techniques. The model uses real-time data from the official NBA API and provides detailed reasoning for each prediction.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ML](https://img.shields.io/badge/ML-Ensemble%20Learning-orange.svg)](https://scikit-learn.org/)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Data Sources](#data-sources)
- [Web Interface](#web-interface)
- [API Integration](#api-integration)
- [Documentation](#documentation)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project implements an ensemble learning system to predict NBA game outcomes, specifically for the OKC Thunder. The model combines Random Forest and Gradient Boosting classifiers to achieve high accuracy predictions with uncertainty quantification.

### Key Highlights

- ✅ **Ensemble Learning**: Random Forest (500 trees) + Gradient Boosting (500 stages)
- ✅ **Real-Time Data**: Automatically collects data from official NBA API
- ✅ **52 Features**: Comprehensive feature engineering including stats, form, and efficiency metrics
- ✅ **Uncertainty-Aware**: Provides error margins and confidence ranges
- ✅ **Interactive Predictions**: Progressive prediction system with weighted historical data
- ✅ **Web Interface**: Streamlit-based web app for easy access
- ✅ **Detailed Explanations**: Transparent reasoning for each prediction

---

## ✨ Features

### Core Functionality

- **Data Collection**: Automated collection from NBA API with robust date handling
- **Feature Engineering**: 52 features including:
  - Team statistics (points, rebounds, assists, etc.)
  - Shooting percentages (FG%, 3P%, FT%)
  - Recent form (last 10 games win rate)
  - Efficiency metrics
  - Point differentials
  - Plus/minus statistics

- **Model Training**: 
  - Random Forest with 500 trees, max depth 20
  - Gradient Boosting with 500 stages, max depth 8
  - Cross-validation for model selection
  - Error margin calculation

- **Interactive Predictions**:
  - Predicts games one at a time
  - Uses predicted games for future predictions (with 0.7x weight)
  - Maintains same real historical data baseline
  - User-controlled workflow

- **Transparency**:
  - Detailed explanation file for each prediction
  - Feature importance analysis
  - Model performance metrics
  - Confidence ranges and error margins

### Web Interface

- **Streamlit App**: Interactive web interface
- **Three Tabs**: Predictions, Model Performance, About
- **Real-Time Updates**: Generate predictions on demand
- **Visualizations**: Charts and metrics display
- **Responsive Design**: Works on desktop and mobile

---

## 📸 Screenshots

### Command Line Interface
```
============================================================
Predicting OKC Thunder Next 5 Games
============================================================

OKC Thunder Recent Performance (from API data):
  Last 10 games: 10-0
  Win rate: 100.0%
  Recent results: W W W W W W W W W W

PREDICTION #1
================================================================================
Date: 2025-12-10
Opponent: Phoenix Suns (PHX)
Location: HOME

Prediction: WIN
Win Probability: 99.5%
Confidence Range: 99.2% - 99.8%
Error Margin: ±0.3%
```

### Web Interface
- Interactive Streamlit app with gradient-styled prediction cards
- Model performance metrics dashboard
- Detailed explanation viewer

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- pip or conda package manager
- NBA API access (free, no API key required)

### Step 1: Clone the Repository

```bash
git clone <your-repo-url>
cd intelligent_systems/project3
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n nba-prediction python=3.11
conda activate nba-prediction
```

### Step 3: Install Dependencies

```bash
# For command-line usage
pip install -r requirements.txt

# For web interface
pip install -r requirements_streamlit.txt
```

### Step 4: Install NBA API

```bash
pip install nba-api
```

---

## ⚡ Quick Start

### 1. Collect Data

```bash
python collect_nba_api_data.py
```

This will:
- Fetch games from NBA API
- Fetch team game logs
- Fetch player game logs (optional, can be slow)
- Fetch team season statistics
- Clean and process data
- Save to `api_collected_data/cleaned/final_game_features.csv`

### 2. Generate Predictions

```bash
python nba_prediction_from_api_data.py
```

This will:
- Load collected data
- Train ensemble models
- Make predictions for next 5 games
- Generate explanation file
- Create visualizations

### 3. Run Web Interface

```bash
streamlit run streamlit_app.py
```

Then open http://localhost:8501 in your browser.

---

## 📖 Usage

### Command Line Usage

#### Data Collection

```bash
python collect_nba_api_data.py
```

**Output:**
- Raw data: `api_collected_data/raw/`
- Cleaned data: `api_collected_data/cleaned/final_game_features.csv`
- Summary report: `api_collected_data/collection_report.json`

#### Generate Predictions

```bash
python nba_prediction_from_api_data.py
```

**Interactive Mode:**
- Makes one prediction at a time
- Prompts: "Would you like to predict the next game? (yes/no)"
- Uses predicted games for subsequent predictions (with reduced weight)

**Output:**
- Predictions displayed in terminal
- Explanation file: `prediction_explanation.txt`
- Visualizations: `model_comparison_api_data.png`, `okc_predictions_api_data.png`

### Web Interface Usage

1. **Start Streamlit:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open Browser:**
   - Navigate to http://localhost:8501

3. **Generate Predictions:**
   - Click "🚀 Generate Predictions" button
   - Wait for model training (1-2 minutes)
   - View predictions in styled cards

4. **Explore:**
   - **Predictions Tab**: See game predictions with probabilities
   - **Model Performance Tab**: View accuracy metrics and feature importance
   - **About Tab**: Learn about the project

### Python API Usage

```python
from nba_prediction_from_api_data import (
    load_api_data,
    preprocess_data,
    train_models,
    predict_okc_games
)

# Load data
game_features_df = load_api_data()

# Preprocess
game_features_df = preprocess_data(game_features_df)

# Train models
X_train, X_test, y_train, y_test = train_test_split(...)
models, results = train_models(X_train, X_test, y_train, y_test)

# Make predictions
predictions_df, predictions_list = predict_okc_games(
    models, results, error_margins, game_features_df
)
```

---

## 📁 Project Structure

```
project3/
├── collect_nba_api_data.py          # Data collection script
├── nba_prediction_from_api_data.py  # Main prediction script
├── streamlit_app.py                 # Web interface
├── requirements.txt                 # CLI dependencies
├── requirements_streamlit.txt       # Web interface dependencies
│
├── api_collected_data/              # Collected API data
│   ├── raw/                         # Raw API responses
│   │   ├── all_games_2025-26.csv
│   │   ├── team_game_logs_2025-26.csv
│   │   ├── player_game_logs_2025-26.csv
│   │   └── team_stats_2025-26.csv
│   └── cleaned/                     # Processed data
│       ├── game_features_cleaned.csv
│       └── final_game_features.csv  # Main dataset
│
├── dataset/                         # Local datasets (optional)
│   ├── Team Stats Per Game.csv
│   ├── Team Summaries.csv
│   └── ...
│
├── nba_api/                         # NBA API library
│   └── ...
│
├── .streamlit/                      # Streamlit config
│   └── config.toml
│
└── Documentation/
    ├── README.md                    # This file
    ├── DESIGN_DECISIONS.md          # Methodology rationale
    ├── API_DATA_WORKFLOW.md         # Data collection workflow
    ├── MODEL_IMPROVEMENTS.md        # Model improvements
    ├── WEB_DEMO_SETUP.md            # Web interface setup
    ├── OPENKORA_INTEGRATION_GUIDE.md # OpenKora showcase guide
    ├── AI_COLLABORATION_REPORT.md   # AI collaboration details
    └── prediction_explanation.txt   # Latest prediction explanations
```

---

## 🧠 Model Architecture

### Ensemble Methods

#### 1. Random Forest
- **Trees**: 500
- **Max Depth**: 20
- **Min Samples Split**: 3
- **Min Samples Leaf**: 1
- **Max Features**: sqrt
- **Class Weight**: balanced

#### 2. Gradient Boosting
- **Stages**: 500
- **Learning Rate**: 0.03
- **Max Depth**: 8
- **Min Samples Split**: 3
- **Min Samples Leaf**: 1
- **Subsample**: 0.85
- **Early Stopping**: Enabled

### Feature Engineering

**52 Features Total:**

1. **Team Statistics (per team):**
   - Points, FG%, 3P%, FT%
   - Rebounds, Assists, Steals, Blocks
   - Turnovers, Personal Fouls
   - Plus/Minus

2. **Derived Features:**
   - Point differential
   - Shooting percentage differentials
   - Rebound differential
   - Assist differential
   - Efficiency metrics

3. **Recent Form:**
   - Team 1 recent win rate (last 5 games)
   - Team 2 recent win rate (last 5 games)
   - Recent win rate differential

4. **Advanced Metrics:**
   - Offensive efficiency
   - Efficiency differential

### Model Selection

- Best model selected based on cross-validation mean score
- 5-fold cross-validation
- Error margins calculated from CV standard deviation
- Adjusted error margins incorporate prediction confidence

---

## 📊 Data Sources

### Primary Source: NBA API

The project uses the official NBA API via the `nba-api` Python library:

- **League Game Finder**: All games for current season
- **Team Game Logs**: Detailed team statistics per game
- **Player Game Logs**: Player-level statistics (optional)
- **League Dash Team Stats**: Season averages

### Data Collection Process

1. **Automatic Season Detection**: Determines current NBA season
2. **Date Filtering**: Only collects games up to current date
3. **Robust Date Parsing**: Handles multiple date formats
4. **Data Cleaning**: Removes duplicates, handles missing values
5. **Feature Engineering**: Creates game-level features
6. **Recent Form Calculation**: Adds rolling win rates

### Data Storage

- **Raw Data**: `api_collected_data/raw/` (CSV files)
- **Cleaned Data**: `api_collected_data/cleaned/final_game_features.csv`
- **Summary Report**: `api_collected_data/collection_report.json`

---

## 🌐 Web Interface

### Streamlit App

The web interface provides an interactive way to use the prediction model.

**Features:**
- One-click prediction generation
- Visual prediction cards with gradients
- Model performance dashboard
- Feature importance charts
- Responsive design

**Tabs:**
1. **Predictions**: Generate and view game predictions
2. **Model Performance**: View accuracy metrics and feature importance
3. **About**: Project information and methodology

### Deployment

**Streamlit Cloud (Recommended):**
1. Push to GitHub
2. Deploy on https://share.streamlit.io
3. Get URL: `https://your-app.streamlit.app`

**Replit:**
1. Upload files to Replit
2. Create `.replit` file
3. Run and get URL

**Railway/Render:**
1. Create `Procfile`
2. Deploy from GitHub
3. Get URL

See `WEB_DEMO_SETUP.md` for detailed instructions.

---

## 🔌 API Integration

### NBA API Endpoints Used

```python
from nba_api.stats.endpoints import (
    leaguegamefinder,      # All games
    teamgamelogs,          # Team game logs
    playergamelogs,        # Player game logs
    leaguedashteamstats    # Team season stats
)
```

### API Parameters

- **Season**: Automatically detected (e.g., "2025-26")
- **Date To**: Current date (filters future games)
- **Date From**: Season start date

### Error Handling

- Graceful fallback if API is unavailable
- Retry logic for rate limits
- User verification step for data accuracy

---

## 📚 Documentation

### Core Documentation

- **README.md**: This file - project overview
- **DESIGN_DECISIONS.md**: Methodology and design choices
- **API_DATA_WORKFLOW.md**: Data collection process
- **MODEL_IMPROVEMENTS.md**: Model evolution and improvements
- **PREDICTION_LOGIC_FIXED.md**: Interactive prediction system

### Setup Guides

- **WEB_DEMO_SETUP.md**: Streamlit web interface setup
- **OPENKORA_INTEGRATION_GUIDE.md**: Showcasing on OpenKora platform
- **QUICK_START_API_DATA.md**: Quick start guide

### Reports

- **AI_COLLABORATION_REPORT.md**: AI collaboration details
- **PROJECT_REPORT.md**: Full project analysis
- **prediction_explanation.txt**: Latest prediction explanations

---

## 🚀 Deployment

### Local Development

```bash
# Activate virtual environment
source venv/bin/activate

# Collect data
python collect_nba_api_data.py

# Run predictions
python nba_prediction_from_api_data.py

# Run web interface
streamlit run streamlit_app.py
```

### Production Deployment

#### Streamlit Cloud

1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy with `streamlit_app.py` as main file
4. Set environment variables if needed

#### Docker (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements_streamlit.txt .
RUN pip install -r requirements_streamlit.txt

COPY . .

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
```

---

## 🧪 Testing

### Manual Testing

1. **Data Collection:**
   ```bash
   python collect_nba_api_data.py
   # Verify: api_collected_data/cleaned/final_game_features.csv exists
   ```

2. **Predictions:**
   ```bash
   python nba_prediction_from_api_data.py
   # Verify: predictions displayed, explanation file created
   ```

3. **Web Interface:**
   ```bash
   streamlit run streamlit_app.py
   # Verify: app loads, predictions generate successfully
   ```

### Expected Outputs

- ✅ Data collection completes without errors
- ✅ Models train with >95% accuracy
- ✅ Predictions generated for all 5 games
- ✅ Explanation file created
- ✅ Visualizations saved

---

## 🔧 Configuration

### Streamlit Configuration

Edit `.streamlit/config.toml`:

```toml
[server]
enableCORS = false
enableXsrfProtection = false
headless = true

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
```

### Model Parameters

Edit model parameters in `nba_prediction_from_api_data.py`:

```python
# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=500,      # Number of trees
    max_depth=20,         # Tree depth
    ...
)

# Gradient Boosting
gb_model = GradientBoostingClassifier(
    n_estimators=500,     # Number of stages
    learning_rate=0.03,   # Learning rate
    ...
)
```

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**
4. **Test thoroughly**
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Areas for Contribution

- Additional ensemble methods
- Feature engineering improvements
- Web interface enhancements
- Documentation improvements
- Bug fixes
- Performance optimizations

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **NBA API**: For providing free access to NBA statistics
- **Scikit-learn**: For excellent ensemble learning implementations
- **Streamlit**: For making web interfaces easy
- **OpenKora**: For the platform to showcase projects

---

## 📧 Contact

For questions, issues, or suggestions:

- **GitHub Issues**: Open an issue on GitHub
- **Email**: [Your email]
- **Project Page**: [Your OpenKora project URL]

---

## 🗺️ Roadmap

### Future Enhancements

- [ ] Real-time prediction updates
- [ ] Additional teams support
- [ ] Player-level predictions
- [ ] Advanced visualizations
- [ ] API endpoint for predictions
- [ ] Mobile app version
- [ ] Historical prediction accuracy tracking

### Known Limitations

- Predictions are based on historical data only
- Doesn't account for injuries or lineup changes
- API rate limits may affect data collection speed
- Model accuracy depends on data quality

---

## 📊 Model Performance

### Current Metrics

- **Random Forest Accuracy**: 100% (CV Mean: 1.0000)
- **Gradient Boosting Accuracy**: 100% (CV Mean: 1.0000)
- **Error Margin**: ±0.3% (adjusted)
- **Features**: 52
- **Training Games**: ~400-500 per season

### Performance Notes

- High accuracy may indicate overfitting on small dataset
- Cross-validation helps assess generalization
- Error margins account for uncertainty
- Model performance varies by season

---

## 🎓 Educational Value

This project demonstrates:

- **Ensemble Learning**: Combining multiple models
- **Feature Engineering**: Creating meaningful features
- **API Integration**: Working with real-world APIs
- **Data Preprocessing**: Cleaning and preparing data
- **Model Evaluation**: Cross-validation and error margins
- **Web Development**: Creating ML web interfaces
- **Documentation**: Comprehensive project documentation

---

## ⚠️ Disclaimer

This project is for educational and demonstration purposes. Predictions are based on statistical models and historical data. Actual game outcomes depend on many factors not captured in the model (injuries, coaching decisions, referee calls, etc.). Use predictions at your own discretion.

---

## 📈 Version History

- **v1.0.0** (2025-12-08): Initial release
  - Ensemble learning implementation
  - NBA API integration
  - Interactive predictions
  - Web interface
  - Comprehensive documentation

---

**Made with ❤️ for NBA fans and ML enthusiasts**

---

*Last Updated: December 2025*
