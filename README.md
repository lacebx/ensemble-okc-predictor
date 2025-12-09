# NBA Game Prediction: OKC Thunder Next 5 Games

## Project Overview

This project implements a machine learning solution using ensemble learning techniques to predict the next 5 game results for the Oklahoma City Thunder (OKC). The OKC Thunder is currently on an impressive 22-1 streak (22 wins, 1 loss) in the 2026 NBA season.

## Project Structure

```
project3/
├── dataset/                          # Dataset directory
│   ├── Team Stats Per Game.csv
│   ├── Team Summaries.csv
│   ├── Opponent Stats Per Game.csv
│   └── ... (other CSV files)
├── nba_game_prediction.ipynb        # Jupyter notebook (main analysis)
├── nba_prediction.py                # Standalone Python script
├── requirements.txt                  # Python dependencies
├── PROJECT_REPORT.md                # Comprehensive project report
└── README.md                        # This file
```

## Requirements

- Python 3.8 or higher
- Required packages (see requirements.txt):
  - pandas >= 1.5.0
  - numpy >= 1.23.0
  - matplotlib >= 3.6.0
  - seaborn >= 0.12.0
  - scikit-learn >= 1.2.0
  - jupyter >= 1.0.0

## Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd /home/lace/Documents/intelligent_systems/project3
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Option 1: Run Python Script
```bash
python nba_prediction.py
```

This will:
- Load and preprocess the data
- Train all ensemble models (Random Forest, Gradient Boosting, Bagging, Voting Classifier)
- Compare model performances
- Generate visualizations
- Make predictions for OKC Thunder's next 5 games

### Option 2: Use Jupyter Notebook
```bash
jupyter notebook nba_game_prediction.ipynb
```

Open the notebook and run all cells for interactive analysis.

## Ensemble Learning Methods Implemented

1. **Random Forest**: Bagging-based ensemble with 100 decision trees
2. **Gradient Boosting**: Sequential boosting with 100 estimators
3. **Bagging**: Bootstrap aggregating with 50 decision trees
4. **Voting Classifier**: Meta-ensemble combining all three methods

## Output Files

After running the script, you'll get:

1. **model_comparison.png**: Bar charts comparing all ensemble methods
2. **feature_importance.png**: Top 15 most important features
3. **confusion_matrices.png**: Confusion matrices for all models
4. **okc_predictions.png**: Predictions visualization for OKC's next 5 games

## Dataset Information

### Target Variable (y)
- **Game Outcome**: Binary classification (Win = 1, Loss = 0)

### Input Features (X)
- Team statistics (points, FG%, rebounds, assists, etc.)
- Advanced metrics (offensive rating, defensive rating, net rating)
- Opponent statistics
- Derived features (differentials between teams)

## Model Performance

The models are evaluated using:
- **Accuracy**: Overall prediction correctness
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Classification Metrics**: Precision, Recall, F1-score

Expected performance: 85-95% accuracy across all ensemble methods.

## Predictions

The model predicts OKC Thunder's next 5 games against:
- Denver Nuggets (DEN)
- Los Angeles Lakers (LAL)
- Houston Rockets (HOU)
- New York Knicks (NYK)
- Sacramento Kings (SAC)

Given OKC's exceptional 22-1 record and strong statistics, the model predicts high win probability for most games.

## Project Report

See `PROJECT_REPORT.md` for:
- Detailed methodology
- Dataset justification
- Model implementation details
- Results analysis
- Challenges and solutions
- Future improvements

## Notes

- The dataset uses season-level statistics, not individual game results
- Matchups are created synthetically based on team statistics
- Predictions are based on current 2026 season data
- Model performance may vary based on data quality and feature engineering

## Troubleshooting

### Issue: Missing data files
**Solution**: Ensure all CSV files are in the `dataset/` directory

### Issue: Import errors
**Solution**: Install all requirements: `pip install -r requirements.txt`

### Issue: Memory errors
**Solution**: Reduce n_estimators in model parameters or use smaller dataset

## Contact

For questions or issues, please refer to the project report or check the code comments.

---

**Project**: Intelligent Systems - Ensemble Learning
**Focus**: NBA Game Prediction using Multiple Ensemble Methods
