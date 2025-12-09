# Quick Start Guide

## Running the Project

### Step 1: Install Dependencies
```bash
cd /home/lace/Documents/intelligent_systems/project3
pip install -r requirements.txt
```

### Step 2: Run the Analysis
```bash
python nba_prediction.py
```

### Step 3: View Results
The script will generate:
- Console output with model comparisons and predictions
- 4 visualization PNG files in the project directory

## What the Script Does

1. **Loads Data**: Reads NBA team statistics from CSV files
2. **Creates Features**: Builds game matchups and feature vectors
3. **Trains Models**: Implements 4 ensemble learning methods
4. **Compares Performance**: Evaluates and compares all models
5. **Makes Predictions**: Predicts OKC Thunder's next 5 games
6. **Generates Visualizations**: Creates comparison charts and prediction graphs

## Expected Output

```
============================================================
NBA Game Prediction Model - OKC Thunder
Ensemble Learning Project
============================================================
Loading datasets...
Team Stats shape: (X, Y)
...

Model Comparison Summary
============================================================
                    accuracy  cv_mean    cv_std
Random Forest         0.XXXX   0.XXXX   0.XXXX
Gradient Boosting    0.XXXX   0.XXXX   0.XXXX
Bagging              0.XXXX   0.XXXX   0.XXXX
Voting Classifier    0.XXXX   0.XXXX   0.XXXX

Best Model: [Model Name]

Predictions for OKC Thunder Next 5 Games
============================================================
Opponent  Opponent Record  Prediction  Win Probability
...
```

## Key Files

- `nba_prediction.py` - Main script (run this)
- `nba_game_prediction.ipynb` - Jupyter notebook version
- `PROJECT_REPORT.md` - Full project documentation
- `README.md` - Detailed project information

## Troubleshooting

**If you get import errors:**
```bash
pip install --upgrade pandas numpy matplotlib seaborn scikit-learn jupyter
```

**If dataset files are missing:**
- Ensure all CSV files are in the `dataset/` directory
- Check that file names match exactly (case-sensitive)

**If you want to modify predictions:**
- Edit the `opponent_abbrevs` list in the `predict_okc_games()` function
- Change line: `opponent_abbrevs = ['DEN', 'LAL', 'HOU', 'NYK', 'SAC']`

## Next Steps

1. Run the script to see predictions
2. Review `PROJECT_REPORT.md` for detailed methodology
3. Experiment with different model parameters
4. Try different opponents for predictions
