# Quick Start - Using NBA API Data

## New Workflow (Recommended)

Instead of using potentially outdated CSV files, we now collect data directly from the NBA API.

## Two-Step Process

### Step 1: Collect Data from NBA API

```bash
python collect_nba_api_data.py
```

**What happens:**
- Fetches all games from NBA API
- Fetches team and player game logs
- Cleans and processes data
- Saves to `api_collected_data/cleaned/final_game_features.csv`

**Time:** 5-10 minutes

**Output:**
```
api_collected_data/
├── raw/              # Raw API responses
├── cleaned/          # Processed data
│   └── final_game_features.csv  ← USE THIS FILE!
└── collection_report.json
```

### Step 2: Run Predictions

```bash
python nba_prediction_from_api_data.py
```

**What happens:**
- Loads cleaned API data
- Trains models
- Makes predictions
- Generates visualizations

**Output:**
- Predictions for OKC's next 5 games
- Enhanced visualizations
- **Correct win counting** (fixed!)

## Why This is Better

✅ **Accurate Data**: Direct from NBA API  
✅ **Current Data**: Always up-to-date  
✅ **Fixed Win Counting**: Correct logic  
✅ **Recent Games Only**: Filtered to today  
✅ **No Outdated CSVs**: Fresh data every time  

## First Time Setup

1. **Collect data:**
   ```bash
   python collect_nba_api_data.py
   ```

2. **Verify data:**
   - Check `api_collected_data/collection_report.json`
   - Verify dates are recent
   - Check OKC games match actual results

3. **Run predictions:**
   ```bash
   python nba_prediction_from_api_data.py
   ```

## Updating Data

Re-run collection when:
- New games have been played
- Before making predictions
- Weekly for fresh data

```bash
python collect_nba_api_data.py  # Updates data
python nba_prediction_from_api_data.py  # Uses updated data
```

## Files Created

### Data Collection:
- `api_collected_data/raw/*.csv` - Raw API data
- `api_collected_data/cleaned/final_game_features.csv` - **Main dataset**
- `api_collected_data/collection_report.json` - Summary

### Predictions:
- `model_comparison_api_data.png` - Model comparison
- `okc_predictions_api_data.png` - Predictions with error bars

## Troubleshooting

### "API data file not found"
**Solution:** Run `collect_nba_api_data.py` first

### "No games found"
**Solution:** Check if season has started, verify date filtering

### "Win count still wrong"
**Solution:** The logic is fixed - verify API data has correct results

---

**Key Point**: Always use `final_game_features.csv` from API collection, not old CSV files!
