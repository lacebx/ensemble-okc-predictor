# API Data Collection Workflow

## Overview

This workflow uses **NBA API** to collect accurate, live data instead of potentially outdated CSV files. All data is fetched from the official NBA API, cleaned, and stored in a structured format.

## Step-by-Step Process

### Step 1: Collect Data from NBA API

Run the data collection script:

```bash
python collect_nba_api_data.py
```

**What it does:**
1. Fetches all NBA games using `LeagueGameFinder`
2. Fetches team game logs using `TeamGameLogs`
3. Fetches player game logs using `PlayerGameLogs`
4. Fetches team season statistics
5. Cleans and processes all data
6. Creates unified dataset: `api_collected_data/cleaned/final_game_features.csv`

**Time:** 5-10 minutes (depending on data size)

**Output:**
- Raw data in `api_collected_data/raw/`
- Cleaned data in `api_collected_data/cleaned/`
- Summary report in `api_collected_data/collection_report.json`

### Step 2: Run Predictions Using API Data

Once data is collected, run predictions:

```bash
python nba_prediction_from_api_data.py
```

**What it does:**
1. Loads cleaned API data
2. Trains Random Forest and Gradient Boosting models
3. Makes predictions for OKC's next 5 games
4. Generates visualizations

**Benefits:**
- ✅ Uses accurate, live NBA API data
- ✅ Correct win counting (fixed logic)
- ✅ Recent games only (filtered to today)
- ✅ No outdated CSV files

## Data Structure

```
api_collected_data/
├── raw/
│   ├── all_games_2025-26.csv          # All games from API
│   ├── team_game_logs_2025-26.csv     # Team game logs
│   ├── player_game_logs_2025-26.csv   # Player game logs
│   └── team_stats_2025-26.csv         # Team season stats
├── cleaned/
│   ├── game_features_cleaned.csv      # Processed game features
│   └── final_game_features.csv        # FINAL DATASET (use this!)
└── collection_report.json              # Summary report
```

## Why This Approach?

### Problems with CSV Files:
- ❌ May be outdated
- ❌ May have errors
- ❌ May not include recent games
- ❌ Hard to verify accuracy

### Benefits of API Data:
- ✅ Always current (fetched live)
- ✅ Official NBA source
- ✅ Includes all recent games
- ✅ Can verify with API test
- ✅ Automatic date filtering

## Data Collection Details

### Endpoints Used:

1. **LeagueGameFinder**
   - Gets all games for the season
   - Filters to games up to today
   - Includes team stats per game

2. **TeamGameLogs**
   - Team-level statistics per game
   - More detailed than LeagueGameFinder
   - Includes advanced metrics

3. **PlayerGameLogs**
   - Player-level statistics per game
   - Can aggregate to team level
   - Provides granular data

4. **LeagueDashTeamStats**
   - Season-level team statistics
   - For context and comparison

### Data Cleaning Process:

1. **Date Parsing**: Handles multiple date formats
2. **Date Filtering**: Only games up to today
3. **Feature Creation**: Combines team stats into matchups
4. **Derived Features**: Calculates differentials and efficiency
5. **Recent Form**: Adds last 5 games win rate
6. **Missing Values**: Fills with median or 0

## Verification

After collection, check:

1. **Collection Report**: `api_collected_data/collection_report.json`
   - Number of games collected
   - Date range
   - Teams included

2. **Data Quality**: Open `final_game_features.csv`
   - Check dates are recent
   - Verify team abbreviations
   - Check win/loss distribution

3. **OKC Games**: Verify OKC's recent games match actual results

## When to Re-collect Data

Re-run `collect_nba_api_data.py` when:
- New games have been played
- You want the most current data
- Predictions seem off (data might be stale)
- Before making important predictions

## Troubleshooting

### "No games found"
- Check if season has started
- Verify date filtering is correct
- Check NBA API is accessible

### "API rate limiting"
- Script includes delays between requests
- If issues persist, wait a few minutes and retry

### "Data seems old"
- Check `collection_report.json` for date range
- Verify `date_to` parameter is set to today
- Re-run collection script

## Next Steps

1. **First Time**: Run `collect_nba_api_data.py`
2. **Verify Data**: Check collection report and sample data
3. **Run Predictions**: Use `nba_prediction_from_api_data.py`
4. **Update Regularly**: Re-collect data weekly or before predictions

---

**Key File**: `api_collected_data/cleaned/final_game_features.csv`
**This is the only file needed for predictions!**
