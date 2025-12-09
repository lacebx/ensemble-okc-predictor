# Testing Guide - API Integrated Model

## Quick Test Checklist

### Before Running
- [ ] Today's date is Dec 8, 2025
- [ ] NBA API is installed: `pip install nba_api`
- [ ] Internet connection is active
- [ ] Local dataset files are present

### During API Test

**What to Look For:**

1. **Current Season**
   - ✅ Should show: "2025-26"
   - ❌ Wrong if shows: "2024-25"

2. **Most Recent Game Date**
   - ✅ Should be: Dec 1-8, 2025 (within last week)
   - ❌ Wrong if shows: April 2025 or earlier

3. **Days Ago**
   - ✅ Should be: 0-7 days
   - ❌ Wrong if shows: 200+ days

4. **Last 5 Games**
   - ✅ Should all be from Nov-Dec 2025
   - ❌ Wrong if shows April 2025 games

5. **Record**
   - ✅ Should match current OKC record (check NBA.com)
   - ❌ Wrong if shows old season record

### Decision Tree

```
API Test Shows:
├─ Recent dates (Dec 2025) + Correct record
│  └─ Type 'y' → Use API data
│
├─ Old dates (April 2025) + Wrong record
│  └─ Type 'n' → Use local dataset
│
└─ Mixed/Unclear
   └─ Type 'n' → Use local dataset (safer)
```

## Expected Output (Correct)

```
Current Date: 2025-12-08
Current Season: 2025-26

2. Testing Game Log Retrieval (Last 5 Games from Today)...
   ✓ Retrieved 25 total games, 25 games up to today
     Most recent game: DEC 07, 2025
     Game date parsed: 2025-12-07
     Days ago: 1
     Result: OKC vs. DEN - W
     Score: 125 points

   Last 5 Games (Most Recent First):
     2025-12-07: OKC vs. DEN - W (125 pts)
     2025-12-05: OKC @ LAL - W (118 pts)
     2025-12-03: OKC vs. PHO - W (122 pts)
     2025-12-01: OKC @ MEM - W (115 pts)
     2025-11-29: OKC vs. MIN - W (130 pts)

3. Testing Season Statistics (Games up to Today)...
   ✓ OKC Thunder Record: 22-1
     Win Percentage: 95.7%
```

## What to Do If API Shows Wrong Data

### Option 1: Use Local Dataset (Recommended)
- Type 'n' when asked
- Script will use local player stats
- More reliable for training data
- Win counting is fixed in local dataset

### Option 2: Check NBA.com Manually
1. Go to NBA.com
2. Check OKC Thunder schedule
3. Verify last 5 games match API output
4. If they don't match, use local dataset

### Option 3: Wait and Retry
- NBA API might have temporary issues
- Try again in a few minutes
- Check if NBA.com is accessible

## After API Test

### If You Type 'y' (API Verified)
- Script uses API data for recent form
- Still uses local dataset for training
- Predictions based on API recent games

### If You Type 'n' (API Not Verified)
- Script uses local dataset only
- Win counting uses fixed function
- Predictions based on local recent games

## Verifying Win Count

After predictions, check:
```
OKC Thunder Recent Performance:
  Last 10 games: X-Y    ← Should match actual
  Win rate: XX.X%
  Recent results: W W W W W W W W W W
```

If win count is still wrong:
1. Check local dataset has correct game results
2. Verify `get_okc_recent_record()` function logic
3. Check that games are sorted by date correctly

## Common Issues

### Issue: "API test failed"
**Solution**: Check internet, try again, or use local dataset

### Issue: "No games found"
**Solution**: Season might be wrong, check `get_current_season()` function

### Issue: "Dates are in the future"
**Solution**: Date parsing issue, check `parse_api_date()` function

### Issue: "Win count still wrong"
**Solution**: Check `get_okc_recent_record()` function, verify game data

---

**Remember**: If in doubt, use local dataset (type 'n'). It's more reliable for training.
