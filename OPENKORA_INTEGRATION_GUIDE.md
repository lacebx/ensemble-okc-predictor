# OpenKora Integration Guide - NBA Prediction Project

## Quick Summary

This guide shows you how to showcase your NBA prediction project in OpenKora Connect using the iframe preview feature.

## What You Need

1. ✅ Streamlit web interface (`streamlit_app.py`) - Already created!
2. ✅ Deployed Streamlit app (on Streamlit Cloud, Replit, or Railway)
3. ✅ OpenKora Connect account (developer profile)

---

## Step-by-Step Instructions

### Step 1: Deploy Your Streamlit App

#### Option A: Streamlit Cloud (Recommended - Easiest)

1. **Push to GitHub:**
   ```bash
   cd /home/lace/Documents/intelligent_systems/project3
   git init
   git add .
   git commit -m "Add Streamlit web interface for NBA predictions"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Main file path: `streamlit_app.py`
   - App URL: `https://your-app-name.streamlit.app`
   - Click "Deploy"
   - Wait 1-2 minutes for deployment

3. **Test your app:**
   - Visit `https://your-app-name.streamlit.app`
   - Make sure it loads correctly
   - Test the "Generate Predictions" button

#### Option B: Replit (Alternative)

1. Go to https://replit.com
2. Create new Python Repl
3. Upload all project files
4. Create `.replit` file:
   ```toml
   run = "streamlit run streamlit_app.py --server.port=8080 --server.address=0.0.0.0"
   ```
5. Click "Run"
6. Get URL: `https://your-repl.repl.co`

---

### Step 2: Add Project to OpenKora

1. **Login to OpenKora:**
   - Go to your OpenKora instance (localhost:8080 or deployed URL)
   - Login with your developer account

2. **Create New Project:**
   - Click "Create Project" or navigate to `/projects/create`
   - Fill in the form:

   **Basic Information:**
   - **Title:** `NBA Game Prediction Model - OKC Thunder`
   - **Description:** `Machine learning model using ensemble learning (Random Forest & Gradient Boosting) to predict NBA game outcomes for the OKC Thunder. Uses real-time data from the official NBA API.`
   - **Long Description:** 
     ```
     This project implements an ensemble learning system to predict NBA game outcomes, 
     specifically for the OKC Thunder. The model uses:
     
     - Random Forest (500 trees, depth 20)
     - Gradient Boosting (500 stages, depth 8)
     - 52 features including team stats, recent form, and efficiency metrics
     - Real-time data from official NBA API
     - Uncertainty-aware predictions with error margins
     - Interactive prediction workflow
     
     The web interface allows users to:
     - Generate predictions for the next 5 games
     - View model performance metrics
     - See detailed reasoning for each prediction
     ```

   **Links:**
   - **GitHub URL:** `https://github.com/yourusername/nba-prediction` (if you have it)
   - **Demo URL:** `https://your-app-name.streamlit.app` (your Streamlit Cloud URL)
   - **Website URL:** (optional) Same as demo URL

   **Tech Stack:**
   - Add tags: `Python`, `Machine Learning`, `Scikit-learn`, `Streamlit`, `NBA API`, `Ensemble Learning`

   **Tags:**
   - `machine-learning`
   - `nba`
   - `predictions`
   - `ensemble-learning`
   - `python`
   - `data-science`

3. **Upload Requirements File:**
   - Click "Upload Requirements File"
   - Select `requirements_streamlit.txt`
   - This helps others understand dependencies

4. **Save Project:**
   - Click "Create Project" or "Save"
   - You'll be redirected to your project page

---

### Step 3: View Your Project Preview

1. **Visit Your Project Page:**
   - Navigate to your project detail page
   - URL: `/projects/your-project-id`

2. **Check Preview:**
   - Click "Live Preview" tab
   - Or scroll to preview in "Overview" tab
   - Your Streamlit app should load in the iframe!

3. **Test Interaction:**
   - Click "Generate Predictions" in the iframe
   - Make sure it works correctly
   - Test all tabs (Predictions, Model Performance, About)

---

## Troubleshooting

### Iframe Not Loading

**Problem:** Preview shows blank or error message

**Solutions:**
1. **Check Streamlit Cloud Settings:**
   - Make sure your app is public
   - Check that it's not blocked by CORS

2. **Test Direct URL:**
   - Open `https://your-app-name.streamlit.app` directly
   - If it works there, it should work in iframe

3. **Use "Open in New Tab":**
   - OpenKora has a fallback button
   - Users can still access your demo

4. **Check Browser Console:**
   - Open browser dev tools (F12)
   - Check for CORS or iframe errors

### App Not Working

**Problem:** "Generate Predictions" button doesn't work

**Solutions:**
1. **Check Data File:**
   - Make sure `api_collected_data/cleaned/final_game_features.csv` exists
   - Run `collect_nba_api_data.py` first if needed

2. **Check Dependencies:**
   - Make sure all packages in `requirements_streamlit.txt` are installed
   - Streamlit Cloud installs them automatically

3. **Check Logs:**
   - In Streamlit Cloud, go to "Manage app" > "Logs"
   - Look for error messages

### Module Import Errors

**Problem:** Can't import prediction modules

**Solutions:**
1. **Check File Structure:**
   - Make sure `nba_prediction_from_api_data.py` is in the same directory
   - All files should be in the repo root

2. **Check Python Version:**
   - Streamlit Cloud uses Python 3.9+
   - Make sure your code is compatible

---

## Best Practices

### For Better Preview Experience:

1. **Optimize Loading:**
   - Cache data collection (don't run API calls on every load)
   - Use session state to cache predictions

2. **Error Handling:**
   - Show helpful error messages
   - Guide users if data is missing

3. **User Experience:**
   - Clear instructions
   - Loading indicators
   - Success/error feedback

### For OpenKora Listing:

1. **Good Description:**
   - Explain what the project does
   - Highlight key features
   - Mention technologies used

2. **Screenshots:**
   - Add screenshots of your app
   - Show predictions in action

3. **Tags:**
   - Use relevant tags
   - Helps discoverability

---

## Example Project Details

Here's what a good OpenKora project listing looks like:

**Title:** NBA Game Prediction Model - OKC Thunder

**Description:**
```
Machine learning model using ensemble learning to predict NBA game outcomes. 
Features interactive web interface, real-time data from NBA API, and detailed 
prediction explanations.
```

**Tech Stack:**
- Python
- Machine Learning
- Scikit-learn
- Streamlit
- NBA API
- Ensemble Learning

**Tags:**
- machine-learning
- nba
- predictions
- python
- data-science
- ensemble-learning

**Demo URL:** `https://your-app.streamlit.app`

---

## Result

Once set up, your NBA prediction project will be:

✅ **Accessible** via web browser  
✅ **Interactive** - users can generate predictions  
✅ **Embedded** in OpenKora iframe  
✅ **Shareable** with demo URL  
✅ **Professional** presentation  

**Perfect for showcasing your ML project!** 🎉

---

## Next Steps

1. ✅ Deploy Streamlit app
2. ✅ Add to OpenKora
3. ✅ Test preview
4. ✅ Share with others!

---

**Need Help?**
- Check `WEB_DEMO_SETUP.md` for detailed setup
- Check Streamlit Cloud docs: https://docs.streamlit.io/streamlit-cloud
- Check OpenKora preview docs: `PROJECT_PREVIEW_IMPLEMENTATION.md`
