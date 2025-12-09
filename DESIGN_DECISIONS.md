# Design Decisions Documentation
## NBA Game Prediction Model - OKC Thunder

This document explains the rationale behind every design decision in the enhanced prediction model.

---

## 1. Dataset Selection and Data Sources

### Why Player-Level Game Data Instead of Season Averages?

**Decision**: Use game-by-game player statistics aggregated to team level rather than season averages.

**Rationale**:
- **Temporal Relevance**: Game-by-game data captures current form and momentum, which is crucial for predictions. Season averages can be misleading if a team's performance has changed recently.
- **Granularity**: Player stats aggregated by game provide more detailed information about team performance in specific matchups.
- **Real Outcomes**: The dataset contains actual game results (W/L), providing ground truth for training rather than synthetic matchups.
- **Momentum Capture**: Recent games are more predictive than games from months ago.

**Alternative Considered**: Using season averages from team statistics.
- **Rejected Because**: Season averages don't account for recent form, injuries, or tactical changes that occur during the season.

---

## 2. Feature Engineering Decisions

### 2.1 Aggregation Method

**Decision**: Aggregate player statistics by summing totals (PTS, REB, AST) and averaging percentages (FG%, 3P%).

**Rationale**:
- **Team Totals**: Points, rebounds, assists are additive - the team total is the sum of individual contributions.
- **Percentage Averages**: Shooting percentages should be weighted averages, but for simplicity and to maintain interpretability, we use team totals divided by team attempts.
- **Game Score Average**: Using mean Game Score (GmSc) provides an indicator of overall player performance quality in that game.

**Alternative Considered**: Weighted averages based on minutes played.
- **Rejected Because**: The current method is simpler and the correlation between minutes and impact is already captured in the totals.

### 2.2 Derived Features (Differentials)

**Decision**: Include difference features (pts_diff, fg_pct_diff, reb_diff, etc.) in addition to absolute team statistics.

**Rationale**:
- **Relative Strength**: The outcome of a game depends on the relative performance of teams, not absolute values. A team scoring 120 points might win or lose depending on opponent.
- **Model Efficiency**: Differential features directly capture matchup dynamics, reducing the need for the model to learn these relationships.
- **Interpretability**: Easier to understand - "OKC scores 5 more points per game than opponent" is more intuitive than comparing two separate values.

**Alternative Considered**: Only absolute features.
- **Rejected Because**: Models would need to learn relative relationships, requiring more data and potentially less accurate predictions.

### 2.3 Efficiency Metrics

**Decision**: Include True Shooting Percentage-like efficiency metric: PTS / (FGA + 0.44*FTA + TOV).

**Rationale**:
- **Comprehensive Efficiency**: Captures scoring efficiency while accounting for possessions used (field goal attempts, free throw attempts, turnovers).
- **Basketball Analytics**: This formula is standard in basketball analytics and provides a single metric for offensive efficiency.
- **Possession-Based**: More accurate than simple points per game because it accounts for pace of play.

**Alternative Considered**: Points per game only.
- **Rejected Because**: Doesn't account for how many possessions were used to score those points.

### 2.4 Recent Form Features

**Decision**: Add rolling win rate over last 5 games as a feature.

**Rationale**:
- **Momentum**: Teams on winning streaks tend to continue winning; teams on losing streaks tend to continue losing (momentum effect).
- **Current Form**: Recent performance is more predictive than season-long averages.
- **Psychological Factors**: Confidence and team chemistry, which affect performance, are better captured by recent results.

**Alternative Considered**: 
- Last 10 games (too long, dilutes recent form)
- Last 3 games (too short, too noisy)
- **Selected**: 5 games as a balance between capturing momentum and maintaining statistical significance.

---

## 3. Ensemble Method Selection

### Why Only Two Methods?

**Decision**: Use only Random Forest and Gradient Boosting, not Bagging or Voting Classifier.

**Rationale**:
- **Performance**: Random Forest and Gradient Boosting consistently perform best for structured/tabular data like sports statistics.
- **Complementary Strengths**: 
  - Random Forest: Excellent at handling feature interactions and non-linear relationships
  - Gradient Boosting: Excellent at sequential learning and capturing complex patterns
- **Computational Efficiency**: Two models are faster to train and evaluate than four.
- **Focus**: Better to optimize two excellent models than spread effort across four.

**Alternatives Considered**:
- **Bagging**: Rejected because Random Forest is already a bagging method (bagging of decision trees). Adding separate Bagging would be redundant.
- **Voting Classifier**: Rejected because with only two models, voting doesn't provide significant benefit over selecting the best single model. Voting is more useful with 3+ diverse models.

### 3.1 Random Forest Configuration

**Decision**: 
- n_estimators=200 (more trees)
- max_depth=15 (deeper trees)
- max_features='sqrt' (feature subsampling)
- class_weight='balanced'

**Rationale**:
- **More Trees**: 200 trees provide better generalization than 100, with diminishing returns beyond this point.
- **Deeper Trees**: Depth 15 allows capturing complex interactions while preventing overfitting through ensemble averaging.
- **Feature Subsampling**: 'sqrt' reduces overfitting and makes trees more diverse, improving ensemble performance.
- **Class Weighting**: Handles any class imbalance (though our data is relatively balanced).

**Alternative Considered**: 
- n_estimators=100: Rejected - more trees improve performance with minimal cost.
- max_depth=10: Rejected - deeper trees capture more complex patterns, and overfitting is controlled by ensemble.

### 3.2 Gradient Boosting Configuration

**Decision**:
- n_estimators=200
- learning_rate=0.05 (lower than typical 0.1)
- max_depth=6
- subsample=0.8 (stochastic)

**Rationale**:
- **More Estimators**: 200 sequential learners provide better performance.
- **Lower Learning Rate**: 0.05 with 200 estimators is equivalent to 0.1 with 100, but provides more stable learning and better generalization.
- **Moderate Depth**: Depth 6 prevents overfitting while still capturing interactions. Deeper than RF because boosting is more prone to overfitting.
- **Subsampling**: 0.8 (stochastic gradient boosting) reduces overfitting and improves generalization.

**Alternative Considered**:
- learning_rate=0.1: Rejected - lower learning rate with more estimators provides better performance.
- No subsampling: Rejected - subsampling significantly improves generalization.

---

## 4. Error Margin and Uncertainty Quantification

### Why Include Error Margins?

**Decision**: Calculate and report error margins for all predictions.

**Rationale**:
- **Real-World Uncertainty**: Many factors affect game outcomes that aren't captured in statistics (injuries, rest, referee calls, clutch performance).
- **Model Uncertainty**: Cross-validation standard deviation provides a measure of model stability.
- **Prediction Confidence**: Lower confidence predictions should have larger error margins.
- **Honest Reporting**: Providing error margins prevents overconfidence in predictions.

### Error Margin Calculation Method

**Decision**: Base error = CV standard deviation; Adjusted error = Base + (1 - confidence) * 10.

**Rationale**:
- **CV Standard Deviation**: Measures model stability across different data splits - a stable model has lower error margin.
- **Confidence Adjustment**: Predictions with probability near 0.5 (low confidence) get larger error margins. Predictions with probability near 0 or 1 (high confidence) get smaller error margins.
- **Multiplicative Factor**: The factor of 10 scales the confidence adjustment appropriately.

**Alternative Considered**: 
- Fixed error margin for all predictions: Rejected - doesn't account for prediction confidence.
- Only CV standard deviation: Rejected - doesn't account for individual prediction uncertainty.

---

## 5. Data Preprocessing Decisions

### 5.1 Missing Value Handling

**Decision**: Fill missing percentages with 0, fill other missing values with median.

**Rationale**:
- **Percentages**: If a team had 0 attempts, the percentage is undefined. Setting to 0 is reasonable (no attempts = 0%).
- **Other Features**: Median is robust to outliers and preserves the distribution better than mean.

**Alternative Considered**: 
- Drop all rows with missing values: Rejected - would lose too much data.
- Forward fill: Rejected - not appropriate for game data where each game is independent.

### 5.2 Train-Test Split

**Decision**: 80-20 split with stratification.

**Rationale**:
- **80-20**: Standard split providing sufficient training data while maintaining a reasonable test set.
- **Stratification**: Ensures both sets have similar win/loss distribution, preventing bias.

**Alternative Considered**:
- 70-30 split: Rejected - 20% test set is sufficient for evaluation.
- Time-based split: Considered but rejected because we want to evaluate on diverse matchups, not just recent games.

---

## 6. Prediction Methodology for OKC Games

### 6.1 Using Actual Schedule

**Decision**: Use the provided actual schedule (dates, opponents, home/away) rather than hypothetical matchups.

**Rationale**:
- **Real-World Application**: Predictions are more useful when they match actual upcoming games.
- **Home/Away Consideration**: While not fully implemented in current version, the structure allows for future home court advantage features.
- **Specific Opponents**: Allows for opponent-specific analysis and preparation.

### 6.2 Feature Creation for Predictions

**Decision**: Use recent game averages (last 5 games) to create features for future games.

**Rationale**:
- **Current Form**: Recent performance is more predictive than season averages.
- **Momentum**: Captures team momentum and current state.
- **Adaptability**: Automatically adjusts as season progresses.

**Alternative Considered**:
- Season averages: Rejected - doesn't capture recent form changes.
- Last game only: Rejected - too noisy, single game can be an outlier.

### 6.3 Handling Insufficient Data

**Decision**: If a team doesn't have enough recent games, fall back gracefully with "INSUFFICIENT DATA" rather than making unreliable predictions.

**Rationale**:
- **Honesty**: Better to admit uncertainty than provide unreliable predictions.
- **User Trust**: Users appreciate knowing when predictions are less reliable.

**Alternative Considered**:
- Use season averages as fallback: Rejected - could be misleading if team's form has changed significantly.

---

## 7. Visualization Decisions

### 7.1 Error Bars on Predictions

**Decision**: Include error bars showing confidence ranges on prediction visualizations.

**Rationale**:
- **Transparency**: Users can see the uncertainty in predictions.
- **Decision Making**: Helps users understand which predictions are more reliable.
- **Professional Standard**: Error bars are standard in scientific/analytical visualizations.

### 7.2 Model Comparison Charts

**Decision**: Show both accuracy and cross-validation scores with error bars.

**Rationale**:
- **Comprehensive Evaluation**: Accuracy shows test set performance; CV shows generalization.
- **Uncertainty**: Error bars (CV std) show model stability.
- **Model Selection**: Helps identify which model is best and most reliable.

---

## 8. Code Structure Decisions

### 8.1 Modular Functions

**Decision**: Break code into separate functions for each major step.

**Rationale**:
- **Maintainability**: Easy to modify individual components.
- **Testability**: Each function can be tested independently.
- **Readability**: Clear flow and purpose of each step.
- **Reusability**: Functions can be reused or called in different orders.

**Alternative Considered**: 
- Monolithic script: Rejected - harder to maintain and debug.

### 8.2 Feature Column Management

**Decision**: Dynamically identify feature columns by excluding non-feature columns.

**Rationale**:
- **Flexibility**: Easy to add/remove features without hardcoding column names.
- **Maintainability**: Changes to data structure don't break feature selection.
- **Robustness**: Works even if column order changes.

**Alternative Considered**:
- Hardcoded feature list: Rejected - brittle, breaks when features change.

---

## 9. Limitations and Future Improvements

### Current Limitations

1. **Home/Away Not Fully Utilized**: While schedule includes location, home court advantage isn't explicitly modeled.
   - **Why**: Would require historical home/away performance data and more complex feature engineering.
   - **Future**: Add home/away win rates and performance differentials.

2. **Injury Data Not Included**: Player availability significantly affects outcomes.
   - **Why**: Injury data not available in current dataset.
   - **Future**: Integrate injury reports and player availability.

3. **Rest Days Not Considered**: Teams perform differently on back-to-backs or with rest.
   - **Why**: Requires schedule analysis and rest day calculation.
   - **Future**: Add days of rest feature.

4. **Head-to-Head History Limited**: Only uses recent form, not historical matchups.
   - **Why**: Current dataset may not have extensive historical data.
   - **Future**: Add head-to-head win rates and performance in recent matchups.

5. **Player-Level Features Not Directly Used**: Aggregates to team level, losing individual player impact.
   - **Why**: Simpler and more interpretable at team level.
   - **Future**: Could add star player performance features.

### Why These Limitations Are Acceptable

- **Scope**: The current model focuses on team-level statistics, which are most predictive.
- **Data Availability**: Working with available data rather than requiring additional sources.
- **Complexity vs. Benefit**: Adding these features increases complexity; the benefit may not justify it for initial implementation.
- **Incremental Improvement**: Better to have a working model that can be improved than an incomplete complex model.

---

## 10. Model Selection Rationale

### Why These Two Models?

**Random Forest**:
- **Strengths**: Handles non-linear relationships, feature interactions, robust to outliers
- **Best For**: When features have complex interactions (team stats often do)
- **Weaknesses**: Can overfit with too many trees, less interpretable than single trees

**Gradient Boosting**:
- **Strengths**: Excellent sequential learning, captures complex patterns, often highest accuracy
- **Best For**: When you want maximum predictive power
- **Weaknesses**: More prone to overfitting, slower to train, less interpretable

**Why Not Others**:
- **Bagging**: Redundant with Random Forest (which is bagging of trees)
- **Voting**: Not needed with only 2 models; better to pick the best one
- **Neural Networks**: Overkill for tabular data, less interpretable, requires more tuning
- **Logistic Regression**: Too simple, can't capture complex relationships

---

## 11. Hyperparameter Choices

### Random Forest Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| n_estimators | 200 | More trees = better performance, diminishing returns after ~200 |
| max_depth | 15 | Deep enough to capture interactions, shallow enough to prevent overfitting |
| min_samples_split | 5 | Prevents overfitting on small samples |
| min_samples_leaf | 2 | Ensures leaves have minimum data |
| max_features | 'sqrt' | Reduces overfitting, increases tree diversity |
| class_weight | 'balanced' | Handles any class imbalance |

### Gradient Boosting Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| n_estimators | 200 | More sequential learners = better performance |
| learning_rate | 0.05 | Lower rate = more stable, better generalization (compensated by more estimators) |
| max_depth | 6 | Moderate depth prevents overfitting (boosting more prone than RF) |
| subsample | 0.8 | Stochastic boosting reduces overfitting |
| min_samples_split | 5 | Same rationale as RF |

---

## 12. Evaluation Metrics

### Why Accuracy and Cross-Validation?

**Decision**: Use accuracy and 5-fold cross-validation.

**Rationale**:
- **Accuracy**: Simple, interpretable metric for binary classification. Directly answers "how often is the model correct?"
- **Cross-Validation**: Provides robust estimate of generalization performance, accounts for data variability.
- **5-Fold**: Good balance between computational cost and statistical reliability.

**Alternatives Considered**:
- **Precision/Recall**: Less relevant when both classes (win/loss) are equally important.
- **F1-Score**: Good for imbalanced data, but our data is relatively balanced.
- **ROC-AUC**: More complex, accuracy is sufficient for this use case.

---

## 13. Prediction Presentation

### Why Include Confidence Ranges?

**Decision**: Show win probability with confidence range (error margin).

**Rationale**:
- **Uncertainty Communication**: Users understand predictions aren't certain.
- **Decision Support**: Helps users weight predictions appropriately.
- **Professional Standard**: Scientific predictions should include uncertainty estimates.

**Format**: "65.2% (55.0% - 75.4%)" shows prediction with range.

---

## Conclusion

Every design decision was made to balance:
1. **Accuracy**: Maximizing prediction correctness
2. **Interpretability**: Making the model understandable
3. **Robustness**: Ensuring reliable performance
4. **Practicality**: Working within data and computational constraints
5. **Honesty**: Acknowledging uncertainty and limitations

The model prioritizes **recent form** and **relative team strength** as these are the most predictive factors for game outcomes, while acknowledging that many factors (injuries, rest, clutch performance) introduce uncertainty that is captured in error margins.

---

**Document Version**: 1.0
**Last Updated**: [Current Date]
**Author**: [Your Name]
