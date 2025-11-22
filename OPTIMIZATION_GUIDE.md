# Hyperparameter Optimization Guide with Optuna + Backtesting

## Overview

This guide explains how to use Optuna to automatically find the best hyperparameters for your Prophet model using backtesting validation.

---

## 🎯 What is Hyperparameter Optimization?

**Hyperparameters** are configuration settings that control how the model learns:
- Seasonality settings
- Prior scales (how flexible the model is)
- Changepoint detection settings

**Optimization** finds the best combination of these settings automatically, instead of manual trial-and-error.

---

## 🔄 What is Backtesting?

**Backtesting** validates the model by:
1. Training on historical data
2. Predicting the next period
3. Comparing predictions to actual values
4. Repeating this process multiple times (folds)

This gives a realistic estimate of model performance on unseen data.

### Example Backtesting with 3 Folds:

```
Full Training Data: [2010 ──────────────────────────────── 2024]

Fold 1: Train[2010─────2020] → Predict[2020-2021] → Evaluate
Fold 2: Train[2010──────────2021] → Predict[2021-2022] → Evaluate  
Fold 3: Train[2010─────────────────2022] → Predict[2022-2023] → Evaluate

Final Score: Average of all 3 evaluations
```

---

## 📊 Hyperparameters Being Optimized

### 1. **Seasonality Settings**

**yearly_seasonality** (True/False)
- Captures 52-week patterns (annual cycles)
- Important for products with seasonal demand

**weekly_seasonality** (True/False)
- Captures 7-day patterns
- Less important for weekly data but can help

**seasonality_mode** ('additive' or 'multiplicative')
- **Additive**: Seasonal effects are constant over time
- **Multiplicative**: Seasonal effects grow with trend
- Usually multiplicative is better for price data

### 2. **Flexibility Parameters**

**changepoint_prior_scale** (0.001 to 0.5)
- Controls trend flexibility
- **Lower** (0.001): Rigid, smooth trends
- **Higher** (0.5): Flexible, can change quickly
- Default: 0.05

**seasonality_prior_scale** (0.01 to 10.0)
- Controls seasonal pattern strength
- **Lower**: Weak seasonality
- **Higher**: Strong seasonality
- Default: 10.0

**holidays_prior_scale** (0.01 to 10.0)
- Controls impact of holiday/event effects
- Default: 10.0

**changepoint_range** (0.8 to 0.95)
- What portion of data to use for detecting trend changes
- 0.8 = only first 80% of data
- 0.95 = first 95% of data

---

## 🚀 How to Run Optimization

### Basic Usage:

```bash
python optimize_prophet.py
```

This will:
1. ✅ Load your data
2. ✅ Run 30 optimization trials with 3 backtesting folds each
3. ✅ Find the best hyperparameters
4. ✅ Generate visualization reports
5. ✅ Save best parameters to `best_params.json`

### Expected Runtime:
- **30 trials × 3 folds** = 90 model trainings
- Approximately **15-30 minutes** depending on your hardware

---

## 📁 Output Files

After running optimization, you'll get:

### 1. `best_params.json`
Contains the optimal hyperparameters:
```json
{
  "best_params": {
    "yearly_seasonality": true,
    "weekly_seasonality": true,
    "seasonality_mode": "multiplicative",
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0,
    "holidays_prior_scale": 5.0,
    "changepoint_range": 0.85
  },
  "best_mape": 1.15,
  "n_trials": 30,
  "n_folds": 3
}
```

### 2. `optimization_results/` Directory

**optimization_history.html** (Interactive)
- Shows MAPE improvement over trials
- Identifies when best parameters were found

**param_importances.html** (Interactive)
- Shows which parameters matter most
- Helps understand what drives performance

**parallel_coordinate.html** (Interactive)
- Visualizes parameter combinations
- Shows relationships between parameters

**slice_plot.html** (Interactive)
- Shows how each parameter affects MAPE
- Helps understand parameter sensitivity

**optimization_summary.png** (Static)
- Summary of optimization results
- Parameter distributions and comparisons

---

## 🔧 Customization

### Adjust Number of Trials

Edit `optimize_prophet.py`:

```python
# More trials = better optimization but slower
N_TRIALS = 50  # Default: 30

# More folds = more robust evaluation but slower
N_FOLDS = 5    # Default: 3
```

### Add More Hyperparameters

In the `_objective` method, add:

```python
params = {
    # ... existing parameters ...
    'uncertainty_samples': trial.suggest_int('uncertainty_samples', 0, 1000),
    # Add any other Prophet parameters
}
```

### Change Optimization Metric

Currently optimizing MAPE. To use RMSE instead:

```python
# In _backtest_model method, replace:
fold_mape = mape(test_fold, pred)
scores.append(fold_mape)

# With:
fold_rmse = rmse(test_fold, pred)
scores.append(fold_rmse)
```

---

## 📈 Using Optimized Parameters

### Method 1: Update forecast_model.py Manually

Copy parameters from `best_params.json` to `forecast_model.py`:

```python
model = Prophet(
    yearly_seasonality=True,  # From best_params.json
    weekly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=5.0,
    changepoint_range=0.85
)
```

### Method 2: Load Automatically

Add to `forecast_model.py`:

```python
import json

# Load optimized parameters
with open('best_params.json', 'r') as f:
    results = json.load(f)
    best_params = results['best_params']

# Create model with optimized parameters
model = Prophet(**best_params, daily_seasonality=False)
```

---

## 🎓 Understanding Optuna

### Tree-structured Parzen Estimator (TPE)

Optuna uses TPE sampler which:
1. **Tries random combinations first** (exploration)
2. **Learns which regions are promising** (exploitation)
3. **Focuses search on good areas**
4. **Finds optimal parameters faster than grid search**

### Pruning (Optional)

Add early stopping for bad trials:

```python
self.study = optuna.create_study(
    direction='minimize',
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()  # Stop bad trials early
)
```

---

## 📊 Interpreting Results

### Good MAPE Scores:
- **< 5%**: Excellent
- **5-10%**: Very good
- **10-20%**: Good
- **> 20%**: Needs improvement

### Parameter Importances:
If optimization shows `changepoint_prior_scale` is most important:
→ Your data has significant trend changes
→ Focus on tuning this parameter more finely

If `seasonality_mode` is most important:
→ Choosing between additive/multiplicative matters a lot
→ Your seasonal patterns vary in amplitude

---

## 🔍 Debugging Tips

### If optimization is too slow:
```python
N_TRIALS = 10  # Reduce trials
N_FOLDS = 2    # Reduce folds
```

### If getting errors:
1. Check data exists: `data.csv` must be present
2. Verify data format: Date column and all GDP columns
3. Check memory: Close other applications

### If results are unstable:
- Increase `N_FOLDS` for more robust evaluation
- Increase `N_TRIALS` to explore more combinations
- Check if data has outliers or anomalies

---

## 🎯 Best Practices

### ✅ DO:

1. **Run optimization before final deployment**
   - Takes time but significantly improves accuracy

2. **Use backtesting for evaluation**
   - More realistic than simple train/test split
   - Reduces overfitting risk

3. **Check parameter importances**
   - Focus tuning efforts on important parameters
   - Ignore parameters with low importance

4. **Save and version optimization results**
   - Keep track of what hyperparameters work
   - Compare different optimization runs

5. **Re-run periodically**
   - As data changes, optimal parameters may change
   - Re-optimize every 6-12 months

### ❌ DON'T:

1. **Don't use test data in optimization**
   - Only use training data (2010-2024)
   - Keep test data (2025-2026) for final evaluation

2. **Don't run too few trials**
   - Minimum 20 trials for meaningful results
   - 50+ trials for production models

3. **Don't ignore warnings**
   - inf values mean model failed
   - Check data quality if many trials fail

4. **Don't over-optimize**
   - Balance optimization time vs. improvement
   - Diminishing returns after 50-100 trials

---

## 🔄 Complete Workflow

### Step 1: Generate Data
```bash
python generate_data.py
```

### Step 2: Run Optimization (NEW!)
```bash
python optimize_prophet.py
```
⏱️ Wait 15-30 minutes

### Step 3: Review Results
- Open HTML files in `optimization_results/`
- Check `best_params.json`

### Step 4: Update Model
- Copy best parameters to `forecast_model.py`
- OR load them automatically

### Step 5: Train Final Model
```bash
python forecast_model.py
```

### Step 6: Compare Results
- Compare MAPE before and after optimization
- Should see improvement!

---

## 📊 Example Results

### Before Optimization (Default Parameters):
```
MAPE: 1.28%
RMSE: 2.61
MAE: 2.14
```

### After Optimization (Expected):
```
MAPE: 1.10-1.20%  (10-15% improvement)
RMSE: 2.30-2.50
MAE: 1.90-2.10
```

---

## 💡 Advanced Topics

### Cross-Validation Strategies

**Expanding Window** (Current):
```python
Fold 1: [────────Train────────][Test]
Fold 2: [─────────Train─────────][Test]
Fold 3: [────────────Train────────────][Test]
```

**Sliding Window** (Alternative):
```python
Fold 1: [────Train────][Test]
Fold 2:     [────Train────][Test]
Fold 3:         [────Train────][Test]
```

### Multi-Objective Optimization

Optimize for both MAPE and speed:

```python
def _objective(self, trial):
    import time
    start = time.time()
    
    mape_score = self._backtest_model(params)
    training_time = time.time() - start
    
    # Combine objectives
    return mape_score + 0.01 * training_time  # Penalize slow models
```

### Hyperparameter Ranges

Based on Prophet documentation:

| Parameter | Range | Default | Tuning Impact |
|-----------|-------|---------|---------------|
| changepoint_prior_scale | 0.001-0.5 | 0.05 | **High** |
| seasonality_prior_scale | 0.01-10.0 | 10.0 | **Medium** |
| holidays_prior_scale | 0.01-10.0 | 10.0 | **Low** |
| changepoint_range | 0.8-0.95 | 0.8 | **Medium** |

---

## 🎉 Summary

You now have:
- ✅ Automatic hyperparameter optimization
- ✅ Backtesting-based evaluation
- ✅ Interactive visualizations
- ✅ Best parameters saved automatically
- ✅ Production-ready optimization pipeline

This will significantly improve your Prophet model's accuracy! 🚀
