# 🎯 Optuna Hyperparameter Optimization - Quick Reference

## What Was Added

### ✅ New Components

1. **`optimize_prophet.py`** - Full optimization script
   - Optuna integration for hyperparameter search
   - Backtesting with expanding windows
   - Automatic visualization generation
   - Best parameters saved to JSON

2. **`test_optimization.py`** - Quick test script
   - 5 trials for fast verification
   - Tests the optimization pipeline
   - Validates setup before full run

3. **`OPTIMIZATION_GUIDE.md`** - Complete documentation
   - How to use Optuna
   - Understanding backtesting
   - Interpreting results
   - Best practices

4. **Updated `requirements.txt`**
   - Added `optuna>=3.0.0`
   - Added `plotly>=5.0.0`

---

## 🚀 Quick Start

### 1. Test the Setup (2 minutes)
```bash
python test_optimization.py
```

### 2. Run Full Optimization (15-30 minutes)
```bash
python optimize_prophet.py
```

### 3. Check Results
- Open `best_params.json` for optimal hyperparameters
- View HTML files in `optimization_results/` for visualizations

### 4. Use Optimized Parameters
Update `forecast_model.py` with parameters from `best_params.json`

---

## 📊 What Gets Optimized

| Hyperparameter | Range | Default | Impact |
|----------------|-------|---------|--------|
| **yearly_seasonality** | True/False | True | Medium |
| **weekly_seasonality** | True/False | True | Low |
| **seasonality_mode** | additive/multiplicative | multiplicative | **High** |
| **changepoint_prior_scale** | 0.001-0.5 | 0.05 | **High** |
| **seasonality_prior_scale** | 0.01-10.0 | 10.0 | Medium |
| **holidays_prior_scale** | 0.01-10.0 | 10.0 | Low |
| **changepoint_range** | 0.8-0.95 | 0.8 | Medium |

---

## 🔄 Backtesting Explained

Instead of a single train/test split, backtesting uses multiple splits:

```
Training Data: [2010 ──────────────────── 2024]

Fold 1: [2010─────────2020] → Test[2020-2021] → MAPE = 1.5%
Fold 2: [2010────────────2021] → Test[2021-2022] → MAPE = 1.3%
Fold 3: [2010───────────────────2022] → Test[2022-2023] → MAPE = 1.4%

Average MAPE: 1.4%  ← This is what Optuna optimizes
```

**Benefits:**
- More robust evaluation
- Reduces overfitting
- Tests model on multiple time periods
- Realistic performance estimate

---

## 📁 Output Files

After running `optimize_prophet.py`:

```
project/
├── best_params.json                    # ⭐ Optimal hyperparameters
└── optimization_results/
    ├── optimization_history.html       # Interactive: MAPE over trials
    ├── param_importances.html          # Interactive: Which params matter most
    ├── parallel_coordinate.html        # Interactive: Parameter relationships
    ├── slice_plot.html                 # Interactive: Parameter sensitivity
    └── optimization_summary.png        # Static: Summary visualization
```

---

## 💡 Using Optimized Parameters

### Option 1: Manual Update

Copy from `best_params.json` to `forecast_model.py`:

```python
model = Prophet(
    yearly_seasonality=True,              # From best_params.json
    weekly_seasonality=True,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.0234,      # Optimized value
    seasonality_prior_scale=8.45,        # Optimized value
    holidays_prior_scale=3.21,           # Optimized value
    changepoint_range=0.87               # Optimized value
)
```

### Option 2: Automatic Load

Add to `forecast_model.py`:

```python
import json

# Load optimized parameters
try:
    with open('best_params.json', 'r') as f:
        results = json.load(f)
        best_params = results['best_params']
    
    model = Prophet(**best_params, daily_seasonality=False)
    print("✓ Using optimized hyperparameters!")
except FileNotFoundError:
    # Fallback to defaults
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    print("⚠ Using default hyperparameters")
```

---

## 🎓 How Optuna Works

### Tree-structured Parzen Estimator (TPE)

1. **Exploration Phase** (First ~10 trials)
   - Tries random parameter combinations
   - Explores the search space

2. **Exploitation Phase** (Later trials)
   - Focuses on promising regions
   - Uses Bayesian optimization
   - Learns from previous trials

3. **Result**
   - Finds optimal parameters faster than grid search
   - Typically needs 30-100 trials

---

## 📈 Expected Improvements

### Before Optimization (Default Parameters)
```
MAPE: 1.28%
RMSE: 2.61
MAE: 2.14
```

### After Optimization (Typical Results)
```
MAPE: 1.10-1.20%  ← 10-15% improvement
RMSE: 2.30-2.50
MAE: 1.90-2.10
```

**Note:** Improvement depends on:
- Data characteristics
- Number of trials
- Initial parameter quality

---

## ⚙️ Customization

### Increase Trials (Better Results, Slower)

Edit `optimize_prophet.py`:
```python
N_TRIALS = 50  # Default: 30
N_FOLDS = 5    # Default: 3
```

### Change Optimization Metric

Currently optimizes MAPE. To use RMSE:

```python
# In _backtest_model method
from darts.metrics import rmse

fold_score = rmse(test_fold, pred)  # Instead of mape
```

### Add More Hyperparameters

```python
def _objective(self, trial):
    params = {
        # ... existing parameters ...
        'uncertainty_samples': trial.suggest_int('uncertainty_samples', 0, 1000),
        'mcmc_samples': trial.suggest_int('mcmc_samples', 0, 500),
    }
```

---

## 🔍 Troubleshooting

### Issue: Optimization is slow
**Solution:** Reduce trials/folds
```python
N_TRIALS = 10  # Quick test
N_FOLDS = 2
```

### Issue: Getting "inf" values
**Solution:** Check data quality
- Ensure data.csv exists
- Verify no missing values
- Check date range covers 2010-2024

### Issue: All trials have similar scores
**Solution:** Widen parameter ranges or check if default parameters are already optimal

### Issue: Out of memory
**Solution:** 
- Close other applications
- Reduce number of folds
- Process smaller data subset

---

## 🎯 Best Practices

### ✅ DO:

1. **Run optimization before production deployment**
2. **Use backtesting for evaluation** (not simple train/test)
3. **Check parameter importances** (focus on what matters)
4. **Save optimization results** (version control)
5. **Re-run periodically** (every 6-12 months)

### ❌ DON'T:

1. **Don't use test data in optimization** (use only 2010-2024)
2. **Don't run too few trials** (minimum 20, prefer 50+)
3. **Don't ignore warnings** (inf values mean issues)
4. **Don't over-optimize** (diminishing returns after 100 trials)

---

## 🔄 Complete Workflow

```bash
# 1. Generate data (if not done)
python generate_data.py

# 2. Quick test (optional, 2 minutes)
python test_optimization.py

# 3. Full optimization (15-30 minutes) ⭐
python optimize_prophet.py

# 4. Review results
#    - Check best_params.json
#    - Open HTML files in optimization_results/

# 5. Update forecast model with best parameters
#    Edit forecast_model.py or use automatic loading

# 6. Run final forecast
python forecast_model.py

# 7. Compare results
#    Before optimization: MAPE = 1.28%
#    After optimization:  MAPE = ~1.15% (expected)
```

---

## 📊 Visualization Guide

### optimization_history.html
- **What:** Shows MAPE improvement over trials
- **Look for:** Downward trend, stable at the end
- **Good sign:** Clear improvement in first 20 trials

### param_importances.html
- **What:** Shows which parameters affect MAPE most
- **Look for:** Parameters with high importance
- **Action:** Focus tuning on top 2-3 parameters

### parallel_coordinate.html
- **What:** Shows parameter combinations and their MAPE
- **Look for:** Patterns in best-performing trials
- **Insight:** Which parameter combinations work together

### slice_plot.html
- **What:** Shows how each parameter individually affects MAPE
- **Look for:** Clear trends or optimal regions
- **Insight:** Parameter sensitivity

---

## 💻 Code Example: Using Optimizer Programmatically

```python
from optimize_prophet import ProphetOptimizer

# Create optimizer
optimizer = ProphetOptimizer(
    data_path='data.csv',
    n_trials=30,
    n_folds=3
)

# Run optimization
best_params, best_score = optimizer.optimize()

# Visualize
optimizer.visualize_optimization()

# Save results
optimizer.save_best_params('my_best_params.json')

# Train final model
best_model = optimizer.train_best_model()

# Make predictions
predictions = best_model.predict(n=104, future_covariates=covariates)
```

---

## 🎉 Summary

You now have:

✅ **Automated hyperparameter optimization**
- Finds best Prophet configuration automatically
- Uses Optuna's intelligent search algorithm

✅ **Backtesting validation**
- Tests on multiple historical periods
- Realistic performance estimates

✅ **Professional visualizations**
- Interactive HTML reports
- Parameter importance analysis

✅ **Production-ready pipeline**
- Easy to run and customize
- Results saved automatically

---

## 📚 Additional Resources

- **Optuna Documentation**: https://optuna.readthedocs.io/
- **Darts Backtesting**: https://unit8co.github.io/darts/generated_api/darts.models.forecasting.forecasting_model.html#darts.models.forecasting.forecasting_model.ForecastingModel.backtest
- **Prophet Parameters**: https://facebook.github.io/prophet/docs/diagnostics.html

---

**Ready to optimize? Run:** `python optimize_prophet.py` 🚀
