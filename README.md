# Time Series Forecasting with Darts Prophet

## Project Overview
This project demonstrates time series forecasting using the Darts Prophet model to predict product prices 104 weeks ahead. The model uses GDP signals from 3 countries (USA, China, EU) as future covariates to improve prediction accuracy.

## Data Description
The dataset contains weekly observations from 2010 to 2027:
- **Target Variable**: Product price (weekly)
- **Future Covariates**: 
  - GDP for USA, China, and EU
  - GDP total (sum of all three)
  - Year-over-year GDP change rates for each country

## Files

### Main Scripts
- `generate_data.py` - Script to generate synthetic time series data
- `forecast_model.py` - Main forecasting script using Darts Prophet
- `optimize_prophet.py` - **NEW!** Hyperparameter optimization with Optuna + Backtesting
- `test_optimization.py` - Quick test for optimization pipeline
- `run_pipeline.py` - Run complete forecasting pipeline

### Data & Results
- `data.csv` - Generated dataset (created by running generate_data.py)
- `predictions.csv` - Model predictions (created by running forecast_model.py)
- `forecast_results.png` - Visualization of forecasting results
- `best_params.json` - **NEW!** Optimized hyperparameters from Optuna
- `optimization_results/` - **NEW!** Optimization visualization reports

### Documentation
- `README.md` - This file
- `PROPHET_GUIDE.md` - Detailed Prophet model guide
- `COVARIATES_GUIDE.md` - Complete guide to covariates
- `OPTIMIZATION_GUIDE.md` - **NEW!** Hyperparameter optimization guide
- `PROJECT_SUMMARY.md` - Project completion summary
- `requirements.txt` - Python package dependencies

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install darts pandas numpy matplotlib prophet scikit-learn
```

## Usage

### Step 1: Generate Mock Data
```bash
python generate_data.py
```
This creates `data.csv` with synthetic time series data from 2010 to 2027.

### Step 2: Optimize Hyperparameters (NEW! Recommended)
```bash
python optimize_prophet.py
```
This will:
- Use Optuna to find optimal Prophet hyperparameters
- Perform backtesting with multiple folds
- Generate optimization visualizations
- Save best parameters to `best_params.json`
- Takes 15-30 minutes

**Quick test** (5 trials, 2 minutes):
```bash
python test_optimization.py
```

### Step 3: Run Forecasting Model
```bash
python forecast_model.py
```
This will:
- Load and prepare the data
- Train a Prophet model with GDP covariates (use optimized params!)
- Generate 104-week forecasts
- Evaluate model performance
- Create visualizations
- Save predictions to `predictions.csv`

## Model Architecture

### Prophet Model
The model uses Facebook's Prophet algorithm (via Darts) with the following features:
- **Yearly seasonality**: Captures annual patterns
- **Weekly seasonality**: Captures weekly patterns
- **Multiplicative seasonality**: Better for data with changing seasonal amplitude
- **Future covariates**: GDP signals from 3 countries
- **Additional encoders**: Month (cyclic) and quarter attributes

### Data Splits
- **Training**: 2010-01-01 to 2024-12-31 (~780 weeks)
- **Test**: 2025-01-01 to 2026-12-31 (~104 weeks)
- **Future covariates**: Available through 2027-12-31

### Prediction Horizon
- **104 weeks** (approximately 2 years)

## Evaluation Metrics
The model is evaluated using:
- **MAPE** (Mean Absolute Percentage Error): Percentage-based error metric
- **RMSE** (Root Mean Squared Error): Penalizes large errors more heavily
- **MAE** (Mean Absolute Error): Average absolute difference

## Results
Results are saved in two formats:
1. **predictions.csv**: Contains date and predicted price for each week
2. **forecast_results.png**: Two-panel visualization showing:
   - Full view of training, test, and predictions
   - Detailed view of test period and forecast

## Key Features
- Realistic synthetic data with trends, seasonality, and noise
- GDP signals as external regressors (future covariates)
- **Automated hyperparameter optimization with Optuna** ⭐ NEW!
- **Backtesting-based model validation** ⭐ NEW!
- Comprehensive model evaluation
- Interactive optimization visualizations
- Professional visualizations
- Easy to extend and customize

## Requirements
- Python 3.8+
- darts >= 0.27.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- prophet >= 1.1.5
- scikit-learn >= 1.3.0
- optuna >= 3.0.0 (for hyperparameter optimization)
- plotly >= 5.0.0 (for interactive visualizations)

## Notes
- The data is synthetic and generated for demonstration purposes
- GDP data in the model represents macroeconomic indicators that might influence product prices
- The model assumes GDP data is available for future periods (common in forecasting scenarios with projected economic indicators)
- Adjust model hyperparameters in `forecast_model.py` for different behavior

## Customization
You can customize the model by modifying parameters in `forecast_model.py`:
- Change seasonality mode (`'additive'` or `'multiplicative'`)
- Adjust seasonality components (yearly, weekly, daily)
- Modify the prediction horizon (change `horizon` variable)
- Add more covariates or features
- Experiment with different train/test splits

## Optimization Features ⭐ NEW!

### Hyperparameter Tuning with Optuna
- Automatically finds best Prophet configuration
- Uses Tree-structured Parzen Estimator (TPE) for efficient search
- Optimizes 7+ hyperparameters simultaneously
- Generates interactive visualization reports

### Backtesting Validation
- Tests model on multiple historical periods
- Expanding window cross-validation
- Realistic performance estimation
- Prevents overfitting

### Optimization Outputs
- `best_params.json` - Optimal hyperparameters
- Interactive HTML reports (optimization history, parameter importance)
- Static PNG summaries
- Automatic model training with best parameters

See `OPTIMIZATION_GUIDE.md` for detailed documentation.

## Future Improvements
- ✅ ~~Add hyperparameter tuning~~ **DONE!**
- ✅ ~~Implement cross-validation/backtesting~~ **DONE!**
- Try other Darts models (ARIMA, LSTM, Transformer, etc.)
- Add confidence intervals
- Add more external regressors
- Implement ensemble methods

## License
This project is for educational and demonstration purposes.
