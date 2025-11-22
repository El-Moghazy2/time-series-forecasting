# Prophet Model Configuration and Usage

## Overview
This project uses **Darts Prophet Model** for time series forecasting. Prophet is a powerful forecasting tool developed by Facebook that works especially well with time series that have strong seasonal patterns and several seasons of historical data.

## Why Prophet?

Prophet is excellent for this task because:
1. **Handles Multiple Seasonality**: Automatically detects yearly and weekly patterns in product prices
2. **Robust to Missing Data**: Can handle gaps in time series data
3. **Future Covariates Support**: Can incorporate GDP signals as external regressors
4. **Multiplicative Seasonality**: Better for data where seasonal amplitude changes over time
5. **Automatic Trend Detection**: Identifies and models long-term trends

## Model Configuration in forecast_model.py

```python
from darts.models import Prophet

model = Prophet(
    yearly_seasonality=True,      # Captures annual patterns (52 weeks)
    weekly_seasonality=True,       # Captures weekly patterns
    daily_seasonality=False,       # Not needed for weekly data
    seasonality_mode='multiplicative',  # Better for changing amplitudes
    add_encoders={
        'cyclic': {'future': ['month']},  # Add cyclical month encoding
        'datetime_attribute': {'future': ['quarter']}  # Add quarter information
    }
)
```

## Key Features

### 1. Future Covariates (GDP Signals)
The model uses GDP data from 3 countries as future covariates:
- USA GDP
- China GDP  
- EU GDP
- GDP Total (sum)
- Year-over-year GDP growth rates

These covariates help the model understand how macroeconomic factors influence product prices.

### 2. Training Configuration
```python
model.fit(
    series=train_target,              # Historical product prices
    future_covariates=train_covariates  # GDP signals (available for future)
)
```

### 3. Prediction with Covariates
```python
predictions = model.predict(
    n=104,                             # 104 weeks ahead
    future_covariates=train_covariates  # Use GDP projections
)
```

## Data Splits

- **Training Period**: 2010-01-01 to 2024-12-31 (~783 weeks)
  - Used to train the Prophet model
  
- **Test Period**: 2025-01-01 to 2026-12-31 (~104 weeks)
  - Used to evaluate model accuracy
  
- **Future Covariates**: Available through 2027-12-31
  - GDP projections used for forecasting

## Prophet's Internal Components

Prophet decomposes the time series into:

1. **Trend Component** (g(t))
   - Models long-term growth patterns
   - Can be linear or logistic

2. **Seasonal Component** (s(t))
   - Yearly seasonality (52-week cycle)
   - Weekly seasonality (7-day cycle for weekly data)

3. **Holiday Component** (h(t))
   - Not used in our model (no holidays specified)

4. **Regressors** (β·x(t))
   - Our GDP covariates
   - Weighted influence on predictions

**Formula**: y(t) = g(t) + s(t) + h(t) + β·x(t) + ε(t)

## Running the Model

### Option 1: Direct Execution
```bash
python forecast_model.py
```

### Option 2: Using the Pipeline
```bash
python run_pipeline.py
```

### Option 3: Step by Step
```bash
# Step 1: Generate data
python generate_data.py

# Step 2: Run Prophet forecasting
python forecast_model.py
```

## Expected Runtime

Prophet model training typically takes:
- **Small datasets (<1000 observations)**: 2-5 minutes
- **Medium datasets (1000-5000)**: 5-15 minutes
- **Large datasets (>5000)**: 15+ minutes

Our dataset has ~940 weekly observations, so expect 3-7 minutes of training time.

## Output Files

After running `forecast_model.py`:

1. **predictions.csv**
   - Date and predicted price for each of 104 weeks
   - Format: `date, predicted_price`

2. **forecast_results.png**
   - Two-panel visualization:
     - Panel 1: Full time series view
     - Panel 2: Zoomed view on test period

## Evaluation Metrics

The model reports three key metrics:

1. **MAPE (Mean Absolute Percentage Error)**
   - Average percentage error
   - Lower is better
   - Easy to interpret (e.g., 5% means predictions are off by 5% on average)

2. **RMSE (Root Mean Squared Error)**
   - Penalizes large errors more heavily
   - In same units as target variable (dollars)

3. **MAE (Mean Absolute Error)**
   - Average absolute error
   - In same units as target variable

## Troubleshooting

### If Prophet takes too long:
- Use `simple_forecast.py` with Exponential Smoothing instead
- Reduce data size in `generate_data.py`

### If you get memory errors:
- The model might be too complex for available RAM
- Try reducing the number of covariates
- Use a simpler seasonality configuration

### If predictions seem off:
- Check data quality with `test_data.py`
- Verify covariates are properly aligned
- Try adjusting `seasonality_mode` to 'additive'

## Model Advantages

✓ Interpretable components (trend, seasonality, covariates)
✓ Robust to outliers and missing data
✓ Automatic hyperparameter selection
✓ Works well with multiple seasonalities
✓ Handles future covariates naturally

## Next Steps

After running the model:
1. Review `forecast_results.png` for visual assessment
2. Check evaluation metrics in console output
3. Analyze `predictions.csv` for specific predictions
4. Compare with simple baseline models
5. Tune hyperparameters if needed

## Advanced Tuning Options

You can modify these Prophet parameters in `forecast_model.py`:

```python
model = Prophet(
    yearly_seasonality=10,           # Fourier order (default: 10)
    weekly_seasonality=3,            # Fourier order (default: 3)
    changepoint_prior_scale=0.05,   # Trend flexibility (default: 0.05)
    seasonality_prior_scale=10.0,   # Seasonality strength (default: 10.0)
    seasonality_mode='multiplicative',  # or 'additive'
)
```

## Resources

- Darts Documentation: https://unit8co.github.io/darts/
- Prophet Paper: https://peerj.com/preprints/3190/
- Prophet Documentation: https://facebook.github.io/prophet/

---

**Note**: The model is currently running. Please wait for it to complete to see the results!
