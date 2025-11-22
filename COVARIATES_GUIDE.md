# Complete Guide to Covariates in Time Series Forecasting

## Overview

Covariates (also called exogenous variables or regressors) are additional features that can improve forecasting accuracy by providing extra information beyond the historical target values.

---

## 📊 Three Types of Covariates

### 1. Future Covariates 🔮

**Definition:** Variables available for both **past AND future** time periods.

#### ✅ Your Current Project Uses These!

**What You're Using:**
- GDP for USA, China, EU
- GDP total
- GDP year-over-year growth rates

**Why They Work:**
- Economic forecasts are published by institutions
- GDP projections available years in advance
- Help model understand macroeconomic influences

#### Other Examples:

**Calendar & Time Features:**
```python
- Day of week (0-6)
- Month of year (1-12)
- Quarter (1-4)
- Week of year (1-52)
- Is weekend (0/1)
- Is holiday (0/1)
- Days to next holiday
- Business days in month
- Season indicators
```

**Economic Indicators:**
```python
- Interest rate forecasts
- Inflation projections
- Exchange rate forecasts
- Stock market indices
- Commodity price forecasts
- Consumer confidence forecasts
- Central bank policy rates
```

**Planned Events:**
```python
- Marketing campaigns (scheduled)
- Product launches (planned)
- Promotional periods (known)
- Store openings/closings (planned)
- Price changes (scheduled)
- Contract renewals (known dates)
```

**Weather Forecasts:**
```python
- Temperature forecast
- Precipitation forecast
- Humidity forecast
- Wind speed forecast
```

**Demographic Projections:**
```python
- Population growth
- Age distribution forecasts
- Migration patterns
- Urbanization rates
```

#### When to Use:
- ✅ Feature is known or can be forecasted for the future
- ✅ Feature is deterministic (like calendar)
- ✅ Reliable forecasts exist (like GDP projections)

#### Supported Models:
- ✅ **Prophet** (your model!)
- ✅ ARIMA, AutoARIMA
- ✅ RNN, LSTM, GRU
- ✅ Transformer, TFT
- ✅ Regression models
- ✅ Tree-based models (XGBoost, LightGBM, CatBoost)

---

### 2. Past Covariates 📍

**Definition:** Variables only available **historically** (not in future).

#### Examples:

**Historical Metrics:**
```python
- Past sales of related products
- Historical website traffic
- Actual weather (not forecasted)
- Past inventory levels
- Historical stock prices
- Yesterday's order count
- Previous day's foot traffic
```

**Lagged Target Features:**
```python
- Price 1 week ago
- Price 4 weeks ago
- Moving average of past prices
- Exponential smoothing of past values
```

**Real-time Measurements:**
```python
- Actual temperature readings
- Actual rainfall
- Real sensor data
- Actual energy consumption
```

**Social Media Metrics:**
```python
- Daily tweet counts (historical)
- Past sentiment scores
- Historical engagement rates
- Previous day's viral trends
```

**Competitor Data:**
```python
- Competitor prices (observed)
- Competitor sales (historical)
- Market share (past periods)
```

#### When to Use:
- ✅ Feature cannot be predicted accurately
- ✅ Real-time or observed data only
- ✅ Historical relationship with target exists

#### Supported Models:
- ✅ ARIMA, AutoARIMA
- ✅ RNN, LSTM, GRU
- ✅ TCN
- ✅ Transformer models
- ✅ Regression models
- ✅ Tree-based models
- ❌ Prophet (does not support past covariates)

---

### 3. Static Covariates 🏷️

**Definition:** Time-invariant features that **never change**.

#### Examples:

**Product Characteristics:**
```python
- Product category
- Product type
- Brand
- Package size
- Color
- Material
- Weight class
```

**Location Features:**
```python
- Store ID
- Region
- City
- Climate zone
- Store size category
- Store type (urban/suburban/rural)
```

**Customer Segments:**
```python
- Customer segment ID
- VIP status
- Account type
- Subscription tier
```

**Entity Metadata:**
```python
- Series ID
- Country
- Currency
- Time zone
- Industry sector
```

#### When to Use:
- ✅ Building models for multiple series
- ✅ Features describe the series itself
- ✅ Using neural network models
- ✅ Want model to learn series-specific patterns

#### Supported Models:
- ✅ Neural network models (RNN, LSTM, GRU)
- ✅ Transformer models
- ✅ TFT (Temporal Fusion Transformer)
- ✅ N-BEATS variants
- ✅ TiDE
- ❌ Statistical models (ARIMA, Prophet)
- ❌ Traditional regression models

---

## 🎯 Examples for Product Price Forecasting

### Recommended Future Covariates:

```python
# Economic (what you're using ✅)
- GDP (USA, China, EU)
- Inflation rate
- Interest rates
- Exchange rates
- Consumer confidence index
- Unemployment rate

# Calendar (easy to add)
- Month (1-12)
- Quarter (1-4)
- Week of year
- Is holiday season
- Is end of quarter

# Cyclical encoding
- sin(2π × month / 12)
- cos(2π × month / 12)

# Market conditions (if available)
- Commodity prices (oil, metals)
- Stock market indices
- Industry indices
```

### Potential Past Covariates:

```python
# Historical metrics
- Competitor prices (observed)
- Past sales volume
- Previous inventory levels
- Historical demand

# Derived features
- Price 1 week ago
- 4-week moving average
- Volatility measures
```

### Static Covariates (for multi-product models):

```python
# Product features
- Product category
- Product tier (premium/standard/budget)
- Target market
- Sales channel
```

---

## 💻 Implementation Examples

### Adding Future Covariates (Extend Your Current Model)

```python
import pandas as pd
from darts import TimeSeries
from darts.models import Prophet

# Load your data
df = pd.read_csv('data.csv')
df['date'] = pd.to_datetime(df['date'])

# Add calendar features
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['is_holiday_season'] = ((df['date'].dt.month == 11) | 
                           (df['date'].dt.month == 12)).astype(int)

# Cyclical encoding
import numpy as np
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Create TimeSeries with extended covariates
target = TimeSeries.from_dataframe(df, 'date', 'product_price', freq='W-FRI')

future_covs = TimeSeries.from_dataframe(
    df, 'date',
    value_cols=['gdp_usa', 'gdp_china', 'gdp_eu',  # Original
                'month', 'quarter',                  # New
                'month_sin', 'month_cos',           # New
                'is_holiday_season'],               # New
    freq='W-FRI'
)

# Train Prophet
model = Prophet()
model.fit(target, future_covariates=future_covs)

# Predict
predictions = model.predict(n=104, future_covariates=future_covs)
```

### Using Past Covariates (Different Model)

```python
from darts.models import RNNModel

# Create past covariates (only historical data)
past_covs = TimeSeries.from_dataframe(
    df_historical, 'date',
    value_cols=['competitor_price', 'daily_sales', 'inventory'],
    freq='W-FRI'
)

# Train RNN model
model = RNNModel(input_chunk_length=52, output_chunk_length=104)
model.fit(target, past_covariates=past_covs)

# Predict (past_covariates only needed up to forecast start)
predictions = model.predict(n=104, past_covariates=past_covs)
```

### Using Static Covariates (Multi-series)

```python
from darts.models import TFTModel

# Create multiple series with static covariates
products_data = []
for product in ['A', 'B', 'C']:
    product_df = df[df['product_id'] == product]
    
    ts = TimeSeries.from_dataframe(
        product_df, 'date', 'price',
        static_covariates=pd.DataFrame({
            'category': [product_category[product]],
            'region': [product_region[product]]
        })
    )
    products_data.append(ts)

# Train on multiple series
model = TFTModel(input_chunk_length=52, output_chunk_length=104)
model.fit(products_data)
```

---

## 📋 Covariate Support Matrix

| Model | Past Covariates | Future Covariates | Static Covariates |
|-------|----------------|-------------------|-------------------|
| **Prophet** ⭐ | ❌ | ✅ | ❌ |
| ARIMA | ✅ | ✅ | ❌ |
| AutoARIMA | ✅ | ✅ | ❌ |
| ExponentialSmoothing | ❌ | ❌ | ❌ |
| RNN/LSTM/GRU | ✅ | ✅ | ✅ |
| TCN | ✅ | ✅ | ✅ |
| Transformer | ✅ | ✅ | ✅ |
| TFT | ✅ | ✅ | ✅ |
| N-BEATS | ❌ | ❌ | ✅ |
| TiDE | ✅ | ✅ | ✅ |
| NHiTS | ✅ | ✅ | ✅ |
| LinearRegression | ✅ | ✅ | ❌ |
| RandomForest | ✅ | ✅ | ❌ |
| XGBoost | ✅ | ✅ | ❌ |
| LightGBM | ✅ | ✅ | ❌ |
| CatBoost | ✅ | ✅ | ❌ |

---

## 🎨 Best Practices

### ✅ DO:

1. **Use deterministic future covariates when possible**
   - Calendar features (always known)
   - Scheduled events
   - Official economic forecasts

2. **Normalize/scale covariates**
   - GDP values are much larger than prices
   - Use standardization or min-max scaling

3. **Check correlation with target**
   - Verify covariates actually relate to target
   - Remove uncorrelated features

4. **Use cyclical encoding for circular features**
   - Month, day of week, hour
   - Preserves circular nature

5. **Start simple, add complexity gradually**
   - Begin with few strong covariates
   - Add more if they improve performance

### ❌ DON'T:

1. **Don't use future-leaking past covariates**
   - No "next day's price" in past covariates
   - Avoid look-ahead bias

2. **Don't include target in covariates**
   - This is circular and invalid

3. **Don't use too many correlated covariates**
   - GDP USA and GDP Total are highly correlated
   - Can cause multicollinearity

4. **Don't forget to align time indices**
   - Covariates must match target's timestamps
   - Missing values cause errors

5. **Don't use unreliable future forecasts**
   - If covariate forecasts are poor, they hurt performance
   - Better to use no covariate than a bad one

---

## 🚀 Quick Start: Run Enhanced Example

Test the enhanced covariate example:

```bash
python covariates_guide.py
```

This will:
1. Load your existing data
2. Add calendar features
3. Add economic indicators
4. Add seasonal events
5. Train Prophet with all covariates
6. Generate predictions
7. Save enhanced dataset

---

## 📚 Further Reading

- **Darts Documentation**: https://unit8co.github.io/darts/userguide/covariates.html
- **Prophet with Regressors**: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
- **Feature Engineering for Time Series**: Look for domain-specific features

---

## 💡 Key Takeaways

1. **Your current model uses Future Covariates** (GDP) - Perfect for Prophet! ✅

2. **Future Covariates = Known in advance** (forecasts, calendar, planned events)

3. **Past Covariates = Historical only** (actual observations, measurements)

4. **Static Covariates = Never change** (categories, IDs, regions)

5. **Prophet only supports Future Covariates** - which is what you need!

6. **Always validate that covariates improve performance** - compare with and without

---

**Your project is already using covariates correctly! 🎉**

The GDP signals as future covariates are an excellent choice for predicting product prices with Prophet.
