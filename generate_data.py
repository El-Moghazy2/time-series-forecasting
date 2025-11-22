"""
Generate synthetic time series data for forecasting model.
Creates weekly product prices and GDP data for 3 countries from 2010 to 2027.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create weekly date range from 2010-01-01 to 2027-12-31
start_date = datetime(2010, 1, 1)
end_date = datetime(2027, 12, 31)
date_range = pd.date_range(start=start_date, end=end_date, freq='W-FRI')

# Generate GDP data for 3 countries (quarterly data interpolated to weekly)
# GDP generally grows with some cyclical patterns
def generate_gdp_series(base_value, growth_rate, volatility, dates):
    """Generate synthetic GDP data with trend and seasonality"""
    n_weeks = len(dates)
    
    # Trend component (exponential growth)
    trend = base_value * (1 + growth_rate) ** (np.arange(n_weeks) / 52)
    
    # Seasonal component (annual cycle)
    seasonal = 0.05 * trend * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    
    # Random noise
    noise = np.random.normal(0, volatility * trend, n_weeks)
    
    # Combine components
    gdp = trend + seasonal + noise
    
    return gdp

# Generate GDP for 3 countries
gdp_usa = generate_gdp_series(base_value=14000, growth_rate=0.025, volatility=0.02, dates=date_range)
gdp_china = generate_gdp_series(base_value=6000, growth_rate=0.065, volatility=0.03, dates=date_range)
gdp_eu = generate_gdp_series(base_value=12000, growth_rate=0.015, volatility=0.018, dates=date_range)

# Generate product price data (influenced by GDP with some lag and additional factors)
def generate_product_price(gdp_usa, gdp_china, gdp_eu, dates):
    """Generate synthetic product price influenced by GDP indicators"""
    n_weeks = len(dates)
    
    # Base price with upward trend
    base_price = 100
    trend = base_price * (1.03) ** (np.arange(n_weeks) / 52)
    
    # Influence from GDP (normalized and weighted)
    gdp_usa_norm = (gdp_usa - gdp_usa.mean()) / gdp_usa.std()
    gdp_china_norm = (gdp_china - gdp_china.mean()) / gdp_china.std()
    gdp_eu_norm = (gdp_eu - gdp_eu.mean()) / gdp_eu.std()
    
    # Weighted GDP influence with lag of 4 weeks
    gdp_influence = np.zeros(n_weeks)
    lag = 4
    if n_weeks > lag:
        gdp_influence[lag:] = (0.4 * gdp_usa_norm[:-lag] + 
                               0.35 * gdp_china_norm[:-lag] + 
                               0.25 * gdp_eu_norm[:-lag])
    
    gdp_effect = 5 * gdp_influence
    
    # Seasonal component (stronger than GDP)
    seasonal = 8 * np.sin(2 * np.pi * np.arange(n_weeks) / 52) + \
               3 * np.sin(4 * np.pi * np.arange(n_weeks) / 52)
    
    # Random noise
    noise = np.random.normal(0, 2, n_weeks)
    
    # Occasional shocks (simulate economic events)
    shocks = np.zeros(n_weeks)
    shock_indices = np.random.choice(n_weeks, size=5, replace=False)
    shocks[shock_indices] = np.random.uniform(-10, 10, 5)
    
    # Combine all components
    price = trend + gdp_effect + seasonal + noise + shocks
    
    # Ensure prices are positive
    price = np.maximum(price, 10)
    
    return price

# Generate product prices
product_price = generate_product_price(gdp_usa, gdp_china, gdp_eu, date_range)

# Create DataFrame
df = pd.DataFrame({
    'date': date_range,
    'product_price': product_price,
    'gdp_usa': gdp_usa,
    'gdp_china': gdp_china,
    'gdp_eu': gdp_eu
})

# Add some derived features
df['gdp_total'] = df['gdp_usa'] + df['gdp_china'] + df['gdp_eu']
df['gdp_usa_yoy_change'] = df['gdp_usa'].pct_change(52) * 100  # Year-over-year % change
df['gdp_china_yoy_change'] = df['gdp_china'].pct_change(52) * 100
df['gdp_eu_yoy_change'] = df['gdp_eu'].pct_change(52) * 100

# Fill NaN values from pct_change
df = df.fillna(method='bfill')

# Save to CSV
df.to_csv('data.csv', index=False)

print(f"Generated data with {len(df)} weekly observations")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\nData shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nLast few rows:")
print(df.tail())
print(f"\nData statistics:")
print(df.describe())

# Save split information
print(f"\n{'='*60}")
print(f"Data splits:")
print(f"  Training: 2010-01-01 to 2024-12-31 (approx {len(df[df['date'] < '2025-01-01'])} weeks)")
print(f"  Test: 2025-01-01 to 2026-12-31 (approx {len(df[(df['date'] >= '2025-01-01') & (df['date'] < '2027-01-01')])} weeks)")
print(f"  Future covariates: 2027-01-01 onwards (approx {len(df[df['date'] >= '2027-01-01'])} weeks)")
print(f"{'='*60}")
