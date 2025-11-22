"""
Example: Adding More Types of Covariates to the Forecasting Model
This demonstrates how to incorporate different covariate types in Prophet.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from darts import TimeSeries
from darts.models import Prophet

def add_calendar_features(df):
    """Add calendar-based future covariates"""
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week
    df['is_year_end'] = ((df['date'].dt.month == 12) & (df['date'].dt.day >= 24)).astype(int)
    df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
    
    # Cyclical encoding for month (continuous representation)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    return df

def add_economic_indicators(df):
    """Add more economic future covariates"""
    n = len(df)
    np.random.seed(42)
    
    # Interest rates (declining trend)
    base_rate = 5.0
    df['interest_rate'] = base_rate - 0.02 * np.arange(n) / 52 + np.random.normal(0, 0.1, n)
    df['interest_rate'] = df['interest_rate'].clip(0.5, 10.0)
    
    # Inflation rate (with cycles)
    base_inflation = 2.5
    df['inflation_rate'] = base_inflation + 0.5 * np.sin(2 * np.pi * np.arange(n) / 52) + np.random.normal(0, 0.3, n)
    
    # Consumer confidence index (0-100 scale)
    base_confidence = 70
    df['consumer_confidence'] = base_confidence + 10 * np.sin(2 * np.pi * np.arange(n) / 52) + np.random.normal(0, 5, n)
    df['consumer_confidence'] = df['consumer_confidence'].clip(30, 100)
    
    # Exchange rate (USD index)
    df['usd_index'] = 100 + 5 * np.sin(2 * np.pi * np.arange(n) / 52) + np.random.normal(0, 2, n)
    
    return df

def add_seasonal_events(df):
    """Add seasonal event indicators (future covariates)"""
    # Holiday seasons (Christmas, Black Friday, etc.)
    df['is_holiday_season'] = ((df['date'].dt.month == 11) | (df['date'].dt.month == 12)).astype(int)
    
    # Summer season
    df['is_summer'] = ((df['date'].dt.month >= 6) & (df['date'].dt.month <= 8)).astype(int)
    
    # Back to school season
    df['is_back_to_school'] = ((df['date'].dt.month == 8) | (df['date'].dt.month == 9)).astype(int)
    
    return df

def create_enhanced_model():
    """Create Prophet model with various covariate types"""
    
    # Load existing data
    print("Loading existing data...")
    df = pd.read_csv('data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Add new covariates
    print("Adding calendar features...")
    df = add_calendar_features(df)
    
    print("Adding economic indicators...")
    df = add_economic_indicators(df)
    
    print("Adding seasonal events...")
    df = add_seasonal_events(df)
    
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"Covariate columns: {df.columns.tolist()}")
    
    # Create TimeSeries objects
    target_series = TimeSeries.from_dataframe(
        df,
        time_col='date',
        value_cols='product_price',
        freq='W-FRI'
    )
    
    # FUTURE COVARIATES: All features available for future predictions
    future_covariate_cols = [
        # Original GDP features
        'gdp_usa', 'gdp_china', 'gdp_eu', 'gdp_total',
        'gdp_usa_yoy_change', 'gdp_china_yoy_change', 'gdp_eu_yoy_change',
        
        # Calendar features
        'month', 'quarter', 'week_of_year', 'is_year_end', 'is_quarter_end',
        'month_sin', 'month_cos',
        
        # Economic indicators
        'interest_rate', 'inflation_rate', 'consumer_confidence', 'usd_index',
        
        # Seasonal events
        'is_holiday_season', 'is_summer', 'is_back_to_school'
    ]
    
    future_covariates = TimeSeries.from_dataframe(
        df,
        time_col='date',
        value_cols=future_covariate_cols,
        freq='W-FRI'
    )
    
    print(f"\nFuture covariates: {len(future_covariate_cols)} features")
    
    # Split data
    train_end = pd.Timestamp('2024-12-31')
    train_target = target_series.split_before(train_end)[0]
    
    # Train Prophet model with enhanced covariates
    print("\nTraining Prophet with enhanced covariates...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        seasonality_mode='multiplicative'
    )
    
    model.fit(
        series=train_target,
        future_covariates=future_covariates
    )
    
    print("✅ Model trained successfully with enhanced covariates!")
    
    # Make predictions
    predictions = model.predict(
        n=104,
        future_covariates=future_covariates
    )
    
    print(f"✅ Generated {len(predictions)} predictions")
    
    return model, predictions, df

# Example of different model types and their covariate support

def covariate_support_by_model():
    """Show which Darts models support which covariate types"""
    
    support_table = {
        'Model': [
            'Prophet', 'ARIMA', 'AutoARIMA', 'ExponentialSmoothing',
            'RNNModel', 'LSTMModel', 'GRUModel', 'BlockRNNModel',
            'TCNModel', 'TransformerModel', 'TFTModel',
            'NHiTS', 'TiDE', 'N-BEATS',
            'LinearRegressionModel', 'RandomForest', 'XGBoostModel',
            'LightGBMModel', 'CatBoostModel'
        ],
        'Past Covariates': [
            '❌', '✅', '✅', '❌',
            '✅', '✅', '✅', '✅',
            '✅', '✅', '✅',
            '✅', '✅', '❌',
            '✅', '✅', '✅',
            '✅', '✅'
        ],
        'Future Covariates': [
            '✅', '✅', '✅', '❌',
            '✅', '✅', '✅', '✅',
            '✅', '✅', '✅',
            '✅', '✅', '❌',
            '✅', '✅', '✅',
            '✅', '✅'
        ],
        'Static Covariates': [
            '❌', '❌', '❌', '❌',
            '✅', '✅', '✅', '✅',
            '✅', '✅', '✅',
            '✅', '✅', '✅',
            '❌', '❌', '❌',
            '❌', '❌'
        ]
    }
    
    df = pd.DataFrame(support_table)
    print("\n" + "="*80)
    print("COVARIATE SUPPORT BY MODEL TYPE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    print("="*80)
    print("ENHANCED COVARIATES EXAMPLE")
    print("="*80)
    
    # Show covariate support
    covariate_support_by_model()
    
    # Train model with enhanced covariates
    print("\n" + "="*80)
    print("TRAINING MODEL WITH ENHANCED COVARIATES")
    print("="*80)
    
    model, predictions, enhanced_df = create_enhanced_model()
    
    # Save enhanced data
    enhanced_df.to_csv('data_enhanced.csv', index=False)
    print("\n✅ Enhanced data saved to 'data_enhanced.csv'")
    
    print("\n" + "="*80)
    print("COVARIATE TYPES SUMMARY")
    print("="*80)
    print("""
    1. FUTURE COVARIATES (used in your model ✅):
       - Available for past AND future
       - Examples: GDP, calendar features, economic forecasts
       - Used by: Prophet, ARIMA, Neural Networks
    
    2. PAST COVARIATES:
       - Only available for historical periods
       - Examples: Past sales, actual weather, historical metrics
       - Used by: ARIMA, RNN, Tree-based models
    
    3. STATIC COVARIATES:
       - Time-invariant features
       - Examples: Product category, store location, region
       - Used by: Neural network models
    """)
    print("="*80)
