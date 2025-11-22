"""
Time Series Forecasting using Darts Prophet Model
Predicts product prices 104 weeks ahead using GDP signals as future covariates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from darts import TimeSeries
from darts.models import Prophet
from darts.metrics import mape, rmse, mae
from darts.utils.statistics import check_seasonality, plot_acf

def load_and_prepare_data(file_path='data.csv'):
    """Load data and split into train/test sets"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total weeks: {len(df)}")
    
    return df

def create_timeseries(df):
    """Convert DataFrame to Darts TimeSeries objects"""
    print("\nCreating TimeSeries objects...")
    
    # Target variable: product price
    target_series = TimeSeries.from_dataframe(
        df,
        time_col='date',
        value_cols='product_price',
        freq='W-FRI'
    )
    
    # Future covariates: GDP signals (available for past and future)
    covariate_series = TimeSeries.from_dataframe(
        df,
        time_col='date',
        value_cols=['gdp_usa', 'gdp_china', 'gdp_eu', 'gdp_total',
                   'gdp_usa_yoy_change', 'gdp_china_yoy_change', 'gdp_eu_yoy_change'],
        freq='W-FRI'
    )
    
    print(f"Target series length: {len(target_series)}")
    print(f"Covariates series length: {len(covariate_series)}")
    
    return target_series, covariate_series

def split_data(target_series, covariate_series, train_end_date='2024-12-31', 
               test_end_date='2026-12-31'):
    """Split data into train and test sets"""
    print("\nSplitting data...")
    
    # Convert string dates to pandas Timestamp
    train_end = pd.Timestamp(train_end_date)
    test_end = pd.Timestamp(test_end_date)
    
    # Split target series
    train_target = target_series.split_before(train_end)[0]
    test_target = target_series.split_before(test_end)[0].split_after(train_end)[1]
    
    # Keep full covariate series (we have future data)
    train_covariates = covariate_series
    
    print(f"Training target length: {len(train_target)} weeks")
    print(f"Training period: {train_target.start_time()} to {train_target.end_time()}")
    print(f"Test target length: {len(test_target)} weeks")
    print(f"Test period: {test_target.start_time()} to {test_target.end_time()}")
    print(f"Available covariates: {len(train_covariates)} weeks")
    
    return train_target, test_target, train_covariates

def analyze_seasonality(series):
    """Analyze seasonality patterns in the data"""
    print("\nAnalyzing seasonality...")
    
    # Check for different seasonal periods
    for period in [52, 26, 13]:  # Yearly, half-yearly, quarterly
        try:
            # Need enough data points (at least 2*period)
            if len(series) >= 2 * period:
                is_seasonal, period_found = check_seasonality(series, m=period, max_lag=min(len(series)//2, period*3), alpha=0.05)
                if is_seasonal:
                    print(f"  Seasonality detected with period {period} weeks")
        except Exception as e:
            print(f"  Could not check seasonality for period {period}: {e}")
    
    return True

def train_prophet_model(train_target, train_covariates):
    """Train Prophet model with future covariates"""
    print("\n" + "="*60)
    print("Training Prophet model...")
    print("="*60)
    
    # Initialize Prophet model
    # Prophet in Darts supports adding regressors (our GDP covariates)
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative',  # Can be 'additive' or 'multiplicative'
        add_encoders={
            'cyclic': {'future': ['month']},
            'datetime_attribute': {'future': ['quarter']}
        }
    )
    
    print("Model configuration:")
    print(f"  Yearly seasonality: True")
    print(f"  Weekly seasonality: True")
    print(f"  Seasonality mode: multiplicative")
    print(f"  Using GDP covariates: {train_covariates.components}")
    
    # Fit the model with future covariates
    print("\nFitting model (this may take a few minutes)...")
    model.fit(
        series=train_target,
        future_covariates=train_covariates
    )
    
    print("Model training completed!")
    
    return model

def make_predictions(model, n_weeks, train_covariates):
    """Generate predictions for n weeks ahead"""
    print(f"\nGenerating {n_weeks}-week forecast...")
    
    # Make predictions
    predictions = model.predict(
        n=n_weeks,
        future_covariates=train_covariates
    )
    
    print(f"Predictions generated: {len(predictions)} weeks")
    print(f"Forecast period: {predictions.start_time()} to {predictions.end_time()}")
    
    return predictions

def evaluate_predictions(test_target, predictions):
    """Evaluate prediction accuracy"""
    print("\n" + "="*60)
    print("Evaluation Metrics")
    print("="*60)
    
    # Calculate metrics on overlapping period
    # Get the overlapping part of predictions and test data
    common_start = max(test_target.start_time(), predictions.start_time())
    common_end = min(test_target.end_time(), predictions.end_time())
    
    test_subset = test_target.slice(common_start, common_end)
    pred_subset = predictions.slice(common_start, common_end)
    
    if len(test_subset) > 0:
        # Calculate metrics
        mape_score = mape(test_subset, pred_subset)
        rmse_score = rmse(test_subset, pred_subset)
        mae_score = mae(test_subset, pred_subset)
        
        print(f"Evaluation period: {common_start} to {common_end}")
        print(f"Number of weeks evaluated: {len(test_subset)}")
        print(f"\nMetrics:")
        print(f"  MAPE (Mean Absolute Percentage Error): {mape_score:.2f}%")
        print(f"  RMSE (Root Mean Squared Error): {rmse_score:.2f}")
        print(f"  MAE (Mean Absolute Error): {mae_score:.2f}")
        
        return {
            'mape': mape_score,
            'rmse': rmse_score,
            'mae': mae_score,
            'n_weeks': len(test_subset)
        }
    else:
        print("Warning: No overlapping period for evaluation")
        return None

def visualize_results(train_target, test_target, predictions, save_path='forecast_results.png'):
    """Visualize the forecasting results"""
    print("\nGenerating visualization...")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot 1: Full view
    ax1 = axes[0]
    train_target.plot(ax=ax1, label='Training Data', color='blue', alpha=0.7)
    test_target.plot(ax=ax1, label='Test Data (Actual)', color='green', alpha=0.7)
    predictions.plot(ax=ax1, label='Predictions', color='red', alpha=0.7)
    ax1.set_title('Product Price Forecasting: Full View', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Product Price', fontsize=12)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed view on test period and forecast
    ax2 = axes[1]
    
    # Show last 52 weeks of training + test + predictions
    train_tail = train_target[-52:]
    train_tail.plot(ax=ax2, label='Recent Training Data', color='blue', alpha=0.7)
    test_target.plot(ax=ax2, label='Test Data (Actual)', color='green', alpha=0.7)
    predictions.plot(ax=ax2, label='104-Week Forecast', color='red', alpha=0.7)
    
    ax2.set_title('Product Price Forecasting: Detailed View (Test Period + Forecast)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Product Price', fontsize=12)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    return fig

def save_predictions(predictions, file_path='predictions.csv'):
    """Save predictions to CSV file"""
    print(f"\nSaving predictions to {file_path}...")
    
    pred_df = predictions.pd_dataframe()
    pred_df.reset_index(inplace=True)
    pred_df.columns = ['date', 'predicted_price']
    pred_df.to_csv(file_path, index=False)
    
    print(f"Predictions saved successfully!")
    print(f"Total predictions: {len(pred_df)}")
    print(f"\nFirst 5 predictions:")
    print(pred_df.head())
    print(f"\nLast 5 predictions:")
    print(pred_df.tail())

def main():
    """Main execution function"""
    print("="*60)
    print("TIME SERIES FORECASTING WITH DARTS PROPHET")
    print("="*60)
    print("Task: Predict product prices 104 weeks ahead")
    print("Model: Prophet with GDP covariates")
    print("="*60)
    
    # Step 1: Load data
    df = load_and_prepare_data('data.csv')
    
    # Step 2: Create TimeSeries objects
    target_series, covariate_series = create_timeseries(df)
    
    # Step 3: Split data
    train_target, test_target, train_covariates = split_data(
        target_series, 
        covariate_series,
        train_end_date='2024-12-31',
        test_end_date='2026-12-31'
    )
    
    # Step 4: Analyze seasonality
    analyze_seasonality(train_target)
    
    # Step 5: Train model
    model = train_prophet_model(train_target, train_covariates)
    
    # Step 6: Make 104-week predictions
    horizon = 104
    predictions = make_predictions(model, horizon, train_covariates)
    
    # Step 7: Evaluate predictions
    metrics = evaluate_predictions(test_target, predictions)
    
    # Step 8: Visualize results
    fig = visualize_results(train_target, test_target, predictions)
    
    # Step 9: Save predictions
    save_predictions(predictions)
    
    print("\n" + "="*60)
    print("FORECASTING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nOutput files generated:")
    print("  1. predictions.csv - Predicted values for 104 weeks")
    print("  2. forecast_results.png - Visualization of results")
    print("\nYou can now analyze the predictions and metrics above.")
    print("="*60)

if __name__ == "__main__":
    main()
