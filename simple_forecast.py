"""
Alternative forecasting model using simpler approaches.
This can be used if Prophet model takes too long or has issues.
Uses basic statistical models from Darts.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from darts import TimeSeries
    from darts.models import ExponentialSmoothing, AutoARIMA
    from darts.metrics import mape, rmse, mae
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False
    print("WARNING: Darts not installed. Please install with: pip install darts")

def load_and_prepare_data(file_path='data.csv'):
    """Load data and split into train/test sets"""
    print("Loading data...")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df

def create_timeseries(df):
    """Convert DataFrame to Darts TimeSeries objects"""
    if not DARTS_AVAILABLE:
        return None, None
        
    print("\nCreating TimeSeries objects...")
    
    # Target variable
    target_series = TimeSeries.from_dataframe(
        df,
        time_col='date',
        value_cols='product_price',
        freq='W-FRI'
    )
    
    # Covariates
    covariate_series = TimeSeries.from_dataframe(
        df,
        time_col='date',
        value_cols=['gdp_usa', 'gdp_china', 'gdp_eu'],
        freq='W-FRI'
    )
    
    return target_series, covariate_series

def split_data(target_series, train_end_date='2024-12-31'):
    """Split data into train and test sets"""
    if not DARTS_AVAILABLE:
        return None, None
        
    print("\nSplitting data...")
    train_end = pd.Timestamp(train_end_date)
    
    train_target = target_series.split_before(train_end)[0]
    test_target = target_series.split_after(train_end)[1]
    
    print(f"Training: {len(train_target)} weeks")
    print(f"Test: {len(test_target)} weeks")
    
    return train_target, test_target

def train_exponential_smoothing(train_target):
    """Train Exponential Smoothing model (fast and simple)"""
    if not DARTS_AVAILABLE:
        return None
        
    print("\n" + "="*60)
    print("Training Exponential Smoothing Model")
    print("="*60)
    
    model = ExponentialSmoothing(seasonal_periods=52)  # 52 weeks in a year
    print("Fitting model...")
    model.fit(train_target)
    print("Model trained successfully!")
    
    return model

def make_predictions(model, n_weeks):
    """Generate predictions"""
    if not DARTS_AVAILABLE or model is None:
        return None
        
    print(f"\nGenerating {n_weeks}-week forecast...")
    predictions = model.predict(n=n_weeks)
    print(f"Predictions generated!")
    
    return predictions

def evaluate_model(test_target, predictions):
    """Evaluate predictions"""
    if not DARTS_AVAILABLE or predictions is None:
        return None
        
    print("\n" + "="*60)
    print("Evaluation Metrics")
    print("="*60)
    
    # Get overlapping period
    common_start = max(test_target.start_time(), predictions.start_time())
    common_end = min(test_target.end_time(), predictions.end_time())
    
    if common_start >= common_end:
        print("No overlapping period for evaluation")
        return None
    
    test_subset = test_target.slice(common_start, common_end)
    pred_subset = predictions.slice(common_start, common_end)
    
    mape_score = mape(test_subset, pred_subset)
    rmse_score = rmse(test_subset, pred_subset)
    mae_score = mae(test_subset, pred_subset)
    
    print(f"Evaluation period: {common_start} to {common_end}")
    print(f"Weeks evaluated: {len(test_subset)}")
    print(f"\nMetrics:")
    print(f"  MAPE: {mape_score:.2f}%")
    print(f"  RMSE: {rmse_score:.2f}")
    print(f"  MAE: {mae_score:.2f}")
    
    return {'mape': mape_score, 'rmse': rmse_score, 'mae': mae_score}

def visualize_results(train_target, test_target, predictions):
    """Visualize results"""
    if not DARTS_AVAILABLE or predictions is None:
        return
        
    print("\nCreating visualization...")
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Full view
    ax1 = axes[0]
    train_target.plot(ax=ax1, label='Training Data', color='blue', alpha=0.7)
    test_target.plot(ax=ax1, label='Test Data', color='green', alpha=0.7)
    predictions.plot(ax=ax1, label='Predictions', color='red', alpha=0.7)
    ax1.set_title('Product Price Forecasting: Exponential Smoothing', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Product Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Zoomed view
    ax2 = axes[1]
    train_tail = train_target[-52:]
    train_tail.plot(ax=ax2, label='Recent Training', color='blue', alpha=0.7)
    test_target.plot(ax=ax2, label='Test Data', color='green', alpha=0.7)
    predictions.plot(ax=ax2, label='Predictions', color='red', alpha=0.7)
    ax2.set_title('Detailed View: Test Period', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Product Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('simple_forecast_results.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to 'simple_forecast_results.png'")

def save_predictions(predictions, filename='simple_predictions.csv'):
    """Save predictions to CSV"""
    if not DARTS_AVAILABLE or predictions is None:
        return
        
    print(f"\nSaving predictions to {filename}...")
    pred_df = predictions.pd_dataframe()
    pred_df.reset_index(inplace=True)
    pred_df.columns = ['date', 'predicted_price']
    pred_df.to_csv(filename, index=False)
    print(f"Saved {len(pred_df)} predictions")

def main():
    """Main execution"""
    print("="*60)
    print("SIMPLE TIME SERIES FORECASTING")
    print("="*60)
    print("Using: Exponential Smoothing (fast alternative)")
    print("="*60)
    
    if not DARTS_AVAILABLE:
        print("\nERROR: Darts library not available.")
        print("Please install: pip install darts")
        return
    
    # Load data
    df = load_and_prepare_data()
    
    # Create time series
    target_series, covariate_series = create_timeseries(df)
    
    # Split data
    train_target, test_target = split_data(target_series)
    
    # Train model
    model = train_exponential_smoothing(train_target)
    
    # Make predictions (104 weeks)
    predictions = make_predictions(model, 104)
    
    # Evaluate
    metrics = evaluate_model(test_target, predictions)
    
    # Visualize
    visualize_results(train_target, test_target, predictions)
    
    # Save predictions
    save_predictions(predictions)
    
    print("\n" + "="*60)
    print("FORECASTING COMPLETED!")
    print("="*60)
    print("\nOutput files:")
    print("  - simple_predictions.csv")
    print("  - simple_forecast_results.png")
    print("="*60)

if __name__ == "__main__":
    main()
