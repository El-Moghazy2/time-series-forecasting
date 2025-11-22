"""
Quick test script to verify the data and basic setup.
This runs faster than the full Prophet model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def test_data_loading():
    """Test that data can be loaded and inspected"""
    print("="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    # Load data
    df = pd.read_csv('data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"\n✓ Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    
    # Basic statistics
    print(f"\nProduct Price Statistics:")
    print(f"  Mean: ${df['product_price'].mean():.2f}")
    print(f"  Std Dev: ${df['product_price'].std():.2f}")
    print(f"  Min: ${df['product_price'].min():.2f}")
    print(f"  Max: ${df['product_price'].max():.2f}")
    
    # Check for missing values
    missing = df.isnull().sum()
    print(f"\nMissing values:")
    for col in df.columns:
        if missing[col] > 0:
            print(f"  {col}: {missing[col]}")
    if missing.sum() == 0:
        print("  None - data is complete!")
    
    return df

def test_data_splits(df):
    """Test data splitting logic"""
    print("\n" + "="*60)
    print("TESTING DATA SPLITS")
    print("="*60)
    
    # Define split dates
    train_end = pd.Timestamp('2024-12-31')
    test_end = pd.Timestamp('2026-12-31')
    
    # Create splits
    train_df = df[df['date'] <= train_end]
    test_df = df[(df['date'] > train_end) & (df['date'] <= test_end)]
    future_df = df[df['date'] > test_end]
    
    print(f"\nTraining data:")
    print(f"  Period: {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"  Weeks: {len(train_df)}")
    
    print(f"\nTest data:")
    print(f"  Period: {test_df['date'].min()} to {test_df['date'].max()}")
    print(f"  Weeks: {len(test_df)}")
    
    print(f"\nFuture covariates:")
    print(f"  Period: {future_df['date'].min()} to {future_df['date'].max()}")
    print(f"  Weeks: {len(future_df)}")
    
    return train_df, test_df, future_df

def test_visualization(df):
    """Create a simple visualization of the data"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATION")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Product Price over time
    ax1 = axes[0, 0]
    ax1.plot(df['date'], df['product_price'], label='Product Price', color='blue', alpha=0.7)
    ax1.set_title('Product Price Over Time', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: GDP Signals
    ax2 = axes[0, 1]
    ax2.plot(df['date'], df['gdp_usa'], label='USA GDP', alpha=0.7)
    ax2.plot(df['date'], df['gdp_china'], label='China GDP', alpha=0.7)
    ax2.plot(df['date'], df['gdp_eu'], label='EU GDP', alpha=0.7)
    ax2.set_title('GDP Signals by Country', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('GDP (Billions)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Product Price Distribution
    ax3 = axes[1, 0]
    ax3.hist(df['product_price'], bins=50, edgecolor='black', alpha=0.7)
    ax3.set_title('Product Price Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: GDP YoY Changes
    ax4 = axes[1, 1]
    ax4.plot(df['date'], df['gdp_usa_yoy_change'], label='USA YoY', alpha=0.7)
    ax4.plot(df['date'], df['gdp_china_yoy_change'], label='China YoY', alpha=0.7)
    ax4.plot(df['date'], df['gdp_eu_yoy_change'], label='EU YoY', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax4.set_title('GDP Year-over-Year Change (%)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('YoY Change (%)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to 'data_exploration.png'")
    
    return fig

def test_correlations(df):
    """Test correlations between variables"""
    print("\n" + "="*60)
    print("TESTING CORRELATIONS")
    print("="*60)
    
    # Calculate correlations with product price
    corr_cols = ['gdp_usa', 'gdp_china', 'gdp_eu', 'gdp_total',
                 'gdp_usa_yoy_change', 'gdp_china_yoy_change', 'gdp_eu_yoy_change']
    
    print("\nCorrelation with Product Price:")
    for col in corr_cols:
        corr = df['product_price'].corr(df[col])
        print(f"  {col}: {corr:.4f}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("QUICK DATA TEST & EXPLORATION")
    print("="*60)
    print("This script quickly validates the data without running the model")
    print("="*60)
    
    # Test 1: Load data
    df = test_data_loading()
    
    # Test 2: Test splits
    train_df, test_df, future_df = test_data_splits(df)
    
    # Test 3: Correlations
    test_correlations(df)
    
    # Test 4: Visualization
    test_visualization(df)
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review 'data_exploration.png' to understand the data")
    print("2. Run 'python forecast_model.py' to train the model and make predictions")
    print("="*60)

if __name__ == "__main__":
    main()
