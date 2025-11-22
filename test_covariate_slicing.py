"""
Test script to demonstrate what happens when you manually slice covariates
vs letting Darts handle it automatically.
"""

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.models import Prophet
import warnings
warnings.filterwarnings('ignore')

def test_automatic_slicing():
    """Test the recommended approach - full covariates to both"""
    print("="*80)
    print("TEST 1: AUTOMATIC SLICING (Recommended)")
    print("="*80)
    
    # Create simple data
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    df = pd.DataFrame({
        'date': dates,
        'price': np.linspace(100, 150, 200) + np.random.randn(200),
        'gdp': np.linspace(1000, 1200, 200) + np.random.randn(200) * 10
    })
    
    # Create TimeSeries
    target = TimeSeries.from_dataframe(df, 'date', 'price', freq='W')
    covariates = TimeSeries.from_dataframe(df, 'date', 'gdp', freq='W')
    
    # Split target only
    train_target = target[:150]  # First 150 weeks
    
    print(f"Target length: {len(target)}")
    print(f"Train target length: {len(train_target)}")
    print(f"Train target range: {train_target.start_time()} to {train_target.end_time()}")
    print(f"Covariates length: {len(covariates)}")
    print(f"Covariates range: {covariates.start_time()} to {covariates.end_time()}")
    
    try:
        # Train with FULL covariates
        model = Prophet()
        model.fit(train_target, future_covariates=covariates)
        print("✓ Fit successful with full covariates")
        
        # Predict with SAME FULL covariates
        predictions = model.predict(n=30, future_covariates=covariates)
        print(f"✓ Predict successful with full covariates")
        print(f"  Predictions length: {len(predictions)}")
        print(f"  Predictions range: {predictions.start_time()} to {predictions.end_time()}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_manual_slicing_correct():
    """Test manual slicing - the 'correct' way if you insist on slicing"""
    print("\n" + "="*80)
    print("TEST 2: MANUAL SLICING - Careful Approach")
    print("="*80)
    
    # Create simple data
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    df = pd.DataFrame({
        'date': dates,
        'price': np.linspace(100, 150, 200) + np.random.randn(200),
        'gdp': np.linspace(1000, 1200, 200) + np.random.randn(200) * 10
    })
    
    # Create TimeSeries
    target = TimeSeries.from_dataframe(df, 'date', 'price', freq='W')
    covariates = TimeSeries.from_dataframe(df, 'date', 'gdp', freq='W')
    
    # Split target
    train_target = target[:150]
    
    # Manual slicing - but carefully!
    # Include enough covariates for training AND prediction
    train_and_future_covariates = covariates[:180]  # 150 training + 30 prediction
    
    print(f"Train target length: {len(train_target)}")
    print(f"Train target range: {train_target.start_time()} to {train_target.end_time()}")
    print(f"Covariates length: {len(train_and_future_covariates)}")
    print(f"Covariates range: {train_and_future_covariates.start_time()} to {train_and_future_covariates.end_time()}")
    
    try:
        model = Prophet()
        model.fit(train_target, future_covariates=train_and_future_covariates)
        print("✓ Fit successful")
        
        # Predict - needs covariates for prediction period
        predictions = model.predict(n=30, future_covariates=train_and_future_covariates)
        print(f"✓ Predict successful")
        print(f"  Predictions length: {len(predictions)}")
        print(f"  Predictions range: {predictions.start_time()} to {predictions.end_time()}")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_manual_slicing_wrong():
    """Test manual slicing - common mistake"""
    print("\n" + "="*80)
    print("TEST 3: MANUAL SLICING - Common Mistake")
    print("="*80)
    
    # Create simple data
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    df = pd.DataFrame({
        'date': dates,
        'price': np.linspace(100, 150, 200) + np.random.randn(200),
        'gdp': np.linspace(1000, 1200, 200) + np.random.randn(200) * 10
    })
    
    # Create TimeSeries
    target = TimeSeries.from_dataframe(df, 'date', 'price', freq='W')
    covariates = TimeSeries.from_dataframe(df, 'date', 'gdp', freq='W')
    
    # Split target
    train_target = target[:150]
    
    # ❌ WRONG: Only include training period covariates
    train_covariates_only = covariates[:150]  # Only training period
    
    print(f"Train target length: {len(train_target)}")
    print(f"Train target range: {train_target.start_time()} to {train_target.end_time()}")
    print(f"Covariates length: {len(train_covariates_only)}")
    print(f"Covariates range: {train_covariates_only.start_time()} to {train_covariates_only.end_time()}")
    
    try:
        model = Prophet()
        model.fit(train_target, future_covariates=train_covariates_only)
        print("✓ Fit successful")
        
        # Try to predict - this will FAIL
        print("\nAttempting to predict with insufficient covariates...")
        predictions = model.predict(n=30, future_covariates=train_covariates_only)
        print(f"✓ Predict successful (unexpected!)")
        
        return True
    except Exception as e:
        print(f"✗ Predict FAILED as expected!")
        print(f"  Error: {type(e).__name__}")
        print(f"  Message: {str(e)[:200]}...")
        print("\n⚠️  This is why you shouldn't slice covariates too short!")
        return False

def test_misaligned_slicing():
    """Test what happens with misaligned time indices"""
    print("\n" + "="*80)
    print("TEST 4: MISALIGNED SLICING - Using wrong portion")
    print("="*80)
    
    # Create simple data
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    df = pd.DataFrame({
        'date': dates,
        'price': np.linspace(100, 150, 200) + np.random.randn(200),
        'gdp': np.linspace(1000, 1200, 200) + np.random.randn(200) * 10
    })
    
    # Create TimeSeries
    target = TimeSeries.from_dataframe(df, 'date', 'price', freq='W')
    covariates = TimeSeries.from_dataframe(df, 'date', 'gdp', freq='W')
    
    # Split target
    train_target = target[:150]  # Weeks 0-149
    
    # ❌ WRONG: Use future-only covariates for training
    # This creates a time mismatch
    future_only_covariates = covariates[150:]  # Weeks 150-199
    
    print(f"Train target range: {train_target.start_time()} to {train_target.end_time()}")
    print(f"Covariates range: {future_only_covariates.start_time()} to {future_only_covariates.end_time()}")
    print("⚠️  Notice: Covariates don't overlap with training period!")
    
    try:
        model = Prophet()
        model.fit(train_target, future_covariates=future_only_covariates)
        print("✓ Fit successful (but model may not learn properly)")
        
        predictions = model.predict(n=30, future_covariates=future_only_covariates)
        print(f"✓ Predict successful")
        
        return True
    except Exception as e:
        print(f"✗ Error: {type(e).__name__}")
        print(f"  Message: {str(e)[:200]}...")
        return False

def compare_results():
    """Compare predictions from automatic vs manual slicing"""
    print("\n" + "="*80)
    print("TEST 5: COMPARING RESULTS - Same predictions?")
    print("="*80)
    
    # Create simple data
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range('2020-01-01', periods=200, freq='W')
    df = pd.DataFrame({
        'date': dates,
        'price': np.linspace(100, 150, 200) + np.random.randn(200),
        'gdp': np.linspace(1000, 1200, 200) + np.random.randn(200) * 10
    })
    
    # Create TimeSeries
    target = TimeSeries.from_dataframe(df, 'date', 'price', freq='W')
    covariates = TimeSeries.from_dataframe(df, 'date', 'gdp', freq='W')
    train_target = target[:150]
    
    # Approach 1: Automatic (full covariates)
    model1 = Prophet()
    model1.fit(train_target, future_covariates=covariates)
    pred1 = model1.predict(n=30, future_covariates=covariates)
    
    # Approach 2: Manual slicing (carefully done)
    manual_covariates = covariates[:180]  # 150 + 30
    model2 = Prophet()
    model2.fit(train_target, future_covariates=manual_covariates)
    pred2 = model2.predict(n=30, future_covariates=manual_covariates)
    
    # Compare
    diff = abs(pred1.values() - pred2.values()).mean()
    print(f"Approach 1 (automatic): Predictions from {pred1.start_time()} to {pred1.end_time()}")
    print(f"Approach 2 (manual):    Predictions from {pred2.start_time()} to {pred2.end_time()}")
    print(f"\nMean absolute difference: {diff:.6f}")
    
    if diff < 0.001:
        print("✓ Results are essentially identical!")
        print("  → Manual slicing works IF done correctly")
        print("  → But automatic is safer and simpler")
    else:
        print(f"⚠️  Results differ by {diff:.6f}")
        print("  → This could indicate alignment issues")

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("TESTING COVARIATE SLICING APPROACHES")
    print("="*80)
    print("\nThis demonstrates what happens with different slicing approaches.")
    print("="*80)
    
    results = {}
    
    # Test 1: Recommended approach
    results['automatic'] = test_automatic_slicing()
    
    # Test 2: Manual slicing done correctly
    results['manual_correct'] = test_manual_slicing_correct()
    
    # Test 3: Manual slicing done wrong
    results['manual_wrong'] = test_manual_slicing_wrong()
    
    # Test 4: Misaligned slicing
    results['misaligned'] = test_misaligned_slicing()
    
    # Test 5: Compare results
    compare_results()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nTest Results:")
    print(f"  1. Automatic slicing (recommended):    {'✓ PASS' if results['automatic'] else '✗ FAIL'}")
    print(f"  2. Manual slicing (correct):           {'✓ PASS' if results['manual_correct'] else '✗ FAIL'}")
    print(f"  3. Manual slicing (wrong):             {'✗ FAIL (expected)' if not results['manual_wrong'] else '? PASS'}")
    print(f"  4. Misaligned slicing:                 {'? Unclear' if results['misaligned'] else '✗ FAIL'}")
    
    print("\n" + "="*80)
    print("CONCLUSIONS")
    print("="*80)
    print("""
1. ✅ AUTOMATIC SLICING (Recommended):
   - Pass full covariates to both fit() and predict()
   - Darts handles alignment automatically
   - Safest and simplest approach
   - Less error-prone

2. ⚠️  MANUAL SLICING (If you must):
   - CAN work if done carefully
   - Must include covariates for BOTH training AND prediction periods
   - Must ensure time indices align properly
   - More error-prone
   - No real benefit over automatic

3. ❌ COMMON MISTAKES:
   - Slicing too short (not including prediction period)
   - Misaligned time indices
   - Using wrong portion of covariates
   - Hard to debug when things go wrong

RECOMMENDATION: Always use full covariates and let Darts handle it!
    """)
    print("="*80)

if __name__ == "__main__":
    main()
