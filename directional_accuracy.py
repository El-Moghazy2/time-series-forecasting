"""
Directional Accuracy Metric for Darts Backtesting

This module provides a directional accuracy metric that can be used with
Darts' model.backtest() method. Directional accuracy measures how often
the model correctly predicts the direction of change (up or down).
"""

import numpy as np
from darts import TimeSeries
from typing import Union, Sequence


def directional_accuracy(
    actual_series: Union[TimeSeries, "Sequence[TimeSeries]"],
    pred_series: Union[TimeSeries, "Sequence[TimeSeries]"],
    intersect: bool = True,
) -> float:
    """
    Calculate directional accuracy between actual and predicted time series.
    
    Directional accuracy measures the percentage of time steps where the model
    correctly predicts whether the value will go up or down compared to the
    previous time step.
    
    Formula:
    --------
    DA = (Number of correct direction predictions / Total predictions) * 100
    
    Where direction is:
    - UP if value(t) > value(t-1)
    - DOWN if value(t) <= value(t-1)
    
    Parameters:
    -----------
    actual_series : TimeSeries or Sequence[TimeSeries]
        The actual time series (ground truth)
    pred_series : TimeSeries or Sequence[TimeSeries]
        The predicted time series
    intersect : bool, default=True
        Whether to only consider the time intersection between actual and predicted series
        
    Returns:
    --------
    float
        Directional accuracy as a percentage (0-100)
        
    Notes:
    ------
    - Returns 100.0 for perfect directional prediction
    - Returns 0.0 if no direction is ever predicted correctly
    - Requires at least 2 time steps to calculate directions
    - Compatible with Darts' model.backtest() method
    
    Example:
    --------
    >>> from darts import TimeSeries
    >>> from darts.models import Prophet
    >>> from directional_accuracy import directional_accuracy
    >>> 
    >>> # Train your model
    >>> model = Prophet()
    >>> model.fit(train_series)
    >>> 
    >>> # Use in backtesting
    >>> backtest_results = model.backtest(
    ...     series=train_series,
    ...     start=0.7,
    ...     forecast_horizon=10,
    ...     metric=directional_accuracy,
    ...     reduction=np.mean
    ... )
    """
    # Handle SeriesGenerator objects from backtest
    # Convert to TimeSeries if needed
    if hasattr(actual_series, '__iter__') and not isinstance(actual_series, TimeSeries):
        # This is a SeriesGenerator or similar iterable
        actual_series = list(actual_series)
    if hasattr(pred_series, '__iter__') and not isinstance(pred_series, TimeSeries):
        # This is a SeriesGenerator or similar iterable
        pred_series = list(pred_series)
    
    # Handle sequences of TimeSeries (for multiple series or stochastic predictions)
    if isinstance(actual_series, list):
        if len(actual_series) != len(pred_series):
            raise ValueError("actual_series and pred_series must have the same length")
        
        # Calculate directional accuracy for each pair and return the mean
        accuracies = [
            directional_accuracy(actual, pred, intersect)
            for actual, pred in zip(actual_series, pred_series)
        ]
        return float(np.mean(accuracies))
    
    # Get numpy arrays from TimeSeries
    from darts.metrics.metrics import _get_values_or_raise
    y_true, y_pred = _get_values_or_raise(actual_series, pred_series, intersect)
    
    # Need at least 2 points to calculate direction
    if len(y_true) < 2:
        raise ValueError("Need at least 2 time steps to calculate directional accuracy")
    
    # Calculate actual directions (change from t-1 to t)
    # Direction is 1 for up, 0 for down/same
    actual_directions = np.diff(y_true, axis=0) > 0
    
    # Calculate predicted directions
    pred_directions = np.diff(y_pred, axis=0) > 0
    
    # Count correct directional predictions
    correct_predictions = np.sum(actual_directions == pred_directions)
    total_predictions = len(actual_directions)
    
    # Calculate percentage
    accuracy = (correct_predictions / total_predictions) * 100.0
    
    return float(accuracy)


def directional_accuracy_with_tolerance(
    actual_series: Union[TimeSeries, "Sequence[TimeSeries]"],
    pred_series: Union[TimeSeries, "Sequence[TimeSeries]"],
    tolerance: float = 0.01,
    intersect: bool = True,
) -> float:
    """
    Calculate directional accuracy with a tolerance threshold for "flat" periods.
    
    This variant treats small changes (below tolerance) as "flat" rather than up/down,
    which can be useful when dealing with noisy data where tiny changes aren't meaningful.
    
    Parameters:
    -----------
    actual_series : TimeSeries or Sequence[TimeSeries]
        The actual time series (ground truth)
    pred_series : TimeSeries or Sequence[TimeSeries]
        The predicted time series
    tolerance : float, default=0.01
        Relative threshold for considering a change as "flat"
        (e.g., 0.01 = 1% change)
    intersect : bool, default=True
        Whether to only consider the time intersection
        
    Returns:
    --------
    float
        Directional accuracy as a percentage (0-100)
        
    Notes:
    ------
    - Directions are: UP (+1), FLAT (0), DOWN (-1)
    - A prediction is correct if it matches the actual direction exactly
    """
    # Handle SeriesGenerator objects from backtest
    if hasattr(actual_series, '__iter__') and not isinstance(actual_series, TimeSeries):
        actual_series = list(actual_series)
    if hasattr(pred_series, '__iter__') and not isinstance(pred_series, TimeSeries):
        pred_series = list(pred_series)
    
    # Handle sequences
    if isinstance(actual_series, list):
        if len(actual_series) != len(pred_series):
            raise ValueError("actual_series and pred_series must have the same length")
        
        accuracies = [
            directional_accuracy_with_tolerance(actual, pred, tolerance, intersect)
            for actual, pred in zip(actual_series, pred_series)
        ]
        return float(np.mean(accuracies))
    
    # Get numpy arrays
    from darts.metrics.metrics import _get_values_or_raise
    y_true, y_pred = _get_values_or_raise(actual_series, pred_series, intersect)
    
    if len(y_true) < 2:
        raise ValueError("Need at least 2 time steps to calculate directional accuracy")
    
    # Calculate percentage changes
    actual_changes = np.diff(y_true, axis=0) / (y_true[:-1] + 1e-10)
    pred_changes = np.diff(y_pred, axis=0) / (y_pred[:-1] + 1e-10)
    
    # Classify directions: 1=up, 0=flat, -1=down
    def classify_direction(changes, tol):
        directions = np.zeros_like(changes)
        directions[changes > tol] = 1    # Up
        directions[changes < -tol] = -1  # Down
        # Changes between -tol and +tol remain 0 (flat)
        return directions
    
    actual_directions = classify_direction(actual_changes, tolerance)
    pred_directions = classify_direction(pred_changes, tolerance)
    
    # Count correct predictions
    correct_predictions = np.sum(actual_directions == pred_directions)
    total_predictions = len(actual_directions)
    
    accuracy = (correct_predictions / total_predictions) * 100.0
    
    return float(accuracy)


# Alias for convenience
da = directional_accuracy


if __name__ == "__main__":
    # Example usage and testing
    print("Directional Accuracy Metric for Darts")
    print("=" * 60)
    
    # Create sample data
    import pandas as pd
    
    dates = pd.date_range('2020-01-01', periods=10, freq='D')
    
    # Actual values: going up and down
    actual_values = [100, 105, 103, 108, 112, 110, 115, 118, 116, 120]
    
    # Predicted values: mostly correct directions
    pred_values = [100, 106, 102, 109, 113, 109, 116, 119, 115, 121]
    
    actual_ts = TimeSeries.from_times_and_values(dates, actual_values)
    pred_ts = TimeSeries.from_times_and_values(dates, pred_values)
    
    # Calculate directional accuracy
    da_score = directional_accuracy(actual_ts, pred_ts)
    
    print(f"\nExample:")
    print(f"Actual values:    {actual_values}")
    print(f"Predicted values: {pred_values}")
    print(f"\nDirectional Accuracy: {da_score:.1f}%")
    
    # Show direction comparison
    print("\nDirection Analysis:")
    print("Time | Actual | Pred | Actual Dir | Pred Dir | Correct")
    print("-" * 60)
    
    for i in range(1, len(actual_values)):
        actual_dir = "UP  " if actual_values[i] > actual_values[i-1] else "DOWN"
        pred_dir = "UP  " if pred_values[i] > pred_values[i-1] else "DOWN"
        correct = "✓" if actual_dir == pred_dir else "✗"
        
        print(f"{i:4d} | {actual_values[i]:6.0f} | {pred_values[i]:4.0f} | "
              f"{actual_dir:9s} | {pred_dir:8s} | {correct}")
    
    print("\n" + "=" * 60)
    print("Usage with model.backtest():")
    print("-" * 60)
    print("""
from directional_accuracy import directional_accuracy
import numpy as np

backtest_results = model.backtest(
    series=train_series,
    start=0.7,
    forecast_horizon=10,
    metric=directional_accuracy,
    reduction=np.mean  # or np.median
)

print(f"Average Directional Accuracy: {backtest_results:.2f}%")
    """)
