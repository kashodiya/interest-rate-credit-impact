"""
Unit tests for LagAnalysisEngine class.
"""

import pytest
import numpy as np
import pandas as pd
from analysis.lag_analysis_engine import LagAnalysisEngine


def test_compute_cross_correlation_basic():
    """Test basic cross-correlation computation."""
    engine = LagAnalysisEngine(max_lag=3)
    
    # Create simple test series
    dates = pd.date_range('2020-01-01', periods=20, freq='M')
    series1 = pd.Series(np.arange(20), index=dates)
    series2 = pd.Series(np.arange(20) * 2, index=dates)
    
    correlations = engine.compute_cross_correlation(series1, series2, max_lag=3)
    
    # Should return array of length max_lag + 1
    assert len(correlations) == 4
    # Perfect positive correlation at lag 0
    assert abs(correlations[0] - 1.0) < 0.01


def test_compute_cross_correlation_different_lengths():
    """Test that different length series raise ValueError."""
    engine = LagAnalysisEngine()
    
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([1, 2, 3])
    
    with pytest.raises(ValueError, match="same length"):
        engine.compute_cross_correlation(series1, series2)


def test_compute_cross_correlation_insufficient_data():
    """Test that series shorter than max_lag raise ValueError."""
    engine = LagAnalysisEngine(max_lag=10)
    
    series1 = pd.Series([1, 2, 3, 4, 5])
    series2 = pd.Series([1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError, match="must be greater than max_lag"):
        engine.compute_cross_correlation(series1, series2)


def test_find_optimal_lag():
    """Test finding optimal lag from correlation array."""
    engine = LagAnalysisEngine()
    
    # Create correlation array where lag 2 has maximum absolute correlation
    correlations = np.array([0.5, 0.6, 0.9, 0.4, 0.3])
    
    optimal_lag = engine.find_optimal_lag(correlations)
    
    assert optimal_lag == 2


def test_find_optimal_lag_negative_correlation():
    """Test finding optimal lag with negative correlations."""
    engine = LagAnalysisEngine()
    
    # Lag 3 has maximum absolute correlation (|-0.95| = 0.95)
    correlations = np.array([0.5, 0.6, 0.7, -0.95, 0.3])
    
    optimal_lag = engine.find_optimal_lag(correlations)
    
    assert optimal_lag == 3


def test_find_optimal_lag_empty_array():
    """Test that empty correlation array raises ValueError."""
    engine = LagAnalysisEngine()
    
    with pytest.raises(ValueError, match="empty"):
        engine.find_optimal_lag(np.array([]))


def test_find_optimal_lag_all_nan():
    """Test that all NaN correlation array raises ValueError."""
    engine = LagAnalysisEngine()
    
    with pytest.raises(ValueError, match="NaN"):
        engine.find_optimal_lag(np.array([np.nan, np.nan, np.nan]))


def test_test_significance():
    """Test p-value calculation for correlation significance."""
    engine = LagAnalysisEngine()
    
    # Strong correlation with many samples should have low p-value
    p_value = engine.test_significance(correlation=0.8, n_samples=100)
    
    assert 0 <= p_value <= 1
    assert p_value < 0.05  # Should be significant


def test_test_significance_weak_correlation():
    """Test p-value for weak correlation."""
    engine = LagAnalysisEngine()
    
    # Weak correlation should have high p-value
    p_value = engine.test_significance(correlation=0.1, n_samples=20)
    
    assert 0 <= p_value <= 1
    assert p_value > 0.05  # Should not be significant


def test_test_significance_invalid_correlation():
    """Test that invalid correlation values raise ValueError."""
    engine = LagAnalysisEngine()
    
    with pytest.raises(ValueError, match="between -1 and 1"):
        engine.test_significance(correlation=1.5, n_samples=100)


def test_test_significance_insufficient_samples():
    """Test that insufficient samples raise ValueError."""
    engine = LagAnalysisEngine()
    
    with pytest.raises(ValueError, match="at least 3 samples"):
        engine.test_significance(correlation=0.5, n_samples=2)


def test_generate_lag_plot():
    """Test lag plot generation."""
    engine = LagAnalysisEngine()
    
    lags = np.array([0, 1, 2, 3, 4])
    correlations = np.array([0.5, 0.6, 0.9, 0.4, 0.3])
    
    fig = engine.generate_lag_plot(lags, correlations, optimal_lag=2)
    
    # Check that figure is created
    assert fig is not None
    # Should have 2 traces (correlation line and optimal lag marker)
    assert len(fig.data) == 2


def test_generate_lag_plot_with_variable_pair():
    """Test lag plot with variable pair names."""
    engine = LagAnalysisEngine()
    
    lags = np.array([0, 1, 2, 3])
    correlations = np.array([0.5, 0.6, 0.7, 0.4])
    
    fig = engine.generate_lag_plot(
        lags, 
        correlations, 
        variable_pair=("Fed Funds Rate", "Total Credit"),
        optimal_lag=2
    )
    
    assert fig is not None
    assert "Fed Funds Rate" in fig.layout.title.text
    assert "Total Credit" in fig.layout.title.text


def test_generate_lag_plot_mismatched_lengths():
    """Test that mismatched lags and correlations raise ValueError."""
    engine = LagAnalysisEngine()
    
    lags = np.array([0, 1, 2])
    correlations = np.array([0.5, 0.6])
    
    with pytest.raises(ValueError, match="same length"):
        engine.generate_lag_plot(lags, correlations)
