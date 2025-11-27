"""
Tests for EDAEngine class.
"""

import pandas as pd
import numpy as np
import pytest
from analysis.eda_engine import EDAEngine


def test_calculate_growth_rates_monthly():
    """Test monthly growth rate calculation."""
    # Create sample data
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    data = {
        'value1': [100, 110, 121, 133.1, 146.41],  # 10% growth each month
        'value2': [200, 220, 242, 266.2, 292.82]   # 10% growth each month
    }
    df = pd.DataFrame(data, index=dates)
    
    engine = EDAEngine()
    result = engine.calculate_growth_rates(df, ['value1', 'value2'], period='monthly')
    
    # Check that result has correct columns
    assert 'value1_growth_rate' in result.columns
    assert 'value2_growth_rate' in result.columns
    
    # Check that first row is NaN (no previous value)
    assert pd.isna(result.iloc[0]['value1_growth_rate'])
    assert pd.isna(result.iloc[0]['value2_growth_rate'])
    
    # Check that subsequent rows have approximately 10% growth
    for i in range(1, len(result)):
        assert abs(result.iloc[i]['value1_growth_rate'] - 10.0) < 0.1
        assert abs(result.iloc[i]['value2_growth_rate'] - 10.0) < 0.1


def test_calculate_growth_rates_quarterly():
    """Test quarterly growth rate calculation."""
    # Create sample data with 6 months
    dates = pd.date_range(start='2020-01-01', periods=6, freq='MS')
    data = {
        'value': [100, 105, 110, 115, 120, 125]
    }
    df = pd.DataFrame(data, index=dates)
    
    engine = EDAEngine()
    result = engine.calculate_growth_rates(df, ['value'], period='quarterly')
    
    # Check that result has correct column
    assert 'value_growth_rate' in result.columns
    
    # First 3 rows should be NaN (no value 3 months ago)
    assert pd.isna(result.iloc[0]['value_growth_rate'])
    assert pd.isna(result.iloc[1]['value_growth_rate'])
    assert pd.isna(result.iloc[2]['value_growth_rate'])
    
    # Fourth row: (115 - 100) / 100 * 100 = 15%
    assert abs(result.iloc[3]['value_growth_rate'] - 15.0) < 0.1
    
    # Fifth row: (120 - 105) / 105 * 100 ≈ 14.29%
    expected = (120 - 105) / 105 * 100
    assert abs(result.iloc[4]['value_growth_rate'] - expected) < 0.1


def test_calculate_growth_rates_invalid_period():
    """Test that invalid period raises ValueError."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    df = pd.DataFrame({'value': [100, 110, 120, 130, 140]}, index=dates)
    
    engine = EDAEngine()
    
    with pytest.raises(ValueError, match="Period must be 'monthly' or 'quarterly'"):
        engine.calculate_growth_rates(df, ['value'], period='yearly')


def test_calculate_growth_rates_missing_column():
    """Test that missing column raises ValueError."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    df = pd.DataFrame({'value1': [100, 110, 120, 130, 140]}, index=dates)
    
    engine = EDAEngine()
    
    with pytest.raises(ValueError, match="Columns not found in DataFrame"):
        engine.calculate_growth_rates(df, ['value1', 'nonexistent'], period='monthly')


def test_calculate_growth_rates_with_nan_values():
    """Test growth rate calculation with NaN values in data."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    data = {
        'value': [100, np.nan, 120, 130, 140]
    }
    df = pd.DataFrame(data, index=dates)
    
    engine = EDAEngine()
    result = engine.calculate_growth_rates(df, ['value'], period='monthly')
    
    # Check that result handles NaN appropriately
    assert 'value_growth_rate' in result.columns
    assert pd.isna(result.iloc[0]['value_growth_rate'])  # First row always NaN
    assert pd.isna(result.iloc[1]['value_growth_rate'])  # NaN in data means no change can be calculated
    assert pd.isna(result.iloc[2]['value_growth_rate'])  # Previous was NaN, so no change can be calculated
    # Row 3: (130 - 120) / 120 * 100 ≈ 8.33%
    assert abs(result.iloc[3]['value_growth_rate'] - 8.333) < 0.01


def test_calculate_growth_rates_preserves_index():
    """Test that result DataFrame preserves the original index."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    df = pd.DataFrame({'value': [100, 110, 120, 130, 140]}, index=dates)
    
    engine = EDAEngine()
    result = engine.calculate_growth_rates(df, ['value'], period='monthly')
    
    # Check that index is preserved
    assert len(result) == len(df)
    assert all(result.index == df.index)


def test_calculate_summary_statistics():
    """Test summary statistics calculation."""
    # Create sample data with known statistics
    data = {
        'value1': [10, 20, 30, 40, 50],
        'value2': [100, 200, 300, 400, 500]
    }
    df = pd.DataFrame(data)
    
    engine = EDAEngine()
    stats = engine.calculate_summary_statistics(df)
    
    # Check that statistics are calculated for both columns
    assert 'value1' in stats
    assert 'value2' in stats
    
    # Check value1 statistics
    assert stats['value1']['mean'] == 30.0
    assert stats['value1']['median'] == 30.0
    assert abs(stats['value1']['std'] - 15.811) < 0.01  # Sample std
    assert stats['value1']['range'] == (10.0, 50.0)
    
    # Check value2 statistics
    assert stats['value2']['mean'] == 300.0
    assert stats['value2']['median'] == 300.0
    assert abs(stats['value2']['std'] - 158.114) < 0.01  # Sample std
    assert stats['value2']['range'] == (100.0, 500.0)


def test_calculate_summary_statistics_with_nan():
    """Test summary statistics with NaN values."""
    data = {
        'value': [10, 20, np.nan, 40, 50]
    }
    df = pd.DataFrame(data)
    
    engine = EDAEngine()
    stats = engine.calculate_summary_statistics(df)
    
    # Check that NaN values are handled (pandas ignores them by default)
    assert 'value' in stats
    assert stats['value']['mean'] == 30.0  # (10+20+40+50)/4
    assert stats['value']['median'] == 30.0
    assert stats['value']['range'] == (10.0, 50.0)


def test_calculate_summary_statistics_non_numeric_columns():
    """Test that non-numeric columns are skipped."""
    data = {
        'numeric': [10, 20, 30],
        'text': ['a', 'b', 'c']
    }
    df = pd.DataFrame(data)
    
    engine = EDAEngine()
    stats = engine.calculate_summary_statistics(df)
    
    # Only numeric column should be in results
    assert 'numeric' in stats
    assert 'text' not in stats


def test_generate_time_series_plot():
    """Test time series plot generation."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    data = {
        'rate1': [1.5, 1.6, 1.7, 1.8, 1.9],
        'rate2': [2.0, 2.1, 2.2, 2.3, 2.4]
    }
    df = pd.DataFrame(data, index=dates)
    
    engine = EDAEngine()
    fig = engine.generate_time_series_plot(df, ['rate1', 'rate2'])
    
    # Check that figure is created
    assert fig is not None
    assert len(fig.data) == 2  # Two traces
    assert fig.data[0].name == 'rate1'
    assert fig.data[1].name == 'rate2'


def test_generate_time_series_plot_with_events():
    """Test time series plot with event annotations."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    data = {
        'rate': [1.5, 1.6, 1.7, 1.8, 1.9]
    }
    df = pd.DataFrame(data, index=dates)
    
    events = [
        {'date': pd.Timestamp('2020-02-01'), 'label': 'Event 1', 'color': 'red'},
        {'date': pd.Timestamp('2020-04-01'), 'label': 'Event 2', 'color': 'blue'}
    ]
    
    engine = EDAEngine()
    fig = engine.generate_time_series_plot(df, ['rate'], events=events)
    
    # Check that figure is created with events
    assert fig is not None
    assert len(fig.data) == 1  # One trace
    # Check that vertical lines were added (shapes in layout)
    assert len(fig.layout.shapes) == 2  # Two event lines


def test_generate_time_series_plot_missing_column():
    """Test that missing column raises ValueError."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    df = pd.DataFrame({'rate1': [1.5, 1.6, 1.7, 1.8, 1.9]}, index=dates)
    
    engine = EDAEngine()
    
    with pytest.raises(ValueError, match="Columns not found in DataFrame"):
        engine.generate_time_series_plot(df, ['rate1', 'nonexistent'])


def test_generate_comparative_plot():
    """Test comparative plot generation."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    data = {
        'fed_funds': [1.5, 1.6, 1.7, 1.8, 1.9],
        'treasury_10y': [2.0, 2.1, 2.2, 2.3, 2.4],
        'total_credit': [4000, 4100, 4200, 4300, 4400],
        'revolving_credit': [1000, 1050, 1100, 1150, 1200]
    }
    df = pd.DataFrame(data, index=dates)
    
    engine = EDAEngine()
    fig = engine.generate_comparative_plot(
        df, 
        rate_cols=['fed_funds', 'treasury_10y'],
        credit_cols=['total_credit', 'revolving_credit']
    )
    
    # Check that figure is created
    assert fig is not None
    assert len(fig.data) == 4  # Four traces total
    
    # Check trace names
    trace_names = [trace.name for trace in fig.data]
    assert 'fed_funds' in trace_names
    assert 'treasury_10y' in trace_names
    assert 'total_credit' in trace_names
    assert 'revolving_credit' in trace_names


def test_generate_comparative_plot_missing_column():
    """Test that missing column raises ValueError."""
    dates = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    data = {
        'rate1': [1.5, 1.6, 1.7, 1.8, 1.9],
        'credit1': [4000, 4100, 4200, 4300, 4400]
    }
    df = pd.DataFrame(data, index=dates)
    
    engine = EDAEngine()
    
    with pytest.raises(ValueError, match="Columns not found in DataFrame"):
        engine.generate_comparative_plot(df, ['rate1'], ['credit1', 'nonexistent'])
