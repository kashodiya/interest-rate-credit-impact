"""
Unit tests for ForecastEngine class.
"""

import pytest
import pandas as pd
import numpy as np
from analysis.forecast_engine import ForecastEngine, ForecastResult, AccuracyMetrics


def test_forecast_engine_initialization():
    """Test ForecastEngine initialization with valid confidence level."""
    engine = ForecastEngine(confidence_level=0.95)
    assert engine.confidence_level == 0.95


def test_forecast_engine_invalid_confidence_level():
    """Test ForecastEngine raises error for invalid confidence level."""
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
        ForecastEngine(confidence_level=1.5)
    
    with pytest.raises(ValueError, match="Confidence level must be between 0 and 1"):
        ForecastEngine(confidence_level=0)


def test_create_arima_model_basic():
    """Test creating ARIMA model with valid time series."""
    engine = ForecastEngine()
    
    # Create sample time series
    dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    values = np.random.randn(50).cumsum() + 100
    series = pd.Series(values, index=dates)
    
    # Create ARIMA model
    model = engine.create_arima_model(series, order=(1, 1, 1))
    
    assert model is not None
    assert hasattr(model, 'forecast')


def test_create_arima_model_too_short():
    """Test ARIMA model raises error for series too short."""
    engine = ForecastEngine()
    
    # Create very short series
    dates = pd.date_range(start='2020-01-01', periods=2, freq='MS')
    series = pd.Series([1.0, 2.0], index=dates)
    
    with pytest.raises(ValueError, match="Series too short"):
        engine.create_arima_model(series, order=(1, 1, 1))


def test_create_arima_model_all_nan():
    """Test ARIMA model raises error for all NaN series."""
    engine = ForecastEngine()
    
    dates = pd.date_range(start='2020-01-01', periods=10, freq='MS')
    series = pd.Series([np.nan] * 10, index=dates)
    
    with pytest.raises(ValueError, match="contains only NaN values"):
        engine.create_arima_model(series, order=(1, 1, 1))


def test_create_arima_model_negative_order():
    """Test ARIMA model raises error for negative order parameters."""
    engine = ForecastEngine()
    
    dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    series = pd.Series(np.random.randn(50), index=dates)
    
    with pytest.raises(ValueError, match="order parameters must be non-negative"):
        engine.create_arima_model(series, order=(-1, 1, 1))


def test_generate_forecast_arima():
    """Test generating forecast from ARIMA model."""
    engine = ForecastEngine(confidence_level=0.95)
    
    # Create sample time series
    dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    values = np.random.randn(50).cumsum() + 100
    series = pd.Series(values, index=dates)
    
    # Create and fit model
    model = engine.create_arima_model(series, order=(1, 1, 1))
    
    # Generate forecast
    forecast = engine.generate_forecast(model, periods=12)
    
    assert isinstance(forecast, ForecastResult)
    assert len(forecast.predicted_values) == 12
    assert len(forecast.lower_bound) == 12
    assert len(forecast.upper_bound) == 12
    assert forecast.confidence_level == 0.95


def test_generate_forecast_invalid_periods():
    """Test forecast raises error for invalid periods."""
    engine = ForecastEngine()
    
    dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    series = pd.Series(np.random.randn(50), index=dates)
    model = engine.create_arima_model(series, order=(1, 1, 1))
    
    with pytest.raises(ValueError, match="Periods must be positive"):
        engine.generate_forecast(model, periods=0)
    
    with pytest.raises(ValueError, match="Periods must be positive"):
        engine.generate_forecast(model, periods=-5)


def test_create_prophet_model_from_series():
    """Test creating Prophet model from Series."""
    engine = ForecastEngine()
    
    # Create sample time series
    dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    values = np.random.randn(50).cumsum() + 100
    series = pd.Series(values, index=dates)
    
    # Create Prophet model
    model = engine.create_prophet_model(series)
    
    assert model is not None
    assert hasattr(model, 'predict')


def test_create_prophet_model_from_dataframe():
    """Test creating Prophet model from DataFrame."""
    engine = ForecastEngine()
    
    # Create sample dataframe in Prophet format
    dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    df = pd.DataFrame({
        'ds': dates,
        'y': np.random.randn(50).cumsum() + 100
    })
    
    # Create Prophet model
    model = engine.create_prophet_model(df)
    
    assert model is not None
    assert hasattr(model, 'predict')


def test_create_prophet_model_invalid_dataframe():
    """Test Prophet model raises error for invalid DataFrame."""
    engine = ForecastEngine()
    
    # DataFrame without required columns
    df = pd.DataFrame({
        'date': pd.date_range(start='2020-01-01', periods=10, freq='MS'),
        'value': np.random.randn(10)
    })
    
    with pytest.raises(ValueError, match="must have 'ds' .* and 'y'"):
        engine.create_prophet_model(df)


def test_create_prophet_model_series_without_datetime_index():
    """Test Prophet model raises error for Series without DatetimeIndex."""
    engine = ForecastEngine()
    
    series = pd.Series([1, 2, 3, 4, 5])
    
    with pytest.raises(ValueError, match="must have DatetimeIndex"):
        engine.create_prophet_model(series)


def test_generate_forecast_prophet():
    """Test generating forecast from Prophet model."""
    engine = ForecastEngine(confidence_level=0.95)
    
    # Create sample time series
    dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    values = np.random.randn(50).cumsum() + 100
    series = pd.Series(values, index=dates)
    
    # Create and fit model
    model = engine.create_prophet_model(series)
    
    # Generate forecast
    forecast = engine.generate_forecast(model, periods=12)
    
    assert isinstance(forecast, ForecastResult)
    assert len(forecast.predicted_values) == 12
    assert len(forecast.lower_bound) == 12
    assert len(forecast.upper_bound) == 12
    assert forecast.confidence_level == 0.95


def test_simulate_scenario():
    """Test scenario simulation with rate changes."""
    engine = ForecastEngine()
    
    # Create sample time series
    dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    values = np.random.randn(50).cumsum() + 100
    series = pd.Series(values, index=dates)
    
    # Create model
    model = engine.create_arima_model(series, order=(1, 1, 1))
    
    # Simulate scenario with +1% rate change
    scenario_forecast = engine.simulate_scenario(model, rate_change=1.0, periods=12)
    
    assert isinstance(scenario_forecast, ForecastResult)
    assert len(scenario_forecast.predicted_values) == 12


def test_simulate_scenario_negative_rate_change():
    """Test scenario simulation with negative rate change."""
    engine = ForecastEngine()
    
    dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    values = np.random.randn(50).cumsum() + 100
    series = pd.Series(values, index=dates)
    
    model = engine.create_arima_model(series, order=(1, 1, 1))
    
    # Simulate scenario with -1% rate change
    scenario_forecast = engine.simulate_scenario(model, rate_change=-1.0, periods=12)
    
    assert isinstance(scenario_forecast, ForecastResult)
    assert len(scenario_forecast.predicted_values) == 12


def test_calculate_accuracy_metrics():
    """Test calculating forecast accuracy metrics."""
    engine = ForecastEngine()
    
    # Create actual and predicted series
    dates = pd.date_range(start='2020-01-01', periods=20, freq='MS')
    actual = pd.Series([100, 102, 105, 103, 107, 110, 108, 112, 115, 113,
                       117, 120, 118, 122, 125, 123, 127, 130, 128, 132], index=dates)
    predicted = pd.Series([101, 103, 104, 104, 108, 109, 109, 113, 114, 114,
                          118, 119, 119, 123, 124, 124, 128, 129, 129, 133], index=dates)
    
    # Calculate metrics
    metrics = engine.calculate_accuracy_metrics(actual, predicted)
    
    assert isinstance(metrics, AccuracyMetrics)
    assert metrics.mae > 0
    assert metrics.rmse > 0
    assert metrics.mape > 0
    assert metrics.rmse >= metrics.mae  # RMSE is always >= MAE


def test_calculate_accuracy_metrics_perfect_prediction():
    """Test accuracy metrics for perfect predictions."""
    engine = ForecastEngine()
    
    dates = pd.date_range(start='2020-01-01', periods=10, freq='MS')
    actual = pd.Series([100, 102, 105, 103, 107, 110, 108, 112, 115, 113], index=dates)
    predicted = actual.copy()
    
    metrics = engine.calculate_accuracy_metrics(actual, predicted)
    
    assert metrics.mae == 0.0
    assert metrics.rmse == 0.0
    assert metrics.mape == 0.0


def test_calculate_accuracy_metrics_mismatched_lengths():
    """Test accuracy metrics with mismatched series lengths."""
    engine = ForecastEngine()
    
    dates1 = pd.date_range(start='2020-01-01', periods=10, freq='MS')
    dates2 = pd.date_range(start='2020-01-01', periods=5, freq='MS')
    
    actual = pd.Series(np.random.randn(10), index=dates1)
    predicted = pd.Series(np.random.randn(5), index=dates2)
    
    # Should work by aligning on common index
    metrics = engine.calculate_accuracy_metrics(actual, predicted)
    assert isinstance(metrics, AccuracyMetrics)


def test_calculate_accuracy_metrics_no_overlap():
    """Test accuracy metrics raises error for non-overlapping series."""
    engine = ForecastEngine()
    
    dates1 = pd.date_range(start='2020-01-01', periods=10, freq='MS')
    dates2 = pd.date_range(start='2021-01-01', periods=10, freq='MS')
    
    actual = pd.Series(np.random.randn(10), index=dates1)
    predicted = pd.Series(np.random.randn(10), index=dates2)
    
    with pytest.raises(ValueError, match="no overlapping indices"):
        engine.calculate_accuracy_metrics(actual, predicted)


def test_calculate_accuracy_metrics_with_nan():
    """Test accuracy metrics handles NaN values."""
    engine = ForecastEngine()
    
    dates = pd.date_range(start='2020-01-01', periods=10, freq='MS')
    actual = pd.Series([100, np.nan, 105, 103, 107, 110, np.nan, 112, 115, 113], index=dates)
    predicted = pd.Series([101, 103, np.nan, 104, 108, 109, 109, np.nan, 114, 114], index=dates)
    
    # Should work by removing NaN values
    metrics = engine.calculate_accuracy_metrics(actual, predicted)
    assert isinstance(metrics, AccuracyMetrics)


def test_generate_forecast_plot():
    """Test generating forecast visualization."""
    engine = ForecastEngine()
    
    # Create historical data
    hist_dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    historical = pd.Series(np.random.randn(50).cumsum() + 100, index=hist_dates)
    
    # Create model and forecast
    model = engine.create_arima_model(historical, order=(1, 1, 1))
    forecast = engine.generate_forecast(model, periods=12)
    
    # Generate plot
    fig = engine.generate_forecast_plot(historical, forecast)
    
    assert fig is not None
    assert len(fig.data) == 4  # historical, forecast, upper bound, lower bound


def test_generate_forecast_plot_empty_historical():
    """Test forecast plot raises error for empty historical series."""
    engine = ForecastEngine()
    
    historical = pd.Series([], dtype=float)
    
    # Create dummy forecast result
    dates = pd.date_range(start='2020-01-01', periods=12, freq='MS')
    forecast = ForecastResult(
        dates=dates,
        predicted_values=pd.Series(np.random.randn(12), index=dates),
        lower_bound=pd.Series(np.random.randn(12), index=dates),
        upper_bound=pd.Series(np.random.randn(12), index=dates),
        confidence_level=0.95
    )
    
    with pytest.raises(ValueError, match="Historical series is empty"):
        engine.generate_forecast_plot(historical, forecast)


def test_generate_forecast_plot_custom_title():
    """Test forecast plot with custom title."""
    engine = ForecastEngine()
    
    hist_dates = pd.date_range(start='2020-01-01', periods=50, freq='MS')
    historical = pd.Series(np.random.randn(50).cumsum() + 100, index=hist_dates)
    
    model = engine.create_arima_model(historical, order=(1, 1, 1))
    forecast = engine.generate_forecast(model, periods=12)
    
    custom_title = "Custom Forecast Title"
    fig = engine.generate_forecast_plot(historical, forecast, title=custom_title)
    
    assert custom_title in fig.layout.title.text
