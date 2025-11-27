"""
ForecastEngine class for time series forecasting using ARIMA and Prophet.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from dataclasses import dataclass
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import warnings


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for forecast evaluation."""
    mae: float  # mean absolute error
    rmse: float  # root mean squared error
    mape: float  # mean absolute percentage error


@dataclass
class ForecastResult:
    """Results from forecast generation."""
    dates: pd.DatetimeIndex
    predicted_values: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    confidence_level: float
    accuracy_metrics: Optional[AccuracyMetrics] = None


class ForecastEngine:
    """Generates time series forecasts using ARIMA or Prophet."""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the ForecastEngine.
        
        Args:
            confidence_level: Confidence level for prediction intervals (default: 0.95)
            
        Raises:
            ValueError: If confidence_level is not between 0 and 1
        """
        if not 0 < confidence_level < 1:
            raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
        
        self.confidence_level = confidence_level
    
    def create_arima_model(self, series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Create ARIMA model using statsmodels.
        
        Args:
            series: Time series data to model
            order: ARIMA order tuple (p, d, q) where:
                   p = autoregressive order
                   d = differencing order
                   q = moving average order
                   
        Returns:
            Fitted ARIMA model object
            
        Raises:
            ValueError: If series is too short or contains all NaN values
            ValueError: If order parameters are negative
        """
        # Validate order parameters
        if any(x < 0 for x in order):
            raise ValueError(f"ARIMA order parameters must be non-negative, got {order}")
        
        # Remove NaN values
        series_clean = series.dropna()
        
        if len(series_clean) == 0:
            raise ValueError("Series contains only NaN values")
        
        # Need at least p+d+q+1 observations
        min_obs = sum(order) + 1
        if len(series_clean) < min_obs:
            raise ValueError(
                f"Series too short for ARIMA{order}. Need at least {min_obs} observations, "
                f"got {len(series_clean)}"
            )
        
        # Fit ARIMA model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = ARIMA(series_clean, order=order)
            fitted_model = model.fit()
        
        return fitted_model
    
    def generate_forecast(self, model, periods: int) -> ForecastResult:
        """
        Generate forecast with confidence intervals.
        
        Args:
            model: Fitted ARIMA or Prophet model
            periods: Number of periods to forecast
            
        Returns:
            ForecastResult object containing predictions and confidence intervals
            
        Raises:
            ValueError: If periods is not positive
            TypeError: If model is not a supported type
        """
        if periods <= 0:
            raise ValueError(f"Periods must be positive, got {periods}")
        
        # Check model type and generate forecast accordingly
        if hasattr(model, 'forecast'):  # ARIMA model
            return self._generate_arima_forecast(model, periods)
        elif hasattr(model, 'predict'):  # Prophet model
            return self._generate_prophet_forecast(model, periods)
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")
    
    def _generate_arima_forecast(self, model, periods: int) -> ForecastResult:
        """Generate forecast from ARIMA model."""
        # Get forecast with confidence intervals
        forecast_result = model.forecast(steps=periods, alpha=1 - self.confidence_level)
        
        # Extract forecast values
        predicted_values = forecast_result
        
        # Get prediction intervals
        pred_intervals = model.get_forecast(steps=periods, alpha=1 - self.confidence_level)
        conf_int = pred_intervals.conf_int()
        
        # Create date index for forecast
        last_date = model.data.dates[-1]
        if isinstance(last_date, pd.Timestamp):
            freq = pd.infer_freq(model.data.dates)
            if freq is None:
                # Default to monthly if frequency cannot be inferred
                freq = 'MS'
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq=freq)[1:]
        else:
            # If dates are not timestamps, create integer index
            forecast_dates = pd.RangeIndex(start=len(model.data.dates), stop=len(model.data.dates) + periods)
        
        # Create series with proper index
        predicted_series = pd.Series(predicted_values.values, index=forecast_dates)
        lower_series = pd.Series(conf_int.iloc[:, 0].values, index=forecast_dates)
        upper_series = pd.Series(conf_int.iloc[:, 1].values, index=forecast_dates)
        
        return ForecastResult(
            dates=forecast_dates,
            predicted_values=predicted_series,
            lower_bound=lower_series,
            upper_bound=upper_series,
            confidence_level=self.confidence_level
        )
    
    def _generate_prophet_forecast(self, model, periods: int) -> ForecastResult:
        """Generate forecast from Prophet model."""
        # Create future dataframe
        future = model.make_future_dataframe(periods=periods, freq='MS')
        
        # Generate forecast
        forecast = model.predict(future)
        
        # Extract only the future predictions (not historical)
        forecast_only = forecast.tail(periods)
        
        # Create date index
        forecast_dates = pd.DatetimeIndex(forecast_only['ds'])
        
        # Extract predictions and confidence intervals
        predicted_series = pd.Series(forecast_only['yhat'].values, index=forecast_dates)
        lower_series = pd.Series(forecast_only['yhat_lower'].values, index=forecast_dates)
        upper_series = pd.Series(forecast_only['yhat_upper'].values, index=forecast_dates)
        
        return ForecastResult(
            dates=forecast_dates,
            predicted_values=predicted_series,
            lower_bound=lower_series,
            upper_bound=upper_series,
            confidence_level=self.confidence_level
        )

    def create_prophet_model(self, df: pd.DataFrame):
        """
        Create Prophet model as alternative to ARIMA.
        
        Args:
            df: DataFrame with columns 'ds' (date) and 'y' (value)
                Or a Series with datetime index (will be converted)
                
        Returns:
            Fitted Prophet model object
            
        Raises:
            ValueError: If df is empty or missing required columns
        """
        # Handle Series input - convert to Prophet format
        if isinstance(df, pd.Series):
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("Series must have DatetimeIndex")
            
            prophet_df = pd.DataFrame({
                'ds': df.index,
                'y': df.values
            })
        else:
            # Validate DataFrame has required columns
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("DataFrame must have 'ds' (date) and 'y' (value) columns")
            prophet_df = df.copy()
        
        # Remove NaN values
        prophet_df = prophet_df.dropna()
        
        if len(prophet_df) == 0:
            raise ValueError("DataFrame contains only NaN values")
        
        # Create and fit Prophet model
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            model = Prophet(
                interval_width=self.confidence_level,
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False
            )
            model.fit(prophet_df)
        
        return model
    
    def simulate_scenario(self, model, rate_change: float, periods: int, 
                         base_series: Optional[pd.Series] = None) -> ForecastResult:
        """
        Simulate credit projections for user-specified interest rate changes.
        
        This method adjusts forecasts based on rate changes by applying a simple
        elasticity-based adjustment to the baseline forecast.
        
        Args:
            model: Fitted ARIMA or Prophet model
            rate_change: Interest rate change in percentage points (e.g., 1.0 for +1%, -1.0 for -1%)
            periods: Number of periods to forecast
            base_series: Optional baseline series for elasticity calculation
            
        Returns:
            ForecastResult with adjusted predictions based on rate scenario
            
        Raises:
            ValueError: If periods is not positive
        """
        if periods <= 0:
            raise ValueError(f"Periods must be positive, got {periods}")
        
        # Generate baseline forecast
        baseline_forecast = self.generate_forecast(model, periods)
        
        # Apply rate change adjustment
        # Simple approach: assume linear relationship where 1% rate change affects credit by some factor
        # This is a simplified scenario simulation - in practice, you'd use elasticity estimates
        # from regression models or historical analysis
        
        # Default elasticity assumption: -0.5 (1% rate increase -> 0.5% credit decrease)
        elasticity = -0.5
        
        # Calculate adjustment factor
        adjustment_factor = 1 + (elasticity * rate_change / 100)
        
        # Apply adjustment to predictions
        adjusted_predictions = baseline_forecast.predicted_values * adjustment_factor
        adjusted_lower = baseline_forecast.lower_bound * adjustment_factor
        adjusted_upper = baseline_forecast.upper_bound * adjustment_factor
        
        return ForecastResult(
            dates=baseline_forecast.dates,
            predicted_values=adjusted_predictions,
            lower_bound=adjusted_lower,
            upper_bound=adjusted_upper,
            confidence_level=self.confidence_level
        )
    
    def calculate_accuracy_metrics(self, actual: pd.Series, predicted: pd.Series) -> AccuracyMetrics:
        """
        Calculate forecast accuracy metrics (MAE, RMSE, MAPE).
        
        Args:
            actual: Series of actual observed values
            predicted: Series of predicted values
            
        Returns:
            AccuracyMetrics object containing MAE, RMSE, and MAPE
            
        Raises:
            ValueError: If actual and predicted have different lengths
            ValueError: If series are empty or contain only NaN values
        """
        # Align series and remove NaN values
        # Check if indices match exactly
        if not actual.index.equals(predicted.index):
            # Try to align by index
            common_index = actual.index.intersection(predicted.index)
            if len(common_index) == 0:
                raise ValueError("Actual and predicted series have no overlapping indices")
            actual = actual.loc[common_index]
            predicted = predicted.loc[common_index]
        
        # Remove NaN values
        valid_mask = ~(actual.isna() | predicted.isna())
        actual_clean = actual[valid_mask]
        predicted_clean = predicted[valid_mask]
        
        if len(actual_clean) == 0:
            raise ValueError("No valid (non-NaN) values to compare")
        
        # Calculate errors
        errors = actual_clean - predicted_clean
        abs_errors = np.abs(errors)
        
        # Mean Absolute Error
        mae = float(np.mean(abs_errors))
        
        # Root Mean Squared Error
        rmse = float(np.sqrt(np.mean(errors ** 2)))
        
        # Mean Absolute Percentage Error
        # Avoid division by zero
        non_zero_mask = (actual_clean != 0).values
        if non_zero_mask.sum() > 0:
            mape = float(np.mean(np.abs(errors.values[non_zero_mask] / actual_clean.values[non_zero_mask]) * 100))
        else:
            mape = float('inf')
        
        return AccuracyMetrics(mae=mae, rmse=rmse, mape=mape)
    
    def generate_forecast_plot(self, historical: pd.Series, forecast_result: ForecastResult,
                              title: str = "Forecast with Confidence Intervals") -> go.Figure:
        """
        Generate forecast visualization with historical data and confidence bands.
        
        Args:
            historical: Historical time series data
            forecast_result: ForecastResult object from generate_forecast()
            title: Plot title
            
        Returns:
            Plotly Figure object with forecast visualization
            
        Raises:
            ValueError: If historical series is empty
        """
        if len(historical) == 0:
            raise ValueError("Historical series is empty")
        
        # Create figure
        fig = go.Figure()
        
        # Add historical data
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        ))
        
        # Add forecast
        fig.add_trace(go.Scatter(
            x=forecast_result.dates,
            y=forecast_result.predicted_values.values,
            mode='lines',
            name='Forecast',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        # Add confidence interval as shaded area
        fig.add_trace(go.Scatter(
            x=forecast_result.dates,
            y=forecast_result.upper_bound.values,
            mode='lines',
            name=f'{int(forecast_result.confidence_level * 100)}% CI Upper',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_result.dates,
            y=forecast_result.lower_bound.values,
            mode='lines',
            name=f'{int(forecast_result.confidence_level * 100)}% CI',
            line=dict(width=0),
            fillcolor='rgba(255, 0, 0, 0.2)',
            fill='tonexty',
            showlegend=True
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
