"""
LagAnalysisEngine class for cross-correlation and lag analysis.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional
import plotly.graph_objects as go


class LagAnalysisEngine:
    """Performs cross-correlation analysis to detect lag effects between time series."""
    
    def __init__(self, max_lag: int = 12):
        """
        Initialize the LagAnalysisEngine.
        
        Args:
            max_lag: Maximum lag period to test (default: 12 months)
        """
        self.max_lag = max_lag
    
    def compute_cross_correlation(self, series1: pd.Series, series2: pd.Series, max_lag: Optional[int] = None) -> np.ndarray:
        """
        Compute cross-correlation between two time series for lag periods from 0 to max_lag.
        
        Args:
            series1: First time series (typically the independent variable, e.g., interest rate)
            series2: Second time series (typically the dependent variable, e.g., credit)
            max_lag: Maximum lag period to compute (uses instance default if not specified)
            
        Returns:
            Array of correlation coefficients for lags 0 to max_lag
            
        Raises:
            ValueError: If series have different lengths or insufficient data points
        """
        if max_lag is None:
            max_lag = self.max_lag
        
        # Validate inputs
        if len(series1) != len(series2):
            raise ValueError(f"Series must have the same length. Got {len(series1)} and {len(series2)}")
        
        if len(series1) <= max_lag:
            raise ValueError(f"Series length ({len(series1)}) must be greater than max_lag ({max_lag})")
        
        # Remove NaN values - align both series
        valid_mask = ~(series1.isna() | series2.isna())
        s1_clean = series1[valid_mask].values
        s2_clean = series2[valid_mask].values
        
        if len(s1_clean) <= max_lag:
            raise ValueError(f"After removing NaN values, series length ({len(s1_clean)}) must be greater than max_lag ({max_lag})")
        
        # Compute correlations for each lag
        correlations = np.zeros(max_lag + 1)
        
        for lag in range(max_lag + 1):
            if lag == 0:
                # No lag - standard correlation
                correlations[lag] = np.corrcoef(s1_clean, s2_clean)[0, 1]
            else:
                # Lag series1 by 'lag' periods
                # series1[t-lag] vs series2[t]
                s1_lagged = s1_clean[:-lag]
                s2_current = s2_clean[lag:]
                
                # Compute correlation
                if len(s1_lagged) > 1:
                    correlations[lag] = np.corrcoef(s1_lagged, s2_current)[0, 1]
                else:
                    correlations[lag] = np.nan
        
        return correlations
    
    def find_optimal_lag(self, correlations: np.ndarray) -> int:
        """
        Identify the lag value that produces the maximum absolute correlation.
        
        Args:
            correlations: Array of correlation coefficients indexed by lag
            
        Returns:
            Lag value (index) with maximum absolute correlation
            
        Raises:
            ValueError: If correlations array is empty or all NaN
        """
        if len(correlations) == 0:
            raise ValueError("Correlations array is empty")
        
        # Find index of maximum absolute correlation, ignoring NaN values
        abs_correlations = np.abs(correlations)
        
        # Check if all values are NaN
        if np.all(np.isnan(abs_correlations)):
            raise ValueError("All correlation values are NaN")
        
        # Find the index of the maximum absolute correlation
        optimal_lag = int(np.nanargmax(abs_correlations))
        
        return optimal_lag
    
    def test_significance(self, correlation: float, n_samples: int, alpha: float = 0.05) -> float:
        """
        Calculate p-value for correlation coefficient significance test.
        
        Uses Fisher's z-transformation to test if correlation is significantly different from zero.
        
        Args:
            correlation: Correlation coefficient to test
            n_samples: Number of samples used to compute the correlation
            alpha: Significance level (default: 0.05)
            
        Returns:
            P-value for the correlation coefficient
            
        Raises:
            ValueError: If n_samples is less than 3 or correlation is outside [-1, 1]
        """
        if n_samples < 3:
            raise ValueError(f"Need at least 3 samples for significance test, got {n_samples}")
        
        if not -1 <= correlation <= 1:
            raise ValueError(f"Correlation must be between -1 and 1, got {correlation}")
        
        # Handle edge cases where correlation is exactly -1 or 1
        if abs(correlation) >= 0.9999:
            return 0.0
        
        # Use t-test for correlation coefficient
        # t = r * sqrt(n-2) / sqrt(1-r^2)
        # where r is the correlation coefficient and n is the number of samples
        t_stat = correlation * np.sqrt(n_samples - 2) / np.sqrt(1 - correlation**2)
        
        # Two-tailed test
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_samples - 2))
        
        return float(p_value)
    
    def generate_lag_plot(self, lags: np.ndarray, correlations: np.ndarray, 
                         variable_pair: tuple = None, optimal_lag: int = None) -> go.Figure:
        """
        Generate visualization showing correlation strength as a function of lag period.
        
        Args:
            lags: Array of lag values
            correlations: Array of correlation coefficients corresponding to each lag
            variable_pair: Optional tuple of (series1_name, series2_name) for plot title
            optimal_lag: Optional optimal lag value to highlight on the plot
            
        Returns:
            Plotly Figure object with lag correlation plot
            
        Raises:
            ValueError: If lags and correlations have different lengths
        """
        if len(lags) != len(correlations):
            raise ValueError(f"Lags and correlations must have the same length. Got {len(lags)} and {len(correlations)}")
        
        # Create figure
        fig = go.Figure()
        
        # Add correlation line
        fig.add_trace(go.Scatter(
            x=lags,
            y=correlations,
            mode='lines+markers',
            name='Correlation',
            line=dict(width=2, color='blue'),
            marker=dict(size=8)
        ))
        
        # Highlight optimal lag if provided
        if optimal_lag is not None and 0 <= optimal_lag < len(correlations):
            fig.add_trace(go.Scatter(
                x=[lags[optimal_lag]],
                y=[correlations[optimal_lag]],
                mode='markers',
                name=f'Optimal Lag ({optimal_lag})',
                marker=dict(size=15, color='red', symbol='star')
            ))
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Update layout
        title = "Cross-Correlation vs Lag Period"
        if variable_pair:
            title = f"Cross-Correlation: {variable_pair[0]} vs {variable_pair[1]}"
        
        fig.update_layout(
            title=title,
            xaxis_title="Lag Period (months)",
            yaxis_title="Correlation Coefficient",
            hovermode='x unified',
            template='plotly_white',
            showlegend=True
        )
        
        # Set y-axis range to [-1, 1] for correlation
        fig.update_yaxes(range=[-1.1, 1.1])
        
        return fig
