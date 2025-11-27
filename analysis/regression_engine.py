"""
RegressionEngine class for building and evaluating regression models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy import stats


@dataclass
class RegressionModel:
    """Specification for a regression model."""
    dependent_var: str
    independent_vars: List[str]
    lags: Dict[str, int]
    X: pd.DataFrame
    y: pd.Series


@dataclass
class Diagnostics:
    """Diagnostic test results for regression model."""
    durbin_watson: float
    breusch_pagan_p: float
    jarque_bera_p: float


@dataclass
class RegressionResults:
    """Results from fitted regression model."""
    coefficients: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    p_values: Dict[str, float]
    residuals: pd.Series
    predicted_values: pd.Series
    diagnostics: Diagnostics
    statsmodels_results: Optional[object] = None


class RegressionEngine:
    """Builds and evaluates multivariate regression models."""
    
    def build_model(self, df: pd.DataFrame, dependent_var: str, 
                   independent_vars: List[str], lags: Dict[str, int] = None) -> RegressionModel:
        """
        Build regression model specification with optional lagged variables.
        
        Args:
            df: DataFrame with time series data indexed by date
            dependent_var: Name of the dependent variable (credit growth)
            independent_vars: List of independent variable names (interest rates)
            lags: Optional dictionary mapping variable names to their optimal lag periods
                 e.g., {'fed_funds_rate': 3} means use fed_funds_rate lagged by 3 periods
                 
        Returns:
            RegressionModel object containing model specification and prepared data
            
        Raises:
            ValueError: If dependent_var or any independent_vars not in DataFrame
            ValueError: If lagged variable would result in insufficient data
        """
        if lags is None:
            lags = {}
        
        # Validate that dependent variable exists
        if dependent_var not in df.columns:
            raise ValueError(f"Dependent variable '{dependent_var}' not found in DataFrame")
        
        # Validate that all independent variables exist
        missing_vars = [var for var in independent_vars if var not in df.columns]
        if missing_vars:
            raise ValueError(f"Independent variables not found in DataFrame: {missing_vars}")
        
        # Create X matrix with lagged variables
        X_data = {}
        max_lag = max(lags.values()) if lags else 0
        
        for var in independent_vars:
            lag_period = lags.get(var, 0)
            
            if lag_period > 0:
                # Create lagged variable
                lagged_series = df[var].shift(lag_period)
                X_data[f"{var}_lag{lag_period}"] = lagged_series
            else:
                # Use variable without lag
                X_data[var] = df[var]
        
        # Create X DataFrame
        X = pd.DataFrame(X_data, index=df.index)
        
        # Get y (dependent variable)
        y = df[dependent_var]
        
        # Remove rows with NaN values (from lagging or original data)
        valid_mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        # Check if we have sufficient data after removing NaN
        if len(X_clean) < len(independent_vars) + 1:
            raise ValueError(
                f"Insufficient data after applying lags. Need at least {len(independent_vars) + 1} "
                f"observations, but only have {len(X_clean)}"
            )
        
        # Add constant term for intercept
        X_clean = sm.add_constant(X_clean)
        
        return RegressionModel(
            dependent_var=dependent_var,
            independent_vars=independent_vars,
            lags=lags,
            X=X_clean,
            y=y_clean
        )
    
    def fit_model(self, model: RegressionModel) -> RegressionResults:
        """
        Fit regression model using statsmodels OLS.
        
        Args:
            model: RegressionModel object from build_model()
            
        Returns:
            RegressionResults object containing coefficients, statistics, and diagnostics
            
        Raises:
            ValueError: If model fitting fails
        """
        # Fit OLS model
        ols_model = sm.OLS(model.y, model.X)
        results = ols_model.fit()
        
        # Extract coefficients
        coefficients = dict(zip(model.X.columns, results.params))
        
        # Extract p-values
        p_values = dict(zip(model.X.columns, results.pvalues))
        
        # Get predicted values and residuals
        predicted_values = pd.Series(results.fittedvalues, index=model.y.index)
        residuals = pd.Series(results.resid, index=model.y.index)
        
        # Calculate diagnostics (will be implemented in task 5.3)
        diagnostics = self.calculate_diagnostics(results)
        
        return RegressionResults(
            coefficients=coefficients,
            r_squared=float(results.rsquared),
            adjusted_r_squared=float(results.rsquared_adj),
            p_values=p_values,
            residuals=residuals,
            predicted_values=predicted_values,
            diagnostics=diagnostics,
            statsmodels_results=results
        )
    
    def calculate_diagnostics(self, results) -> Diagnostics:
        """
        Calculate diagnostic tests for regression model.
        
        Performs three key diagnostic tests:
        - Durbin-Watson: Tests for autocorrelation in residuals (values near 2 indicate no autocorrelation)
        - Breusch-Pagan: Tests for heteroscedasticity (p > 0.05 suggests homoscedasticity)
        - Jarque-Bera: Tests for normality of residuals (p > 0.05 suggests normal distribution)
        
        Args:
            results: Fitted statsmodels OLS results object
            
        Returns:
            Diagnostics object containing test statistics and p-values
        """
        # Durbin-Watson test for autocorrelation
        dw_stat = durbin_watson(results.resid)
        
        # Breusch-Pagan test for heteroscedasticity
        # Returns: (lm_stat, lm_pvalue, f_stat, f_pvalue)
        bp_test = het_breuschpagan(results.resid, results.model.exog)
        bp_pvalue = bp_test[1]
        
        # Jarque-Bera test for normality (already in statsmodels results)
        jb_stat, jb_pvalue = stats.jarque_bera(results.resid)
        
        return Diagnostics(
            durbin_watson=float(dw_stat),
            breusch_pagan_p=float(bp_pvalue),
            jarque_bera_p=float(jb_pvalue)
        )
    
    def generate_prediction_plot(self, actual: pd.Series, predicted: pd.Series) -> go.Figure:
        """
        Generate visualization showing actual versus predicted values.
        
        Creates a plot with:
        - Actual values as a line
        - Predicted values as a line
        - Perfect prediction reference line (45-degree line)
        
        Args:
            actual: Series of actual observed values
            predicted: Series of predicted values from the model
            
        Returns:
            Plotly Figure object with actual vs predicted plot
            
        Raises:
            ValueError: If actual and predicted have different lengths or indices
        """
        # Validate inputs
        if len(actual) != len(predicted):
            raise ValueError(f"Actual and predicted must have same length. Got {len(actual)} and {len(predicted)}")
        
        # Align indices if they differ
        if not actual.index.equals(predicted.index):
            # Find common index
            common_index = actual.index.intersection(predicted.index)
            if len(common_index) == 0:
                raise ValueError("Actual and predicted series have no overlapping indices")
            actual = actual.loc[common_index]
            predicted = predicted.loc[common_index]
        
        # Create figure with two subplots: time series and scatter
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Time Series: Actual vs Predicted", "Scatter: Actual vs Predicted"),
            horizontal_spacing=0.12
        )
        
        # Left plot: Time series
        fig.add_trace(
            go.Scatter(
                x=actual.index,
                y=actual.values,
                mode='lines',
                name='Actual',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=predicted.index,
                y=predicted.values,
                mode='lines',
                name='Predicted',
                line=dict(color='red', width=2, dash='dash')
            ),
            row=1, col=1
        )
        
        # Right plot: Scatter plot with 45-degree reference line
        fig.add_trace(
            go.Scatter(
                x=actual.values,
                y=predicted.values,
                mode='markers',
                name='Predictions',
                marker=dict(color='green', size=6, opacity=0.6),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Add 45-degree reference line (perfect prediction)
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Regression Model: Actual vs Predicted Values",
            template='plotly_white',
            showlegend=True,
            height=500,
            hovermode='closest'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="Actual", row=1, col=2)
        fig.update_yaxes(title_text="Predicted", row=1, col=2)
        
        return fig
