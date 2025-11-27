"""
SensitivityAnalyzer class for analyzing differential impacts across credit categories.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import plotly.graph_objects as go
from scipy import stats


@dataclass
class ThresholdResults:
    """Results from threshold effect testing."""
    thresholds: List[float]
    elasticities_below: List[float]
    elasticities_above: List[float]
    significant_thresholds: List[float]


class SensitivityAnalyzer:
    """Analyzes differential impacts of interest rate changes across credit categories."""
    
    def calculate_elasticity(self, credit_series: pd.Series, rate_series: pd.Series) -> float:
        """
        Calculate credit-rate elasticity measure.
        
        Elasticity measures the percentage change in credit divided by the percentage change in rates.
        This implementation uses the arc elasticity formula for time series data.
        
        Args:
            credit_series: Time series of credit values (e.g., revolving credit, non-revolving credit)
            rate_series: Time series of interest rate values
            
        Returns:
            Elasticity coefficient (percentage change in credit / percentage change in rate)
            
        Raises:
            ValueError: If series have different lengths or insufficient data points
            ValueError: If series contain only NaN values after alignment
        """
        # Validate inputs
        if len(credit_series) != len(rate_series):
            raise ValueError(
                f"Series must have the same length. Got credit: {len(credit_series)}, rate: {len(rate_series)}"
            )
        
        # Align series and remove NaN values
        valid_mask = ~(credit_series.isna() | rate_series.isna())
        credit_clean = credit_series[valid_mask]
        rate_clean = rate_series[valid_mask]
        
        if len(credit_clean) < 2:
            raise ValueError(
                f"Need at least 2 valid observations for elasticity calculation, got {len(credit_clean)}"
            )
        
        # Calculate percentage changes
        credit_pct_change = credit_clean.pct_change(fill_method=None).dropna()
        rate_pct_change = rate_clean.pct_change(fill_method=None).dropna()
        
        # Align after pct_change (which creates NaN in first row)
        valid_mask_pct = ~(credit_pct_change.isna() | rate_pct_change.isna())
        credit_pct_clean = credit_pct_change[valid_mask_pct]
        rate_pct_clean = rate_pct_change[valid_mask_pct]
        
        if len(credit_pct_clean) == 0:
            raise ValueError("No valid percentage changes to calculate elasticity")
        
        # Remove infinite values that can occur from division by zero in pct_change
        finite_mask = np.isfinite(credit_pct_clean) & np.isfinite(rate_pct_clean)
        credit_pct_finite = credit_pct_clean[finite_mask]
        rate_pct_finite = rate_pct_clean[finite_mask]
        
        if len(credit_pct_finite) == 0:
            raise ValueError("No finite percentage changes to calculate elasticity")
        
        # Calculate elasticity as the ratio of average percentage changes
        # Filter out periods where rate change is near zero to avoid extreme elasticities
        rate_threshold = 0.001  # 0.1% threshold
        significant_rate_changes = np.abs(rate_pct_finite) > rate_threshold
        
        if significant_rate_changes.sum() == 0:
            # If no significant rate changes, return 0 elasticity
            return 0.0
        
        # Calculate elasticity for periods with significant rate changes
        elasticities = credit_pct_finite[significant_rate_changes] / rate_pct_finite[significant_rate_changes]
        
        # Return mean elasticity
        elasticity = float(np.mean(elasticities))
        
        return elasticity
    
    def rank_sensitivities(self, elasticities: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Order credit types by their correlation strength with interest rate variables.
        
        Args:
            elasticities: Dictionary mapping credit category names to their elasticity values
            
        Returns:
            List of tuples (credit_category, elasticity) ordered by absolute elasticity (highest to lowest)
            
        Raises:
            ValueError: If elasticities dictionary is empty
        """
        if not elasticities:
            raise ValueError("Elasticities dictionary is empty")
        
        # Sort by absolute value of elasticity (descending)
        ranked = sorted(
            elasticities.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return ranked
    
    def test_threshold_effects(self, df: pd.DataFrame, credit_col: str, 
                              rate_col: str, thresholds: List[float]) -> ThresholdResults:
        """
        Test for non-linear threshold effects where credit response changes at specific rate levels.
        
        This method splits the data at each threshold and calculates elasticity separately
        for periods above and below the threshold to detect non-linear responses.
        
        Args:
            df: DataFrame containing credit and rate time series
            credit_col: Name of credit column
            rate_col: Name of interest rate column
            thresholds: List of rate threshold values to test (e.g., [2.0, 3.0, 5.0] for 2%, 3%, 5%)
            
        Returns:
            ThresholdResults object containing elasticities above/below each threshold
            and list of statistically significant thresholds
            
        Raises:
            ValueError: If credit_col or rate_col not in DataFrame
            ValueError: If thresholds list is empty
            ValueError: If insufficient data for threshold testing
        """
        # Validate inputs
        if credit_col not in df.columns:
            raise ValueError(f"Credit column '{credit_col}' not found in DataFrame")
        
        if rate_col not in df.columns:
            raise ValueError(f"Rate column '{rate_col}' not found in DataFrame")
        
        if not thresholds:
            raise ValueError("Thresholds list is empty")
        
        # Get series
        credit_series = df[credit_col]
        rate_series = df[rate_col]
        
        # Remove NaN values
        valid_mask = ~(credit_series.isna() | rate_series.isna())
        credit_clean = credit_series[valid_mask]
        rate_clean = rate_series[valid_mask]
        
        if len(credit_clean) < 10:
            raise ValueError(
                f"Need at least 10 valid observations for threshold testing, got {len(credit_clean)}"
            )
        
        # Test each threshold
        elasticities_below = []
        elasticities_above = []
        significant_thresholds = []
        
        for threshold in thresholds:
            # Split data at threshold
            below_mask = rate_clean <= threshold
            above_mask = rate_clean > threshold
            
            # Check if we have enough data in each segment
            if below_mask.sum() < 2 or above_mask.sum() < 2:
                # Not enough data to calculate elasticity in one or both segments
                elasticities_below.append(np.nan)
                elasticities_above.append(np.nan)
                continue
            
            # Calculate elasticity for each segment
            try:
                elasticity_below = self.calculate_elasticity(
                    credit_clean[below_mask],
                    rate_clean[below_mask]
                )
            except ValueError:
                elasticity_below = np.nan
            
            try:
                elasticity_above = self.calculate_elasticity(
                    credit_clean[above_mask],
                    rate_clean[above_mask]
                )
            except ValueError:
                elasticity_above = np.nan
            
            elasticities_below.append(elasticity_below)
            elasticities_above.append(elasticity_above)
            
            # Test if difference is significant
            # A threshold is significant if elasticities differ substantially
            if not (np.isnan(elasticity_below) or np.isnan(elasticity_above)):
                # Simple heuristic: difference > 0.5 in absolute elasticity
                if abs(elasticity_above - elasticity_below) > 0.5:
                    significant_thresholds.append(threshold)
        
        return ThresholdResults(
            thresholds=thresholds,
            elasticities_below=elasticities_below,
            elasticities_above=elasticities_above,
            significant_thresholds=significant_thresholds
        )
    
    def generate_sensitivity_comparison(self, sensitivities: Dict[str, float], 
                                       title: str = "Credit Sensitivity Comparison") -> go.Figure:
        """
        Generate comparative visualization showing response magnitudes across credit categories.
        
        Creates a bar chart comparing elasticity values across different credit types.
        
        Args:
            sensitivities: Dictionary mapping credit category names to their elasticity values
            title: Plot title
            
        Returns:
            Plotly Figure object with sensitivity comparison bar chart
            
        Raises:
            ValueError: If sensitivities dictionary is empty
        """
        if not sensitivities:
            raise ValueError("Sensitivities dictionary is empty")
        
        # Rank sensitivities for ordered display
        ranked = self.rank_sensitivities(sensitivities)
        
        # Extract categories and values
        categories = [item[0] for item in ranked]
        values = [item[1] for item in ranked]
        
        # Create color scheme based on sign of elasticity
        colors = ['red' if v < 0 else 'green' for v in values]
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker=dict(
                color=colors,
                line=dict(color='black', width=1)
            ),
            text=[f"{v:.2f}" for v in values],
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Elasticity: %{y:.3f}<extra></extra>'
        ))
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        
        # Update layout
        fig.update_layout(
            title=title,
            xaxis_title="Credit Category",
            yaxis_title="Elasticity (% change in credit / % change in rate)",
            template='plotly_white',
            showlegend=False,
            height=500
        )
        
        # Rotate x-axis labels if many categories
        if len(categories) > 5:
            fig.update_xaxes(tickangle=-45)
        
        return fig
