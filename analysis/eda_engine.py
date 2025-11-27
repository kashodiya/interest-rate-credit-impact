"""
EDAEngine class for exploratory data analysis.
"""

import pandas as pd
from typing import List, Dict, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class EDAEngine:
    """Performs exploratory data analysis and generates visualizations."""
    
    def calculate_growth_rates(self, df: pd.DataFrame, columns: List[str], period: str = 'monthly') -> pd.DataFrame:
        """
        Calculate percentage changes for specified columns.
        
        Args:
            df: DataFrame with time series data indexed by date
            columns: List of column names to calculate growth rates for
            period: Either 'monthly' or 'quarterly' for the calculation period
            
        Returns:
            DataFrame with growth rates (percentage changes) for specified columns
            
        Raises:
            ValueError: If period is not 'monthly' or 'quarterly'
            ValueError: If any specified column is not in the DataFrame
        """
        if period not in ['monthly', 'quarterly']:
            raise ValueError(f"Period must be 'monthly' or 'quarterly', got '{period}'")
        
        # Validate that all columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Create result DataFrame with same index
        result_df = pd.DataFrame(index=df.index)
        
        # Calculate growth rates for each column
        for col in columns:
            if period == 'monthly':
                # Monthly percentage change: (current - previous) / previous * 100
                result_df[f"{col}_growth_rate"] = df[col].pct_change(fill_method=None) * 100
            elif period == 'quarterly':
                # Quarterly percentage change: compare to value 3 months ago
                result_df[f"{col}_growth_rate"] = df[col].pct_change(periods=3, fill_method=None) * 100
        
        return result_df
    
    def generate_time_series_plot(self, df: pd.DataFrame, columns: List[str], events: List[dict] = None) -> go.Figure:
        """
        Generate time series line plots with optional event annotations.
        
        Args:
            df: DataFrame with time series data indexed by date
            columns: List of column names to plot
            events: Optional list of event dictionaries with keys:
                   - 'date': date of the event
                   - 'label': text label for the event
                   - 'color': optional color for the annotation (default: 'red')
                   
        Returns:
            Plotly Figure object with time series line chart
            
        Raises:
            ValueError: If any specified column is not in the DataFrame
        """
        # Validate that all columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each column
        for col in columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[col],
                mode='lines',
                name=col,
                line=dict(width=2)
            ))
        
        # Add event annotations if provided
        if events:
            shapes = []
            annotations = []
            
            for event in events:
                event_date = event.get('date')
                event_label = event.get('label', '')
                event_color = event.get('color', 'red')
                
                # Add vertical line shape
                shapes.append(dict(
                    type="line",
                    x0=event_date,
                    x1=event_date,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(
                        color=event_color,
                        width=2,
                        dash="dash"
                    ),
                    opacity=0.5
                ))
                
                # Add annotation
                annotations.append(dict(
                    x=event_date,
                    y=1,
                    yref="paper",
                    text=event_label,
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(color=event_color)
                ))
            
            fig.update_layout(shapes=shapes, annotations=annotations)
        
        # Update layout
        fig.update_layout(
            title="Time Series Plot",
            xaxis_title="Date",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, dict]:
        """
        Calculate mean, median, std, range for all numeric columns.
        
        Args:
            df: DataFrame with numeric data
            
        Returns:
            Dictionary mapping column names to statistics dictionaries containing:
            - mean: arithmetic mean
            - median: median value
            - std: standard deviation
            - range: tuple of (min, max)
        """
        statistics = {}
        
        # Get only numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        for col in numeric_cols:
            # Skip columns with all NaN values
            if df[col].isna().all():
                continue
                
            statistics[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'range': (float(df[col].min()), float(df[col].max()))
            }
        
        return statistics
    
    def generate_comparative_plot(self, df: pd.DataFrame, rate_cols: List[str], credit_cols: List[str]) -> go.Figure:
        """
        Generate comparative visualizations showing credit trends alongside interest rate changes.
        Uses dual y-axes to display both rate and credit variables on the same chart.
        
        Args:
            df: DataFrame with time series data indexed by date
            rate_cols: List of interest rate column names (plotted on left y-axis)
            credit_cols: List of credit column names (plotted on right y-axis)
            
        Returns:
            Plotly Figure object with comparative plot using dual y-axes
            
        Raises:
            ValueError: If any specified column is not in the DataFrame
        """
        # Validate that all columns exist
        missing_cols = [col for col in rate_cols + credit_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
        
        # Create figure with secondary y-axis
        from plotly.subplots import make_subplots
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add rate traces on primary y-axis
        for col in rate_cols:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(width=2)
                ),
                secondary_y=False
            )
        
        # Add credit traces on secondary y-axis
        for col in credit_cols:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode='lines',
                    name=col,
                    line=dict(width=2, dash='dash')
                ),
                secondary_y=True
            )
        
        # Update layout
        fig.update_layout(
            title="Comparative Analysis: Interest Rates vs Consumer Credit",
            hovermode='x unified',
            template='plotly_white'
        )
        
        # Set axis titles
        fig.update_xaxes(title_text="Date")
        fig.update_yaxes(title_text="Interest Rate (%)", secondary_y=False)
        fig.update_yaxes(title_text="Credit (Billions $)", secondary_y=True)
        
        return fig
