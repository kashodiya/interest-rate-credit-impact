"""
Configuration file for Interest Rate and Consumer Credit Analysis System.
Contains data paths and analysis parameters.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List


@dataclass
class DataPaths:
    """Paths to data files and output directories."""
    # Input data paths
    # Use real data files if available, otherwise fall back to sample files
    h15_dataset: str = "data/h15_interest_rates.csv"
    g19_dataset: str = "data/g19_consumer_credit.csv"
    
    # Output paths
    output_dir: str = "output"
    exports_dir: str = "output/exports"
    visualizations_dir: str = "output/visualizations"


@dataclass
class AnalysisParameters:
    """Parameters for analysis configuration."""
    # Lag analysis parameters
    max_lag_periods: int = 12  # Maximum lag in months
    correlation_significance_level: float = 0.05
    
    # Regression parameters
    test_size: float = 0.2  # Proportion of data for testing
    
    # Forecasting parameters
    forecast_periods: int = 12  # Number of periods to forecast
    confidence_level: float = 0.95
    arima_order: tuple = (1, 1, 1)  # (p, d, q) for ARIMA
    
    # Visualization parameters
    figure_width: int = 1200
    figure_height: int = 600
    export_dpi: int = 300
    
    # Economic events for annotation
    economic_events: List[dict] = None
    
    def __post_init__(self):
        if self.economic_events is None:
            self.economic_events = [
                {"date": "2008-09-15", "label": "Lehman Brothers Collapse"},
                {"date": "2020-03-11", "label": "COVID-19 Pandemic Declared"},
                {"date": "2022-03-16", "label": "Fed Rate Hike Cycle Begins"}
            ]


@dataclass
class Config:
    """Main configuration object."""
    data_paths: DataPaths = None
    analysis_params: AnalysisParameters = None
    
    def __post_init__(self):
        if self.data_paths is None:
            self.data_paths = DataPaths()
        if self.analysis_params is None:
            self.analysis_params = AnalysisParameters()


# Default configuration instance
default_config = Config()
