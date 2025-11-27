# Configuration Examples

This document provides examples of common configuration scenarios for the Interest Rate and Consumer Credit Analysis System.

## Basic Configuration

The default configuration in `config.py` is suitable for most use cases:

```python
from config import default_config

# Access configuration
print(f"H.15 Dataset: {default_config.data_paths.h15_dataset}")
print(f"Max Lag: {default_config.analysis_params.max_lag_periods}")
```

## Custom Data Paths

### Using Your Own Datasets

```python
from config import Config, DataPaths

# Create custom data paths
custom_paths = DataPaths(
    h15_dataset="data/my_h15_data.csv",
    g19_dataset="data/my_g19_data.csv",
    output_dir="my_output",
    exports_dir="my_output/exports"
)

# Create config with custom paths
my_config = Config(data_paths=custom_paths)
```

### Using Absolute Paths

```python
from pathlib import Path

custom_paths = DataPaths(
    h15_dataset=str(Path.home() / "Documents" / "data" / "h15.csv"),
    g19_dataset=str(Path.home() / "Documents" / "data" / "g19.csv")
)
```

## Analysis Parameters

### Short-Term Analysis (3-Month Lag)

For analyzing short-term effects:

```python
from config import Config, AnalysisParameters

short_term_params = AnalysisParameters(
    max_lag_periods=3,  # Only look at 0-3 month lags
    forecast_periods=6,  # 6-month forecast
    confidence_level=0.90  # 90% confidence intervals
)

config = Config(analysis_params=short_term_params)
```

### Long-Term Analysis (24-Month Lag)

For analyzing long-term effects:

```python
long_term_params = AnalysisParameters(
    max_lag_periods=24,  # Look at up to 2-year lags
    forecast_periods=24,  # 2-year forecast
    confidence_level=0.95,
    arima_order=(2, 1, 2)  # More complex ARIMA model
)

config = Config(analysis_params=long_term_params)
```

### High-Precision Forecasting

For more accurate forecasts with tighter confidence intervals:

```python
precision_params = AnalysisParameters(
    forecast_periods=12,
    confidence_level=0.99,  # 99% confidence (wider bands)
    arima_order=(3, 1, 3),  # More complex model
    test_size=0.3  # Use more data for testing
)
```

## Visualization Settings

### High-Resolution Exports

For publication-quality visualizations:

```python
viz_params = AnalysisParameters(
    figure_width=1600,
    figure_height=900,
    export_dpi=600  # High DPI for print
)
```

### Compact Visualizations

For presentations or web display:

```python
compact_params = AnalysisParameters(
    figure_width=800,
    figure_height=450,
    export_dpi=150  # Lower DPI for web
)
```

## Economic Events

### Custom Event Annotations

Add your own economic events to visualizations:

```python
custom_events = [
    {"date": "2008-09-15", "label": "Lehman Brothers Collapse"},
    {"date": "2020-03-11", "label": "COVID-19 Pandemic"},
    {"date": "2022-03-16", "label": "Fed Rate Hike Cycle"},
    {"date": "2023-03-10", "label": "Silicon Valley Bank Failure"}
]

event_params = AnalysisParameters(
    economic_events=custom_events
)
```

### No Event Annotations

To disable event annotations:

```python
no_events_params = AnalysisParameters(
    economic_events=[]
)
```

## Complete Custom Configuration

### Example: Financial Crisis Analysis

Analyzing the 2008 financial crisis period:

```python
from config import Config, DataPaths, AnalysisParameters

crisis_config = Config(
    data_paths=DataPaths(
        h15_dataset="data/h15_2006_2010.csv",
        g19_dataset="data/g19_2006_2010.csv",
        output_dir="output/crisis_analysis"
    ),
    analysis_params=AnalysisParameters(
        max_lag_periods=18,
        forecast_periods=12,
        confidence_level=0.95,
        arima_order=(2, 1, 2),
        economic_events=[
            {"date": "2007-08-09", "label": "BNP Paribas Freezes Funds"},
            {"date": "2008-03-16", "label": "Bear Stearns Collapse"},
            {"date": "2008-09-15", "label": "Lehman Brothers Bankruptcy"},
            {"date": "2008-10-03", "label": "TARP Signed"},
            {"date": "2009-03-09", "label": "Market Bottom"}
        ]
    )
)
```

### Example: COVID-19 Impact Analysis

Analyzing the pandemic's impact on credit:

```python
covid_config = Config(
    data_paths=DataPaths(
        h15_dataset="data/h15_2019_2023.csv",
        g19_dataset="data/g19_2019_2023.csv",
        output_dir="output/covid_analysis"
    ),
    analysis_params=AnalysisParameters(
        max_lag_periods=12,
        forecast_periods=18,
        confidence_level=0.90,
        economic_events=[
            {"date": "2020-03-11", "label": "WHO Declares Pandemic"},
            {"date": "2020-03-27", "label": "CARES Act Signed"},
            {"date": "2020-04-01", "label": "Unemployment Peak"},
            {"date": "2021-03-11", "label": "American Rescue Plan"},
            {"date": "2022-03-16", "label": "Fed Begins Rate Hikes"}
        ]
    )
)
```

### Example: Recent Rate Hike Cycle

Analyzing the 2022-2023 rate hike cycle:

```python
rate_hike_config = Config(
    data_paths=DataPaths(
        h15_dataset="data/h15_2021_2024.csv",
        g19_dataset="data/g19_2021_2024.csv",
        output_dir="output/rate_hike_analysis"
    ),
    analysis_params=AnalysisParameters(
        max_lag_periods=6,  # Shorter lag for recent data
        forecast_periods=12,
        confidence_level=0.95,
        arima_order=(1, 1, 1),
        economic_events=[
            {"date": "2022-03-16", "label": "First Rate Hike (+0.25%)"},
            {"date": "2022-05-04", "label": "Rate Hike (+0.50%)"},
            {"date": "2022-06-15", "label": "Rate Hike (+0.75%)"},
            {"date": "2022-07-27", "label": "Rate Hike (+0.75%)"},
            {"date": "2022-09-21", "label": "Rate Hike (+0.75%)"},
            {"date": "2022-11-02", "label": "Rate Hike (+0.75%)"},
            {"date": "2022-12-14", "label": "Rate Hike (+0.50%)"},
            {"date": "2023-02-01", "label": "Rate Hike (+0.25%)"}
        ]
    )
)
```

## Using Custom Configuration in Code

### In Main Script

```python
from config import Config, DataPaths, AnalysisParameters

# Create custom config
my_config = Config(
    data_paths=DataPaths(
        h15_dataset="data/my_data.csv",
        g19_dataset="data/my_credit.csv"
    ),
    analysis_params=AnalysisParameters(
        max_lag_periods=6,
        forecast_periods=12
    )
)

# Use in data loading
from data_processing.loader import DataLoader

loader = DataLoader()
h15_data = loader.load_h15_dataset(my_config.data_paths.h15_dataset)
g19_data = loader.load_g19_dataset(my_config.data_paths.g19_dataset)
```

### In Analysis Engines

```python
from analysis.lag_analysis_engine import LagAnalysisEngine

# Use config parameters
lag_engine = LagAnalysisEngine()
correlations = lag_engine.compute_cross_correlation(
    series1,
    series2,
    max_lag=my_config.analysis_params.max_lag_periods
)
```

### In Forecasting

```python
from analysis.forecast_engine import ForecastEngine

forecast_engine = ForecastEngine(
    confidence_level=my_config.analysis_params.confidence_level
)

forecast = forecast_engine.generate_forecast(
    series,
    periods=my_config.analysis_params.forecast_periods,
    method='arima'
)
```

## Environment-Specific Configuration

### Development Configuration

```python
dev_config = Config(
    data_paths=DataPaths(
        h15_dataset="data/sample_h15_interest_rates.csv",
        g19_dataset="data/sample_g19_consumer_credit.csv",
        output_dir="output/dev"
    ),
    analysis_params=AnalysisParameters(
        max_lag_periods=6,
        forecast_periods=6,
        export_dpi=150  # Lower quality for faster dev
    )
)
```

### Production Configuration

```python
prod_config = Config(
    data_paths=DataPaths(
        h15_dataset="data/production/h15_full.csv",
        g19_dataset="data/production/g19_full.csv",
        output_dir="output/production",
        exports_dir="output/production/exports"
    ),
    analysis_params=AnalysisParameters(
        max_lag_periods=12,
        forecast_periods=12,
        confidence_level=0.95,
        export_dpi=600  # High quality for reports
    )
)
```

## Tips

1. **Start with defaults**: The default configuration works well for most analyses
2. **Adjust lag periods**: Use shorter lags (3-6) for recent data, longer (12-24) for historical
3. **Confidence levels**: 95% is standard, use 90% for tighter bands, 99% for wider
4. **ARIMA order**: Start with (1,1,1), increase complexity if needed
5. **Test size**: 20% is standard, increase to 30% for more robust validation
6. **Export DPI**: 300 for print, 150 for web, 600 for publication
7. **Economic events**: Add events relevant to your analysis period

## Validation

Always validate your configuration before running analysis:

```python
def validate_config(config):
    """Validate configuration parameters."""
    assert config.analysis_params.max_lag_periods > 0, "Max lag must be positive"
    assert 0 < config.analysis_params.confidence_level < 1, "Confidence must be between 0 and 1"
    assert config.analysis_params.forecast_periods > 0, "Forecast periods must be positive"
    assert config.analysis_params.export_dpi > 0, "DPI must be positive"
    print("✓ Configuration validated successfully")

validate_config(my_config)
```
