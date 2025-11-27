# Interest Rate and Consumer Credit Analysis System

A Python-based data analysis application that examines the relationship between federal interest rates and consumer borrowing behavior using Federal Reserve datasets (H.15 and G.19).

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package and environment management.

### Install uv

If you don't have `uv` installed:

```bash
# On Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# On macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install Dependencies

```bash
# uv will automatically create a virtual environment and install dependencies
uv sync

# Or if you prefer to use requirements.txt
uv pip install -r requirements.txt
```

Note: With `uv`, you don't need to manually activate the virtual environment. Just use `uv run` to execute commands.

## Project Structure

```
interest-rate-credit-analysis/
├── data_processing/       # Data loading, validation, and merging
├── analysis/              # Statistical analysis engines
├── dashboard/             # Plotly Dash web interface
├── tests/                 # Unit and property-based tests
├── data/                  # Federal Reserve datasets (H.15, G.19)
├── output/                # Analysis results and exports
├── config.py              # Configuration settings
└── main.py                # Main application orchestrator
```

## Quick Start

### 1. Download Real Federal Reserve Data (Automated)

The easiest way to get started is to download real data from FRED (Federal Reserve Economic Data):

```bash
# Download data from 2015 to present
uv run python download_data.py --start 2015-01-01

# Or download from 2010 for more historical data
uv run python download_data.py --start 2010-01-01

# Or specify a custom date range
uv run python download_data.py --start 2020-01-01 --end 2024-12-31
```

This will automatically download:
- **H.15 Interest Rates**: Fed Funds Rate, Treasury Yields (1Y, 2Y, 10Y), Prime Rate
- **G.19 Consumer Credit**: Total, Revolving, and Non-Revolving Credit

Files are saved to:
- `data/h15_interest_rates.csv`
- `data/g19_consumer_credit.csv`

**Note**: Sample data is included in `data/sample_*.csv` files for testing.

### 2. Run the Analysis Pipeline

The analysis pipeline loads data, performs validation, merging, EDA, lag analysis, regression, and forecasting:

```bash
uv run python main.py
```

This will:
- Load H.15 (interest rates) and G.19 (consumer credit) datasets
- Validate data quality
- Merge datasets on matching time periods
- Perform exploratory data analysis
- Calculate cross-correlations and optimal lags
- Build regression models
- Generate forecasts

### 5. Launch the Interactive Dashboard

After running the analysis, launch the dashboard to explore results interactively:

```bash
uv run python run_dashboard.py
```

The dashboard will be available at: http://127.0.0.1:8050

The dashboard includes:
- **Time Series Panel**: Interactive charts showing interest rates and credit trends
- **Correlation Panel**: Heatmaps and scatter plots exploring relationships
- **Regression Panel**: Model coefficients, R-squared, and diagnostic tests
- **Forecast Panel**: Predicted credit trends with confidence intervals
- **Scenario Simulator**: Interactive sliders to adjust rate assumptions and see projected impacts

### 3. Alternative: Manual Data Download

If you prefer to download data manually or the automated script doesn't work:

1. **Visit Federal Reserve Data Download Portal:**
   - H.15: https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15
   - G.19: https://www.federalreserve.gov/datadownload/Choose.aspx?rel=G19

2. **Configure download settings:**
   - Frequency: Monthly
   - File Format: CSV
   - File Structure: Dates down the first column

3. **Save files as:**
   - `data/h15_interest_rates.csv`
   - `data/g19_consumer_credit.csv`

See `DOWNLOAD_DATA_GUIDE.md` for detailed manual download instructions.

### 4. Using Sample Data

Sample datasets are included for testing:
- `data/sample_h15_interest_rates.csv` - Interest rates from 2020-2023
- `data/sample_g19_consumer_credit.csv` - Consumer credit from 2020-2023

To use sample data instead of real data, update `config.py`:

```python
# In config.py
h15_dataset: str = "data/sample_h15_interest_rates.csv"
g19_dataset: str = "data/sample_g19_consumer_credit.csv"
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run specific test file
uv run pytest tests/test_data_processing.py

# Run property-based tests with verbose output
uv run pytest -v tests/
```

## Development Commands

```bash
# Run any Python script
uv run python script.py

# Install additional packages
uv pip install package-name

# Update dependencies
uv pip install --upgrade -r requirements.txt
```

## Data Sources

### Federal Reserve Data Download Portal

- **H.15 Selected Interest Rates**: https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15
  - Federal Funds Rate
  - Treasury Yields (1Y, 2Y, 10Y)
  - Bank Prime Loan Rate

- **G.19 Consumer Credit**: https://www.federalreserve.gov/datadownload/Choose.aspx?rel=G19
  - Total Consumer Credit Outstanding
  - Revolving Credit (credit cards)
  - Non-Revolving Credit (auto loans, student loans)

See `data/README.md` for detailed download instructions and data format specifications.

## Analytical Techniques

### 1. Exploratory Data Analysis (EDA)
- **Time Series Visualization**: Line plots showing interest rates and credit trends over time
- **Growth Rate Calculation**: Monthly and quarterly percentage changes
- **Summary Statistics**: Mean, median, standard deviation, and range for all variables
- **Comparative Analysis**: Side-by-side visualization of rates and credit

### 2. Time Series Correlation & Lag Analysis
- **Cross-Correlation**: Compute correlation coefficients for lag periods 0-12 months
- **Optimal Lag Detection**: Identify the lag that produces maximum correlation
- **Statistical Significance**: P-value calculation for correlations
- **Visualization**: Correlation vs. lag plots

### 3. Regression Modeling
- **Model Structure**: Credit growth as dependent variable, interest rates as independent variables
- **Lagged Variables**: Incorporate optimal lags from correlation analysis
- **Diagnostics**: Durbin-Watson, Breusch-Pagan, Jarque-Bera tests
- **Model Evaluation**: R-squared, adjusted R-squared, coefficient p-values
- **Visualization**: Actual vs. predicted values

### 4. Forecasting
- **ARIMA Models**: Time series forecasting with confidence intervals
- **Prophet Models**: Alternative forecasting approach
- **Scenario Simulation**: Adjust interest rate assumptions to see projected credit impacts
- **Accuracy Metrics**: MAE, RMSE, MAPE on historical test data
- **Visualization**: Forecasts with confidence bands

### 5. Sensitivity Analysis
- **Elasticity Calculation**: Measure credit responsiveness to rate changes
- **Category Comparison**: Compare revolving vs. non-revolving credit sensitivity
- **Threshold Detection**: Identify non-linear effects at specific rate levels

## Interpreting Results

### Understanding Lag Analysis
- **Optimal Lag = 0**: Credit responds immediately to rate changes
- **Optimal Lag = 3**: Credit responds with a 3-month delay
- **Negative Correlation**: Higher rates → Lower credit growth (expected)
- **Positive Correlation**: May indicate other economic factors at play

### Understanding Regression Results
- **R-squared > 0.7**: Strong explanatory power
- **P-value < 0.05**: Statistically significant relationship
- **Durbin-Watson ≈ 2.0**: No autocorrelation (good)
- **Negative Coefficients**: Higher rates → Lower credit (expected)

### Understanding Forecasts
- **Confidence Intervals**: 95% confidence means we expect actual values to fall within the band 95% of the time
- **Wider Bands**: More uncertainty in predictions
- **MAE/RMSE**: Lower values indicate better forecast accuracy

## Exporting Results

The system can export analysis results in various formats:

```python
from dashboard.export_manager import ExportManager

export_manager = ExportManager()

# Export datasets
export_manager.export_dataset(df, "processed_data.csv")

# Export visualizations
export_manager.export_visualization(fig, "chart.png", dpi=300)

# Export model results
export_manager.export_model_results(regression_results, "model.json")

# Export forecasts
export_manager.export_forecast(forecast_results, "forecast.csv")
```

Exports are saved to the `output/exports/` directory by default.

## Configuration

Edit `config.py` to customize:

- **Data Paths**: Input and output file locations
- **Analysis Parameters**: 
  - Max lag periods (default: 12 months)
  - Forecast periods (default: 12 months)
  - Confidence level (default: 95%)
  - ARIMA order (default: (1,1,1))
- **Visualization Settings**: Figure dimensions, DPI, colors
- **Economic Events**: Dates and labels for chart annotations

## Troubleshooting

### "File not found" errors
- Ensure data files exist in the `data/` directory
- Check that file paths in `config.py` are correct
- Use sample data files for testing

### Dashboard won't start
- Ensure port 8050 is not in use by another application
- Try a different port: `dashboard.run(port=8051)`
- Check that analysis completed successfully first

### Tests failing
- Run `uv sync` to ensure all dependencies are installed
- Check Python version (requires 3.9+)
- Some property-based tests use randomization and may occasionally fail

### Memory issues with large datasets
- Reduce the date range of your datasets
- Increase system RAM allocation
- Process data in chunks if needed
