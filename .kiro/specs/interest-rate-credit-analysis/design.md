# Design Document

## Overview

The Interest Rate and Consumer Credit Analysis System is a Python-based data analysis application that examines the relationship between federal interest rates and consumer borrowing behavior. The system ingests Federal Reserve datasets (H.15 and G.19), performs comprehensive statistical analysis including exploratory data analysis, time series correlation with lag detection, multivariate regression modeling, and forecasting. Results are presented through an interactive dashboard built with Plotly Dash, enabling analysts to explore insights, test scenarios, and export findings.

The system follows a modular architecture with clear separation between data ingestion, analysis engines, and presentation layers. This design ensures maintainability, testability, and extensibility for future analytical techniques.

## Architecture

The system is organized into the following layers:

```
┌─────────────────────────────────────────────────────────┐
│                    Dashboard Layer                       │
│              (Plotly Dash Web Interface)                 │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                   Analysis Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐│
│  │   EDA    │  │   Lag    │  │Regression│  │Forecast │││
│  │  Engine  │  │ Analysis │  │  Engine  │  │ Engine  │││
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘││
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                  Data Processing Layer                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   Loader     │  │  Validator   │  │    Merger    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                    Data Storage                          │
│              (CSV files from Fed Reserve)                │
└─────────────────────────────────────────────────────────┘
```

**Layer Responsibilities:**

- **Data Processing Layer**: Loads, validates, and merges Federal Reserve datasets
- **Analysis Layer**: Performs statistical computations and modeling
- **Dashboard Layer**: Provides interactive visualization and user controls

## Components and Interfaces

### 1. Data Processing Components

#### DataLoader
Responsible for reading Federal Reserve datasets from CSV files.

```python
class DataLoader:
    def load_h15_dataset(self, filepath: str) -> pd.DataFrame
    def load_g19_dataset(self, filepath: str) -> pd.DataFrame
```

**Inputs**: File paths to H.15 and G.19 CSV files
**Outputs**: Pandas DataFrames with parsed time series data
**Dependencies**: pandas library

#### DataValidator
Validates dataset integrity and reports issues.

```python
class DataValidator:
    def validate_columns(self, df: pd.DataFrame, required_columns: List[str]) -> ValidationResult
    def validate_numeric_values(self, df: pd.DataFrame, columns: List[str]) -> ValidationResult
    def detect_missing_data(self, df: pd.DataFrame) -> ValidationResult
```

**Inputs**: DataFrames and validation rules
**Outputs**: ValidationResult objects containing pass/fail status and issue details
**Dependencies**: pandas, numpy

#### DataMerger
Merges H.15 and G.19 datasets on time periods.

```python
class DataMerger:
    def merge_datasets(self, h15_df: pd.DataFrame, g19_df: pd.DataFrame) -> pd.DataFrame
    def align_time_periods(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]
```

**Inputs**: Two DataFrames with time series data
**Outputs**: Single merged DataFrame with aligned time periods
**Dependencies**: pandas

### 2. Analysis Components

#### EDAEngine
Performs exploratory data analysis and generates visualizations.

```python
class EDAEngine:
    def calculate_growth_rates(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame
    def generate_time_series_plot(self, df: pd.DataFrame, columns: List[str], events: List[Event]) -> Figure
    def calculate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Statistics]
    def generate_comparative_plot(self, df: pd.DataFrame, rate_cols: List[str], credit_cols: List[str]) -> Figure
```

**Inputs**: Merged dataset, column specifications, economic event markers
**Outputs**: Statistical summaries and Plotly Figure objects
**Dependencies**: pandas, numpy, plotly

#### LagAnalysisEngine
Performs cross-correlation analysis to detect lag effects.

```python
class LagAnalysisEngine:
    def compute_cross_correlation(self, series1: pd.Series, series2: pd.Series, max_lag: int) -> np.ndarray
    def find_optimal_lag(self, correlations: np.ndarray) -> int
    def test_significance(self, correlation: float, n_samples: int) -> float
    def generate_lag_plot(self, lags: np.ndarray, correlations: np.ndarray) -> Figure
```

**Inputs**: Time series pairs, maximum lag period
**Outputs**: Correlation coefficients, optimal lag values, significance levels, visualizations
**Dependencies**: numpy, scipy, plotly

#### RegressionEngine
Builds and evaluates multivariate regression models.

```python
class RegressionEngine:
    def build_model(self, df: pd.DataFrame, dependent_var: str, independent_vars: List[str], lags: Dict[str, int]) -> RegressionModel
    def fit_model(self, model: RegressionModel) -> RegressionResults
    def calculate_diagnostics(self, results: RegressionResults) -> Diagnostics
    def generate_prediction_plot(self, actual: pd.Series, predicted: pd.Series) -> Figure
```

**Inputs**: Dataset, variable specifications, lag periods
**Outputs**: Fitted model, coefficients, R-squared, p-values, residuals, diagnostics
**Dependencies**: statsmodels, pandas, numpy

#### ForecastEngine
Generates time series forecasts using ARIMA or Prophet.

```python
class ForecastEngine:
    def create_arima_model(self, series: pd.Series, order: Tuple[int, int, int]) -> ARIMAModel
    def create_prophet_model(self, df: pd.DataFrame) -> ProphetModel
    def generate_forecast(self, model: Union[ARIMAModel, ProphetModel], periods: int) -> ForecastResult
    def simulate_scenario(self, model: Union[ARIMAModel, ProphetModel], rate_change: float, periods: int) -> ForecastResult
    def calculate_accuracy_metrics(self, actual: pd.Series, predicted: pd.Series) -> AccuracyMetrics
```

**Inputs**: Time series data, model parameters, forecast horizon, scenario specifications
**Outputs**: Forecasted values, confidence intervals, accuracy metrics
**Dependencies**: statsmodels, prophet, pandas, numpy

#### SensitivityAnalyzer
Analyzes differential impacts across credit categories.

```python
class SensitivityAnalyzer:
    def calculate_elasticity(self, credit_series: pd.Series, rate_series: pd.Series) -> float
    def rank_sensitivities(self, elasticities: Dict[str, float]) -> List[Tuple[str, float]]
    def test_threshold_effects(self, df: pd.DataFrame, credit_col: str, rate_col: str, thresholds: List[float]) -> ThresholdResults
    def generate_sensitivity_comparison(self, sensitivities: Dict[str, float]) -> Figure
```

**Inputs**: Credit and rate time series, threshold values
**Outputs**: Elasticity measures, rankings, threshold test results, visualizations
**Dependencies**: pandas, numpy, scipy

### 3. Dashboard Components

#### DashboardApp
Main Plotly Dash application coordinating all visualizations and interactions.

```python
class DashboardApp:
    def __init__(self, analysis_results: AnalysisResults)
    def create_time_series_panel(self) -> html.Div
    def create_correlation_panel(self) -> html.Div
    def create_regression_panel(self) -> html.Div
    def create_forecast_panel(self) -> html.Div
    def create_scenario_simulator(self) -> html.Div
    def register_callbacks(self)
    def run(self, host: str, port: int)
```

**Inputs**: Compiled analysis results
**Outputs**: Interactive web dashboard
**Dependencies**: dash, plotly, pandas

#### ExportManager
Handles exporting analysis results in various formats.

```python
class ExportManager:
    def export_dataset(self, df: pd.DataFrame, filepath: str, format: str)
    def export_visualization(self, fig: Figure, filepath: str, format: str, dpi: int)
    def export_model_results(self, results: RegressionResults, filepath: str, format: str)
    def export_forecast(self, forecast: ForecastResult, filepath: str)
```

**Inputs**: Data objects, file paths, format specifications
**Outputs**: Files written to disk (CSV, PNG, SVG, JSON)
**Dependencies**: pandas, plotly, json

## Data Models

### Core Data Structures

#### TimeSeriesData
```python
@dataclass
class TimeSeriesData:
    date: pd.DatetimeIndex
    fed_funds_rate: pd.Series
    treasury_1y: pd.Series
    treasury_2y: pd.Series
    treasury_10y: pd.Series
    prime_rate: pd.Series
    total_credit: pd.Series
    revolving_credit: pd.Series
    non_revolving_credit: pd.Series
```

#### ValidationResult
```python
@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue]
    
@dataclass
class ValidationIssue:
    row: Optional[int]
    column: str
    issue_type: str  # 'missing', 'invalid_type', 'out_of_range'
    message: str
```

#### CorrelationResult
```python
@dataclass
class CorrelationResult:
    variable_pair: Tuple[str, str]
    correlations: np.ndarray  # correlation at each lag
    lags: np.ndarray
    optimal_lag: int
    max_correlation: float
    p_value: float
```

#### RegressionResults
```python
@dataclass
class RegressionResults:
    coefficients: Dict[str, float]
    r_squared: float
    adjusted_r_squared: float
    p_values: Dict[str, float]
    residuals: pd.Series
    predicted_values: pd.Series
    diagnostics: Diagnostics

@dataclass
class Diagnostics:
    durbin_watson: float  # autocorrelation test
    breusch_pagan_p: float  # heteroscedasticity test
    jarque_bera_p: float  # normality test
```

#### ForecastResult
```python
@dataclass
class ForecastResult:
    dates: pd.DatetimeIndex
    predicted_values: pd.Series
    lower_bound: pd.Series
    upper_bound: pd.Series
    confidence_level: float
    accuracy_metrics: Optional[AccuracyMetrics]

@dataclass
class AccuracyMetrics:
    mae: float  # mean absolute error
    rmse: float  # root mean squared error
    mape: float  # mean absolute percentage error
```

#### ElasticityResult
```python
@dataclass
class ElasticityResult:
    credit_category: str
    rate_variable: str
    elasticity: float
    confidence_interval: Tuple[float, float]
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Federal Reserve dataset parsing completeness
*For any* valid Federal Reserve dataset file (H.15 or G.19), parsing should extract all required columns specified for that dataset type without data loss.
**Validates: Requirements 1.1, 1.2**

### Property 2: Validation error reporting accuracy
*For any* dataset with missing or invalid data at specific locations, the validation report should identify the exact row and column positions of all issues.
**Validates: Requirements 1.4**

### Property 3: Dataset merge preserves time alignment
*For any* pair of H.15 and G.19 datasets with overlapping time periods, the merged dataset should contain only matching time periods with correctly aligned values from both sources.
**Validates: Requirements 1.5**

### Property 4: Growth rate calculation correctness
*For any* time series, the calculated growth rates should equal the percentage change between consecutive time periods.
**Validates: Requirements 2.2**

### Property 5: Summary statistics accuracy
*For any* numeric variable in a dataset, the calculated mean, median, standard deviation, and range should match the mathematically correct values for that distribution.
**Validates: Requirements 2.4**

### Property 6: Visualization data completeness
*For any* generated visualization, the plot object should contain all data series specified in the visualization request.
**Validates: Requirements 2.1, 2.5, 3.3, 4.5, 5.5, 7.4**

### Property 7: Cross-correlation lag coverage
*For any* pair of time series, cross-correlation analysis should compute correlation coefficients for all lag periods from 0 to the specified maximum lag.
**Validates: Requirements 3.1**

### Property 8: Optimal lag identification
*For any* set of correlation coefficients across lag periods, the identified optimal lag should correspond to the lag with the maximum absolute correlation value.
**Validates: Requirements 3.2**

### Property 9: Regression model structure correctness
*For any* regression model built by the system, credit growth should be the dependent variable and interest rate variables (with optional lags) should be the independent variables.
**Validates: Requirements 4.1, 4.2**

### Property 10: Regression results completeness
*For any* fitted regression model, the results should include coefficients, R-squared, p-values for all predictors, residuals, and diagnostic test results (Durbin-Watson, Breusch-Pagan, Jarque-Bera).
**Validates: Requirements 4.3, 4.4, 6.3**

### Property 11: Forecast output structure
*For any* generated forecast, the output should include predicted values, lower confidence bounds, upper confidence bounds, and the confidence level used.
**Validates: Requirements 5.2**

### Property 12: Forecast accuracy metrics calculation
*For any* forecast with corresponding actual values, the calculated MAE and RMSE should equal the mathematically correct error measures between predicted and actual values.
**Validates: Requirements 5.4**

### Property 13: Scenario simulation responsiveness
*For any* user-specified interest rate change scenario, the system should generate credit projections that reflect the specified rate adjustment.
**Validates: Requirements 5.3**

### Property 14: Elasticity calculation correctness
*For any* credit category and interest rate variable pair, the calculated elasticity should equal the percentage change in credit divided by the percentage change in rates.
**Validates: Requirements 7.1**

### Property 15: Sensitivity ranking consistency
*For any* set of credit categories with calculated correlation strengths, the ranking should order categories from highest to lowest absolute correlation with interest rates.
**Validates: Requirements 7.2**

### Property 16: Export format compliance
*For any* export request, the generated file should conform to the specified format (CSV, PNG, SVG, JSON, or text) and contain all required data elements for that export type.
**Validates: Requirements 8.1, 8.2, 8.3, 8.4**

## Error Handling

The system implements comprehensive error handling at each layer:

### Data Processing Errors

**File Not Found**: When dataset files are missing, the system shall raise a `FileNotFoundError` with the expected file path and suggest checking the Federal Reserve DDP portal URL.

**Parse Errors**: When CSV parsing fails due to malformed data, the system shall raise a `DataParseError` indicating the line number and nature of the parsing issue.

**Validation Failures**: When datasets fail validation, the system shall return a `ValidationResult` object with `is_valid=False` and a list of all detected issues, allowing the user to review and correct data problems.

**Merge Conflicts**: When H.15 and G.19 datasets have no overlapping time periods, the system shall raise a `MergeError` indicating the date ranges of each dataset.

### Analysis Errors

**Insufficient Data**: When time series are too short for requested analysis (e.g., lag analysis with max_lag exceeding series length), the system shall raise an `InsufficientDataError` specifying the minimum required data points.

**Convergence Failures**: When regression or forecast models fail to converge, the system shall raise a `ConvergenceError` with diagnostic information and suggest parameter adjustments.

**Singular Matrix**: When regression encounters multicollinearity, the system shall raise a `SingularMatrixError` and suggest removing correlated predictors.

**Invalid Parameters**: When users specify invalid parameters (e.g., negative lag values, confidence levels outside [0,1]), the system shall raise a `ValueError` with clear parameter constraints.

### Dashboard Errors

**Missing Data**: When dashboard attempts to display results before analysis is complete, the system shall display a user-friendly message indicating which analysis steps need to be run first.

**Export Failures**: When file export fails due to permissions or disk space, the system shall catch the exception and display an error message with the specific system error and suggested remedies.

### Error Recovery

The system implements graceful degradation:
- If optional visualizations fail, the system continues with numerical results
- If one credit category analysis fails, the system processes remaining categories
- If export to one format fails, the system attempts alternative formats

All errors are logged with timestamps, context, and stack traces to facilitate debugging.

## Testing Strategy

The system employs a dual testing approach combining unit tests and property-based tests to ensure comprehensive correctness validation.

### Unit Testing Approach

Unit tests verify specific examples, integration points, and edge cases:

**Data Processing Tests**:
- Loading sample H.15 and G.19 files from the Federal Reserve
- Handling empty datasets
- Merging datasets with partial time overlap
- Validating datasets with specific known issues

**Analysis Tests**:
- Computing growth rates for known time series
- Cross-correlation with manually calculated expected values
- Regression on synthetic data with known coefficients
- Forecast accuracy on historical test sets

**Dashboard Tests**:
- Rendering dashboard components with sample data
- Callback functions responding to user interactions
- Export functions creating files with expected content

**Framework**: pytest for Python unit tests

### Property-Based Testing Approach

Property-based tests verify universal properties across randomly generated inputs, providing broader coverage than example-based tests. The system uses **Hypothesis** as the property-based testing library for Python.

**Configuration**: Each property-based test shall run a minimum of 100 iterations to ensure thorough exploration of the input space.

**Test Tagging**: Each property-based test shall include a comment explicitly referencing the correctness property from this design document using the format: `# Feature: interest-rate-credit-analysis, Property {number}: {property_text}`

**Property Implementation**: Each correctness property listed in the Correctness Properties section shall be implemented by a single property-based test.

**Key Property Tests**:

1. **Parsing Properties**: Generate random valid Federal Reserve CSV structures and verify all required columns are extracted
2. **Validation Properties**: Generate datasets with random missing/invalid values and verify error reports are accurate
3. **Merge Properties**: Generate random time series pairs and verify merge alignment
4. **Calculation Properties**: Generate random time series and verify growth rates, statistics, correlations, and elasticities are mathematically correct
5. **Model Properties**: Generate random datasets and verify regression models have correct structure and complete results
6. **Forecast Properties**: Generate random time series and verify forecast outputs have required structure
7. **Export Properties**: Generate random analysis results and verify exports contain all required elements

**Generators**: Custom Hypothesis strategies shall be implemented for:
- Valid Federal Reserve dataset structures
- Time series with configurable properties (length, trend, seasonality)
- Regression model specifications
- Forecast parameters

**Test Organization**: Property-based tests shall be co-located with unit tests in test files, clearly marked with the property reference comment.

### Integration Testing

Integration tests verify end-to-end workflows:
- Loading data → EDA → visualization pipeline
- Loading data → lag analysis → regression with lagged variables
- Loading data → forecasting → scenario simulation
- Complete analysis → dashboard rendering → export

### Test Coverage Goals

- Minimum 85% code coverage for core analysis logic
- 100% coverage of error handling paths
- All 16 correctness properties implemented as property-based tests
- All critical user workflows covered by integration tests

