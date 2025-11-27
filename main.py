"""
Main application orchestrator for Interest Rate and Consumer Credit Analysis System.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import sys

from config import default_config
from data_processing.loader import DataLoader
from data_processing.validator import DataValidator
from data_processing.merger import DataMerger
from analysis.eda_engine import EDAEngine
from analysis.lag_analysis_engine import LagAnalysisEngine
from analysis.regression_engine import RegressionEngine
from analysis.forecast_engine import ForecastEngine
from analysis.sensitivity_analyzer import SensitivityAnalyzer


@dataclass
class AnalysisResults:
    """Container for all analysis results."""
    merged_data: Any = None
    eda_results: Optional[Dict] = None
    lag_results: Optional[Dict] = None
    regression_results: Any = None
    forecast_results: Any = None
    sensitivity_results: Optional[Dict] = None


def main():
    """
    Main analysis pipeline that coordinates:
    - Data loading, validation, and merging
    - EDA, lag analysis, regression, forecasting, and sensitivity analysis
    - Compilation of results into AnalysisResults object
    """
    print("Interest Rate and Consumer Credit Analysis System")
    print("=" * 60)
    
    results = AnalysisResults()
    
    try:
        # Step 1: Load data
        print("\n[1/7] Loading datasets...")
        loader = DataLoader()
        
        try:
            h15_data = loader.load_h15_dataset(default_config.data_paths.h15_dataset)
            print(f"  ✓ Loaded H.15 dataset: {len(h15_data)} records")
        except FileNotFoundError as e:
            print(f"  ✗ Error loading H.15 dataset: {e}")
            print(f"  Please ensure the file exists at: {default_config.data_paths.h15_dataset}")
            return None
        
        try:
            g19_data = loader.load_g19_dataset(default_config.data_paths.g19_dataset)
            print(f"  ✓ Loaded G.19 dataset: {len(g19_data)} records")
        except FileNotFoundError as e:
            print(f"  ✗ Error loading G.19 dataset: {e}")
            print(f"  Please ensure the file exists at: {default_config.data_paths.g19_dataset}")
            return None
        
        # Step 2: Validate data
        print("\n[2/7] Validating datasets...")
        validator = DataValidator()
        
        # Validate H.15 dataset
        h15_required_cols = ['fed_funds_rate', 'treasury_1y', 'treasury_10y', 'prime_rate']
        h15_col_validation = validator.validate_columns(h15_data, h15_required_cols)
        
        if not h15_col_validation.is_valid:
            print(f"  ✗ H.15 validation failed: {len(h15_col_validation.issues)} issues found")
            for issue in h15_col_validation.issues[:5]:
                print(f"    - {issue.message}")
            return None
        
        h15_numeric_validation = validator.validate_numeric_values(h15_data, h15_required_cols)
        if not h15_numeric_validation.is_valid:
            print(f"  ✗ H.15 has non-numeric values: {len(h15_numeric_validation.issues)} issues found")
            for issue in h15_numeric_validation.issues[:5]:
                print(f"    - {issue.message}")
            return None
        
        print("  ✓ H.15 dataset validated")
        
        # Validate G.19 dataset
        g19_required_cols = ['total_credit', 'revolving_credit', 'non_revolving_credit']
        g19_col_validation = validator.validate_columns(g19_data, g19_required_cols)
        
        if not g19_col_validation.is_valid:
            print(f"  ✗ G.19 validation failed: {len(g19_col_validation.issues)} issues found")
            for issue in g19_col_validation.issues[:5]:
                print(f"    - {issue.message}")
            return None
        
        g19_numeric_validation = validator.validate_numeric_values(g19_data, g19_required_cols)
        if not g19_numeric_validation.is_valid:
            print(f"  ✗ G.19 has non-numeric values: {len(g19_numeric_validation.issues)} issues found")
            for issue in g19_numeric_validation.issues[:5]:
                print(f"    - {issue.message}")
            return None
        
        print("  ✓ G.19 dataset validated")
        
        # Step 3: Merge datasets
        print("\n[3/7] Merging datasets...")
        merger = DataMerger()
        merged_data = merger.merge_datasets(h15_data, g19_data)
        print(f"  ✓ Merged dataset: {len(merged_data)} records")
        results.merged_data = merged_data
        
        # Step 4: Exploratory Data Analysis
        print("\n[4/7] Performing exploratory data analysis...")
        eda_engine = EDAEngine()
        
        # Calculate growth rates
        credit_cols = ['total_credit', 'revolving_credit', 'non_revolving_credit']
        growth_rates = eda_engine.calculate_growth_rates(merged_data, credit_cols)
        print(f"  ✓ Calculated growth rates for {len(credit_cols)} credit variables")
        
        # Calculate summary statistics
        summary_stats = eda_engine.calculate_summary_statistics(merged_data)
        print(f"  ✓ Generated summary statistics for {len(summary_stats)} variables")
        
        results.eda_results = {
            'growth_rates': growth_rates,
            'summary_stats': summary_stats
        }
        
        # Step 5: Lag Analysis
        print("\n[5/7] Performing lag analysis...")
        lag_engine = LagAnalysisEngine()
        
        lag_results = {}
        rate_cols = ['fed_funds_rate', 'treasury_10y']
        
        for rate_col in rate_cols:
            for credit_col in credit_cols[:1]:  # Just total credit for main pipeline
                if rate_col in merged_data.columns and credit_col in merged_data.columns:
                    correlations = lag_engine.compute_cross_correlation(
                        merged_data[rate_col],
                        merged_data[credit_col],
                        max_lag=default_config.analysis_params.max_lag_periods
                    )
                    optimal_lag = lag_engine.find_optimal_lag(correlations)
                    lag_results[f"{rate_col}_vs_{credit_col}"] = {
                        'optimal_lag': optimal_lag,
                        'max_correlation': correlations[optimal_lag]
                    }
                    print(f"  ✓ {rate_col} vs {credit_col}: optimal lag = {optimal_lag} months")
        
        results.lag_results = lag_results
        
        # Step 6: Regression Analysis
        print("\n[6/7] Building regression model...")
        regression_engine = RegressionEngine()
        
        try:
            # Build and fit regression model
            dependent_var = 'total_credit'
            independent_vars = ['fed_funds_rate', 'treasury_10y']
            
            # Use optimal lags from lag analysis
            lags = {}
            for var in independent_vars:
                key = f"{var}_vs_{dependent_var}"
                if key in lag_results:
                    lags[var] = lag_results[key]['optimal_lag']
            
            model = regression_engine.build_model(
                merged_data,
                dependent_var=dependent_var,
                independent_vars=independent_vars,
                lags=lags
            )
            
            regression_results = regression_engine.fit_model(model)
            print(f"  ✓ Model R-squared: {regression_results.r_squared:.4f}")
            print(f"  ✓ Durbin-Watson: {regression_results.diagnostics.durbin_watson:.4f}")
            
            results.regression_results = regression_results
        except Exception as e:
            print(f"  ✗ Regression analysis failed: {e}")
            print("  Continuing without regression results...")
        
        # Step 7: Forecasting
        print("\n[7/7] Generating forecasts...")
        forecast_engine = ForecastEngine(
            confidence_level=default_config.analysis_params.confidence_level
        )
        
        try:
            # Create ARIMA model for total credit
            arima_model = forecast_engine.create_arima_model(
                merged_data['total_credit'],
                order=default_config.analysis_params.arima_order
            )
            
            # Generate forecast
            forecast_results = forecast_engine.generate_forecast(
                arima_model,
                periods=default_config.analysis_params.forecast_periods
            )
            print(f"  ✓ Generated {default_config.analysis_params.forecast_periods}-period forecast")
            print(f"  ✓ Confidence level: {forecast_results.confidence_level*100:.0f}%")
            
            results.forecast_results = forecast_results
        except Exception as e:
            print(f"  ✗ Forecasting failed: {e}")
            print("  Continuing without forecast results...")
        
        # Success
        print("\n" + "=" * 60)
        print("✓ Analysis pipeline completed successfully!")
        print("=" * 60)
        
        return results
        
    except Exception as e:
        print(f"\n✗ Error in analysis pipeline: {e}")
        print(f"  Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    results = main()
    
    if results is None:
        print("\nAnalysis failed. Please check the errors above.")
        sys.exit(1)
    else:
        print("\nAnalysis results are ready.")
        print("\nTo launch the dashboard with these results, run:")
        print("  uv run python run_dashboard.py")
        sys.exit(0)
