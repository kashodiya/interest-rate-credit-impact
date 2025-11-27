"""
ExportManager class for exporting analysis results.
"""

import pandas as pd
import plotly.graph_objects as go
import json
from pathlib import Path
from datetime import datetime
from typing import Union


class ExportManager:
    """Handles exporting analysis results in various formats."""
    
    def __init__(self, output_dir: str = "output/exports"):
        """
        Initialize ExportManager.
        
        Args:
            output_dir: Base directory for exports (default: "output/exports")
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_dataset(self, df: pd.DataFrame, filepath: str, format: str = "csv"):
        """
        Save processed datasets in CSV format with timestamps.
        
        Args:
            df: DataFrame to export
            filepath: Path where file should be saved (relative to output_dir or absolute)
            format: Export format, currently only "csv" is supported
            
        Raises:
            ValueError: If format is not supported
            ValueError: If DataFrame is empty
        """
        if df.empty:
            raise ValueError("Cannot export empty DataFrame")
        
        if format.lower() != "csv":
            raise ValueError(f"Unsupported format '{format}'. Only 'csv' is supported for datasets.")
        
        # Resolve filepath
        filepath = self._resolve_filepath(filepath)
        
        # Add timestamp to filename if not already present
        filepath = self._add_timestamp_to_path(filepath)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        df.to_csv(filepath, index=True)
        
        return str(filepath)
    
    def export_visualization(self, fig: go.Figure, filepath: str, 
                            format: str = "png", dpi: int = 300):
        """
        Save charts in PNG or SVG format with configurable DPI.
        
        Args:
            fig: Plotly Figure object to export
            filepath: Path where file should be saved (relative to output_dir or absolute)
            format: Export format - "png" or "svg"
            dpi: Dots per inch for PNG export (default: 300)
            
        Raises:
            ValueError: If format is not supported
            ValueError: If dpi is not positive
        """
        if format.lower() not in ["png", "svg"]:
            raise ValueError(f"Unsupported format '{format}'. Supported formats: 'png', 'svg'")
        
        if dpi <= 0:
            raise ValueError(f"DPI must be positive, got {dpi}")
        
        # Resolve filepath
        filepath = self._resolve_filepath(filepath)
        
        # Ensure correct extension
        filepath = filepath.with_suffix(f".{format.lower()}")
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format.lower() == "png":
            # Calculate width and height based on DPI
            # Plotly uses pixels, so we need to scale
            scale = dpi / 72  # 72 is the default DPI
            fig.write_image(str(filepath), format="png", scale=scale)
        else:  # svg
            fig.write_image(str(filepath), format="svg")
        
        return str(filepath)
    
    def export_model_results(self, results, filepath: str, format: str = "json"):
        """
        Generate summary reports in JSON or text format containing coefficients, 
        statistics, and diagnostics.
        
        Args:
            results: RegressionResults object or similar model results
            filepath: Path where file should be saved (relative to output_dir or absolute)
            format: Export format - "json" or "text"
            
        Raises:
            ValueError: If format is not supported
            ValueError: If results object doesn't have required attributes
        """
        if format.lower() not in ["json", "text", "txt"]:
            raise ValueError(f"Unsupported format '{format}'. Supported formats: 'json', 'text'")
        
        # Resolve filepath
        filepath = self._resolve_filepath(filepath)
        
        # Ensure correct extension
        if format.lower() == "json":
            filepath = filepath.with_suffix(".json")
        else:
            filepath = filepath.with_suffix(".txt")
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract results data
        results_dict = self._extract_model_results(results)
        
        # Export based on format
        if format.lower() == "json":
            with open(filepath, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
        else:  # text
            with open(filepath, 'w') as f:
                f.write(self._format_results_as_text(results_dict))
        
        return str(filepath)
    
    def export_forecast(self, forecast, filepath: str):
        """
        Save predicted values with confidence intervals in tabular format.
        
        Args:
            forecast: ForecastResult object containing predictions and confidence intervals
            filepath: Path where file should be saved (relative to output_dir or absolute)
            
        Raises:
            ValueError: If forecast object doesn't have required attributes
        """
        # Resolve filepath
        filepath = self._resolve_filepath(filepath)
        
        # Ensure CSV extension
        filepath = filepath.with_suffix(".csv")
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Create DataFrame from forecast
        forecast_df = self._extract_forecast_data(forecast)
        
        # Export to CSV
        forecast_df.to_csv(filepath, index=True)
        
        return str(filepath)
    
    def _resolve_filepath(self, filepath: str) -> Path:
        """
        Resolve filepath relative to output_dir if not absolute.
        
        Args:
            filepath: File path as string
            
        Returns:
            Resolved Path object
        """
        path = Path(filepath)
        if not path.is_absolute():
            path = self.output_dir / path
        return path
    
    def _add_timestamp_to_path(self, filepath: Path) -> Path:
        """
        Add timestamp to filename if not already present.
        
        Args:
            filepath: Path object
            
        Returns:
            Path with timestamp added to stem
        """
        # Check if filename already has a timestamp pattern
        stem = filepath.stem
        if not any(char.isdigit() for char in stem[-8:]):
            # Add timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_stem = f"{stem}_{timestamp}"
            filepath = filepath.with_stem(new_stem)
        return filepath
    
    def _extract_model_results(self, results) -> dict:
        """
        Extract model results into a dictionary.
        
        Args:
            results: RegressionResults or similar object
            
        Returns:
            Dictionary containing model results
            
        Raises:
            ValueError: If results object doesn't have required attributes
        """
        results_dict = {
            "export_timestamp": datetime.now().isoformat(),
            "model_type": type(results).__name__
        }
        
        # Try to extract common attributes
        if hasattr(results, 'coefficients'):
            results_dict['coefficients'] = results.coefficients
        
        if hasattr(results, 'r_squared'):
            results_dict['r_squared'] = results.r_squared
        
        if hasattr(results, 'adjusted_r_squared'):
            results_dict['adjusted_r_squared'] = results.adjusted_r_squared
        
        if hasattr(results, 'p_values'):
            results_dict['p_values'] = results.p_values
        
        if hasattr(results, 'diagnostics'):
            diag = results.diagnostics
            results_dict['diagnostics'] = {
                'durbin_watson': diag.durbin_watson if hasattr(diag, 'durbin_watson') else None,
                'breusch_pagan_p': diag.breusch_pagan_p if hasattr(diag, 'breusch_pagan_p') else None,
                'jarque_bera_p': diag.jarque_bera_p if hasattr(diag, 'jarque_bera_p') else None
            }
        
        # If no attributes were found, raise error
        if len(results_dict) == 2:  # Only timestamp and model_type
            raise ValueError("Results object doesn't have any recognized attributes (coefficients, r_squared, etc.)")
        
        return results_dict
    
    def _format_results_as_text(self, results_dict: dict) -> str:
        """
        Format results dictionary as human-readable text.
        
        Args:
            results_dict: Dictionary containing model results
            
        Returns:
            Formatted text string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL RESULTS SUMMARY")
        lines.append("=" * 60)
        lines.append(f"Export Timestamp: {results_dict.get('export_timestamp', 'N/A')}")
        lines.append(f"Model Type: {results_dict.get('model_type', 'N/A')}")
        lines.append("")
        
        # Coefficients
        if 'coefficients' in results_dict:
            lines.append("COEFFICIENTS:")
            lines.append("-" * 60)
            for var, coef in results_dict['coefficients'].items():
                lines.append(f"  {var:30s}: {coef:12.6f}")
            lines.append("")
        
        # Model Statistics
        lines.append("MODEL STATISTICS:")
        lines.append("-" * 60)
        if 'r_squared' in results_dict:
            lines.append(f"  R-squared:                    {results_dict['r_squared']:12.6f}")
        if 'adjusted_r_squared' in results_dict:
            lines.append(f"  Adjusted R-squared:           {results_dict['adjusted_r_squared']:12.6f}")
        lines.append("")
        
        # P-values
        if 'p_values' in results_dict:
            lines.append("P-VALUES:")
            lines.append("-" * 60)
            for var, pval in results_dict['p_values'].items():
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                lines.append(f"  {var:30s}: {pval:12.6f} {sig}")
            lines.append("")
        
        # Diagnostics
        if 'diagnostics' in results_dict:
            lines.append("DIAGNOSTIC TESTS:")
            lines.append("-" * 60)
            diag = results_dict['diagnostics']
            if diag.get('durbin_watson') is not None:
                lines.append(f"  Durbin-Watson:                {diag['durbin_watson']:12.6f}")
            if diag.get('breusch_pagan_p') is not None:
                lines.append(f"  Breusch-Pagan p-value:        {diag['breusch_pagan_p']:12.6f}")
            if diag.get('jarque_bera_p') is not None:
                lines.append(f"  Jarque-Bera p-value:          {diag['jarque_bera_p']:12.6f}")
            lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def _extract_forecast_data(self, forecast) -> pd.DataFrame:
        """
        Extract forecast data into a DataFrame.
        
        Args:
            forecast: ForecastResult object
            
        Returns:
            DataFrame with forecast data
            
        Raises:
            ValueError: If forecast object doesn't have required attributes
        """
        required_attrs = ['dates', 'predicted_values', 'lower_bound', 'upper_bound']
        missing_attrs = [attr for attr in required_attrs if not hasattr(forecast, attr)]
        
        if missing_attrs:
            raise ValueError(f"Forecast object missing required attributes: {missing_attrs}")
        
        # Create DataFrame
        df = pd.DataFrame({
            'predicted_value': forecast.predicted_values.values,
            'lower_bound': forecast.lower_bound.values,
            'upper_bound': forecast.upper_bound.values
        }, index=forecast.dates)
        
        # Add confidence level if available
        if hasattr(forecast, 'confidence_level'):
            df.attrs['confidence_level'] = forecast.confidence_level
        
        # Add accuracy metrics if available
        if hasattr(forecast, 'accuracy_metrics') and forecast.accuracy_metrics is not None:
            metrics = forecast.accuracy_metrics
            df.attrs['mae'] = metrics.mae if hasattr(metrics, 'mae') else None
            df.attrs['rmse'] = metrics.rmse if hasattr(metrics, 'rmse') else None
            df.attrs['mape'] = metrics.mape if hasattr(metrics, 'mape') else None
        
        return df
