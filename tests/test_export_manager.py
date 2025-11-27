"""
Unit tests for ExportManager class.
"""

import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import json
import tempfile
import shutil
from datetime import datetime

from dashboard.export_manager import ExportManager
from analysis.regression_engine import RegressionResults, Diagnostics
from analysis.forecast_engine import ForecastResult, AccuracyMetrics


@pytest.fixture
def temp_output_dir():
    """Create a temporary output directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def export_manager(temp_output_dir):
    """Create ExportManager instance with temporary directory."""
    return ExportManager(output_dir=temp_output_dir)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range('2020-01-01', periods=10, freq='MS')
    return pd.DataFrame({
        'value1': np.random.randn(10),
        'value2': np.random.randn(10)
    }, index=dates)


@pytest.fixture
def sample_figure():
    """Create a sample Plotly figure for testing."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines'))
    return fig


@pytest.fixture
def sample_regression_results():
    """Create sample regression results for testing."""
    dates = pd.date_range('2020-01-01', periods=10, freq='MS')
    return RegressionResults(
        coefficients={'const': 1.5, 'var1': 0.8, 'var2': -0.3},
        r_squared=0.85,
        adjusted_r_squared=0.82,
        p_values={'const': 0.001, 'var1': 0.01, 'var2': 0.05},
        residuals=pd.Series(np.random.randn(10), index=dates),
        predicted_values=pd.Series(np.random.randn(10), index=dates),
        diagnostics=Diagnostics(
            durbin_watson=1.95,
            breusch_pagan_p=0.15,
            jarque_bera_p=0.25
        )
    )


@pytest.fixture
def sample_forecast_result():
    """Create sample forecast result for testing."""
    dates = pd.date_range('2024-01-01', periods=12, freq='MS')
    values = np.random.randn(12) + 100
    return ForecastResult(
        dates=dates,
        predicted_values=pd.Series(values, index=dates),
        lower_bound=pd.Series(values - 5, index=dates),
        upper_bound=pd.Series(values + 5, index=dates),
        confidence_level=0.95,
        accuracy_metrics=AccuracyMetrics(mae=2.5, rmse=3.2, mape=1.8)
    )


class TestExportDataset:
    """Tests for export_dataset method."""
    
    def test_export_csv_basic(self, export_manager, sample_dataframe):
        """Test basic CSV export functionality."""
        filepath = export_manager.export_dataset(sample_dataframe, "test_data.csv")
        
        # Verify file was created
        assert Path(filepath).exists()
        
        # Verify content can be read back
        df_loaded = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert len(df_loaded) == len(sample_dataframe)
        assert list(df_loaded.columns) == list(sample_dataframe.columns)
    
    def test_export_adds_timestamp(self, export_manager, sample_dataframe):
        """Test that timestamp is added to filename."""
        filepath = export_manager.export_dataset(sample_dataframe, "test_data.csv")
        
        # Check that filename contains timestamp pattern
        filename = Path(filepath).stem
        assert any(char.isdigit() for char in filename)
    
    def test_export_empty_dataframe_raises_error(self, export_manager):
        """Test that exporting empty DataFrame raises ValueError."""
        empty_df = pd.DataFrame()
        
        with pytest.raises(ValueError, match="Cannot export empty DataFrame"):
            export_manager.export_dataset(empty_df, "test.csv")
    
    def test_export_unsupported_format_raises_error(self, export_manager, sample_dataframe):
        """Test that unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            export_manager.export_dataset(sample_dataframe, "test.xlsx", format="xlsx")


class TestExportVisualization:
    """Tests for export_visualization method."""
    
    def test_export_png_basic(self, export_manager, sample_figure):
        """Test basic PNG export functionality."""
        filepath = export_manager.export_visualization(sample_figure, "test_plot.png", format="png")
        
        # Verify file was created
        assert Path(filepath).exists()
        assert Path(filepath).suffix == ".png"
    
    def test_export_svg_basic(self, export_manager, sample_figure):
        """Test basic SVG export functionality."""
        filepath = export_manager.export_visualization(sample_figure, "test_plot.svg", format="svg")
        
        # Verify file was created
        assert Path(filepath).exists()
        assert Path(filepath).suffix == ".svg"
    
    def test_export_with_custom_dpi(self, export_manager, sample_figure):
        """Test PNG export with custom DPI."""
        filepath = export_manager.export_visualization(sample_figure, "test_plot.png", format="png", dpi=150)
        
        # Verify file was created
        assert Path(filepath).exists()
    
    def test_export_invalid_dpi_raises_error(self, export_manager, sample_figure):
        """Test that invalid DPI raises ValueError."""
        with pytest.raises(ValueError, match="DPI must be positive"):
            export_manager.export_visualization(sample_figure, "test.png", dpi=-100)
    
    def test_export_unsupported_format_raises_error(self, export_manager, sample_figure):
        """Test that unsupported format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported format"):
            export_manager.export_visualization(sample_figure, "test.pdf", format="pdf")


class TestExportModelResults:
    """Tests for export_model_results method."""
    
    def test_export_json_basic(self, export_manager, sample_regression_results):
        """Test basic JSON export of model results."""
        filepath = export_manager.export_model_results(sample_regression_results, "results.json", format="json")
        
        # Verify file was created
        assert Path(filepath).exists()
        assert Path(filepath).suffix == ".json"
        
        # Verify content can be read back
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert 'coefficients' in data
        assert 'r_squared' in data
        assert 'diagnostics' in data
    
    def test_export_text_basic(self, export_manager, sample_regression_results):
        """Test basic text export of model results."""
        filepath = export_manager.export_model_results(sample_regression_results, "results.txt", format="text")
        
        # Verify file was created
        assert Path(filepath).exists()
        assert Path(filepath).suffix == ".txt"
        
        # Verify content contains expected sections
        with open(filepath, 'r') as f:
            content = f.read()
        
        assert "COEFFICIENTS:" in content
        assert "MODEL STATISTICS:" in content
        assert "DIAGNOSTIC TESTS:" in content
    
    def test_export_includes_all_results(self, export_manager, sample_regression_results):
        """Test that all result components are exported."""
        filepath = export_manager.export_model_results(sample_regression_results, "results.json", format="json")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Check coefficients
        assert data['coefficients']['const'] == 1.5
        assert data['coefficients']['var1'] == 0.8
        
        # Check statistics
        assert data['r_squared'] == 0.85
        assert data['adjusted_r_squared'] == 0.82
        
        # Check diagnostics
        assert data['diagnostics']['durbin_watson'] == 1.95
    
    def test_export_invalid_results_raises_error(self, export_manager):
        """Test that exporting invalid results object raises ValueError."""
        invalid_results = object()
        
        with pytest.raises(ValueError, match="doesn't have any recognized attributes"):
            export_manager.export_model_results(invalid_results, "results.json")


class TestExportForecast:
    """Tests for export_forecast method."""
    
    def test_export_forecast_basic(self, export_manager, sample_forecast_result):
        """Test basic forecast export functionality."""
        filepath = export_manager.export_forecast(sample_forecast_result, "forecast.csv")
        
        # Verify file was created
        assert Path(filepath).exists()
        assert Path(filepath).suffix == ".csv"
        
        # Verify content can be read back
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        assert len(df) == 12
        assert 'predicted_value' in df.columns
        assert 'lower_bound' in df.columns
        assert 'upper_bound' in df.columns
    
    def test_export_forecast_preserves_values(self, export_manager, sample_forecast_result):
        """Test that forecast values are preserved correctly."""
        filepath = export_manager.export_forecast(sample_forecast_result, "forecast.csv")
        
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Check that values match (approximately, due to CSV round-trip)
        np.testing.assert_array_almost_equal(
            df['predicted_value'].values,
            sample_forecast_result.predicted_values.values,
            decimal=5
        )
    
    def test_export_forecast_missing_attributes_raises_error(self, export_manager):
        """Test that exporting invalid forecast object raises ValueError."""
        # Create object with missing attributes
        class InvalidForecast:
            dates = pd.date_range('2024-01-01', periods=5)
            predicted_values = pd.Series([1, 2, 3, 4, 5])
            # Missing lower_bound and upper_bound
        
        with pytest.raises(ValueError, match="missing required attributes"):
            export_manager.export_forecast(InvalidForecast(), "forecast.csv")


class TestExportManagerIntegration:
    """Integration tests for ExportManager."""
    
    def test_multiple_exports_to_same_directory(self, export_manager, sample_dataframe, sample_figure):
        """Test that multiple exports can coexist in the same directory."""
        # Export dataset
        csv_path = export_manager.export_dataset(sample_dataframe, "data.csv")
        
        # Export visualization
        png_path = export_manager.export_visualization(sample_figure, "plot.png")
        
        # Verify both files exist
        assert Path(csv_path).exists()
        assert Path(png_path).exists()
    
    def test_export_creates_subdirectories(self, export_manager, sample_dataframe):
        """Test that export creates necessary subdirectories."""
        filepath = export_manager.export_dataset(sample_dataframe, "subdir/nested/data.csv")
        
        # Verify file was created in nested directory
        assert Path(filepath).exists()
        assert "subdir" in str(filepath)
        assert "nested" in str(filepath)
