"""
Unit tests for DashboardApp class.
"""

import pytest
import pandas as pd
import numpy as np
from dash import html

from dashboard.dashboard_app import DashboardApp


@pytest.fixture
def sample_merged_data():
    """Create sample merged data for testing."""
    dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    return pd.DataFrame({
        'fed_funds_rate': np.random.uniform(0, 5, 24),
        'treasury_1y': np.random.uniform(0, 5, 24),
        'treasury_10y': np.random.uniform(1, 6, 24),
        'prime_rate': np.random.uniform(2, 7, 24),
        'total_credit': np.random.uniform(4000, 4500, 24),
        'revolving_credit': np.random.uniform(1000, 1200, 24),
        'non_revolving_credit': np.random.uniform(3000, 3300, 24)
    }, index=dates)


@pytest.fixture
def analysis_results_with_data(sample_merged_data):
    """Create mock analysis results object."""
    class MockAnalysisResults:
        def __init__(self, merged_data):
            self.merged_data = merged_data
    
    return MockAnalysisResults(sample_merged_data)


@pytest.fixture
def analysis_results_empty():
    """Create mock analysis results with no data."""
    class MockAnalysisResults:
        pass
    
    return MockAnalysisResults()


class TestDashboardAppInitialization:
    """Tests for DashboardApp initialization."""
    
    def test_dashboard_initializes_with_results(self, analysis_results_with_data):
        """Test that dashboard initializes with analysis results."""
        dashboard = DashboardApp(analysis_results_with_data)
        
        assert dashboard.analysis_results is not None
        assert dashboard.app is not None
    
    def test_dashboard_has_layout(self, analysis_results_with_data):
        """Test that dashboard has a layout configured."""
        dashboard = DashboardApp(analysis_results_with_data)
        
        assert dashboard.app.layout is not None


class TestTimeSeriesPanel:
    """Tests for create_time_series_panel method."""
    
    def test_creates_time_series_panel_with_data(self, analysis_results_with_data):
        """Test that time series panel is created with valid data."""
        dashboard = DashboardApp(analysis_results_with_data)
        panel = dashboard.create_time_series_panel()
        
        # Check that panel is an html.Div
        assert isinstance(panel, html.Div)
        
        # Check that panel contains children
        assert hasattr(panel, 'children')
        assert len(panel.children) > 0
    
    def test_time_series_panel_handles_missing_data(self, analysis_results_empty):
        """Test that time series panel handles missing data gracefully."""
        dashboard = DashboardApp(analysis_results_empty)
        panel = dashboard.create_time_series_panel()
        
        # Should return a div with error message
        assert isinstance(panel, html.Div)
    
    def test_time_series_panel_contains_graphs(self, analysis_results_with_data):
        """Test that time series panel contains graph components."""
        dashboard = DashboardApp(analysis_results_with_data)
        panel = dashboard.create_time_series_panel()
        
        # Convert to string to check for graph components
        panel_str = str(panel)
        assert 'Graph' in panel_str or 'dcc.Graph' in panel_str


class TestCorrelationPanel:
    """Tests for create_correlation_panel method."""
    
    def test_creates_correlation_panel_with_data(self, analysis_results_with_data):
        """Test that correlation panel is created with valid data."""
        dashboard = DashboardApp(analysis_results_with_data)
        panel = dashboard.create_correlation_panel()
        
        # Check that panel is an html.Div
        assert isinstance(panel, html.Div)
        
        # Check that panel contains children
        assert hasattr(panel, 'children')
        assert len(panel.children) > 0
    
    def test_correlation_panel_handles_missing_data(self, analysis_results_empty):
        """Test that correlation panel handles missing data gracefully."""
        dashboard = DashboardApp(analysis_results_empty)
        panel = dashboard.create_correlation_panel()
        
        # Should return a div with error message
        assert isinstance(panel, html.Div)
    
    def test_correlation_panel_contains_heatmap(self, analysis_results_with_data):
        """Test that correlation panel contains heatmap."""
        dashboard = DashboardApp(analysis_results_with_data)
        panel = dashboard.create_correlation_panel()
        
        # Convert to string to check for graph components
        panel_str = str(panel)
        assert 'Graph' in panel_str or 'correlation-heatmap' in panel_str


@pytest.fixture
def analysis_results_with_regression(sample_merged_data):
    """Create mock analysis results with regression results."""
    from analysis.regression_engine import RegressionResults, Diagnostics
    
    class MockAnalysisResults:
        def __init__(self, merged_data):
            self.merged_data = merged_data
            self.regression_results = RegressionResults(
                coefficients={'const': 1.5, 'fed_funds_rate': -0.8, 'treasury_10y': 0.3},
                r_squared=0.75,
                adjusted_r_squared=0.72,
                p_values={'const': 0.001, 'fed_funds_rate': 0.01, 'treasury_10y': 0.05},
                residuals=pd.Series(np.random.randn(24)),
                predicted_values=pd.Series(np.random.randn(24)),
                diagnostics=Diagnostics(
                    durbin_watson=1.95,
                    breusch_pagan_p=0.15,
                    jarque_bera_p=0.25
                )
            )
    
    return MockAnalysisResults(sample_merged_data)


class TestRegressionPanel:
    """Tests for create_regression_panel method."""
    
    def test_creates_regression_panel_with_data(self, analysis_results_with_regression):
        """Test that regression panel is created with valid data."""
        dashboard = DashboardApp(analysis_results_with_regression)
        panel = dashboard.create_regression_panel()
        
        # Check that panel is an html.Div
        assert isinstance(panel, html.Div)
        
        # Check that panel contains children
        assert hasattr(panel, 'children')
        assert len(panel.children) > 0
    
    def test_regression_panel_handles_missing_data(self, analysis_results_empty):
        """Test that regression panel handles missing data gracefully."""
        dashboard = DashboardApp(analysis_results_empty)
        panel = dashboard.create_regression_panel()
        
        # Should return a div with error message
        assert isinstance(panel, html.Div)
    
    def test_regression_panel_contains_coefficients(self, analysis_results_with_regression):
        """Test that regression panel displays coefficients."""
        dashboard = DashboardApp(analysis_results_with_regression)
        panel = dashboard.create_regression_panel()
        
        # Convert to string to check for content
        panel_str = str(panel)
        assert 'Coefficient' in panel_str or 'const' in panel_str


@pytest.fixture
def analysis_results_with_forecast(sample_merged_data):
    """Create mock analysis results with forecast results."""
    from analysis.forecast_engine import ForecastResult, AccuracyMetrics
    
    dates = pd.date_range('2024-01-01', periods=12, freq='MS')
    values = np.random.randn(12) + 4200
    
    class MockAnalysisResults:
        def __init__(self, merged_data):
            self.merged_data = merged_data
            self.forecast_results = ForecastResult(
                dates=dates,
                predicted_values=pd.Series(values, index=dates),
                lower_bound=pd.Series(values - 50, index=dates),
                upper_bound=pd.Series(values + 50, index=dates),
                confidence_level=0.95,
                accuracy_metrics=AccuracyMetrics(mae=25.5, rmse=32.1, mape=1.8)
            )
    
    return MockAnalysisResults(sample_merged_data)


class TestForecastPanel:
    """Tests for create_forecast_panel method."""
    
    def test_creates_forecast_panel_with_data(self, analysis_results_with_forecast):
        """Test that forecast panel is created with valid data."""
        dashboard = DashboardApp(analysis_results_with_forecast)
        panel = dashboard.create_forecast_panel()
        
        # Check that panel is an html.Div
        assert isinstance(panel, html.Div)
        
        # Check that panel contains children
        assert hasattr(panel, 'children')
        assert len(panel.children) > 0
    
    def test_forecast_panel_handles_missing_data(self, analysis_results_empty):
        """Test that forecast panel handles missing data gracefully."""
        dashboard = DashboardApp(analysis_results_empty)
        panel = dashboard.create_forecast_panel()
        
        # Should return a div with error message
        assert isinstance(panel, html.Div)
    
    def test_forecast_panel_contains_chart(self, analysis_results_with_forecast):
        """Test that forecast panel contains chart."""
        dashboard = DashboardApp(analysis_results_with_forecast)
        panel = dashboard.create_forecast_panel()
        
        # Convert to string to check for content
        panel_str = str(panel)
        assert 'Graph' in panel_str or 'forecast-chart' in panel_str


class TestScenarioSimulator:
    """Tests for create_scenario_simulator method."""
    
    def test_creates_scenario_simulator_with_data(self, analysis_results_with_forecast):
        """Test that scenario simulator is created with valid data."""
        dashboard = DashboardApp(analysis_results_with_forecast)
        panel = dashboard.create_scenario_simulator()
        
        # Check that panel is an html.Div
        assert isinstance(panel, html.Div)
        
        # Check that panel contains children
        assert hasattr(panel, 'children')
        assert len(panel.children) > 0
    
    def test_scenario_simulator_handles_missing_data(self, analysis_results_empty):
        """Test that scenario simulator handles missing data gracefully."""
        dashboard = DashboardApp(analysis_results_empty)
        panel = dashboard.create_scenario_simulator()
        
        # Should return a div with error message
        assert isinstance(panel, html.Div)
    
    def test_scenario_simulator_contains_slider(self, analysis_results_with_forecast):
        """Test that scenario simulator contains slider control."""
        dashboard = DashboardApp(analysis_results_with_forecast)
        panel = dashboard.create_scenario_simulator()
        
        # Convert to string to check for content
        panel_str = str(panel)
        assert 'Slider' in panel_str or 'rate-adjustment-slider' in panel_str


class TestDashboardCallbacks:
    """Tests for register_callbacks method."""
    
    def test_register_callbacks_executes(self, analysis_results_with_forecast):
        """Test that register_callbacks can be called without errors."""
        dashboard = DashboardApp(analysis_results_with_forecast)
        
        # Should not raise any errors
        dashboard.register_callbacks()
    
    def test_callbacks_registered_on_app(self, analysis_results_with_forecast):
        """Test that callbacks are registered on the Dash app."""
        dashboard = DashboardApp(analysis_results_with_forecast)
        dashboard.register_callbacks()
        
        # Check that app has callbacks
        assert hasattr(dashboard.app, 'callback_map')
        # Callbacks should be registered
        assert len(dashboard.app.callback_map) > 0


class TestDashboardRun:
    """Tests for run method."""
    
    def test_run_method_exists(self, analysis_results_with_forecast):
        """Test that run method exists and is callable."""
        dashboard = DashboardApp(analysis_results_with_forecast)
        
        # Check that run method exists
        assert hasattr(dashboard, 'run')
        assert callable(dashboard.run)
    
    def test_run_updates_layout(self, analysis_results_with_forecast):
        """Test that run method updates the layout with all panels."""
        dashboard = DashboardApp(analysis_results_with_forecast)
        
        # Mock the run method to prevent actual server start
        original_run = dashboard.app.run
        dashboard.app.run = lambda **kwargs: None
        
        try:
            # Call run (won't actually start server due to mock)
            dashboard.run()
            
            # Check that layout was updated
            assert dashboard.app.layout is not None
        finally:
            # Restore original method
            dashboard.app.run = original_run
