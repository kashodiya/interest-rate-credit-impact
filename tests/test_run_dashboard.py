"""
Unit tests for run_dashboard script.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from main import AnalysisResults


@pytest.fixture
def mock_analysis_results():
    """Create mock analysis results."""
    dates = pd.date_range('2020-01-01', periods=24, freq='MS')
    merged_data = pd.DataFrame({
        'fed_funds_rate': np.random.uniform(0, 5, 24),
        'treasury_10y': np.random.uniform(1, 6, 24),
        'total_credit': np.random.uniform(4000, 4500, 24),
        'revolving_credit': np.random.uniform(1000, 1200, 24),
        'non_revolving_credit': np.random.uniform(3000, 3300, 24)
    }, index=dates)
    
    return AnalysisResults(
        merged_data=merged_data,
        eda_results={'test': 'data'},
        lag_results={'test': 'data'},
        regression_results=None,
        forecast_results=None
    )


class TestRunDashboard:
    """Tests for run_dashboard script."""
    
    @patch('run_dashboard.main')
    @patch('run_dashboard.DashboardApp')
    def test_run_dashboard_with_successful_analysis(self, mock_dashboard_class, mock_main, mock_analysis_results):
        """Test that run_dashboard launches dashboard when analysis succeeds."""
        # Setup mocks
        mock_main.return_value = mock_analysis_results
        
        mock_dashboard_instance = Mock()
        mock_dashboard_class.return_value = mock_dashboard_instance
        mock_dashboard_instance.run = Mock()
        
        # Import and run
        from run_dashboard import run_dashboard
        
        # This will try to run the dashboard, but we've mocked it
        try:
            run_dashboard()
        except SystemExit:
            pass  # Expected when dashboard.run is mocked
        
        # Verify main was called
        assert mock_main.called
        
        # Verify dashboard was created with results
        mock_dashboard_class.assert_called_once_with(mock_analysis_results)
    
    @patch('run_dashboard.main')
    def test_run_dashboard_exits_on_failed_analysis(self, mock_main):
        """Test that run_dashboard exits when analysis fails."""
        # Setup mock to return None (failure)
        mock_main.return_value = None
        
        # Import and run
        from run_dashboard import run_dashboard
        
        # Should exit with code 1
        with pytest.raises(SystemExit) as exc_info:
            run_dashboard()
        
        assert exc_info.value.code == 1
