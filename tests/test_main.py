"""
Unit tests for main analysis pipeline.
"""

import pytest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

from main import AnalysisResults, main


@pytest.fixture
def mock_sample_data():
    """Create mock sample data."""
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


class TestAnalysisResults:
    """Tests for AnalysisResults dataclass."""
    
    def test_analysis_results_initialization(self):
        """Test that AnalysisResults can be initialized."""
        results = AnalysisResults()
        
        assert results.merged_data is None
        assert results.eda_results is None
        assert results.lag_results is None
        assert results.regression_results is None
        assert results.forecast_results is None
        assert results.sensitivity_results is None
    
    def test_analysis_results_with_data(self, mock_sample_data):
        """Test that AnalysisResults can store data."""
        results = AnalysisResults(
            merged_data=mock_sample_data,
            eda_results={'test': 'data'}
        )
        
        assert results.merged_data is not None
        assert results.eda_results == {'test': 'data'}


class TestMainPipeline:
    """Tests for main analysis pipeline."""
    
    def test_main_returns_none_on_missing_files(self):
        """Test that main returns None when data files are missing."""
        # This will fail to load files and return None
        result = main()
        
        # Should return None due to missing files
        assert result is None
    
    @patch('main.DataLoader')
    @patch('main.DataValidator')
    @patch('main.DataMerger')
    def test_main_pipeline_structure(self, mock_merger, mock_validator, mock_loader, mock_sample_data):
        """Test that main pipeline calls all components in order."""
        # Setup mocks
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.load_h15_dataset.return_value = mock_sample_data
        mock_loader_instance.load_g19_dataset.return_value = mock_sample_data
        
        mock_validator_instance = Mock()
        mock_validator.return_value = mock_validator_instance
        
        from data_processing.validator import ValidationResult
        mock_validator_instance.validate_dataset.return_value = ValidationResult(
            is_valid=True,
            issues=[]
        )
        
        mock_merger_instance = Mock()
        mock_merger.return_value = mock_merger_instance
        mock_merger_instance.merge_datasets.return_value = mock_sample_data
        
        # Run main
        result = main()
        
        # Verify components were called
        assert mock_loader_instance.load_h15_dataset.called
        assert mock_loader_instance.load_g19_dataset.called
        assert mock_validator_instance.validate_dataset.called
        assert mock_merger_instance.merge_datasets.called
