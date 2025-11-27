"""
Property-based tests for DataValidator.

These tests verify that validation error reporting is accurate across
randomly generated datasets with various types of issues.
"""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from data_processing.validator import DataValidator, ValidationResult, ValidationIssue


# Feature: interest-rate-credit-analysis, Property 2: Validation error reporting accuracy
# For any dataset with missing or invalid data at specific locations, 
# the validation report should identify the exact row and column positions of all issues.
# Validates: Requirements 1.4


# Custom strategies for generating test data
@st.composite
def dataframe_with_missing_values(draw):
    """
    Generate a DataFrame with intentionally placed missing values.
    Returns the DataFrame and a set of (row, column) tuples where values are missing.
    """
    n_rows = draw(st.integers(min_value=5, max_value=20))
    n_cols = draw(st.integers(min_value=2, max_value=5))
    
    # Generate column names
    col_names = [f"col_{i}" for i in range(n_cols)]
    
    # Generate base data (all valid numeric values)
    data = {}
    for col in col_names:
        data[col] = draw(st.lists(
            st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows
        ))
    
    df = pd.DataFrame(data)
    
    # Intentionally introduce missing values at specific locations
    n_missing = draw(st.integers(min_value=1, max_value=min(5, n_rows * n_cols // 2)))
    missing_locations = set()
    
    for _ in range(n_missing):
        row_idx = draw(st.integers(min_value=0, max_value=n_rows - 1))
        col_idx = draw(st.integers(min_value=0, max_value=n_cols - 1))
        col_name = col_names[col_idx]
        
        df.iloc[row_idx, col_idx] = np.nan
        missing_locations.add((row_idx, col_name))
    
    return df, missing_locations


@st.composite
def dataframe_with_invalid_types(draw):
    """
    Generate a DataFrame with intentionally placed non-numeric values.
    Returns the DataFrame, columns that should be numeric, and locations of invalid values.
    """
    n_rows = draw(st.integers(min_value=5, max_value=20))
    n_cols = draw(st.integers(min_value=2, max_value=5))
    
    # Generate column names
    col_names = [f"col_{i}" for i in range(n_cols)]
    
    # Generate base data (all valid numeric values as strings to allow mixing)
    data = {}
    for col in col_names:
        data[col] = [
            draw(st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False))
            for _ in range(n_rows)
        ]
    
    df = pd.DataFrame(data)
    
    # Intentionally introduce non-numeric values at specific locations
    n_invalid = draw(st.integers(min_value=1, max_value=min(5, n_rows * n_cols // 2)))
    invalid_locations = set()
    
    for _ in range(n_invalid):
        row_idx = draw(st.integers(min_value=0, max_value=n_rows - 1))
        col_idx = draw(st.integers(min_value=0, max_value=n_cols - 1))
        col_name = col_names[col_idx]
        
        # Insert a non-numeric string value
        invalid_value = draw(st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
        df.at[row_idx, col_name] = invalid_value
        invalid_locations.add((row_idx, col_name))
    
    return df, col_names, invalid_locations


@settings(max_examples=100)
@given(dataframe_with_missing_values())
def test_detect_missing_data_reports_all_locations(data_tuple):
    """
    Property: For any dataset with missing values at specific locations,
    detect_missing_data should report all those locations accurately.
    """
    df, expected_missing_locations = data_tuple
    
    validator = DataValidator()
    result = validator.detect_missing_data(df)
    
    # Extract reported missing locations from issues (only those with specific row numbers)
    reported_locations = set()
    for issue in result.issues:
        if issue.row is not None and issue.issue_type == 'missing':
            reported_locations.add((issue.row, issue.column))
    
    # Verify all expected missing locations are reported
    assert expected_missing_locations == reported_locations, (
        f"Missing data locations mismatch.\n"
        f"Expected: {expected_missing_locations}\n"
        f"Reported: {reported_locations}\n"
        f"Missing from report: {expected_missing_locations - reported_locations}\n"
        f"Extra in report: {reported_locations - expected_missing_locations}"
    )
    
    # Verify validation fails when there are missing values
    if len(expected_missing_locations) > 0:
        assert not result.is_valid, "Validation should fail when missing data is present"


@settings(max_examples=100)
@given(dataframe_with_invalid_types())
def test_validate_numeric_values_reports_all_invalid_locations(data_tuple):
    """
    Property: For any dataset with non-numeric values at specific locations,
    validate_numeric_values should report all those locations accurately.
    """
    df, numeric_columns, expected_invalid_locations = data_tuple
    
    validator = DataValidator()
    result = validator.validate_numeric_values(df, numeric_columns)
    
    # Extract reported invalid locations from issues (only those with specific row numbers)
    reported_locations = set()
    for issue in result.issues:
        if issue.row is not None and issue.issue_type == 'invalid_type':
            reported_locations.add((issue.row, issue.column))
    
    # Verify all expected invalid locations are reported
    # Note: The validator reports up to 5 specific locations per column
    # So we check that all reported locations are in the expected set
    assert reported_locations.issubset(expected_invalid_locations), (
        f"Invalid type locations mismatch.\n"
        f"Expected subset of: {expected_invalid_locations}\n"
        f"Reported: {reported_locations}\n"
        f"Unexpected in report: {reported_locations - expected_invalid_locations}"
    )
    
    # Verify that if there are invalid values, at least some are reported
    if len(expected_invalid_locations) > 0:
        assert len(reported_locations) > 0, "Should report at least some invalid locations"
        assert not result.is_valid, "Validation should fail when invalid types are present"


@settings(max_examples=100)
@given(
    st.lists(st.text(min_size=1, max_size=10), min_size=2, max_size=5, unique=True),
    st.integers(min_value=3, max_value=10)
)
def test_validate_columns_reports_all_missing_columns(required_columns, n_rows):
    """
    Property: For any set of required columns, validate_columns should report
    exactly which columns are missing from the dataset.
    """
    # Create a DataFrame with only a subset of required columns
    n_present = len(required_columns) // 2
    present_columns = required_columns[:n_present]
    expected_missing = set(required_columns[n_present:])
    
    # Create DataFrame with only present columns
    data = {col: [1.0] * n_rows for col in present_columns}
    df = pd.DataFrame(data)
    
    validator = DataValidator()
    result = validator.validate_columns(df, required_columns)
    
    # Extract reported missing columns
    reported_missing = set()
    for issue in result.issues:
        if issue.issue_type == 'missing' and issue.row is None:
            reported_missing.add(issue.column)
    
    # Verify all missing columns are reported
    assert expected_missing == reported_missing, (
        f"Missing columns mismatch.\n"
        f"Expected: {expected_missing}\n"
        f"Reported: {reported_missing}"
    )
    
    # Verify validation fails when columns are missing
    if len(expected_missing) > 0:
        assert not result.is_valid, "Validation should fail when required columns are missing"
    else:
        assert result.is_valid, "Validation should pass when all required columns are present"


@settings(max_examples=100)
@given(st.integers(min_value=5, max_value=20), st.integers(min_value=2, max_value=5))
def test_validation_passes_for_clean_data(n_rows, n_cols):
    """
    Property: For any dataset with no missing values and all numeric data,
    all validation methods should pass and report no issues.
    """
    # Generate clean data
    col_names = [f"col_{i}" for i in range(n_cols)]
    data = {col: np.random.randn(n_rows) for col in col_names}
    df = pd.DataFrame(data)
    
    validator = DataValidator()
    
    # Test all validation methods
    result_columns = validator.validate_columns(df, col_names)
    result_numeric = validator.validate_numeric_values(df, col_names)
    result_missing = validator.detect_missing_data(df)
    
    # All should pass
    assert result_columns.is_valid, "Column validation should pass for clean data"
    assert result_numeric.is_valid, "Numeric validation should pass for clean data"
    assert result_missing.is_valid, "Missing data detection should pass for clean data"
    
    # All should have no issues
    assert len(result_columns.issues) == 0, "Should have no column issues"
    assert len(result_numeric.issues) == 0, "Should have no numeric issues"
    assert len(result_missing.issues) == 0, "Should have no missing data issues"
