"""
Property-based tests for DataMerger.

These tests verify that dataset merging preserves time alignment across
randomly generated time series datasets.
"""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
from data_processing.merger import DataMerger, MergeError


# Feature: interest-rate-credit-analysis, Property 3: Dataset merge preserves time alignment
# For any pair of H.15 and G.19 datasets with overlapping time periods, 
# the merged dataset should contain only matching time periods with correctly aligned values from both sources.
# Validates: Requirements 1.5


# Custom strategies for generating test data
@st.composite
def datetime_range(draw, min_date=None, max_date=None):
    """Generate a random datetime range."""
    if min_date is None:
        min_date = datetime(2000, 1, 1)
    if max_date is None:
        max_date = datetime(2023, 12, 31)
    
    start = draw(st.datetimes(min_value=min_date, max_value=max_date))
    # Generate end date that's at least 30 days after start
    # Make sure we have enough room for the end date
    min_end = start + timedelta(days=30)
    if min_end > max_date:
        # If we can't fit 30 days, just use a smaller range
        min_end = start + timedelta(days=1)
    
    # Only generate end date if there's room
    if min_end <= max_date:
        end = draw(st.datetimes(
            min_value=min_end,
            max_value=max_date
        ))
    else:
        end = start + timedelta(days=1)
    
    return start, end


@st.composite
def dataframe_with_datetime_index(draw, prefix="col", min_cols=1, max_cols=5):
    """
    Generate a DataFrame with a datetime index and numeric columns.
    Returns the DataFrame.
    """
    # Generate date range
    start_date, end_date = draw(datetime_range())
    
    # Generate frequency (monthly or daily)
    freq = draw(st.sampled_from(['MS', 'D']))  # Month start or Daily
    
    # Create date range
    date_index = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Ensure we have at least a few dates
    assume(len(date_index) >= 5)
    
    # Generate number of columns
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    
    # Generate column names
    col_names = [f"{prefix}_{i}" for i in range(n_cols)]
    
    # Generate data for each column
    data = {}
    for col in col_names:
        # Generate mostly valid floats, with some NaN values mixed in
        values = []
        for _ in range(len(date_index)):
            # 10% chance of NaN
            if draw(st.booleans()) and draw(st.integers(min_value=0, max_value=9)) == 0:
                values.append(np.nan)
            else:
                values.append(draw(st.floats(
                    min_value=-10000,
                    max_value=10000,
                    allow_nan=False,
                    allow_infinity=False
                )))
        data[col] = values
    
    df = pd.DataFrame(data, index=date_index)
    df.index.name = 'date'
    
    return df


@st.composite
def overlapping_dataframes(draw):
    """
    Generate two DataFrames with overlapping datetime indices.
    Returns (df1, df2, expected_common_dates).
    """
    # Generate a common date range first
    n_dates = draw(st.integers(min_value=10, max_value=30))
    start_date = draw(st.datetimes(
        min_value=datetime(2010, 1, 1),
        max_value=datetime(2020, 1, 1)
    ))
    
    # Create base date range
    base_dates = pd.date_range(start=start_date, periods=n_dates, freq='MS')
    
    # For df1, use a subset of dates (may include all)
    n_df1 = draw(st.integers(min_value=n_dates // 2, max_value=n_dates))
    df1_indices = draw(st.lists(
        st.integers(min_value=0, max_value=n_dates - 1),
        min_size=n_df1,
        max_size=n_df1,
        unique=True
    ))
    df1_dates = [base_dates[i] for i in sorted(df1_indices)]
    
    # For df2, use a different subset with guaranteed overlap
    # Ensure at least 50% overlap
    min_overlap = max(3, n_dates // 3)
    overlap_indices = df1_indices[:min_overlap]  # Take first min_overlap from df1
    
    # Add some additional dates from base_dates
    remaining_indices = [i for i in range(n_dates) if i not in df1_indices]
    n_additional = draw(st.integers(min_value=0, max_value=min(5, len(remaining_indices))))
    if n_additional > 0 and remaining_indices:
        additional_indices = draw(st.lists(
            st.sampled_from(remaining_indices),
            min_size=n_additional,
            max_size=n_additional,
            unique=True
        ))
        df2_indices = sorted(set(overlap_indices + additional_indices))
    else:
        df2_indices = sorted(overlap_indices)
    
    df2_dates = [base_dates[i] for i in df2_indices]
    
    # Create df1
    n_cols_df1 = draw(st.integers(min_value=1, max_value=3))
    df1_data = {}
    for i in range(n_cols_df1):
        df1_data[f"h15_{i}"] = np.random.randn(len(df1_dates))
    df1 = pd.DataFrame(df1_data, index=pd.DatetimeIndex(df1_dates))
    df1.index.name = 'date'
    
    # Create df2
    n_cols_df2 = draw(st.integers(min_value=1, max_value=3))
    df2_data = {}
    for i in range(n_cols_df2):
        df2_data[f"g19_{i}"] = np.random.randn(len(df2_dates))
    df2 = pd.DataFrame(df2_data, index=pd.DatetimeIndex(df2_dates))
    df2.index.name = 'date'
    
    # Calculate expected common dates
    expected_common_dates = sorted(set(df1.index).intersection(set(df2.index)))
    
    return df1, df2, expected_common_dates


@settings(max_examples=100)
@given(overlapping_dataframes())
def test_merge_preserves_time_alignment(data_tuple):
    """
    Property: For any pair of datasets with overlapping time periods,
    the merged dataset should contain only matching time periods with
    correctly aligned values from both sources.
    """
    df1, df2, expected_common_dates = data_tuple
    
    merger = DataMerger()
    merged_df = merger.merge_datasets(df1, df2)
    
    # Property 1: Merged dataset should contain exactly the common dates
    assert len(merged_df) == len(expected_common_dates), (
        f"Merged dataset length mismatch.\n"
        f"Expected {len(expected_common_dates)} rows, got {len(merged_df)} rows"
    )
    
    # Convert to sets for comparison
    merged_dates_set = set(merged_df.index)
    expected_dates_set = set(expected_common_dates)
    
    assert merged_dates_set == expected_dates_set, (
        f"Merged dataset dates mismatch.\n"
        f"Missing dates: {expected_dates_set - merged_dates_set}\n"
        f"Extra dates: {merged_dates_set - expected_dates_set}"
    )
    
    # Property 2: Values from df1 should be correctly aligned in merged dataset
    for col in df1.columns:
        assert col in merged_df.columns, f"Column {col} from df1 missing in merged dataset"
        
        for date in expected_common_dates:
            original_value = df1.loc[date, col]
            merged_value = merged_df.loc[date, col]
            
            # Handle NaN comparison
            if pd.isna(original_value):
                assert pd.isna(merged_value), (
                    f"Value mismatch for {col} at {date}: "
                    f"expected NaN, got {merged_value}"
                )
            else:
                assert original_value == merged_value or (
                    pd.isna(original_value) and pd.isna(merged_value)
                ), (
                    f"Value mismatch for {col} at {date}: "
                    f"expected {original_value}, got {merged_value}"
                )
    
    # Property 3: Values from df2 should be correctly aligned in merged dataset
    for col in df2.columns:
        assert col in merged_df.columns, f"Column {col} from df2 missing in merged dataset"
        
        for date in expected_common_dates:
            original_value = df2.loc[date, col]
            merged_value = merged_df.loc[date, col]
            
            # Handle NaN comparison
            if pd.isna(original_value):
                assert pd.isna(merged_value), (
                    f"Value mismatch for {col} at {date}: "
                    f"expected NaN, got {merged_value}"
                )
            else:
                assert original_value == merged_value or (
                    pd.isna(original_value) and pd.isna(merged_value)
                ), (
                    f"Value mismatch for {col} at {date}: "
                    f"expected {original_value}, got {merged_value}"
                )
    
    # Property 4: Merged dataset should be sorted chronologically
    assert merged_df.index.is_monotonic_increasing, (
        "Merged dataset should be sorted in chronological order"
    )


@st.composite
def non_overlapping_dataframes(draw):
    """
    Generate two DataFrames with non-overlapping datetime indices.
    """
    # Generate first DataFrame
    start1, end1 = draw(datetime_range(
        min_date=datetime(2000, 1, 1),
        max_date=datetime(2010, 12, 31)
    ))
    
    date_index1 = pd.date_range(start=start1, end=end1, freq='MS')
    assume(len(date_index1) >= 5)
    
    n_cols1 = draw(st.integers(min_value=1, max_value=3))
    data1 = {f"col1_{i}": np.random.randn(len(date_index1)) for i in range(n_cols1)}
    df1 = pd.DataFrame(data1, index=date_index1)
    df1.index.name = 'date'
    
    # Generate second DataFrame with dates after df1
    start2 = end1 + timedelta(days=365)  # At least 1 year gap
    end2 = start2 + timedelta(days=365)
    
    date_index2 = pd.date_range(start=start2, end=end2, freq='MS')
    assume(len(date_index2) >= 5)
    
    n_cols2 = draw(st.integers(min_value=1, max_value=3))
    data2 = {f"col2_{i}": np.random.randn(len(date_index2)) for i in range(n_cols2)}
    df2 = pd.DataFrame(data2, index=date_index2)
    df2.index.name = 'date'
    
    return df1, df2


@settings(max_examples=100)
@given(non_overlapping_dataframes())
def test_merge_raises_error_for_non_overlapping_datasets(data_tuple):
    """
    Property: For any pair of datasets with no overlapping time periods,
    merge_datasets should raise a MergeError.
    """
    df1, df2 = data_tuple
    
    merger = DataMerger()
    
    try:
        merged_df = merger.merge_datasets(df1, df2)
        # If we get here, the merge succeeded when it shouldn't have
        assert False, (
            f"Expected MergeError for non-overlapping datasets, but merge succeeded.\n"
            f"df1 range: {df1.index.min()} to {df1.index.max()}\n"
            f"df2 range: {df2.index.min()} to {df2.index.max()}\n"
            f"Merged dataset has {len(merged_df)} rows"
        )
    except MergeError as e:
        # This is expected - verify the error message contains useful information
        error_msg = str(e)
        assert "No overlapping time periods" in error_msg, (
            f"MergeError should mention 'No overlapping time periods', got: {error_msg}"
        )
        # Verify date ranges are mentioned in error
        assert str(df1.index.min().date()) in error_msg or str(df1.index.max().date()) in error_msg, (
            f"MergeError should mention df1 date range, got: {error_msg}"
        )


@settings(max_examples=100)
@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=2, max_value=5),
    st.integers(min_value=2, max_value=5)
)
def test_merge_preserves_all_columns(n_dates, n_cols_df1, n_cols_df2):
    """
    Property: For any pair of datasets, the merged dataset should contain
    all columns from both source datasets.
    """
    # Create date range
    dates = pd.date_range(start='2020-01-01', periods=n_dates, freq='MS')
    
    # Create df1
    df1_cols = [f"h15_{i}" for i in range(n_cols_df1)]
    df1_data = {col: np.random.randn(n_dates) for col in df1_cols}
    df1 = pd.DataFrame(df1_data, index=dates)
    df1.index.name = 'date'
    
    # Create df2 with same dates
    df2_cols = [f"g19_{i}" for i in range(n_cols_df2)]
    df2_data = {col: np.random.randn(n_dates) for col in df2_cols}
    df2 = pd.DataFrame(df2_data, index=dates)
    df2.index.name = 'date'
    
    merger = DataMerger()
    merged_df = merger.merge_datasets(df1, df2)
    
    # Verify all columns are present
    expected_columns = set(df1_cols + df2_cols)
    actual_columns = set(merged_df.columns)
    
    assert expected_columns == actual_columns, (
        f"Column mismatch in merged dataset.\n"
        f"Expected: {expected_columns}\n"
        f"Actual: {actual_columns}\n"
        f"Missing: {expected_columns - actual_columns}\n"
        f"Extra: {actual_columns - expected_columns}"
    )
