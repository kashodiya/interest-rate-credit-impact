"""
Property-based tests for EDAEngine visualization methods.

These tests verify that visualizations contain all requested data series
across randomly generated datasets and column specifications.
"""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime, timedelta
from analysis.eda_engine import EDAEngine


# Feature: interest-rate-credit-analysis, Property 6: Visualization data completeness
# For any generated visualization, the plot object should contain all data series 
# specified in the visualization request.
# Validates: Requirements 2.1, 2.5


# Custom strategies for generating test data
@st.composite
def datetime_index_dataframe(draw, min_periods=10, max_periods=50, min_cols=1, max_cols=10):
    """
    Generate a DataFrame with datetime index and numeric columns.
    Returns the DataFrame.
    """
    n_periods = draw(st.integers(min_value=min_periods, max_value=max_periods))
    n_cols = draw(st.integers(min_value=min_cols, max_value=max_cols))
    
    # Generate date range
    start_date = draw(st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2020, 1, 1)
    ))
    
    date_index = pd.date_range(start=start_date, periods=n_periods, freq='MS')
    
    # Generate column names
    col_names = [f"series_{i}" for i in range(n_cols)]
    
    # Generate data for each column
    data = {}
    for col in col_names:
        # Generate numeric values (mostly valid, some NaN)
        values = []
        for _ in range(n_periods):
            # 5% chance of NaN
            if draw(st.booleans()) and draw(st.integers(min_value=0, max_value=19)) == 0:
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
    
    return df


@st.composite
def dataframe_with_column_subset(draw):
    """
    Generate a DataFrame and a subset of its columns to plot.
    Returns (df, columns_to_plot).
    """
    df = draw(datetime_index_dataframe(min_cols=2, max_cols=10))
    
    # Select a random subset of columns (at least 1, at most all)
    n_cols_to_plot = draw(st.integers(min_value=1, max_value=len(df.columns)))
    columns_to_plot = draw(st.lists(
        st.sampled_from(df.columns.tolist()),
        min_size=n_cols_to_plot,
        max_size=n_cols_to_plot,
        unique=True
    ))
    
    return df, columns_to_plot


@st.composite
def dataframe_with_rate_credit_columns(draw):
    """
    Generate a DataFrame with separate rate and credit columns.
    Returns (df, rate_cols, credit_cols).
    """
    n_periods = draw(st.integers(min_value=10, max_value=50))
    n_rate_cols = draw(st.integers(min_value=1, max_value=5))
    n_credit_cols = draw(st.integers(min_value=1, max_value=5))
    
    # Generate date range
    start_date = draw(st.datetimes(
        min_value=datetime(2000, 1, 1),
        max_value=datetime(2020, 1, 1)
    ))
    
    date_index = pd.date_range(start=start_date, periods=n_periods, freq='MS')
    
    # Generate rate columns (typically 0-10%)
    rate_cols = [f"rate_{i}" for i in range(n_rate_cols)]
    data = {}
    for col in rate_cols:
        data[col] = draw(st.lists(
            st.floats(min_value=0, max_value=10, allow_nan=False, allow_infinity=False),
            min_size=n_periods,
            max_size=n_periods
        ))
    
    # Generate credit columns (typically large values in billions)
    credit_cols = [f"credit_{i}" for i in range(n_credit_cols)]
    for col in credit_cols:
        data[col] = draw(st.lists(
            st.floats(min_value=1000, max_value=10000, allow_nan=False, allow_infinity=False),
            min_size=n_periods,
            max_size=n_periods
        ))
    
    df = pd.DataFrame(data, index=date_index)
    
    return df, rate_cols, credit_cols


@settings(max_examples=100, deadline=None)
@given(dataframe_with_column_subset())
def test_time_series_plot_contains_all_requested_series(data_tuple):
    """
    Property: For any DataFrame and subset of columns, generate_time_series_plot
    should create a figure containing exactly those data series.
    """
    df, columns_to_plot = data_tuple
    
    engine = EDAEngine()
    fig = engine.generate_time_series_plot(df, columns_to_plot)
    
    # Property 1: Figure should have exactly the number of traces as requested columns
    assert len(fig.data) == len(columns_to_plot), (
        f"Figure should have {len(columns_to_plot)} traces, but has {len(fig.data)}"
    )
    
    # Property 2: Each requested column should have a corresponding trace
    trace_names = [trace.name for trace in fig.data]
    for col in columns_to_plot:
        assert col in trace_names, (
            f"Column '{col}' was requested but not found in figure traces.\n"
            f"Requested: {columns_to_plot}\n"
            f"Found: {trace_names}"
        )
    
    # Property 3: No extra traces should be present
    for trace_name in trace_names:
        assert trace_name in columns_to_plot, (
            f"Trace '{trace_name}' found in figure but was not requested.\n"
            f"Requested: {columns_to_plot}\n"
            f"Found: {trace_names}"
        )
    
    # Property 4: Each trace should contain data from the DataFrame
    for i, col in enumerate(columns_to_plot):
        trace = fig.data[i]
        assert trace.name == col, f"Trace {i} should be named '{col}', got '{trace.name}'"
        
        # Verify trace has data points
        assert len(trace.x) > 0, f"Trace '{col}' should have x-axis data"
        assert len(trace.y) > 0, f"Trace '{col}' should have y-axis data"
        assert len(trace.x) == len(trace.y), f"Trace '{col}' x and y data should have same length"


@settings(max_examples=100)
@given(dataframe_with_column_subset())
def test_time_series_plot_with_events_contains_all_series(data_tuple):
    """
    Property: For any DataFrame with event annotations, the figure should still
    contain all requested data series.
    """
    df, columns_to_plot = data_tuple
    
    # Generate random events within the date range
    n_events = np.random.randint(1, 4)
    events = []
    for i in range(n_events):
        # Pick a random date from the DataFrame's index
        event_date = np.random.choice(df.index)
        events.append({
            'date': event_date,
            'label': f'Event {i}',
            'color': np.random.choice(['red', 'blue', 'green'])
        })
    
    engine = EDAEngine()
    fig = engine.generate_time_series_plot(df, columns_to_plot, events=events)
    
    # Property 1: Figure should have exactly the number of traces as requested columns
    # (events are added as shapes, not traces)
    assert len(fig.data) == len(columns_to_plot), (
        f"Figure should have {len(columns_to_plot)} traces, but has {len(fig.data)}"
    )
    
    # Property 2: Each requested column should have a corresponding trace
    trace_names = [trace.name for trace in fig.data]
    for col in columns_to_plot:
        assert col in trace_names, (
            f"Column '{col}' was requested but not found in figure traces.\n"
            f"Requested: {columns_to_plot}\n"
            f"Found: {trace_names}"
        )
    
    # Property 3: Event annotations should be present as shapes
    assert len(fig.layout.shapes) == n_events, (
        f"Figure should have {n_events} event shapes, but has {len(fig.layout.shapes)}"
    )


@settings(max_examples=100)
@given(dataframe_with_rate_credit_columns())
def test_comparative_plot_contains_all_requested_series(data_tuple):
    """
    Property: For any DataFrame with rate and credit columns, generate_comparative_plot
    should create a figure containing all requested rate and credit series.
    """
    df, rate_cols, credit_cols = data_tuple
    
    engine = EDAEngine()
    fig = engine.generate_comparative_plot(df, rate_cols, credit_cols)
    
    # Property 1: Figure should have traces for all rate and credit columns
    total_expected_traces = len(rate_cols) + len(credit_cols)
    assert len(fig.data) == total_expected_traces, (
        f"Figure should have {total_expected_traces} traces "
        f"({len(rate_cols)} rate + {len(credit_cols)} credit), "
        f"but has {len(fig.data)}"
    )
    
    # Property 2: Each requested rate column should have a corresponding trace
    trace_names = [trace.name for trace in fig.data]
    for col in rate_cols:
        assert col in trace_names, (
            f"Rate column '{col}' was requested but not found in figure traces.\n"
            f"Requested rates: {rate_cols}\n"
            f"Found: {trace_names}"
        )
    
    # Property 3: Each requested credit column should have a corresponding trace
    for col in credit_cols:
        assert col in trace_names, (
            f"Credit column '{col}' was requested but not found in figure traces.\n"
            f"Requested credits: {credit_cols}\n"
            f"Found: {trace_names}"
        )
    
    # Property 4: No extra traces should be present
    expected_cols = set(rate_cols + credit_cols)
    actual_cols = set(trace_names)
    assert expected_cols == actual_cols, (
        f"Trace names mismatch.\n"
        f"Expected: {expected_cols}\n"
        f"Actual: {actual_cols}\n"
        f"Missing: {expected_cols - actual_cols}\n"
        f"Extra: {actual_cols - expected_cols}"
    )
    
    # Property 5: Each trace should contain data
    for trace in fig.data:
        assert len(trace.x) > 0, f"Trace '{trace.name}' should have x-axis data"
        assert len(trace.y) > 0, f"Trace '{trace.name}' should have y-axis data"
        assert len(trace.x) == len(trace.y), (
            f"Trace '{trace.name}' x and y data should have same length"
        )


@settings(max_examples=100)
@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=1, max_value=5)
)
def test_time_series_plot_preserves_data_order(n_periods, n_cols):
    """
    Property: For any DataFrame, the traces in the figure should appear in the
    same order as the requested columns.
    """
    # Create DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='MS')
    col_names = [f"col_{i}" for i in range(n_cols)]
    data = {col: np.random.randn(n_periods) for col in col_names}
    df = pd.DataFrame(data, index=dates)
    
    engine = EDAEngine()
    fig = engine.generate_time_series_plot(df, col_names)
    
    # Property: Traces should be in the same order as requested columns
    trace_names = [trace.name for trace in fig.data]
    assert trace_names == col_names, (
        f"Trace order should match requested column order.\n"
        f"Expected: {col_names}\n"
        f"Actual: {trace_names}"
    )


@settings(max_examples=100)
@given(
    st.integers(min_value=10, max_value=50),
    st.integers(min_value=1, max_value=3),
    st.integers(min_value=1, max_value=3)
)
def test_comparative_plot_separates_rate_and_credit_series(n_periods, n_rate_cols, n_credit_cols):
    """
    Property: For any comparative plot, rate series should be distinguishable
    from credit series (e.g., by line style or axis assignment).
    """
    # Create DataFrame
    dates = pd.date_range(start='2020-01-01', periods=n_periods, freq='MS')
    
    rate_cols = [f"rate_{i}" for i in range(n_rate_cols)]
    credit_cols = [f"credit_{i}" for i in range(n_credit_cols)]
    
    data = {}
    for col in rate_cols:
        data[col] = np.random.uniform(0, 10, n_periods)
    for col in credit_cols:
        data[col] = np.random.uniform(1000, 10000, n_periods)
    
    df = pd.DataFrame(data, index=dates)
    
    engine = EDAEngine()
    fig = engine.generate_comparative_plot(df, rate_cols, credit_cols)
    
    # Property: Credit series should have dashed lines (as per implementation)
    for trace in fig.data:
        if trace.name in credit_cols:
            assert trace.line.dash == 'dash', (
                f"Credit trace '{trace.name}' should have dashed line style"
            )
        elif trace.name in rate_cols:
            # Rate traces should have solid lines (default or explicitly set)
            assert trace.line.dash is None or trace.line.dash == 'solid', (
                f"Rate trace '{trace.name}' should have solid line style"
            )
