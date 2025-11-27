"""
DataMerger class for merging H.15 and G.19 datasets.
"""

import pandas as pd
from typing import Tuple


class MergeError(Exception):
    """Raised when datasets cannot be merged due to incompatible time periods."""
    pass


class DataMerger:
    """Merges H.15 and G.19 datasets on time periods."""
    
    def align_time_periods(self, df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align time periods between two DataFrames.
        
        Finds overlapping dates between the two DataFrames and returns
        both DataFrames filtered to only include those dates.
        
        Args:
            df1: First DataFrame with datetime index
            df2: Second DataFrame with datetime index
            
        Returns:
            Tuple of aligned DataFrames (df1_aligned, df2_aligned)
            
        Raises:
            MergeError: If there are no overlapping time periods
        """
        # Ensure both DataFrames have datetime indices
        if not isinstance(df1.index, pd.DatetimeIndex):
            raise ValueError("df1 must have a DatetimeIndex")
        if not isinstance(df2.index, pd.DatetimeIndex):
            raise ValueError("df2 must have a DatetimeIndex")
        
        # Find overlapping dates
        common_dates = df1.index.intersection(df2.index)
        
        if len(common_dates) == 0:
            df1_range = f"{df1.index.min()} to {df1.index.max()}"
            df2_range = f"{df2.index.min()} to {df2.index.max()}"
            raise MergeError(
                f"No overlapping time periods found. "
                f"Dataset 1 range: {df1_range}, "
                f"Dataset 2 range: {df2_range}"
            )
        
        # Filter both DataFrames to common dates
        df1_aligned = df1.loc[common_dates].sort_index()
        df2_aligned = df2.loc[common_dates].sort_index()
        
        return df1_aligned, df2_aligned
    
    def merge_datasets(self, h15_df: pd.DataFrame, g19_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge H.15 and G.19 datasets on matching time periods.
        
        Aligns the datasets by time period and combines them into a single DataFrame.
        Missing data in the merged result is preserved as NaN values.
        
        Args:
            h15_df: H.15 dataset DataFrame with datetime index
            g19_df: G.19 dataset DataFrame with datetime index
            
        Returns:
            Merged DataFrame with aligned time periods containing all columns
            from both datasets
            
        Raises:
            MergeError: If datasets have no overlapping time periods
        """
        # First align the time periods
        h15_aligned, g19_aligned = self.align_time_periods(h15_df, g19_df)
        
        # Merge the aligned DataFrames on their indices
        # Using outer join to preserve all data, though they should be aligned
        merged_df = pd.merge(
            h15_aligned,
            g19_aligned,
            left_index=True,
            right_index=True,
            how='outer'
        )
        
        # Sort by date to ensure chronological order
        merged_df = merged_df.sort_index()
        
        return merged_df
