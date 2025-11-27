"""
DataLoader class for loading Federal Reserve datasets.
"""

import pandas as pd
from pathlib import Path


class DataParseError(Exception):
    """Raised when CSV parsing fails due to malformed data."""
    pass


class DataLoader:
    """Loads Federal Reserve H.15 and G.19 datasets from CSV files."""
    
    def load_h15_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load H.15 Selected Interest Rates dataset.
        
        Extracts Fed Funds Rate, Treasury Yields (1Y, 2Y, 10Y), and Bank Prime Loan Rate.
        
        Args:
            filepath: Path to H.15 CSV file
            
        Returns:
            DataFrame with parsed time series data indexed by date
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            DataParseError: If CSV parsing fails
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(
                f"H.15 dataset file not found at {filepath}. "
                f"Please download from Federal Reserve DDP portal: "
                f"https://www.federalreserve.gov/datadownload/"
            )
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Parse date column - try common date column names
            date_columns = ['Date', 'date', 'DATE', 'TIME_PERIOD', 'Time Period']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                raise DataParseError(
                    f"No date column found in H.15 dataset. "
                    f"Expected one of: {date_columns}"
                )
            
            # Convert date column to datetime and set as index
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            df.index.name = 'date'
            
            # Extract required columns - map common column name variations
            column_mapping = {
                'fed_funds_rate': ['fed_funds_rate', 'FEDFUNDS', 'Fed Funds Rate', 'Federal Funds Rate', 'DFF'],
                'treasury_1y': ['treasury_1y', 'DGS1', 'Treasury 1Y', '1-Year Treasury', 'GS1'],
                'treasury_2y': ['treasury_2y', 'DGS2', 'Treasury 2Y', '2-Year Treasury', 'GS2'],
                'treasury_10y': ['treasury_10y', 'DGS10', 'Treasury 10Y', '10-Year Treasury', 'GS10'],
                'prime_rate': ['prime_rate', 'DPRIME', 'Bank Prime Loan Rate', 'Prime Rate', 'PRIME']
            }
            
            result_df = pd.DataFrame(index=df.index)
            
            for standard_name, possible_names in column_mapping.items():
                found = False
                for col_name in possible_names:
                    if col_name in df.columns:
                        result_df[standard_name] = pd.to_numeric(df[col_name], errors='coerce')
                        found = True
                        break
                
                if not found:
                    # Create column with NaN if not found
                    result_df[standard_name] = pd.NA
            
            return result_df
            
        except pd.errors.ParserError as e:
            raise DataParseError(f"Failed to parse H.15 CSV file: {str(e)}")
        except Exception as e:
            raise DataParseError(f"Error loading H.15 dataset: {str(e)}")
    
    def load_g19_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load G.19 Consumer Credit dataset.
        
        Extracts Total Consumer Credit, Revolving Credit, and Non-Revolving Credit.
        
        Args:
            filepath: Path to G.19 CSV file
            
        Returns:
            DataFrame with parsed time series data indexed by date
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            DataParseError: If CSV parsing fails
        """
        if not Path(filepath).exists():
            raise FileNotFoundError(
                f"G.19 dataset file not found at {filepath}. "
                f"Please download from Federal Reserve DDP portal: "
                f"https://www.federalreserve.gov/datadownload/"
            )
        
        try:
            # Read CSV file
            df = pd.read_csv(filepath)
            
            # Parse date column - try common date column names
            date_columns = ['Date', 'date', 'DATE', 'TIME_PERIOD', 'Time Period']
            date_col = None
            for col in date_columns:
                if col in df.columns:
                    date_col = col
                    break
            
            if date_col is None:
                raise DataParseError(
                    f"No date column found in G.19 dataset. "
                    f"Expected one of: {date_columns}"
                )
            
            # Convert date column to datetime and set as index
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            df.index.name = 'date'
            
            # Extract required columns - map common column name variations
            column_mapping = {
                'total_credit': ['total_credit', 'TOTALSL', 'Total Consumer Credit', 'Total Credit', 'CONSUMER'],
                'revolving_credit': ['revolving_credit', 'REVOLSL', 'Revolving Credit', 'Revolving', 'REVOL'],
                'non_revolving_credit': ['non_revolving_credit', 'NONREVSL', 'Non-Revolving Credit', 'Nonrevolving', 'NONREVOL']
            }
            
            result_df = pd.DataFrame(index=df.index)
            
            for standard_name, possible_names in column_mapping.items():
                found = False
                for col_name in possible_names:
                    if col_name in df.columns:
                        result_df[standard_name] = pd.to_numeric(df[col_name], errors='coerce')
                        found = True
                        break
                
                if not found:
                    # Create column with NaN if not found
                    result_df[standard_name] = pd.NA
            
            return result_df
            
        except pd.errors.ParserError as e:
            raise DataParseError(f"Failed to parse G.19 CSV file: {str(e)}")
        except Exception as e:
            raise DataParseError(f"Error loading G.19 dataset: {str(e)}")
