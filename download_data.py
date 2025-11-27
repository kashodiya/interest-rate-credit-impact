"""
Script to automatically download Federal Reserve data from FRED.

This script downloads H.15 interest rate data and G.19 consumer credit data
from the Federal Reserve Economic Data (FRED) database.

Usage:
    uv run python download_data.py
    
Optional arguments:
    --start YYYY-MM-DD    Start date (default: 2010-01-01)
    --end YYYY-MM-DD      End date (default: today)
"""

import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, timedelta
from pathlib import Path
import argparse


def download_h15_data(start_date, end_date, output_file="data/h15_interest_rates.csv"):
    """
    Download H.15 Selected Interest Rates from FRED.
    
    Args:
        start_date: Start date for data download
        end_date: End date for data download
        output_file: Path to save the CSV file
    """
    print("Downloading H.15 Interest Rates data from FRED...")
    
    # FRED series codes for interest rates
    series = {
        'fed_funds_rate': 'DFF',           # Federal Funds Effective Rate
        'treasury_1y': 'DGS1',             # 1-Year Treasury Constant Maturity Rate
        'treasury_2y': 'DGS2',             # 2-Year Treasury Constant Maturity Rate
        'treasury_10y': 'DGS10',           # 10-Year Treasury Constant Maturity Rate
        'prime_rate': 'DPRIME'             # Bank Prime Loan Rate
    }
    
    # Download each series
    data_frames = {}
    for col_name, series_code in series.items():
        try:
            print(f"  Downloading {col_name} ({series_code})...")
            df = pdr.DataReader(series_code, 'fred', start_date, end_date)
            df.columns = [col_name]
            data_frames[col_name] = df
        except Exception as e:
            print(f"  ⚠ Warning: Could not download {col_name}: {e}")
            # Create empty series with NaN
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            data_frames[col_name] = pd.DataFrame({col_name: pd.NA}, index=date_range)
    
    # Combine all series
    combined_df = pd.concat(data_frames.values(), axis=1)
    
    # Resample to monthly frequency (end of month)
    combined_df = combined_df.resample('MS').mean()  # MS = Month Start
    
    # Reset index to make date a column
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}")
    print(f"  ✓ Downloaded {len(combined_df)} monthly records")
    
    return combined_df


def download_g19_data(start_date, end_date, output_file="data/g19_consumer_credit.csv"):
    """
    Download G.19 Consumer Credit data from FRED.
    
    Args:
        start_date: Start date for data download
        end_date: End date for data download
        output_file: Path to save the CSV file
    """
    print("\nDownloading G.19 Consumer Credit data from FRED...")
    
    # FRED series codes for consumer credit
    series = {
        'total_credit': 'TOTALSL',         # Total Consumer Credit Outstanding
        'revolving_credit': 'REVOLSL',     # Revolving Consumer Credit Outstanding
        'non_revolving_credit': 'NONREVSL' # Non-Revolving Consumer Credit Outstanding
    }
    
    # Download each series
    data_frames = {}
    for col_name, series_code in series.items():
        try:
            print(f"  Downloading {col_name} ({series_code})...")
            df = pdr.DataReader(series_code, 'fred', start_date, end_date)
            df.columns = [col_name]
            data_frames[col_name] = df
        except Exception as e:
            print(f"  ⚠ Warning: Could not download {col_name}: {e}")
            # Create empty series with NaN
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            data_frames[col_name] = pd.DataFrame({col_name: pd.NA}, index=date_range)
    
    # Combine all series
    combined_df = pd.concat(data_frames.values(), axis=1)
    
    # Resample to monthly frequency (end of month)
    combined_df = combined_df.resample('MS').mean()  # MS = Month Start
    
    # Reset index to make date a column
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Date'}, inplace=True)
    
    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to {output_file}")
    print(f"  ✓ Downloaded {len(combined_df)} monthly records")
    
    return combined_df


def main():
    """Main function to download all Federal Reserve data."""
    parser = argparse.ArgumentParser(
        description='Download Federal Reserve data from FRED'
    )
    parser.add_argument(
        '--start',
        type=str,
        default='2010-01-01',
        help='Start date in YYYY-MM-DD format (default: 2010-01-01)'
    )
    parser.add_argument(
        '--end',
        type=str,
        default=datetime.now().strftime('%Y-%m-%d'),
        help='End date in YYYY-MM-DD format (default: today)'
    )
    parser.add_argument(
        '--h15-output',
        type=str,
        default='data/h15_interest_rates.csv',
        help='Output file for H.15 data (default: data/h15_interest_rates.csv)'
    )
    parser.add_argument(
        '--g19-output',
        type=str,
        default='data/g19_consumer_credit.csv',
        help='Output file for G.19 data (default: data/g19_consumer_credit.csv)'
    )
    
    args = parser.parse_args()
    
    # Parse dates
    start_date = pd.to_datetime(args.start)
    end_date = pd.to_datetime(args.end)
    
    print("=" * 60)
    print("Federal Reserve Data Download")
    print("=" * 60)
    print(f"Date Range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Data Source: FRED (Federal Reserve Economic Data)")
    print("=" * 60)
    
    try:
        # Download H.15 data
        h15_df = download_h15_data(start_date, end_date, args.h15_output)
        
        # Download G.19 data
        g19_df = download_g19_data(start_date, end_date, args.g19_output)
        
        print("\n" + "=" * 60)
        print("✓ Download completed successfully!")
        print("=" * 60)
        print(f"\nH.15 Data: {len(h15_df)} records")
        print(f"  Columns: {', '.join(h15_df.columns)}")
        print(f"\nG.19 Data: {len(g19_df)} records")
        print(f"  Columns: {', '.join(g19_df.columns)}")
        print("\nYou can now run the analysis:")
        print("  uv run python main.py")
        
    except Exception as e:
        print(f"\n✗ Error downloading data: {e}")
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Verify FRED is accessible: https://fred.stlouisfed.org/")
        print("  3. Try a different date range")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
