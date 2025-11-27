# Federal Reserve Data Download Guide

## Quick Links

- **H.15 Interest Rates**: https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15
- **G.19 Consumer Credit**: https://www.federalreserve.gov/datadownload/Choose.aspx?rel=G19

## H.15 Series to Download

| Series Code | Description |
|-------------|-------------|
| `RIFSPFF_N.M` | Federal Funds Rate |
| `RIFLGFCY01_N.M` | 1-Year Treasury Constant Maturity Rate |
| `RIFLGFCY02_N.M` | 2-Year Treasury Constant Maturity Rate |
| `RIFLGFCY10_N.M` | 10-Year Treasury Constant Maturity Rate |
| `RIFSPBLP_N.M` | Bank Prime Loan Rate |

## G.19 Series to Download

| Series Code | Description |
|-------------|-------------|
| `TOTALSL` | Total Consumer Credit Outstanding |
| `REVOLSL` | Revolving Credit Outstanding |
| `NONREVSL` | Non-Revolving Credit Outstanding |

## Download Settings

For both H.15 and G.19:
- **Frequency**: Monthly
- **File Format**: CSV
- **File Structure**: Dates down the first column
- **Date Range**: Your choice (recommend at least 5 years, e.g., 2010-2024)

## File Naming

Save downloaded files as:
- `data/h15_interest_rates.csv`
- `data/g19_consumer_credit.csv`

## Column Name Mapping

The system automatically recognizes these column name variations:

### H.15 Columns
- Fed Funds Rate: `fed_funds_rate`, `FEDFUNDS`, `Fed Funds Rate`, `Federal Funds Rate`, `DFF`, `RIFSPFF_N.M`
- Treasury 1Y: `treasury_1y`, `DGS1`, `Treasury 1Y`, `1-Year Treasury`, `GS1`, `RIFLGFCY01_N.M`
- Treasury 2Y: `treasury_2y`, `DGS2`, `Treasury 2Y`, `2-Year Treasury`, `GS2`, `RIFLGFCY02_N.M`
- Treasury 10Y: `treasury_10y`, `DGS10`, `Treasury 10Y`, `10-Year Treasury`, `GS10`, `RIFLGFCY10_N.M`
- Prime Rate: `prime_rate`, `DPRIME`, `Bank Prime Loan Rate`, `Prime Rate`, `PRIME`, `RIFSPBLP_N.M`

### G.19 Columns
- Total Credit: `total_credit`, `TOTALSL`, `Total Consumer Credit`, `Total Credit`, `CONSUMER`
- Revolving Credit: `revolving_credit`, `REVOLSL`, `Revolving Credit`, `Revolving`, `REVOL`
- Non-Revolving Credit: `non_revolving_credit`, `NONREVSL`, `Non-Revolving Credit`, `Nonrevolving`, `NONREVOL`

## After Downloading

1. Place files in `data/` folder
2. Run analysis: `uv run python main.py`
3. Launch dashboard: `uv run python run_dashboard.py`

## Troubleshooting

### Column Names Don't Match
If the Federal Reserve changes their column naming:
1. Open your downloaded CSV file
2. Check the actual column names
3. Update `data_processing/loader.py` to add the new names to the `column_mapping` dictionaries

### Date Format Issues
The system expects dates in format: `YYYY-MM-DD` or `YYYY-MM-01`
If dates are in a different format, the loader will attempt to parse them automatically.

### Missing Data
The system handles missing data (NaN values) gracefully. Missing values will be:
- Reported during validation
- Excluded from calculations where necessary
- Preserved in merged datasets

## Alternative: FRED API

For automated downloads, consider using the FRED (Federal Reserve Economic Data) API:
- Website: https://fred.stlouisfed.org/
- API Docs: https://fred.stlouisfed.org/docs/api/fred/
- Python Package: `pip install fredapi`

Example series codes for FRED:
- `DFF` - Federal Funds Rate
- `DGS1`, `DGS2`, `DGS10` - Treasury Yields
- `DPRIME` - Prime Rate
- `TOTALSL`, `REVOLSL`, `NONREVSL` - Consumer Credit
