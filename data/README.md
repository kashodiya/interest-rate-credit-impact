# Federal Reserve Data

This directory contains Federal Reserve datasets for the Interest Rate and Consumer Credit Analysis System.

## Data Sources

### H.15 Selected Interest Rates
- **Source**: Federal Reserve Statistical Release H.15
- **URL**: https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15
- **Description**: Contains selected interest rates including:
  - Federal Funds Rate
  - Treasury Yields (1-year, 2-year, 10-year)
  - Bank Prime Loan Rate

### G.19 Consumer Credit
- **Source**: Federal Reserve Statistical Release G.19
- **URL**: https://www.federalreserve.gov/datadownload/Choose.aspx?rel=G19
- **Description**: Contains consumer credit data including:
  - Total Consumer Credit Outstanding
  - Revolving Credit (e.g., credit cards)
  - Non-Revolving Credit (e.g., auto loans, student loans)

## Download Instructions

1. Visit the Federal Reserve Data Download Portal:
   - H.15: https://www.federalreserve.gov/datadownload/Choose.aspx?rel=H15
   - G.19: https://www.federalreserve.gov/datadownload/Choose.aspx?rel=G19

2. Select the following options:
   - **Data Range**: Select your desired date range (recommend at least 5 years)
   - **Frequency**: Monthly
   - **File Format**: CSV
   - **File Structure**: Dates down the first column

3. For H.15, select these series:
   - Federal Funds Effective Rate (FEDFUNDS)
   - 1-Year Treasury Constant Maturity Rate (DGS1)
   - 2-Year Treasury Constant Maturity Rate (DGS2)
   - 10-Year Treasury Constant Maturity Rate (DGS10)
   - Bank Prime Loan Rate (DPRIME)

4. For G.19, select these series:
   - Total Consumer Credit Outstanding (TOTALSL)
   - Revolving Credit Outstanding (REVOLSL)
   - Non-Revolving Credit Outstanding (NONREVSL)

5. Download the CSV files and place them in this directory:
   - `data/h15_interest_rates.csv`
   - `data/g19_consumer_credit.csv`

## Sample Data

Sample datasets are provided for testing purposes:
- `sample_h15_interest_rates.csv` - Sample H.15 data (2020-2023)
- `sample_g19_consumer_credit.csv` - Sample G.19 data (2020-2023)

These sample files contain synthetic data for demonstration purposes only.

## Data Format

### H.15 CSV Format
```
Date,fed_funds_rate,treasury_1y,treasury_2y,treasury_10y,prime_rate
2020-01-01,1.55,1.54,1.57,1.88,4.75
2020-02-01,1.58,1.42,1.38,1.57,4.75
...
```

### G.19 CSV Format
```
Date,total_credit,revolving_credit,non_revolving_credit
2020-01-01,4192.5,1089.2,3103.3
2020-02-01,4201.3,1095.7,3105.6
...
```

## Notes

- All interest rates are in percentages (e.g., 1.55 = 1.55%)
- All credit amounts are in billions of dollars
- Dates should be in YYYY-MM-DD format
- Missing values should be left blank or marked as NaN
