## 🎯 **Project Overview**

**Title**: *How Do Interest Rates Shape Consumer Borrowing?*  
**Objective**: Analyze the relationship between federal interest rates and consumer credit trends (credit cards, auto loans, student loans) using Federal Reserve data.

---

## 📦 **Datasets to Use**

1. **H.15 – Selected Interest Rates**  
   - Fed Funds Rate (Effective)
   - Treasury Yields (1Y, 2Y, 10Y)
   - Bank Prime Loan Rate

2. **G.19 – Consumer Credit**  
   - Total Consumer Credit Outstanding
   - Revolving Credit (e.g., credit cards)
   - Non-Revolving Credit (e.g., auto, student loans)

You can download both from the [Federal Reserve DDP portal](https://www.federalreserve.gov/datadownload/).

---

## 🧪 **Analytical Techniques**

### 1. **Exploratory Data Analysis (EDA)**
- Visualize trends over time (line plots)
- Compare growth rates of credit vs. interest rate changes
- Identify major economic events (e.g., 2008, 2020) as annotations

### 2. **Time Series Correlation & Lag Analysis**
- Use **cross-correlation** to detect lag between rate changes and credit response
- Identify whether credit reacts immediately or with delay (e.g., 3-month lag)

### 3. **Regression Modeling**
- Build **multivariate regression** models:
  - Dependent variable: Credit growth (monthly or quarterly)
  - Independent variables: Fed Funds Rate, Prime Rate, Treasury Yields
- Include lagged variables if needed

### 4. **Forecasting**
- Use **ARIMA** or **Prophet** to forecast:
  - Future consumer credit trends under different interest rate scenarios
- Scenario modeling: What happens if rates rise/fall by 1%?

---

## 📊 **Dashboard Showcase Ideas**

You can build this in **Power BI**, **Tableau**, or **Plotly Dash**. Key components:

| Section | Description |
|--------|-------------|
| **Time Series Viewer** | Interactive line charts for interest rates and credit categories |
| **Correlation Explorer** | Heatmap or scatter plots showing correlation and lag effects |
| **Regression Insights** | Model summary, coefficients, and R² values |
| **Forecast Panel** | Forecasted credit trends with confidence intervals |
| **Scenario Simulator** | Slider to simulate rate changes and see projected credit impact |

---

## 🧠 **Insights to Highlight**

- Does consumer credit **contract or expand** when rates rise?
- Which type of credit is **most sensitive** to rate changes?
- Are there **non-linear effects** or thresholds?
- How long does it take for rate changes to affect borrowing?

---

## 🛠️ **Tools & Libraries**

- **Python**: `pandas`, `statsmodels`, `prophet`, `matplotlib`, `seaborn`, `plotly`
- **R**: `forecast`, `ggplot2`, `dplyr`, `shiny`
- **BI Tools**: Tableau, Power BI, or Looker Studio

---

Would you like help with:
- A **starter notebook** in Python or R?
- Guidance on **data cleaning and merging** H.15 and G.19?
- A **dashboard wireframe** or mockup?

Let me know how you'd like to proceed!
