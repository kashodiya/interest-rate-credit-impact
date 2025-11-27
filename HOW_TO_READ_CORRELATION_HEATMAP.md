# How to Read the Correlation Heatmap

## What is a Correlation Heatmap?

A correlation heatmap is a visual representation showing how strongly pairs of variables are related to each other. In your dashboard, it shows relationships between interest rates and consumer credit.

## Understanding the Colors

### Color Scale (Red-Blue)
- **Dark Blue** = Strong Positive Correlation (+1.0)
  - When one variable increases, the other increases
  - Example: Treasury 1Y and Treasury 10Y rates move together
  
- **White/Light** = No Correlation (0.0)
  - Variables are independent of each other
  - No predictable relationship
  
- **Dark Red** = Strong Negative Correlation (-1.0)
  - When one variable increases, the other decreases
  - Example: Interest rates up → Consumer credit down

### Correlation Values
- **+1.0** = Perfect positive correlation
- **+0.7 to +1.0** = Strong positive correlation
- **+0.3 to +0.7** = Moderate positive correlation
- **-0.3 to +0.3** = Weak or no correlation
- **-0.7 to -0.3** = Moderate negative correlation
- **-1.0 to -0.7** = Strong negative correlation
- **-1.0** = Perfect negative correlation

## How to Read the Heatmap

### 1. The Diagonal (Always 1.0)
The diagonal line from top-left to bottom-right shows each variable correlated with itself.
- These are always **1.0** (perfect correlation)
- They're always dark blue
- **Ignore these** - they don't tell us anything useful

### 2. Mirror Image
The heatmap is symmetrical:
- Top-right triangle = mirror of bottom-left triangle
- Correlation of A→B = Correlation of B→A
- You only need to read one half

### 3. Finding Relationships

**Look for Interest Rate vs Credit relationships:**

#### Fed Funds Rate vs Total Credit
- **Find "fed_funds_rate" row**
- **Look across to "total_credit" column**
- The color and number show the relationship

**Example Interpretations:**

| Correlation | Color | Meaning |
|-------------|-------|---------|
| -0.65 | Red | When Fed raises rates, credit tends to decrease |
| +0.45 | Blue | When rates go up, credit also goes up (unusual!) |
| -0.15 | White | Weak relationship, rates don't predict credit well |

## Real-World Examples from Your Data

### Expected Patterns:

1. **Interest Rates Correlate with Each Other**
   - Fed Funds Rate ↔ Prime Rate: **Strong positive** (~0.95)
   - Treasury 1Y ↔ Treasury 10Y: **Strong positive** (~0.85)
   - **Why?** All rates tend to move together with Fed policy

2. **Interest Rates vs Consumer Credit**
   - Fed Funds Rate ↔ Total Credit: **Negative** (~-0.40 to -0.70)
   - **Why?** Higher rates make borrowing expensive → less credit
   
3. **Credit Types Correlate with Each Other**
   - Total Credit ↔ Revolving Credit: **Strong positive** (~0.95)
   - Total Credit ↔ Non-Revolving Credit: **Strong positive** (~0.98)
   - **Why?** Total credit is the sum of both types

### Surprising Patterns to Look For:

1. **Positive Rate-Credit Correlation**
   - If you see positive correlation between rates and credit
   - **Possible reasons:**
     - Strong economy → Fed raises rates AND people borrow more
     - Time lag effects not captured
     - Other economic factors dominating

2. **Weak Correlations**
   - If correlations are near zero
   - **Possible reasons:**
     - Relationship is non-linear
     - Time lags are important (use lag analysis!)
     - Other factors are more important

## Step-by-Step: Reading Your Heatmap

### Step 1: Identify the Variables
Look at the labels on both axes:
- **Interest Rates**: fed_funds_rate, treasury_1y, treasury_2y, treasury_10y, prime_rate
- **Credit Variables**: total_credit, revolving_credit, non_revolving_credit

### Step 2: Pick a Relationship to Examine
Example: "How does Fed Funds Rate relate to Total Credit?"

### Step 3: Find the Intersection
- Find "fed_funds_rate" on one axis
- Find "total_credit" on the other axis
- Look at where they intersect

### Step 4: Read the Value
- Hover over the cell to see the exact number
- Note the color intensity

### Step 5: Interpret
- **Negative value (red)**: Inverse relationship
  - Higher rates → Lower credit (expected!)
- **Positive value (blue)**: Direct relationship
  - Higher rates → Higher credit (investigate why!)
- **Near zero (white)**: No clear relationship
  - Need to look at lag analysis or other factors

## Common Questions

### Q: Why are some correlations positive when we expect negative?
**A:** Several reasons:
1. **Time lags**: Credit responds to rates with a delay (check lag analysis panel)
2. **Economic growth**: Strong economy causes both high rates AND high credit
3. **Confounding factors**: Other variables affecting both
4. **Data period**: Specific time period may show unusual patterns

### Q: What's a "good" correlation value?
**A:** Depends on context:
- **Economics/Finance**: 0.3-0.5 is often considered meaningful
- **Physical sciences**: Usually expect >0.7
- **Your analysis**: -0.4 to -0.7 for rate-credit is realistic and useful

### Q: Should I worry about low correlations?
**A:** Not necessarily:
- Low correlation doesn't mean no relationship
- Relationship might be non-linear
- Time lags might be important (check lag analysis!)
- Multiple factors might be at play

### Q: Which correlations matter most?
**A:** Focus on:
1. **Fed Funds Rate ↔ Credit variables** (main policy tool)
2. **Treasury 10Y ↔ Credit variables** (long-term borrowing costs)
3. **Prime Rate ↔ Credit variables** (consumer lending rates)

## Using Heatmap with Other Panels

### 1. Heatmap + Scatter Plots
- Heatmap shows **strength** of relationship
- Scatter plots show **shape** of relationship
- Use together to understand the full picture

### 2. Heatmap + Lag Analysis
- Heatmap shows **current** correlation
- Lag analysis shows **delayed** correlation
- Credit might respond to rates with 3-6 month delay

### 3. Heatmap + Regression
- Heatmap shows **pairwise** relationships
- Regression shows **combined** effect of multiple rates
- Regression accounts for interactions between variables

## Pro Tips

### 1. Look for Clusters
- Groups of variables with similar colors
- Indicates they move together as a group

### 2. Compare Across Credit Types
- Does revolving credit respond differently than non-revolving?
- Different sensitivities suggest different borrowing behaviors

### 3. Check Rate Relationships
- If Treasury rates don't correlate with Fed Funds Rate
- Might indicate market expectations diverging from Fed policy

### 4. Time Period Matters
- Correlations can change over time
- 2008 crisis vs 2020 pandemic vs 2022 rate hikes
- Consider downloading different time periods to compare

## Example Analysis Workflow

1. **Start with the heatmap** - Get overview of all relationships
2. **Identify interesting patterns** - Strong correlations or surprises
3. **Check scatter plots** - Visualize the specific relationships
4. **Review lag analysis** - See if time delays explain weak correlations
5. **Examine regression** - Understand combined effects
6. **Test scenarios** - Use simulator to predict impacts

## Quick Reference Card

| Correlation | Strength | Color | Interpretation |
|-------------|----------|-------|----------------|
| 0.9 to 1.0 | Very Strong | Dark Blue | Almost perfect positive relationship |
| 0.7 to 0.9 | Strong | Blue | Strong positive relationship |
| 0.4 to 0.7 | Moderate | Light Blue | Moderate positive relationship |
| 0.1 to 0.4 | Weak | Very Light Blue | Weak positive relationship |
| -0.1 to 0.1 | None | White | No meaningful relationship |
| -0.4 to -0.1 | Weak | Very Light Red | Weak negative relationship |
| -0.7 to -0.4 | Moderate | Light Red | Moderate negative relationship |
| -0.9 to -0.7 | Strong | Red | Strong negative relationship |
| -1.0 to -0.9 | Very Strong | Dark Red | Almost perfect negative relationship |

## What to Report

When sharing findings from the heatmap:

1. **State the correlation value**: "Fed Funds Rate and Total Credit have a correlation of -0.65"

2. **Describe the strength**: "This is a moderate to strong negative correlation"

3. **Explain the meaning**: "When the Fed raises rates, consumer credit tends to decrease"

4. **Add context**: "This makes economic sense as higher rates make borrowing more expensive"

5. **Note limitations**: "However, this doesn't account for time lags, which our lag analysis shows are important"

## Further Reading

For deeper understanding:
- Check the **Lag Analysis panel** for time-delayed relationships
- Review the **Regression panel** for multivariate relationships
- Use the **Scenario Simulator** to test predictions
- See `README.md` section "Interpreting Results" for more guidance
