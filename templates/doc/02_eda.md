# 02 — EDA (Exploratory Data Analysis)

## 1) Data Health
- Row/Column counts
- Missingness summary (global + key columns)
- Duplicates & key integrity
- Target distribution & base rate

## 2) Target Analysis
- Overall distribution
- By key segments (e.g. property / team / channel)
- Time trend (if applicable)
- Potential label issues (outliers, unexplainable points)

## 3) Feature Analysis
### Numeric
- Distribution / skewness
- Winsorize / log transform candidate
- Correlation with target (watch for leakage)

### Categorical
- Cardinality
- Rare levels
- Target mean encoding risk (must be fold-safe)

### Date/Time (if applicable)
- Seasonality
- Rolling stats (must be computed fold-safe)
- Time gaps / missing intervals

## 4) Data Leakage & Bias Checks
- Leakage candidates list (specific column names + rationale)
- Bias / fairness considerations (if applicable)

## 5) EDA Conclusions → Actions
- What to drop
- What to transform
- What to engineer
- Hypotheses to test in modeling