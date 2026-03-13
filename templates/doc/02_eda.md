# 02 — EDA (Exploratory Data Analysis)

## 1) Data Health
- Row/Column counts
- Missingness summary（全局 + 重要欄位）
- Duplicates & key integrity
- Target distribution & base rate

## 2) Target Analysis
- Overall distribution
- By key segments（例如 property / team / channel）
- Time trend（如適用）
- Potential label issues（極端值、不可解釋點）

## 3) Feature Analysis
### Numeric
- Distribution / skewness
- Winsorize / log transform candidate
- Correlation with target（注意 leakage）

### Categorical
- Cardinality
- Rare levels
- Target mean encoding risk（需 fold-safe）

### Date/Time (if applicable)
- Seasonality
- Rolling stats (must be computed fold-safe)
- Time gaps / missing intervals

## 4) Data Leakage & Bias Checks
- Leakage candidates list（具體欄位名 + 理由）
- Bias / fairness considerations（如適用）

## 5) EDA Conclusions → Actions
- What to drop
- What to transform
- What to engineer
- Hypotheses to test in modeling