# EDA Report — {PROJECT_NAME}

> **Generated**: 2026-06-11 09:30:00
> **Data source**: `projects/{PROJECT_NAME}/data/processed/train.parquet`
> **Config**: `projects/{PROJECT_NAME}/configs/baseline.yaml`

---

## 1. Data Overview

- **Rows**: 120,000
- **Columns**: 25
- **Target column**: `target`
- **Target dtype**: int64

### Column List
```
   0. id (int64)
   1. feature_0 (float64)
   2. feature_1 (float64)
   3. feature_2 (float64)
   4. feature_3 (float64)
   5. feature_4 (float64)
   6. feature_5 (float64)
   7. feature_6 (float64)
   8. feature_7 (float64)
   9. feature_8 (float64)
  10. feature_9 (float64)
  11. feature_10 (float64)
  12. feature_11 (float64)
  13. feature_12 (float64)
  14. feature_13 (float64)
  15. feature_14 (float64)
  16. feature_15 (float64)
  17. feature_16 (float64)
  18. feature_17 (float64)
  19. feature_18 (float64)
  20. feature_19 (float64)
  21. feature_20 (float64)
  22. feature_21 (float64)
  23. feature_22 (float64)
  24. target (int64)
```

### Data Type Summary
- **int64**: 2
- **float64**: 23

---

## 2. Missing Values

**Overall missing rate**: 0.00%

No missing values found.

---

## 3. Duplicates

- **Duplicate rows**: 0 (0.00%)

---

## 4. Target Variable Analysis

- **Column**: `target`
- **Dtype**: int64
- **Non-null count**: 120,000 / 120,000
- **Unique classes**: 2

| Class | Count | Proportion |
|-------|-------|-----------|
| 0 | 96,000 | 80.00% |
| 1 | 24,000 | 20.00% |

- **Imbalance ratio (major/minor)**: 4.00

---

## 5. Numeric Features

**Total numeric features**: 23

| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max | Skew | Missing% |
|--------|-------|------|-----|-----|-----|-----|-----|-----|------|---------|
| `feature_0` | 120,000 | 0.5123 | 0.2891 | -3.2140 | 0.3102 | 0.5200 | 0.7201 | 3.4512 | 0.32 | 0.00% |
| `feature_1` | 120,000 | -0.0234 | 0.9502 | -4.8920 | -0.6700 | -0.0100 | 0.6200 | 4.1230 | 0.12 | 0.00% |
| `feature_2` | 120,000 | 0.1500 | 0.1800 | -2.0000 | 0.0500 | 0.1500 | 0.2600 | 3.5000 | 1.50 | 0.00% |
| `feature_3` | 120,000 | 0.4500 | 0.3200 | -1.5000 | 0.2500 | 0.4500 | 0.6500 | 2.1000 | 0.08 | 0.00% |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
| `feature_22` | 120,000 | 0.0000 | 1.0000 | -4.0000 | -0.6700 | 0.0000 | 0.6700 | 4.0000 | 0.05 | 0.00% |

---

## 6. Categorical Features

No categorical features found.

---

## 7. Feature Correlations

**Total numeric features for correlation**: 23

### Top Correlations with Target (`target`)

| Feature | Correlation |
|---------|-------------|
| `feature_5` | 0.1832 |
| `feature_12` | 0.1245 |
| `feature_1` | 0.0891 |
| `feature_18` | 0.0782 |
| `feature_3` | 0.0654 |
| `feature_9` | -0.0521 |
| `feature_15` | -0.0412 |
| `feature_7` | 0.0389 |
| `feature_20` | -0.0315 |
| `feature_10` | 0.0287 |
| ... and 13 more features

---

## 8. Data Leakage Check

No obvious leak columns detected.

---

## 9. CV Strategy Validation

- **CV Type**: StratifiedKFold
- **N Splits**: 5
- **Shuffle**: True
- **Random State**: 42

---

## 10. Feature Engineering Suggestions

### General Suggestions
- Create interaction features (cross features)
- Try group aggregations (group by categorical, aggregate numeric)
- Check for time-based features if datetime columns exist
- Run featuretools for automated feature engineering

---

## 11. Summary

| Aspect | Value |
|--------|-------|
| Dataset size | 120,000 rows x 25 cols |
| Target column | `target` |
| Overall missing rate | 0.00% |
| Numeric features | 23 |
| Categorical features | 0 |
| Duplicate rows | 0 (0.00%) |
| Target classes | 2 |
| Majority class | 80.0% |

---