# 04 — Modeling

## 1) Baseline (Must Have)
- Model: {{BASELINE_MODEL}}
- **Default rule**: If task is **tabular classification/regression** (non-time-series, non-ranking), baseline model should be **LightGBM**.
- Override baseline model only when task constraints require it (e.g., ranking objective, strict time-series modeling constraints, or user-specified requirement).
- Feature set id: {{FEATURE_SET_ID}}
- CV: per `doc/03_cv_strategy.md`
- Baseline score: {{BASELINE_SCORE}}

## 2) Feature Pipeline (Fold-safe)
- Missing:
- Numeric transform:
- Categorical encoding:
- Text (if any):
- Time-series feature policy (if any): lag/rolling computed fold-safe
- Ranking feature policy (if any): query-level features computed fold-safe

### 2.1 Feature Engineering (Create New Features)
- Goal: Systematically create new features for **numeric / categorical / datetime / geometric** columns while staying **fold-safe and avoiding leakage**.
- Numeric (continuous):
  - Ratios: e.g. `x1 / (x2 + 1)`, share of total.
  - Nonlinear transforms: log / sqrt / clipping / winsorization (confirm heavy tail in EDA first).
  - Interactions: `x1 * x2`, `x1 / x2`, bucket then one-hot and interact with other variables.
- Categorical:
  - Target encoding / count encoding for high-cardinality columns (**must be fold-safe**; fit encoder on train fold only).
  - Combined columns: `country + device`, `channel + weekday`, etc.
- Datetime:
  - Cyclical: hour-of-day / day-of-week / month-of-year; add sin/cos encoding if needed.
  - Relative time: days since event, distance from current time (must avoid future leakage).
  - Lag / rolling features: `value_t-1`, `rolling_mean_7d` (compute within each fold train window, then apply to valid).
- Geometric / Spatial (if coordinates/distance columns exist):
  - Distance: Euclidean / Haversine between points.
  - Direction: bearing / quadrant.
  - Area aggregates: mean, density by grid/admin unit (**train fold data only**).

### 2.2 Auto Feature / AutoML Libraries
- Allowed and recommended under these conditions:
  - **featuretools**: For automated feature synthesis (multi-table / time-series); generated features must:
    - Be fit per **fold** (each train fold fit separately, then transform valid) to avoid leakage.
    - Be written into `src/features.py` as a pipeline for reproducibility.
  - **pycaret** (or other AutoML):
    - Use for quick baseline and feature combo exploration.
    - If adopting a pipeline/features: document settings, model family, key params in this doc and `runs/<run_id>/params.json`.
- Principle:
  - Auto feature / AutoML serves as a **candidate features / pipeline generator**; once adopted, implement as explicit sklearn/LightGBM pipeline in this repo.
  - All generated features must satisfy `AGENT_RULES.md` fold-safe / no-leakage rules.

### 2.3 Data Wide Search (when minimal context)

When the user **only drops a dataset** without filling `00_problem_statement.md` or `01_data_card.md`, the agent can run **Data Wide Search** to bootstrap context and strengthen feature engineering.

**Triggers**:
- Problem statement or Data Card is empty, placeholder-only, or missing.
- User explicitly asks: "help me understand this data", "suggest features", "what can I do with this dataset?"

**Flow**:
1. **Infer minimal context from data**: `df.info()`, schema scan, sample rows; infer target, ID, datetime columns and task type.
2. **Web / domain search**: If dataset/domain hints exist → search similar problems, Kaggle notebooks, domain best practices; collect feature candidates (ratios, aggregates, time transforms, encodings).
3. **User–agent interaction**: Summarize findings and propose inferred target, task type, feature candidates; ask user to confirm or correct; update docs per feedback.
4. **Document and implement**: Write inferred content into `doc/*`; record Data Wide Search findings and adopted hypotheses in this section.

**This project's Data Wide Search record** (if executed):
- Search keywords / sources:
- Discovered feature candidates (and adoption status):
- User confirmations / corrections:

### 2.4 Target Transform (per task and distribution)
- Purpose: Make target match model/metric assumptions (0/1, continuous, reasonable scale); record in config/run; invert at inference.
- **Binary classification**:
  - If target is string/category (e.g. `Yes`/`No`, `0`/`1` strings), **must convert to numeric 0/1 first**.
  - Convention: `pos_label` (e.g. `Yes`) → 1, other → 0; if not specified use `sorted(unique)`: first → 0, second → 1.
  - Record in `configs/*.yaml` `task.target_positive_label`; after conversion run StratifiedKFold / train / evaluate; predicted proba = positive class proba.
- **Multiclass**:
  - For string labels, use `LabelEncoder` or fixed mapping to 0..K-1; write mapping to run artifacts for consistent inference.
- **Regression**:
  - **Large or right-skewed values**: Apply `log1p` to target, train, then `np.expm1(pred)` at inference; or choose `log`/`sqrt`/winsorize per EDA and document in doc + params.
  - **Non-negative with zeros**: Prefer `log1p` to avoid log(0).
  - Transform and inverse must be fixed in pipeline (same for train/valid/test); if metric is in original scale, compute after inverse.
- **Record**:
  - In this section of `doc/04_modeling.md`: target transform used (binary pos_label, regression log1p or not).
  - In `runs/<run_id>/params.json` or `artifacts/target_mapping.json` for evaluate / infer.

## 3) Model Families & When to Use
### EDA-driven model decision (CatBoost vs XGBoost)
- After EDA, run a controlled comparison between **CatBoost** and **XGBoost** using the same:
  - fold definition from `doc/03_cv_strategy.md`
  - feature set
  - metric definition
  - seed and evaluation protocol
- Prefer **CatBoost** when:
  - many categorical columns remain after preprocessing
  - high-cardinality categoricals are important
  - missing values are frequent and simple preprocessing is desired
  - quick robust baseline is needed with minimal feature preprocessing
- Prefer **XGBoost** when:
  - features are mostly numeric or already well-encoded
  - sparse/high-dimensional transformed features are used
  - tighter control over tree growth/regularization is needed
  - large-scale training speed/memory tradeoff is better in project tests
- Decision rule:
  - choose the model with better CV primary metric mean (and acceptable std/runtime)
  - if scores are effectively tied, prefer the faster/more stable model
  - record the final decision and evidence in `runs/<run_id>/notes.md` and `results.json`

### Classification (binary / multiclass)
- objectives: logistic / softmax
- class imbalance handling: class_weight / scale_pos_weight
- calibration (optional): isotonic/sigmoid (must use train split only)

### Regression
- objectives: l2 / l1 / huber
- target transform: log1p (if needed)

### Ranking
- query_id (group key) required
- objective: pairwise / ndcg
- metric: NDCG@K / MAP@K
- ensure query integrity in CV

### Time-series
- time-based CV required
- no future leakage (strict)
- horizon defined in problem statement

## 4) Hyperparameter Recording (Authoritative Format)
Each model must record:
- params:
- training:
- seed:
- early stopping:
- feature set id:
- runtime (train/infer):
- memory usage (if available)

## 5) KFold / CV Training Procedure
1) build folds from `doc/03_cv_strategy.md`
2) for each fold:
   - fit preprocessors on train fold
   - train model
   - predict valid fold => store OOF
3) aggregate metrics => mean/std
4) store artifacts:
   - OOF preds
   - fold models (or refit on full train)
   - feature list + importance

## 6) Ablations
Record each run's change in a traceable way:
- Change:
- Reason:
- Expected impact:
- Observed impact:
- Decision: keep/discard
