# 04 — Modeling

## 1) Baseline (Must Have)
- Model: {{BASELINE_MODEL}}
- Feature set id: {{FEATURE_SET_ID}}
- CV: per `03_cv_strategy.md`
- Baseline score: {{BASELINE_SCORE}}

## 2) Feature Pipeline (Fold-safe)
- Missing:
- Numeric transform:
- Categorical encoding:
- Text (if any):
- Time-series feature policy (if any): lag/rolling computed fold-safe
- Ranking feature policy (if any): query-level features computed fold-safe

## 3) Model Families & When to Use
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
每個模型都要記錄：
- params:
- training:
- seed:
- early stopping:
- feature set id:
- runtime (train/infer):
- memory usage (if available)

## 5) KFold / CV Training Procedure
1) build folds from `03_cv_strategy.md`
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
每次 run 的 change 以「可追溯」方式記：
- Change:
- Reason:
- Expected impact:
- Observed impact:
- Decision: keep/discard
