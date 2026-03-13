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

### 2.1 Feature Engineering (Create New Features)
- Goal: 系統性地為 **數值 / 類別 / 日期時間 / 幾何型數值** 建立新特徵，同時保持 **fold-safe、不引入 leakage**。
- Numeric（連續數值）:
  - 比例 / 比值：例如 `x1 / (x2 + 1)`、占總額比例。
  - 非線性變換：log / sqrt / clipping / winsorization（先在 EDA 中確認 heavy tail）。
  - 交互項：`x1 * x2`、`x1 / x2`、bucket 後 one-hot 再與其他變數交互。
- Categorical:
  - 高基數欄位的 target encoding / count encoding（**必須 fold-safe**，只用 train fold 訓練 encoder）。
  - 組合欄位：`country + device`、`channel + weekday` 等。
- Datetime:
  - 週期性：hour-of-day / day-of-week / month-of-year，必要時加 sin/cos encoding。
  - 相對時間：與某事件的天數差、距離當前時間的距離（必須避免穿越未來）。
  - 滾動 / lag 特徵：`value_t-1`, `rolling_mean_7d`（在每 fold train window 內計算，再套用到 valid）。
- Geometric / Spatial（如有座標、距離類欄位）:
  - 距離：兩點間歐幾里得距離 / Haversine 距離。
  - 方向：方位角 / quadrant。
  - 區域聚合：以地理格網/行政區為單位的平均值、密度（**只能用 train fold 資料**）。

### 2.2 Auto Feature / AutoML Libraries
- 允許且建議在下列前提下使用自動化工具：
  - **featuretools**：做自動化特徵合成（特別是多表 / 時序關係），但生成特徵必須：
    - 以 **fold 為單位** fit（每個 train fold 單獨 fit，再 transform valid），避免泄漏。
    - 以 pipeline 形式寫入 `src/features.py`，確保可重現。
  - **pycaret**（或其他 AutoML 框架）：
    - 可用於快速探索 baseline 模型與特徵組合。
    - 若採用其中的 pipeline/特徵，需明確在本檔記錄：使用的設定、模型家族、重要參數，並在 `runs/<run_id>/params.json` 中落地。
- 原則：
  - Auto feature / AutoML 僅作為 **產生 candidate features / pipeline 的工具**，一旦決定採用，需在本 repo 中以明確的 sklearn/LightGBM pipeline 方式實作。
  - 任意由這些工具產生的特徵，必須滿足 `AGENT_RULES.md` 的所有 fold-safe / no-leakage 要求。

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
