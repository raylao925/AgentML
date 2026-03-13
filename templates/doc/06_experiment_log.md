# 06 — Experiment Log (Ledger + Run Artifacts)

> 目的：
> - `results.json`：存放「結構化」實驗紀錄（給 agent/程式解析、排序、挑 best run）
> - `runs/<run_id>/`：存放每次 run 的完整產物（params / metrics / artifacts / notes）
>
> 原則：**每一次 run（KEEP 或 DISCARD）都必須落地記錄**，確保可追溯、可重現、可比較。

---

## A) Run Folder Contract (Must Produce)

每次實驗必須建立一個 folder：`runs/<run_id>/`，最少包含：

- `runs/<run_id>/params.json`  
  - 實際生效的 config（含 model / CV / feature flags / seed）
- `runs/<run_id>/metrics.json`  
  - per-fold 指標 + 聚合（mean/std）+ 重要診斷（如 calibration / ndcg@k breakdown）
- `runs/<run_id>/notes.md`  
  - 必須包含：Hypothesis、What changed、Why、Outcome、Decision、Next step
- `runs/<run_id>/artifacts/`（按需要）
  - `model.*`（可為 pkl / cbm / txt / onnx 等）
  - `feature_list.json`
  - `oof_predictions.*`（建議 parquet/csv）
  - `test_predictions.*`（只有 final/提交時才允許）
- `runs/<run_id>/plots/`（可選）
  - 重要圖（feature importance、ROC、PR、residuals、calibration curve、time split diagnostics）

> 建議：所有路徑都以 project root 作相對路徑，避免搬 project 時失效。

---

## B) Ledger File: `results.json`

- 檔案位置：`projects/<project_slug>/results.json`
- 檔案格式：**JSON Array**（每個元素 = 一次 run 的 record）
- 更新策略：**append-only**（只追加，不覆蓋歷史；如需修正，新增 correction 記錄並註明）
- 排序建議：依 `datetime` 由舊到新（或由新到舊，但要固定一種）
- 最小要求：任何 run 都要寫入一筆 record（KEEP / DISCARD 都一樣）

> NOTE（可選）：如你需要「更容易 append & 支援並發」，可以改用 `results.jsonl`（每行一個 JSON object），但本 project 先以 `results.json` 為主。

---

## C) `results.json` Schema (Recommended)

每個 record **必須**包含下列欄位（可按需要增減，但建議保留核心欄位）：

```json
{
  "run_id": "20260313_1040_lgbm_baseline",
  "datetime": "2026-03-13T10:40:00+08:00",

  "task": {
    "family": "classification_binary",
    "target": "is_cancel",
    "primary_metric": "AUC",
    "secondary_metrics": ["LogLoss", "Brier", "F1"]
  },

  "data": {
    "data_version": "raw_snapshot_20260301",
    "row_count_train": 123456,
    "row_count_valid": 30864,
    "schema_hash": "sha1:xxxx",
    "leakage_policy_ref": "docs/01_data_card.md"
  },

  "cv": {
    "cv_type": "GroupKFold",
    "n_splits": 5,
    "random_state": 42,
    "group_key": "customer_id",
    "time_col": null,
    "gap_or_embargo": null
  },

  "features": {
    "feature_set_id": "fs_baseline_v1",
    "feature_version": "v1.0.3",
    "feature_flags": {
      "use_target_encoding": false,
      "use_lag_features": false,
      "use_rolling_features": false
    },
    "dropped_columns": ["leak_col_1", "post_event_status"]
  },

  "model": {
    "name": "LightGBM",
    "objective": "binary",
    "params": {
      "learning_rate": 0.05,
      "num_leaves": 64,
      "min_data_in_leaf": 50
    }
  },

  "metrics": {
    "primary": {
      "name": "AUC",
      "mean": 0.7812,
      "std": 0.0041,
      "per_fold": [0.7790, 0.7851, 0.7803, 0.7820, 0.7800]
    },
    "secondary": {
      "LogLoss_mean": 0.4920,
      "Brier_mean": 0.1830
    }
  },

  "resources": {
    "train_seconds": 312.4,
    "infer_seconds": 0.86,
    "peak_memory_gb": 4.2,
    "model_size_mb": 18.7
  },

  "artifacts": {
    "run_dir": "runs/20260313_1040_lgbm_baseline/",
    "params_path": "runs/20260313_1040_lgbm_baseline/params.json",
    "metrics_path": "runs/20260313_1040_lgbm_baseline/metrics.json",
    "notes_path": "runs/20260313_1040_lgbm_baseline/notes.md",
    "model_path": "runs/20260313_1040_lgbm_baseline/artifacts/model.pkl",
    "oof_path": "runs/20260313_1040_lgbm_baseline/artifacts/oof_predictions.parquet",
    "plots_dir": "runs/20260313_1040_lgbm_baseline/plots/"
  },

  "decision": {
    "status": "keep",
    "rule_version": "keep_discard_v1",
    "reason": "Primary metric improved with stable std; resources within limits."
  },

  "notes_short": "Baseline established. Next: try class_weight + TE-safe encoding."
}
```

### C.1) Task-specific Add-ons (Optional)

#### Ranking 任務建議加：
- `task.query_key`（例如 query_id / session_id）
- `metrics.primary.per_query`（如你要存更細）
- `model.objective`（pairwise/listwise、ndcg）

#### Time-series 任務建議加：
- `cv.time_col`、`cv.gap_or_embargo`
- `task.horizon`
- `data.time_range_train/valid`

---

## D) Keep/Discard Rule (Default)

> 目標：讓 agent 可以**自動**、**一致**、**可審計**地決策，避免主觀判斷。

### D.1 Default Decision Logic

1) 找到目前 `results.json` 中 **best KEEP** 的基準（按 `metrics.primary.mean` 排序；ranking/regression 需按方向）
2) 計算本 run 與 best baseline 的差異：`delta = new_mean - best_mean`（或 RMSE 用 best_mean - new_mean）
3) 按以下規則判斷：

#### ✅ KEEP（預設）
- `delta >= improve_threshold`
- 且 `std` 沒有顯著惡化（例如 std 增幅 <= 20%）
- 且 `resources` 不超出上限（train/infer/memory/model_size）

#### ✅ KEEP（Tie-break，預設可選）
- `abs(delta) <= tie_margin`
- 但 secondary 指標明顯改善（例如 LogLoss/Brier/latency）
- 且未違反任何硬約束（尤其 leakage / test rule）

#### ❌ DISCARD（預設）
- 不滿足 KEEP 條件，或
- 出現任何 leakage 風險 / CV 規則違反 / 使用 test 調參

### D.2 Suggested Defaults (可在 `AGENT_RULES.md` 覆寫)
- `improve_threshold`：
  - classification AUC: +0.001 ~ +0.002
  - regression RMSE: 相對改善 ≥ 0.2%（或絕對值視 scale）
  - ranking NDCG@10: +0.002
- `tie_margin`：0.0002（視資料量調整）
- `resources limits`：由 project 在 `configs/baseline.yaml` 或 `AGENT_RULES.md` 指定

---

## E) How to Write `results.json` (Append Policy)

### E.1 Append-only 行為要求
- 不覆蓋舊 record
- 若 run 失敗（crash / timeout），也要寫入 record：
  - `decision.status = "discard"`
  - `decision.reason = "runtime_error: ..."`
  - `metrics` 可留空或寫 `null`

### E.2 Minimal Required Fields
任何 record 最少要有：
- `run_id`
- `datetime`
- `task.family`
- `cv.cv_type`
- `model.name`
- `metrics.primary.mean`（若成功）
- `decision.status`
- `artifacts.run_dir`

---

## F) Companion File: `runs/<run_id>/notes.md` Template

每個 run 的 notes.md 建議用以下格式：

```md
# Run {{run_id}}

## Task Summary (from 00_problem_statement)
- task:
- target:
- primary metric:
- constraints:

## Hypothesis
（我預期改動會令指標 ↑/↓，原因）

## What Changed
- files changed:
- config diff:
- feature diff:
- model diff:

## Results
- primary metric (mean/std):
- secondary:
- resource usage:

## Decision
- KEEP / DISCARD
- reason:

## Next Step
（下一個最值得試的方向）
```