# {{PROJECT_NAME}}

## 1) Problem
- Task type: (binary / multiclass / regression / ranking)
- Target: `{{TARGET_COLUMN}}`
- Primary metric: `{{PRIMARY_METRIC}}`
- Business goal: {{BUSINESS_GOAL}}

## 2) Data
- Data source: {{SOURCE}}
- Data version: {{DATA_VERSION}}
- Row count: {{N_ROWS}}
- Feature count: {{N_FEATURES}}
- Time range (if applicable): {{TIME_RANGE}}

詳見：`docs/01_data_card.md`

## 3) Approach Summary
- Baseline: {{BASELINE_MODEL}}
- CV strategy: {{CV_STRATEGY}}
- Best single model: {{BEST_SINGLE_MODEL}}
- Ensemble: {{ENSEMBLE_METHOD}}
- Notes: {{KEY_INSIGHTS}}

## 4) How to Run
- Train: `python src/train.py --config configs/baseline.yaml`
- Evaluate: `python src/evaluate.py --run_id <run_id>`
- Infer: `python src/infer.py --model runs/<run_id>/artifacts/model.pkl --input ...`

## 5) Results
- Ledger: `results.json`  （append-only，schema 見 `docs/06_experiment_log.md`）
- Artifacts: `runs/<run_id>/`

---

## 6) Src Contract（給 Agent 的程式介面）

當本範本被複製到 `projects/<project_slug>/` 後，預期的 `src/` 介面為：

- `data.py`
  - 讀取 `configs/baseline.yaml` 的 `data` 與 `cv` 區塊。
  - 依 `docs/03_cv_strategy.md` 建立 folds（StratifiedKFold / GroupKFold / TimeSeriesSplit 等）。
  - 僅允許在這裡做 split / schema 探勘 / profiling，不可在其他模組私改 CV 規則。

- `features.py`
  - 依 `features.flags` 與 `features.params` 構建 **fold-safe** 特徵 pipeline。
  - 開關包含：scaler / onehot / target encoding / featuretools（auto feature）等。
  - 只能使用 train fold fit、再對 valid/test transform，嚴禁 leakage。

- `train.py`
  - 入口：`python src/train.py --config configs/baseline.yaml [--run_id ...]`。
  - 流程：讀 baseline config → 載入資料 → 建 folds → 跑 CV → 寫入：
    - `runs/<run_id>/params.json`、`metrics.json`、`notes.md`
    - `runs/<run_id>/artifacts/*`（model、OOF、feature_list、dataset_profile_* 等）
  - 同時 append 一筆 record 到 `results.json`。

- `evaluate.py`
  - 入口：`python src/evaluate.py --run_id <run_id>`。
  - 從 `runs/<run_id>/artifacts/oof_predictions.*` 讀 OOF，重算 primary/secondary metrics（metric 定義固定）。

- `infer.py`
  - 入口：`python src/infer.py --run_id <run_id> [--output ...]`。
  - 讀 `artifacts/model.pkl` 與 config 的 `data.id_cols`，對 test 做推論並產生 submission（預設 `artifacts/submission.csv`，含 id + prediction）。

- `ensemble.py`（*可選*）
  - 建議做為 ensemble 入口：只讀 `runs/*/artifacts/oof_predictions.*` 與 `metrics.json`，依 `docs/05_ensemble.md` 產生加權平均或 stacking，並寫入新的 `runs/<run_id_ensemble>/` 與 `results.json`。

> 對 OpenClaw / Agent 的建議：  
> - 儘量只透過 `configs/*.yaml`、`docs/*` 與 `src/train.py`、`src/ensemble.py` 介面來驅動實驗。  
> - 避免直接改動 metric 定義、CV 實作或 ledger 格式。
