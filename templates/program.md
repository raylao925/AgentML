# program.md — Autonomous ML Protocol (Project: {{PROJECT_NAME}})

## 0) Mission
你是一個 Auto-ML Agent。你的任務是在**不引入 data leakage** 的前提下，
對 {{TASK_FAMILY}} 任務（tabular / time-series / ranking / classification / regression）
最大化 {{PRIMARY_METRIC}}（並追蹤 {{SECONDARY_METRICS}}）。

projects/<project_slug>/依照structure.md Folder Layout自動生成folders和files, 如果folder/files 不存在:

你可以自動：
- 修改 `configs/*.yaml`（模型、超參、CV、特徵開關）
- 修改 `src/features.py`（新增/移除/修正特徵）
- 修改 `src/train.py`（模型訓練流程、early stopping、loss、ranking objective 等）
- 修改 `src/evaluate.py`（但不可改 metric 定義本身，只可修 bug / 提升效率）
- 修改 `src/ensemble.py`（根據baseline cv score, 進行AutoML流程, 得出ensemble版本, 目標是ensemble的cv score會更好）

你必須遵守 `AGENT_RULES.md` 的所有硬約束。

---

## 1) Task Understanding (Read First)
你必須先閱讀並抽取：
1) `docs/*.md`：整個 ML Task 內容
2) `configs/baseline.yaml`：baseline 的全部設定

然後寫出「你理解的任務摘要」到 `runs/<run_id>/notes.md` 的最頂部。

## 1.5 CV Bootstrap (Only if CV is missing) — Lock-in Mode (1)

Before any modeling experiments, the agent MUST ensure `docs/03_cv_strategy.md` exists and is authoritative.

### Step 1 — Detect CV Authority
- If user prompt explicitly specifies CV / group_key / time_col / query_id → treat as P0
- Else if `docs/03_cv_strategy.md` exists and non-empty → treat as P1
- Else → treat as P2 (auto-infer)

### Step 2 — If P2 (Auto-infer CV)
The agent MUST:
1) Run minimal EDA / schema scan:
   - df.info() summary (dtypes, non-null counts)
   - candidate id/group columns (high-cardinality keys)
   - candidate time columns (datetime-like names/types)
   - target distribution and duplicates by group
2) Infer appropriate CV strategy:
   - time-based split if temporal leakage risk or time_col is present and task is temporal
   - GroupKFold if group leakage risk is present (same entity repeats)
   - ranking requires query_id integrity (GroupKFold by query_id)
   - StratifiedKFold for classification if no time/group constraint overrides
3) Write `docs/03_cv_strategy.md`:
   - include the inferred rule + evidence + rationale
   - set "Lock-in: true" section
4) From this point onward, treat CV as P1 (locked). No silent changes.

### Step 3 — Proceed to Baseline
- After CV is locked, run the baseline experiment and log to `results.json`.

---

## 2) Hard Constraints (Must Not Break)
1) **No Leakage**：不得使用 `docs/01_data_card.md` 標記為 leak 的欄位；不得做任何會令 valid fold 看見未來資訊的計算。
2) **CV/Split 規則不可私改**：所有比較必須使用 `docs/03_cv_strategy.md` 定義的 split。若要改 CV，只能先更新 `docs/03_cv_strategy.md` 並清楚寫理由。
3) **Test set 禁止用於調參**：test 只可以用於 final report（或 submission 生成），不可用來挑模型、挑特徵、挑 threshold。
4) **Metric 定義固定**：PRIMARY_METRIC 計算方式不可改（除非修正明顯 bug，並記錄原因）。
5) **Reproducibility**：每次 run 必須固定 seed，並記錄 data_version、feature_version、code_hash。

---

## 3) Allowed Search Space (What to Explore)
### 3.1 Feature work
- 缺失值處理、encoding、scaling（必須 fold-safe）
- 時間序列特徵：lag/rolling（必須按時間+fold 計算，避免穿越）
- group aggregation（必須只用 train fold 資料計算）

### 3.2 Model work
- Tabular: LightGBM/XGBoost/CatBoost/LogReg/ElasticNet
- Multi-class: softmax objectives / one-vs-rest
- Ranking: pairwise/listwise（如 LGBMRanker/XGBRanker）
- Time-series: 以「時間切分」CV + 可用 tree/linear/seq model（若可行）

### 3.3 Optimization work
- early stopping、class weight、calibration（如分類）
- ensembling（見 `docs/05_ensemble.md`；必須用 OOF 設計）

---

## 4) Experiment Loop (Autonomous)
每次迭代你必須做：

### Step A — Plan
- 根據現有 `results.json` 找到 best run
- 選一個「最可能提升」的變更（一次只做 1~2 個改動，方便 attribution）
- 在 `runs/<run_id>/notes.md` 寫出：
  - Hypothesis
  - Expected direction
  - What changed（file + key diff）

### Step B — Execute
- 產生 run_id：`YYYYMMDD_HHMM_<shortdesc>`
- 跑 training + evaluation
- 保存：
  - `runs/<run_id>/params.json`（實際生效 config）
  - `runs/<run_id>/metrics.json`（每 fold + aggregate）
  - `runs/<run_id>/artifacts/*`（模型、特徵列表、OOF、重要圖）
  - `runs/<run_id>/notes.md`

### Step C — Log to Ledger (`results.json`)
把 run 結果寫入 `results.json`（格式見 `docs/06_experiment_log.md` 的 Spec）。

### Step D — Keep / Discard
- Keep 條件（預設，可在 `AGENT_RULES.md` 調整）：
  - primary_metric_mean 提升 >= {{IMPROVE_THRESHOLD}}
  - 或者 primary_metric_mean 相若但 secondary 改善顯著（且符合限制）
- 若 discard：仍要記錄原因（overfit / variance 大 / speed 太慢 / leakage risk）

---

## 5) Output Contract (Must Produce)
每個 run 都必須產出：
- 一個 `runs/<run_id>/` folder（params、metrics、notes）
- 在 `results.json` 增加一筆 record
- 更新必要 docs（若你改了 CV/特徵/模型設計，必須同步更新對應 docs）