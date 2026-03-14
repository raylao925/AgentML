# Configs Template Guide (for OpenClaw Agents)

本目錄提供 **configs 範本**，讓 OpenClaw / Auto-ML agents 知道：

- `baseline.yaml`：單次 experiment 的「完整可執行設定」。
- `search_space.yaml`：允許探索的模型 / 特徵 / 超參空間與 guardrails。

專案在 `projects/<project_slug>/configs/` 會各自複製一份，再由 agent 依賽題修改。

---

## 1. `baseline.yaml` — 單次 Run 的「真實 config」

**用途**：

- 定義 **task / data / cv / features / model / training / output** 的所有細節。
- `src/train.py` 直接讀這個 YAML，照此執行一個 run。
- 每次 run 的完整 config snapshot 會被存到 `runs/<run_id>/params.json`。

**關鍵區塊：**

- `project`：
  - `name`：project slug（例如 `playground-series-s6e3`）。
  - `seed`：全域亂數種子（CV + 模型）。

- `task`：
  - `family`：`classification_binary` / `classification_multiclass` / `regression` / `ranking` / `time_series`。
  - `target`：target 欄位名（例如 `Churn`）。
  - `target_positive_label`（選填）：binary 時哪個 label 視為 1（例如 `"Yes"`）。
  - `primary_metric` / `secondary_metrics`：與 `doc/00_problem_statement.md`、`doc/06_experiment_log.md` 一致。

- `data`：
  - `train_path` / `test_path`：資料路徑（相對於 project root）。
  - `id_cols`：ID 欄位（不進模型、在 infer 時會帶到 submission）。
  - `drop_cols`：明確禁止使用的欄位（含 leakage）。
  - `time_col` / `group_key`：若有 time-based / group-based CV，在這裡與 `03_cv_strategy.md` 對齊。

- `cv`：
  - `cv_type`：如 `StratifiedKFold`、`GroupKFold`、`TimeSeriesSplit`。
  - `n_splits`、`shuffle`、`random_state`：固定 CV 設定。
  - `stratify_col`：常設為 target；若為 `null` 則由程式自動用 target。
  - `time_col` / `group_key` / `gap_or_embargo`：與 `doc/03_cv_strategy.md` 一致；agent 不可私改，只能在 doc 說明後同步修改。

- `features`：
  - `feature_set_id`：特徵版本 ID（如 `fs_baseline_v1` / `fs_ft_v1`）。
  - `flags`：開關（例如 `use_scaler`、`use_onehot`、`use_featuretools` 等），由 agent 在 search 時切換。
  - `params`：各種 feature 模組的參數，例如 target encoding、lag/rolling、featuretools 等。

- `model`：
  - `name`：`LightGBM` / `XGBoost` / `CatBoost` / `LogisticRegression` / `ElasticNet` / `LGBMRanker` / `XGBRanker`。
  - `objective`：`binary` / `multiclass` / `regression` / `ranking`。
  - `params`：模型超參（num_leaves / learning_rate / reg_lambda 等）。

- `training`：
  - early stopping、num_boost_round、eval_at 等訓練相關設定。

- `output`：
  - `results_path`：experiment ledger（通常為 `results.json`）。
  - `runs_dir`：run artifacts 目錄（例如 `runs/`）。
  - `save_oof` / `save_model`：是否儲存 OOF 與 model artifacts。

**Agent 如何「安全地」修改 `baseline.yaml`：**

- **允許修改**：
  - `features.flags` / `features.params`（開關與超參）。
  - `model.name` / `model.params`（在 `search_space.yaml` 限定的 model 範圍內）。
  - `training` 中 early stopping / num_boost_round 等（需記錄在 params.json）。

- **禁止直接修改**：
  - `task.primary_metric` 的定義（除非修 bug）。
  - `cv` 區塊（如要變更，需先更新 `doc/03_cv_strategy.md` 並在 notes 中寫清楚）。
  - 使用 test set 做 tuning（違反 AGENT_RULES）。

---

## 2. `search_space.yaml` — 可探索空間與 Guardrails

**用途**：

- 告訴 agent「哪些維度可以被搜尋、範圍是什麼」。
- 定義 keep / discard 規則的 threshold（improve_threshold、tie_margin 等）。

**核心區塊：**

- `policy`：
  - `improve_threshold`：不同任務（classification AUC / ranking NDCG@10 / regression RMSE）要提升多少才算「顯著改善」。
  - `tie_margin`：與最佳 run 差異在此範圍內時視為「平手」。
  - `std_worsen_ratio_max`：允許 CV std 增加的上限（預設 1.2 = 20%）。
  - `resources_limits`：訓練時間 / 推論時間 / memory / model size 上限。

- `allowed.models_by_task`：
  - 不同 `task.family` 可以選擇的 model 名稱列表。
  - Agent 在切換 `model.name` 時必須遵守這個白名單。

- `allowed.feature_flags`：
  - 每個 `features.flags.*` 可以取的布林值範圍。
  - Agent 在單次 run 內通常只改 1~2 個 flag，以便 attribution。

- `search.hyperparams`：
  - 各 model 對應的超參空間與型別（int / uniform / log_uniform）。
  - Agent 可以在這個範圍內做 grid / random / Bayesian 等搜尋，但不能超出邊界。

- `guardrails`：
  - `forbidden_changes`：明確禁止的行為（如改 metric 定義、用 test 做 tuning、私改 CV）。
  - `require_docs_update_if_changed`：某些欄位（如 `cv`、`features.flags`、`model.name`）如果變更，必須同步更新對應 doc。

**Agent 如何使用 `search_space.yaml`：**

- 在「選下一步實驗」時：
  - 從 `allowed.models_by_task` 中挑可用 model family。
  - 從 `allowed.feature_flags` 找出可以 flip 的 flags（例如把 `use_featuretools` 從 `false` 改成 `true`）。
  - 從 `search.hyperparams.<ModelName>` 中抽樣新的超參組合。

- 在「判斷 run 是否 KEEP」時：
  - 用 `policy.improve_threshold` / `tie_margin` / `std_worsen_ratio_max`，比較本次 run 的 primary metric 與歷史 best run。
  - 若違反 `resources_limits` 或 `guardrails`，即使分數好也標記為 DISCARD。

---

## 3. 建議給 OpenClaw / Agent 的使用方式

1. **讀取 `configs/baseline.yaml` 與 `configs/search_space.yaml`：**
   - 解析 task family / target / metric。
   - 確認 `data.train_path` / `test_path` 與 id / drop 欄位。

2. **決定本次實驗要動哪些「維度」：**
   - 例如：
     - 只改 `features.flags.use_featuretools`（baseline → 加入 auto feature）。
     - 或固定 features，只改 `LightGBM` 的 `learning_rate`、`num_leaves`。

3. **產生新的 config（程式中）並執行 run：**
   - 把變更 merge 到 baseline config 中（不直接覆蓋原始 YAML，可在程式中組合），再寫入 `runs/<run_id>/params.json` 作為 snapshot。

4. **根據 `search_space.policy` 判斷 KEEP / DISCARD：**
   - 並 append 一筆 record 到 `results.json`，包含：
     - 本次使用的 config（摘要）。
     - primary / secondary metrics（含 per-fold）。
     - decision.status 與 decision.reason。

此 README 即是給 OpenClaw / Auto-ML agent 的「configs 說明書」，讓 agent 清楚知道：

- 哪些欄位可以安全修改、範圍是什麼。
- 哪些欄位屬於 protocol/metric/CV 的硬約束，必須透過 doc 更新流程處理。
- 如何把一次實驗的改動，乾淨地映射到 `runs/<run_id>/params.json` 與 `results.json`。

