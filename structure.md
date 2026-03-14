# Project Structure (Agent-Friendly)

本 Project 採用「Markdown 驅動 + 結構化 Ledger」的方式管理實驗。
核心概念：
- `program.md` 定義研究 protocol（agent 讀取後自動改 config/feature/model）
- `docs/` 固定研究流程（Problem → Data → EDA → CV → Modeling → Ensemble → Deploy）
- `runs/<run_id>/` 存每次實驗的完整產物
- `results.json` 作為 experiment ledger（每次 run 一筆記錄，用於排序/挑 best/keep-discard）

---

## Folder Layout

```text
projects/<project_slug>/
  program.md                 # Agent research protocol（自動化核心）
  AGENT_RULES.md             # 不可違反規則（leakage/CV/test/metric）
  README.md                  # Project 概覽（人類閱讀）

  docs/
    00_problem_statement.md  # 任務定義（支援 tabular/time-series/ranking/multiclass/binary）
    01_data_card.md          # Data schema + leakage checklist
    02_eda.md                # EDA 模板（結論→行動）
    03_cv_strategy.md        # CV/切分權威文件（group key/time col 由 user 定義）
    04_modeling.md           # 建模、超參、KFold 流程、ablation
    05_ensemble.md           # ensemble 設計（OOF stacking/blending）
    06_experiment_log.md     # ledger 規格（results.json schema + keep/discard）
    07_deployment_or_submission.md  # 推理/部署/提交

  configs/
    baseline.yaml            # 可跑 baseline（agent 以此為起點）
    search_space.yaml        # agent 可探索邊界（可選模型/可調參/feature flags）

  src/
    data.py                  # load/clean/split（遵守 docs/03_cv_strategy.md）
    features.py              # 特徵工程（必須 fold-safe；支援手工 + auto feature）
    train.py                 # 訓練入口（讀 configs，寫 runs/<run_id> + results.json）
    evaluate.py              # metric 計算（口徑固定；重算 OOF 指標）
    infer.py                 # 推理/輸出 submission（依 data.id_cols 加上 prediction）
    ensemble.py              # （可選）ensemble 入口，依 05_ensemble.md 使用 OOF 做 stacking/blending

  data/
    raw/                     # 原始資料（通常 gitignore）
    interim/
    processed/               # train/test parquet/csv 等

  runs/
    <run_id>/
      params.json            # 實際生效 config snapshot
      metrics.json           # per-fold + aggregate metrics
      notes.md               # hypothesis/變更/結果/決策
      artifacts/             # model、oof preds、feature list 等
      plots/                 # 圖表（可選）

  results.json               # Experiment ledger（append-only；keep/discard 決策依據）