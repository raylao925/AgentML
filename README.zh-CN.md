# AgentML

> [English](README.md) | 中文

AgentML 是一個 **基於 OpenClaw 的 agent 驅動** 框架，用結構化、可重現的工作流執行 AutoML。
每個資料集/問題獨立在專案資料夾中，由 Markdown 文件管理，並透過機器可讀的實驗 ledger 記錄。

---

## Repo 概覽

```text
ClawML/
  README.md           # 本檔（英文版）
  README.zh-CN.md     # 中文版
  SKILL.md            # Agent 技能：專案設定、CV、leakage、Data Wide Search、訓練等
  structure.md        # 專案資料夾結構（agent-friendly）
  projects.md         # 專案索引
  requirements.txt    # 基礎依賴（numpy, pandas, sklearn, lightgbm 等）

  templates/          # 複製到 projects/<project_slug>/ 建立新專案
    AGENT_RULES.md
    configs/          # baseline.yaml, search_space.yaml
    doc/              # 00-07 markdown 範本
    src/              # data.py, features.py, train.py, evaluate.py, infer.py, ensemble.py, train_multi_model.py, infer_ensemble.py

  projects/           # 以資料集為單位的專案資料夾（見 structure.md）
    <project_slug>/
```

---

## AgentML 提供什麼

- **單一專案隔離**：每個資料集/問題對應一個獨立專案資料夾，有各自的 docs、configs、runs 與 ledger
- **Markdown 驅動流程**：跨專案一致的結構：
  - Problem Statement → Data Card → EDA → CV Strategy → Modeling → Ensemble → Deployment/Submission
- **自動執行（Agent 自動化）**：Agent 讀取 `program.md`，可在 `search_space.yaml` 內自動修改 **config / feature / model**
- **Data Wide Search**：當使用者 **只丟入資料集**、未填 docs 時，Agent 可執行網域搜尋來補足情境並強化特徵工程（見下）
- **可重現實驗**：CV 策略一旦建立即視為權威並 **鎖定**；所有 run 產出一致的 artifacts 與結構化 log
- **結構化 Ledger**：所有 run 以 append-only 寫入 `results.json`，便於篩選、排序與自動化

---

## 專案結構（Agent-Friendly）

每個資料集/問題對應一個專案資料夾：

```text
projects/<project_slug>/
  program.md                 # Agent 協議（自動化迴圈）
  AGENT_RULES.md             # 不可違反規則（leakage/CV/test/logging）
  README.md                  # 專案概覽（人類閱讀）

  docs/
    00_problem_statement.md  # 任務定義（tabular/time-series/ranking/multiclass/binary）
    01_data_card.md          # Data schema + leakage checklist
    02_eda.md                # EDA 範本（insights → actions）
    03_cv_strategy.md        # CV 權威（建立後 lock-in）
    04_modeling.md           # Modeling + params + CV 流程 + ablations
    05_ensemble.md           # Ensemble 設計（OOF stacking/blending）
    06_experiment_log.md     # Ledger 規格（results.json schema + keep/discard）
    07_deployment_or_submission.md  # Inference/deploy/submission

  configs/
    baseline.yaml            # 可執行 baseline（起點）
    search_space.yaml        # 可探索邊界（models/params/feature flags）

  src/
    data.py                  # load/clean/split（須遵守 docs/03_cv_strategy.md）
    features.py              # 特徵工程（須 fold-safe）
    train.py                 # 訓練入口（讀 configs）
    evaluate.py              # 指標計算（定義固定）
    infer.py                 # 推理/submission

  data/
    raw/                     # 原始資料（通常 gitignore）
    interim/
    processed/

  runs/
    <run_id>/
      params.json            # 生效 config snapshot
      metrics.json           # per-fold + aggregate metrics
      notes.md               # hypothesis/change/outcome/decision/next step
      artifacts/             # model, oof preds, feature list, etc.
      plots/                 # 可選 plots

  results.json               # append-only 實驗 ledger
```

---

## 快速開始

### 1) 建立新專案
將專案範本複製到新資料夾：

```bash
mkdir -p projects/<project_slug>/
cp -r templates/. projects/<project_slug>/
```

### 2) 填寫權威文件（或使用 Data Wide Search）

**選項 A — 完整控制**：至少完成 `docs/00_problem_statement.md` 和 `docs/01_data_card.md`。

**選項 B — 最小設定**：將 train/test 資料放入 `data/raw/` 或 `data/processed/`，執行 Agent。Agent 會進行 Data Wide Search 推斷情境、提出特徵並與你確認（見下方「Data Wide Search」）。

CV 可二擇一：
- **使用者指定**：填寫 `docs/03_cv_strategy.md`（建議，便於嚴格控制）
- **Auto-infer + Lock-in**：留空或缺失，由 Agent 從 EDA/`df.info()` 推斷並寫入一次（見「CV Bootstrapping」）

### 3) 執行 Baseline（手動）
執行一次 baseline config 產生第一筆 ledger：

```bash
python projects/<project_slug>/src/train.py --config projects/<project_slug>/configs/baseline.yaml
```

會產生：
- `projects/<project_slug>/runs/<run_id>/...`
- `projects/<project_slug>/results.json`（append）

### 4) 執行自動迴圈（OpenClaw）
用 OpenClaw 執行專案協議：

```bash
# placeholder：依 OpenClaw runner 語法調整
openclaw run projects/<project_slug>/program.md
```

Agent 會：
1) 讀取 `program.md` + `docs/*` + `configs/*`
2) 在 `search_space.yaml` 內提出並套用 bounded change（config/feature/model）
3) 使用鎖定的 CV 策略訓練與評估
4) append 一筆 record 到 `results.json`
5) 標記 keep/discard 並迭代

---

## CV Bootstrapping（Auto-infer + Lock-in）

若使用者未指定 CV，且 `docs/03_cv_strategy.md` 缺失或為空：

1) Agent 進行最小資料理解（EDA / `df.info()` / schema scan）
2) 推斷安全的 CV 策略（time-based / group-based / stratified 等）
3) **寫入 `docs/03_cv_strategy.md` 並鎖定**
4) 後續 run 必須完全遵守

一旦鎖定，Agent 不得私下改動：
- GroupKFold → KFold
- time-based split → random split
- ranking query 完整性

詳見 `AGENT_RULES.md` 的權威與 lock-in 政策。

---

## Data Wide Search（最小情境 Bootstrapping）

當使用者 **只將資料集放入** `projects/<project_slug>/data/`，未填寫 `docs/00_problem_statement.md` 或 `01_data_card.md` 時，Agent 可透過 **wide search** 補足情境：

1. **從資料推斷**：執行 `df.info()`、schema scan、取樣；辨識可能的 target、ID、datetime 欄位；推測任務類型
2. **網域/競賽搜尋**：搜尋類似問題、Kaggle notebook、領域最佳實踐；蒐集特徵構想（比例、聚合、時序轉換、編碼）
3. **與使用者互動**：摘要發現；提出推測的 target、任務類型、特徵候選；請使用者確認或修正
4. **撰寫與實作**：將推斷內容寫入 `docs/*`；在 `docs/04_modeling.md` 加入「Data Wide Search」小節；在 `src/features.py` 實作 fold-safe 特徵

完整流程見 `SKILL.md` §1.5 與 `templates/doc/04_modeling.md` §2.3。

---

## 實驗追蹤

### Ledger
- `projects/<project_slug>/results.json`（append-only）

### 每 Run 的 Artifacts
- `projects/<project_slug>/runs/<run_id>/`
  - `params.json`（生效 config snapshot）
  - `metrics.json`（per-fold + aggregate）
  - `notes.md`（hypothesis/change/outcome/decision）
  - `artifacts/`（model, OOF predictions, feature list）
  - `plots/`（可選）

Ledger 可用於：
- 排序 best `decision == "keep"` 的 runs
- 審計變更（notes + params snapshot）
- 依規則自動化 keep/discard

---

## 慣例與護欄

- **Test 不用於 tuning**：Test 僅用於 final report / submission
- **Fold-safe 預處理**：在 train fold 上 fit，再 transform valid fold
- **每次 1–2 類改動**：每 run 最多 1–2 類（feature family OR param tweak OR model swap）
- **Append-only logs**：所有 run（含失敗）都必須寫入 `results.json`

---

## 專案索引

見 `projects.md` 取得專案清單。
