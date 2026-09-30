我已完成對整個 repo 的架構盤點。本文 **§1.1–§1.3、§2 部分證據行** 保留當時 As-Is 快照（路徑多寫 `templates/`、`.clinerules/`、`projects/playground-*`）；**以 §1.4、§2.0、§8 為準** 對齊目前工作區。以下是 system analysis 與「skill + template」重組藍圖。

> **Payload 路徑約定（全文）**：下文若寫 `templates/`，在現行 repo 中均指 `.agents/skills/agentml/assets/project-template/`（根目錄 `templates/` 僅剩 `README.md` 重定向 shim）。

---

# AgentML 架構分析與重組藍圖（OpenClaw → Skill + Template）

## 0. 核心結論（先講重點）

| # | 結論 | 依據 |
|---|---|---|
| 1 | 這個 repo 目前是**「文件完整、runtime 不完整」**：`templates/src/train.py`、`evaluate.py`、`ensemble.py` 是 stub，**整條 ML pipeline 從未真正跑通**（`projects/*/runs/` 不存在、config 指向不存在的 `data/processed/*.parquet`、ledger 只有 placeholder 假記錄） | `templates/src/train.py:142` `preds = np.zeros(...)`；`templates/src/evaluate.py:56` `raise NotImplementedError`；`runs/` 缺失 |
| 2 | 真正的痛點不是 OpenClaw，而是**規則被複製了 6 份**（root `SKILL.md`、`structure.md`、`templates/AGENT_RULES.md`、`templates/program.md`、`.clinerules/*`、`templates/README.md`），導致編號已經漂移（`03_cv_strategy` vs `04_cv_strategy`、`docs/` vs `doc/`） | `templates/doc/04_cv_strategy.md:1` 標題仍寫 `# 03 — CV Strategy`；`AGENT_RULES.md:14` 指向不存在的 `docs/` |
| 3 | 因此重組的核心原則應是：**「規則住在 Skill，狀態住在 Template 實例」**，並用可執行 guardrail 取代純文字規則 | 見 §4.5 / §4.7 |
| 4 | 從 Kaggle 擴展到 customer dataset，缺的不是模型，而是**資料入口層 + 任務發現協議 + 隱私/交付契約**（目前 `SKILL.md` §1.5 的 Data Wide Search 假設了「Kaggle 式有公開線索」） | `SKILL.md:45-66`、`templates/doc/00_problem_statement.md` 假設使用者已知 target/metric |
| 5 | `.clinerules/` 是 `templates/` 的**舊快照**，且它才是當前 agent 實際讀到的規則 → 造成 agent 讀到過期編號與錯誤 CLI | `.clinerules/env.md:22` 用 `--ensemble hill_climb --run_id`，實際 CLI 是 `--ensemble_run_id`（`templates/src/infer_ensemble.py:62`） |

---

## 1. 現況系統分析（As-Is）

### 1.1 六個平面（Layer Map）

```
[分發平面]  README / README.zh-CN / structure.md / projects.md
                │
[規則平面]  SKILL.md(root) ── .clinerules/{AGENT_RULES,program,env,kaggle_process}.md
                │                     └─(stale copy of templates/)
                │
[技能平面]  .agents/skills/agentml/{SKILL.md, references/workflow.md,
                                   scripts/init_agentml_project.py, assets/checklist.md}
                │
[模板平面]  templates/{program.md, AGENT_RULES.md, doc/00..08, configs/*.yaml, src/*.py, memory/*, results.json}
                │
[實例平面]  projects/<slug>/{...完全相同的一份 copy...},  data/raw/*.csv
                │
[證據平面]  runs/<run_id>/{params,metrics,notes,artifacts,plots}  +  results.json
```

### 1.2 檔案職責盤點

| 檔案 | 意圖角色 | 實際狀態 |
|---|---|---|
| `SKILL.md` (root) | agent 技能總表 | 內容詳細（0~13 節），但**不在 skill 載入路徑上**（`.agents/skills/` 才是），且與 `.agents/skills/agentml/SKILL.md` 重複 |
| `structure.md` | 目錄契約 | 仍是舊編號（`03_cv_strategy`、`04_modeling`） |
| `templates/program.md` | agent 自主迴圈協議 | 已更新為 `04_cv_strategy`，但與 `AGENT_RULES.md` 的 `docs/` 路徑衝突 |
| `templates/AGENT_RULES.md` | 硬約束 | 路徑寫成 `docs/`（實際是 `doc/`） |
| `templates/doc/*` | 研究狀態 | `03_features_engineering.md` 是**空檔（1 行）**；`08` 標題誤植為 `07` |
| `templates/configs/baseline.yaml` | 參數契約 | `train_path: data/processed/train.parquet` **不存在**（實例只有 `data/raw/*.csv`） |
| `templates/src/evaluate.py` | 指標權威 | **`NotImplementedError`**，且**無 CLI**（README 卻寫 `--run_id`） |
| `templates/src/train.py` | 訓練入口 | **stub**：`preds = np.zeros(len(valid_idx))`，不 fit 模型、不存 `model.pkl` |
| `templates/src/train_multi_model.py` | 多模型訓練 | **真實現**（LGBM/XGB/CatBoost/LogReg/ElasticNet + GPU 偵測 + 平行）但依賴 `evaluate.compute_metrics` → 第一個 fold 就崩 |
| `templates/src/ensemble.py` | 集成 | 只有均權；stacking `NotImplementedError`；不寫 ledger；**不產生 `ensemble_metadata.json`** |
| `templates/src/infer_ensemble.py` | 集成推論 | **真實現（含 hill climbing）**，但**讀的 `ensemble_metadata.json` 沒有任何生產者** → 永久 FileNotFoundError |
| `templates/src/features.py` | fold-safe 特徵 | ColumnTransformer 部分可用；`FeaturetoolsTransformer` 是 stub；無 target encoding 實作 |
| `templates/src/eda.py` | EDA 報告 | 可用但為 module-level script（無法 import）、`--no-plots` 是**死旗標**（無任何繪圖碼） |
| `requirements.txt` | 依賴 | 缺 `joblib`（`infer.py:67` 使用）、缺 `kaggle`、無 lock 檔 |
| `.clinerules/` | 工作區規則 | **舊快照**（且有 mojibake 的 `kaggle_process.md`） |

### 1.3 兩個實例專案的實況（這是最關鍵的證據）

```
projects/playground-series-s6e6/src/*.py  ==  templates/src/*.py   (9/9 檔案 MD5 完全相同)
projects/playground-series-s6e9/src/*.py  ==  templates/src/*.py   (9/9 完全相同)
projects/*/results.json  → 1 筆 placeholder 記錄 run_id="YYYYMMDD_HHMM_model_desc", AUC mean=0.8
projects/*/runs/         → 不存在
projects/*/data/processed, data/interim → 不存在（資料只在 data/raw/*.csv）
projects/playground-series-s6e9/data/raw/projects/ → 誤下載產生的巢狀目錄
```

**推論**：目前「agent 自行修改 src/*.py」這件事**從未真正發生過**；repo 是設計稿 + 骨架，不是可運行的系統。這點會直接影響重組順序 —— 必須先讓它「跑得起來」，否則 skill 化只是把沒跑通的東西包裝得更好看。

### 1.4 工作區增量（2026-09，對齊目前 tree）

| 項目 | 快照當時（§1.1–§1.3） | 目前工作區 |
|---|---|---|
| 模板 payload | 根目錄 `templates/**` | **SSOT**：`.agents/skills/agentml/assets/project-template/**`；`templates/` 僅 `README.md`（MOVED） |
| 工作區規則 | `.clinerules/` 為 agent 實際來源 | **已不在 tree**；Kaggle 流程併入 `references/sources/kaggle.md` |
| 實例證據 | `projects/playground-s6e6/s6e9` MD5 對照 | **`projects/` 可為空**（未納版控或未 checkout）；證據邏輯仍適用於任何由 init 產生的 slug |
| `doc/04_cv_strategy.md` | 標題誤寫 `03` | payload 內 H1 已為 **`# 04 —`**（以 `check_ssot.py` R3/R4 驗證） |
| `doc/03_features_engineering.md` | 空檔 | payload 內 **已有 feature registry 模板** |
| `doc/08_deployment_or_submission.md` | 標題誤寫 `07` | payload 內 H1 已為 **`# 08 —`** |
| `AGENT_RULES.md` CV 路徑 | 寫 `docs/` | payload 使用 **`doc/04_cv_strategy.md`** |
| 工具鏈 | 僅 init | **`init_agentml_project.py`**、**`sync_project.py`**（payload→實例，不覆寫 `data/`/`runs/`）、**`check_ssot.py`** |
| Runtime | 同 §1.2 stub 描述 | **仍未閉環**（`evaluate.py` / `train.py` / `ensemble.py` 契約見 §2 P0） |
| Skill 入口 | 與 root `SKILL.md` 重疊 | `.agents/skills/agentml/SKILL.md` = **腳手架**；root `SKILL.md` = **完整 lifecycle 目錄**（Phase 2 拆入 `references/` 待做） |

---

## 2. 缺陷清單（Findings，依嚴重度）

### 2.0 路徑勘誤（讀 §1 時必對照）

| 本文舊寫法 | 現行路徑 / 狀態 |
|---|---|
| `templates/src/*`、`templates/doc/*` | `.agents/skills/agentml/assets/project-template/src/*`、`.../doc/*` |
| `.clinerules/env.md` CLI 範例 | 已移除；修正目標：`references/sources/kaggle.md`（仍含 `claw_env`、錯誤 `--run_id` 等待 Phase 0 hygiene） |
| `projects/*/results.json` placeholder | 模板 payload 的 `results.json` 已為 **`[]`**；舊實例若仍存在假 ledger 需手動清空或 `--reset-ledger` |
| `init` 僅替換 `{{PROJECT_NAME}}` | **`baseline.yaml` 內 `{{TARGET_COLUMN}}` 等仍須 agent / Task Discovery 填寫**（非 init 自動化） |

### P0 — 阻斷（不修就無法完成任何真實 run）

| ID | 缺陷 | 證據 | 修法 |
|---|---|---|---|
| P0-1 | 指標層缺失 | `evaluate.py:56 NotImplementedError` | 實作 AUC/LogLoss/Brier/Acc/F1、RMSE/MAE/R²、NDCG@K/MAP@K（含 multiclass macro/micro/weighted）＋ CLI `--run_id` |
| P0-2 | 訓練層缺失 | `train.py:142` 全零 OOF | 抽出 `src/models.py` 統一 model factory（重用 `train_multi_model` 的 builder），`train.py` 與 `train_multi_model.py` 共用 |
| P0-3 | 指標未串接 | `train_multi_model.py:421` 呼叫 `eval_mod.compute_metrics` | 依 P0-1 解決；否則「已實作」的多模型訓練必然中斷 |
| P0-4 | 契約斷裂 | `infer_ensemble.py:155` 讀 `ensemble_metadata.json`，全 repo 無生產者 | 讓 `ensemble.py` 寫出 `ensemble_metadata.json`（run_ids/weights/oof/metrics/config） |
| P0-5 | 資料路徑失效 | `baseline.yaml:18` → `data/processed/train.parquet`（不存在） | `data.py` 加 path resolver：processed 不存在時 fallback `data/raw/`，並於 `data_card.md` 記錄解析結果 |
| P0-6 | 假 ledger 污染「最佳 run」 | `projects/*/results.json` AUC=0.8 假記錄 | 模板 `results.json` 改為 `[]`＋ schema header；新增 `agentml ledger verify` |

### P1 — 契約/SSoT 崩壞

| ID | 缺陷 | 證據 |
|---|---|---|
| P1-1 | 文件編號漂移 | `doc/04_cv_strategy.md` 標題寫 `03`；`doc/08_...` 標題寫 `07`；root `README.md:163`、`structure.md`、`references/workflow.md:34`、`init_agentml_project.py:108` 全部寫 `03_cv_strategy` |
| P1-2 | 目錄名不一致 | `AGENT_RULES.md` 用 `docs/`（不存在），其餘用 `doc/` |
| P1-3 | 空/佔位文件 | `doc/03_features_engineering.md` 只有 `# 03 — `；`program.md`/`README.md` 的 `{{...}}` 佔位符在實例中未替換 |
| P1-4 | CLI 文件與實作不符 | `.clinerules/env.md:22` vs `infer_ensemble.py:62`；README `evaluate.py --run_id` 無對應 main |
| P1-5 | 規則不可驗證 | `AGENT_RULES.md` 全是散文；CV lock-in 無 hash、無 CI 檢查 |
| P1-6 | 規則多處重複 | 同一條 CV 規則存在於 6 個檔案（見 §0 結論 2） |
| P1-7 | 舊快照被當成規則來源 | `.clinerules/` 整份是 `templates/` 的舊版 |
| P1-8 | 可重現性不足 | `train.py:80-81` 僅 `np.random.seed`（未 seed `random`/`PYTHONHASHSEED`）；`code_hash`/`data_version` 未實際計算 |

### P2 — 品質/擴充性

`catboost_info/` 落在 repo root（建議 `allow_writing_files=False`）；`kaggle.md` 有 Big5/UTF-8 mojibake 且引用 s6e4（實例是 s6e6/s6e9）；`eda.py` 為 module-level script；`data.py:184` 就地排序 `train_df`（副作用）；`requirements.txt` 無 lock；無 `tests/`；無 `src/__init__.py`（只能用 `python src/x.py`，不能用 `-m`）。

---

## 3. 目標架構（To-Be）：Skill + Template 三平面

### 3.1 架構原則

> **規則住在 Skill；狀態住在 Template 實例；證據住在 runs/results.json。三者之間只能透過「契約檔」耦合。**

```
┌─────────────────────────── Skill 平面（唯一規則來源，版控、可升級）────────────┐
│ .agents/skills/agentml/                                                       │
│   SKILL.md                 ← 只有 front-matter + 入口決策樹（薄）              │
│   references/              ← progressive disclosure，規則分頁                 │
│     00_contract.md         ← 檔案/CLI/資料夾契約（原 structure.md）            │
│     01_lifecycle.md        ← 生命週期與 gate（原 program.md 通用部分）         │
│     02_policy.md           ← cv authority / leakage / test / metric（原 AGENT_RULES）│
│     03_task_discovery.md   ← 任務發現協議（customer 模式核心）★新增            │
│     04_ingestion.md        ← 資料源 adapter 契約（kaggle/local/db/oss）★新增   │
│     05_features.md / 06_modeling.md / 07_ensemble.md / 08_delivery.md          │
│     09_guardrails.md       ← 可執行護欄清單                                   │
│     sources/kaggle.md      ← Kaggle 專用 recipe（原 kaggle.md，修編碼）        │
│   assets/project-template/ ← 唯一的模板 payload（原 templates/）              │
│   scripts/agentml.py       ← 統一 CLI（new/doctor/eda/cv-lock/run/ensemble/…） │
│   assets/checklist.md, schemas/*.json                                        │
└───────────────────────────────────────────────────────────────────────────────┘
                    │ copies (agentml new)
                    ▼
┌─────────────────────────── Project 平面（使用者/客戶資料，gitignored）────────┐
│ projects/<slug>/                                                              │
│   project.yaml        ← ★新增：skill_version / policy_version / mode(kaggle|customer) │
│   doc/00..08_*.md     ← 任務狀態（編號固定、標題一致、無重複規則）             │
│   configs/{baseline,search_space}.yaml                                        │
│   src/*.py            ← agent 可自行修改區（writable zone）                    │
│   memory/*.md, data/{raw,interim,processed}, reports/                         │
└───────────────────────────────────────────────────────────────────────────────┘
                    │ produces
                    ▼
┌─────────────────────────── Evidence 平面 ─────────────────────────────────────┐
│ runs/<run_id>/{params.json,metrics.json,notes.md,artifacts/,plots/}           │
│ results.json（append-only，schema 由 skill 的 schemas/ledger.schema.json 驗證）│
└───────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Skill 的 front-matter（沿用現有 `.agents/skills/` 慣例）

現有 `.agents/skills/agentml/SKILL.md` 已是正確格式，擴充即可：

```yaml
---
name: agentml
description: Run an end-to-end, leakage-safe AutoML lifecycle (Kaggle or customer tabular data) by scaffolding a project from the AgentML template, then reading/modifying its docs, configs and src to execute EDA, CV lock-in, training, ensembling and delivery. Use when the user drops a dataset, asks to build/keep improving a model, or asks for a standardized ML project folder.
argument-hint: [project_slug] [--mode kaggle|customer] [--setup-venv] [--data <path>]
user-invocable: true
---
```

### 3.3 Frozen / Writable 契約（本重組最重要的機制）

| 區 | 檔案 | 誰可改 | 理由 |
|---|---|---|---|
| **Frozen（skill 擁有）** | `doc/04_cv_strategy.md` 的 `lock_in` 區塊、`evaluate.py` 的指標語義、`results.json` schema、`baseline.yaml` 的 `task.target/primary_metric` | 僅使用者要求 + 文件更新 | 保證跨 run 可比性 |
| **Semi（需記錄才可改）** | `configs/*.yaml`、`src/features.py`、`src/models.py`（參數）、`src/eda.py` | agent 可改，必須在 `runs/<run_id>/notes.md` 記錄 diff | 探索空間在 `search_space.yaml` 內 |
| **Free（agent 自由）** | `src/train*.py` 的流程、`plots/`、新 feature 函式 | agent 可改 | 這是 agent 的主要工作區 |
| **Forbidden** | `data/raw/*`、test 路徑用於調參、target 進 `feature_cols` | 任何人都不可 | `AGENT_RULES` §1/§3 |

實作方式：在每個 frozen 檔案開頭加 `# AGENTML:FROZEN v1 <sha256>`，`agentml doctor` 與 train 前 preflight 驗證 hash。

### 3.4 SSoT 規則表（每個事實只有一個家）

| 事實 | 唯一住處 | 其他檔案只能引用 |
|---|---|---|
| CV 規則 | `projects/<slug>/doc/04_cv_strategy.md`（含 `locked_hash`） | skill `references/02_policy.md`（說明機制，不寫具體值） |
| 指標定義 | `projects/<slug>/src/evaluate.py` + `doc/00_problem_statement.md` | ledger、README |
| 允許探索範圍 | `configs/search_space.yaml` | skill `references/06_modeling.md` |
| 目錄/CLI 契約 | skill `references/00_contract.md` | root `README.md`、`projects.md` 只放指標連結 |
| keep/discard 門檻 | `configs/search_space.yaml:policy` | `doc/06_experiment_log.md`（說明） |
| Kaggle 流程 | skill `references/sources/kaggle.md` | 專案 `project.yaml.mode: kaggle` |

### 3.5 統一 CLI（取代散落的 9 個 script 入口）

```
agentml new <slug> --mode customer|kaggle [--setup-venv]
agentml doctor                      # 環境/契約/護欄/依賴自檢
agentml ingest --source data_sources/<x>.yaml   # 資料落地 + manifest
agentml eda                         # 產生 reports/eda_report.md + 更新 doc/02
agentml cv-lock                     # P0/P1/P2 判定 → 寫入 doc/04 + 記錄 hash
agentml run   --config configs/baseline.yaml [--models ...]
agentml ensemble --run_ids ... --method hill_climb
agentml infer --run_id ... --out deliverables/scored.csv
agentml ledger verify|best|report
agentml guardrails                  # 獨立 CI 入口
```

### 3.6 Reference 拆分清單（Phase 2 待辦）

| 新檔 | 來源（現況） |
|---|---|
| `references/00_contract.md` | `structure.md` + README 目錄表 |
| `references/01_lifecycle.md` | `assets/project-template/program.md` 通用循環 |
| `references/02_policy.md` | payload `AGENT_RULES.md` 機制說明 + root `SKILL.md` 政策節 |
| `references/03_task_discovery.md` | root `SKILL.md` §1.5（customer 改寫） |
| `references/04_ingestion.md` | 本文件 §4.2 + future `ingest.py` |
| `references/05_features.md` … `08_delivery.md` | root `SKILL.md` 對應建模/集成/交付節 |
| `references/09_guardrails.md` | §6 DoD + Phase 4 檢查項 |
| root `SKILL.md` | 保留 5–10 行指標連結 skill + references |

---

## 4. Kaggle → Customer Dataset 泛化設計

### 4.1 模式差異（必須顯式建模，不能靠 prompt 猜）

| 面向 | Kaggle 模式 | Customer 模式 |
|---|---|---|
| 任務定義 | 官方 metric + sample_submission 明確 | **未知，需訪談/推斷** → `references/03_task_discovery.md` |
| Target | 檔案給定 | 需定義 label window、決策時點、延遲可得性 |
| 資料入口 | `kaggle competitions download` | 檔案/DB/S3/Excel → `data_sources/*.yaml` |
| 評估 | LB 分數 | **成本函數**（FN/FP 成本、人工覆核率）、上線 KPI |
| 交付 | `submission.csv` | 批次評分檔 / API contract / 決策報表 + 閾值 |
| 隱私 | 公開 | **PII、需去識別化、不得外傳原始資料** |
| CV | 依競賽結構 | 依**業務時間軸與實體**（客戶/門市/案件） |

### 4.2 新增 `data_sources/<name>.yaml`（取代 `kaggle_process.md` 手抄）

```yaml
source_type: local_files | kaggle_competition | database | object_store | api
mode: customer
target: { column: churn_flag, positive_label: "Yes", label_window: "90d", available_after: "T+30d" }
entity: { primary_key: customer_id, time_col: snapshot_date }
privacy: { pii_columns: [name, email, phone], hash: sha256_salt_env, allow_external_search: false }
paths: { raw: data/raw, processed: data/processed }
splits_hint: { group_key: customer_id, time_col: snapshot_date }   # 供 cv-lock 取用（P0）
```

→ 由 `src/ingest.py` 實作 adapter，輸出 `data/processed/{train,test}.parquet` + `reports/data_manifest.json`（row count / schema / hash）。這一步同時解掉 P0-5。

### 4.3 Task Discovery 協議（customer 模式強制關卡）

1. **Schema 掃描**：dtype、cardinality、null、datetime 候選、PK 候選。
2. **推斷假設**：task family、target 候選、metric 候選、CV 候選（附信心度）。
3. **訪談清單（must-ask）**：預測什麼＋在什麼時點？標籤怎麼來、多久後可得？漏判(FP)與漏抓(FN)哪個貴？推論是批次還是即時？有沒有 PII/合規限制？
4. **確認寫回** `doc/00_problem_statement.md`（`confirmed_by_user: true` 才可進入 modeling）。
5. 若 `allow_external_search: true` 才允許 web/domain 搜尋（Kaggle 式的 Data Wide Search），否則**僅用 schema metadata**。

> 這一節就是「擴展到 customer dataset」的真正缺口：現在的 `SKILL.md §1.5` 把 wide search 當成預設，對客戶資料既不必要也可能違規。

### 4.4 交付形態抽象化

`doc/08_deployment_or_submission.md` 泛化為三選一：`submission_csv`（Kaggle）/ `batch_scoring`（客戶，含 `score_date` + `threshold` + 可解釋欄位）/ `api_contract`（輸入輸出 schema + 延遲預算）。`infer.py` → `src/deliver.py --mode`。

### 4.5 Init / Sync 契約（已落地腳本）

| 操作 | 行為 | Agent 須知 |
|---|---|---|
| `init_agentml_project.py <slug>` | `copytree` payload → `projects/<slug>/`；建立 `data/{raw,interim,processed}`、`runs/`；替換 `README.md` / `program.md` 的 `{{PROJECT_NAME}}` | 任務級 placeholder 在 yaml / doc 中，init **不**代填 |
| `sync_project.py <slug>` | 覆寫 `AGENT_RULES.md`、`program.md`、`README.md`、`src/*`、`configs/*` | **不碰** `data/`、`runs/`、`results.json`（除非 `--reset-ledger`） |
| `sync_project.py ... --include-docs` | 可覆寫 `doc/` | 會抹掉專案內已寫研究狀態 → 預設不用 |
| `sync_project.py ... --include-memory` | 可覆寫 `memory/` | 同上 |
| `check_ssot.py [--project path]` | R1–R7：payload 完整性、禁 `docs/`、doc 編號、禁 live 引用 legacy `templates/`、ledger、placeholder | CI 目標入口；本地需 `python` 在 PATH |

---

## 5. 遷移計畫（含舊→新路徑映射）

### Phase 0 — SSoT 止血（純文件，1 小時，零行為風險）
- 全 repo 統一 `doc/`（不是 `docs/`）與 `00..08` 編號；修 `04`/`08` 標題、`structure.md`、`AGENT_RULES.md`、root `README*.md`、`references/workflow.md`、`init_agentml_project.py:108`。
- `.clinerules/` 只留 `env.md`（並修 CLI 範例為 `--ensemble_run_id`）；刪除其餘舊快照，改由 skill 生成。**→ 工作區已無 `.clinerules/`。**
- `kaggle.md` 修編碼並移入 skill `references/sources/kaggle.md`；刪除 root `kaggle_process.ipynb`（或轉成 reference）。**→ 已移入 `references/sources/kaggle.md`；§5.1 hygiene 未完成。**
- `templates/results.json` → `[]`；`projects/*/results.json` 移成 `results.placeholder.json`。**→ payload `results.json` 已為 `[]`。**
- payload 遷至 `assets/project-template/` + `sync_project.py` + `check_ssot.py`。**→ 已落地（見 §8）。**

### 5.1 Kaggle 文件 hygiene（Phase 0 剩餘）

- [x] `references/sources/kaggle.md`：移除 `conda activate claw_env`、本機硬編碼路徑
- [x] CLI 範例與 `infer_ensemble.py` 對齊（`--ensemble_run_id`，非 `--ensemble hill_climb --run_id`）
- [x] 全文 UTF-8；範例競賽 slug 與 repo 實例一致或改為占位
- [x] `kaggle_init.ipynb` kernel 名稱去 OpenClaw 命名
- [x] `requirements.txt` 註解「OpenClaw」改為 AgentML

### Phase 1 — 讓它真的能跑（P0 全修，最高優先）
`src/models.py`（model factory）+ `evaluate.py` 實作 + `train.py` 接上 + `ensemble.py` 產 `ensemble_metadata.json` + `data.py` path resolver/target mapping + `src/doctor.py` + `tests/test_smoke.py`（200 筆合成資料、2 folds、斷言 metric 正確與 ledger schema）。

### Phase 2 — Skill 化重組
`templates/` → `.agents/skills/agentml/assets/project-template/`（單一擁有者）；root `SKILL.md` 內容拆入 `references/*.md`（保留 root `SKILL.md` 為 5 行指標，避免破壞既有連結）；9 個 script 收斂為 `scripts/agentml.py` 子命令（舊 script 保留薄 wrapper 一版以兼容 `env.md` 指令）；`project.yaml` 加入 `skill_version/policy_version`。

### Phase 3 — Customer 泛化
`data_sources/*.yaml`、`src/ingest.py`、`references/03_task_discovery.md`、`references/04_ingestion.md`、`src/deliver.py`、`doc/01_data_card.md` 增 PII 欄位、`references/02_policy.md` 增合規條款。

### Phase 4 — 可執行護欄 + CI
`scripts/guardrails.py`（下列檢查）＋ GitHub Actions：
1. target 不在 `feature_cols`；2. `doc/04_cv_strategy.md` 的 `locked_hash` 與 `data.py` split 實作 hash 一致；3. 任何模組引用 `test_path` 於 tuning 路徑 → fail；4. ledger schema 驗證；5. `{{...}}` 佔位符殘留檢查；6. group/query 不得跨 fold（實際重跑 split 驗證）；7. fold-safe 檢查（fit 只用到 train fold 的 AST 掃描啟發式）。

### Phase 5 — 既有專案遷移
`agentml migrate projects/<slug>`：補 `project.yaml`、重編號 doc、修正 config 路徑、驗證 ledger。

### 舊 → 新 映射

| 舊路徑 | 新路徑 |
|---|---|
| `SKILL.md`(root, 內容) | `.agents/skills/agentml/references/{00..09}.md` |
| `structure.md` | `references/00_contract.md`（root 留 1 行指標） |
| `templates/**` | `.agents/skills/agentml/assets/project-template/**` |
| `.clinerules/{AGENT_RULES,program,README,doc,configs,skills}/*` | 刪除（由 skill 提供）；保留 `env.md` |
| `kaggle.md` | `references/sources/kaggle.md`（UTF-8 重寫） |
| `templates/src/{train,evaluate,ensemble}.py` | 重寫；新增 `src/models.py`、`src/doctor.py`、`src/ingest.py`、`src/deliver.py` |

---

## 6. 驗收標準（Definition of Done）

1. `agentml new demo --mode customer` 後，`agentml doctor` 全綠、`agentml run` 在 200 列合成資料上 < 60s 完成，產出 `runs/<id>/{params,metrics,notes}.json` 與 OOF。
2. `agentml run` 對合成資料的 AUC 與 sklearn 直接計算值誤差 < 1e-9（測試斷言）。
3. `agentml guardrails` 對「故意把 target 放進 feature_cols」能失敗。
4. `results.json` 通過 schema 驗證；placeholder 記錄會被 `ledger verify` 拒絕。
5. `agentml ensemble --method hill_climb` 能端到端跑完並產出可提交的交付檔（`ensemble_metadata.json` 存在）。
6. 全 repo `grep -r "03_cv_strategy\|docs/0" ` 為 0 命中；無殘留 `{{...}}`。
7. 每個 `doc/*.md` 編號 == 檔名前綴（自動檢查）。

---

## 8. 執行狀態看板（滾動更新）

**驗證命令**（payload）：`python .agents/skills/agentml/scripts/check_ssot.py`  
**驗證命令**（實例）：`python .agents/skills/agentml/scripts/check_ssot.py --project projects/<slug>`

> **實測紀錄（2026-10-01，Windows / `claw_env`，於現行工作區）**
> - `check_ssot.py`（payload）→ `[PASS] 0 problem(s) / 107 note(s)`
> - `tests/test_smoke.py` → `Ran 1 test in 8.1s ... OK`（train → evaluate → ensemble 全鏈）
> - `findstr /s /i "03_cv_strategy"` → 0 命中；`"openclaw"` / `"claw_env"`（tracked text）→ 0 命中
> - payload `results.json` = `[]`；`projects/*/results.json` = `[]`
> - `guardrails.py --payload` → `9 checks, 0 fail`（G2/G5 於 payload 模式自動放寬為 WARN）
> - scaffold 探針 `agentml new _probe_customer --mode customer` → `project.yaml` 正確蓋章（mode/created）；`agentml cv-lock` 後 G2 `locked_hash matches`；未填佔位符時 G5 正確 FAIL（探針已刪）
> - E2E 探針（ingest → train → deliver）→ ingest：target 由 `yes/no` 二值化、PII `email` sha256、`reports/data_manifest.json`；deliver 三模式（`batch_scoring` auto-threshold=0.4755 / `api_contract` 4 features / `submission_csv`）全通過
> - `agentml.py` 13 個子命令可用；`doctor` → `0 core problem(s)`

### Phase 0 — SSoT 止血
- [x] payload 位於 `assets/project-template/`
- [x] 根 `templates/` 僅 shim README（`templates/README.md` = MOVED 重定向）
- [x] `check_ssot.py` / `sync_project.py` / `init_agentml_project.py`（+ `.ps1`）皆存在
- [x] payload doc `00..08` 編號與 H1 一致（`04`=`# 04 — CV Strategy`、`08`=`# 08 — Deployment / Submission`）；`AGENT_RULES.md` 使用 `doc/04_cv_strategy.md`
- [x] `doc/03_features_engineering.md` 已非空檔（含 feature registry 模板）
- [x] `references/sources/kaggle.md` 存在
- [x] Kaggle 文件 hygiene（§5.1）：無 `claw_env`、CLI 已對齊 `--ensemble_run_id`、UTF-8
- [x] root `structure.md` / `README.md` 與 doc 編號完全一致（`03_cv_strategy` = 0 命中）
- [x] `.clinerules/` 僅剩 `env.md` 且 CLI 範例已修（`--ensemble_run_id`）
- [x] payload `results.json` = `[]`；`projects/*/results.json` = `[]`（P0-6 假 ledger 已清）

### Phase 1 — 讓它真的能跑（P0）
- [x] P0-1 `evaluate.compute_metrics`（AUC/LogLoss/Brier/Acc/F1/RMSE/MAE/R²/NDCG@K/MAP@K）+ CLI `--run_id`
- [x] P0-2 `models.py` model factory + `train.py` 真實 per-fold OOF（模型確實 fit、存 `model.pkl`）
- [x] P0-3 `train_multi_model.py` 指標串接（依賴 P0-1，已解）
- [x] P0-4 `ensemble.py` 寫出 `ensemble_metadata.json`（生產者存在，smoke 斷言通過）
- [x] P0-5 `data.py` path resolver（`resolve_data_path`：processed 不存在時回退 `data/raw/*`）
- [x] `tests/test_smoke.py` 本地綠燈（200 列合成、2 folds、train→evaluate→ensemble、ledger schema）
- [x] CI 掛 `check_ssot`（`.github/workflows/ci.yml`：check_ssot + guardrails --payload + unittest）

### Phase 2 — Skill 化重組
- [x] root `SKILL.md` 內容 → `references/00..09`（新增 10 頁；root `SKILL.md` 保留 §0 環境 + §0.5 索引，§2–9 收斂為指標）
- [x] `scripts/agentml.py` 統一 CLI（new/sync/doctor/ingest/eda/cv-lock/run/ensemble/infer/deliver/ledger/guardrails/check-ssot；原 script 仍可直接呼叫）
- [x] `project.yaml`（mode / skill.version / policy.version；`new --mode` 蓋章、`sync` 僅補缺不清空）

### Phase 3 — Customer 泛化
- [x] `data_sources/*.yaml`（`README.md` + `example_customer.yaml`）+ `src/ingest.py`（local_files / kaggle_competition / database / stub；PII sha256、target 二值化、`reports/data_manifest.json`）
- [x] `references/03_task_discovery.md`（customer 模式必問清單 + Lifecycle Gate A）
- [x] `src/deliver.py`（submission_csv / batch_scoring（OOF auto-threshold）/ api_contract）
- [x] `references/04_ingestion.md`（adapter 契約）+ `08_delivery.md`

### Phase 4 — 可執行護欄 + CI
- [x] `scripts/guardrails.py`（G1..G9；payload 模式自動放寬 G2/G5）
- [x] GitHub Actions（`.github/workflows/ci.yml`）

### Phase 5 — 既有專案遷移
- [ ] `agentml migrate projects/<slug>`（補 `project.yaml`、重編號 doc、驗證 ledger）

### P2 殘項（非阻斷）
- [x] `requirements.txt` 補 `joblib` / `kaggle` / `openpyxl`（另註解 `sqlalchemy` 選用）
- [x] `src/__init__.py` 已建立（payload）
- [x] 清理 payload `src/__pycache__/*.pyc`（`.gitignore` 已含 `__pycache__/`）

**當前 sprint 建議**（與 §7 一致）：Phase 2（references 拆分 + `agentml.py`）與 Phase 4（`guardrails.py` + CI）並行；customer ingestion（Phase 3）仍應在 smoke 綠燈後進行（前置條件已具備）。

> **勘誤（本次實測）**：§1.4 表列 `.clinerules/`「已不在 tree」**不準確** —— 該目錄仍存在，僅保留 `env.md`（Phase 0 動作本身已完成，僅描述與現況不符）。§1.4 其餘「runtime 仍未閉環」描述亦已過時：P0-1/2/4/5 均已落地並通過 smoke。§1.3 的 `projects/*/runs` 不存在、config 指向不存在的 parquet 等敘述屬舊快照。

---

## 7. 風險與需你決策的點

| 議題 | 選項 A（推薦） | 選項 B |
|---|---|---|
| 模板 payload 位置 | 移入 skill（單一擁有者）**→ 已採用 A，payload 已在 `assets/project-template/`** | 保留 root `templates/` + skill 引用（雙來源，持續漂移風險） |
| 是否拆多個 skill | 1 個 `agentml` + references 分頁 | 拆成 `agentml-eda/-cv/-modeling/-ensemble` 子技能（鏈式呼叫複雜） |
| `projects/` 版控 | 維持 gitignored（客戶資料不外洩） | 改為只提交 `project.yaml` + 合成 smoke 專案 |
| 是否保留 OpenClaw 相容 | 保留 `program.md` 檔名，但內容由 skill 生成（相容舊指令） | 全面改為 `AGENT_PROTOCOL.md`（破壞相容） |
| Kaggle 支援 | 保留為 `mode: kaggle` adapter | 移到獨立 repo |

**建議下一步**：Phase 0 文件 hygiene（§5.1）與 **Phase 1 P0 runtime**（§8 看板）並行；**customer ingestion（Phase 3）應在 smoke test 綠燈之後**。進度以 **§8** 勾選為準，勿再依 §1.3 的 `templates/` 路徑操作。