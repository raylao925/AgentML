# Kaggle CLI 常用命令大全

> **Scope**: Kaggle ingestion/submission recipe for `mode: kaggle` projects.
> Consolidated from the former root `kaggle.md` + `.clinerules/kaggle_process.md`
> (single source of truth = this skill).
> Automation: see `scripts/kaggle_init.ipynb` (lists playground competitions, creates
> `projects/<project_slug>/data/raw`, downloads + extracts, copies the template payload).

## Policy (MUST)
1. Data goes to `projects/<project_slug>/data/raw/` — raw data is gitignored, keep it there.
2. **Submission happens only when the user explicitly asks.** Never auto-submit.
3. Record every submission message/score in `doc/08_deployment_or_submission.md` and `results.json`.

## Environment

Use your project virtualenv or system Python with `kaggle` installed (`pip install kaggle`).
Run scripts with `python` / `py` from the **project root** (e.g. `projects/<slug>/`), not a hard-coded conda path.

---

## 1) 競賽下載

### 下載競賽資料
```bash
# 建立目錄結構
mkdir projects\{project_slug}\data\raw

# 切換到 raw 目錄
cd projects\{project_slug}\data\raw

# 下載競賽 ZIP
kaggle competitions download -c {competition_id}
```

**範例：**
```bash
mkdir projects\playground-series-s6e6\data\raw
cd projects\playground-series-s6e6\data\raw
kaggle competitions download -c playground-series-s6e6
```

### 解壓縮
```bash
# 解壓到當前目錄
tar -xf competition_id.zip
```
或用 PowerShell：
```powershell
Expand-Archive -Path {competition_id}.zip -DestinationPath .
```

---

## 2) 競賽提交

### 提交預測檔案
```bash
kaggle competitions submit -c {competition_id} -f {submission_path} -m "提交備註"
```

**提交後必做（LB 反饋環）**：把成績回填 ledger — 在該 run 的 `results.json` 記錄加上 `lb`
區塊（schema 見 payload `doc/06_experiment_log.md` §H：`score` / `rank` / `submitted_at` /
`cv_minus_lb`）。`cv_minus_lb` 的絕對值 > CV std → 標記 investigate（分布漂移或 OOF 過擬合），
**不得**因此更動已鎖定的 CV；公開榜用法受 `doc/00_problem_statement.md` 的提交預算約束。

**參數說明：**
- `-c`：競賽 ID
- `-f`：提交檔案路徑（CSV）
- `-m`：提交訊息（雙引號包覆）

**範例：**
```bash
kaggle competitions submit -c {competition_id} -f submissions/submission.csv -m "LGBM baseline v1"
```

---

## 3) 查看競賽狀態與排行榜

### 查看競賽列表
```bash
# 列出 userHasEntered True
kaggle competitions list | findstr /I "true"

# 列出所有競賽
kaggle competitions list

# 列出進行中的競賽
kaggle competitions list --category active

# 列出 playground, sort by latest deadline
kaggle competitions list --category playground --sort-by latestDeadline
```

### 查看競賽詳情
```bash
kaggle competitions list -c {competition_id}
```

### 查看排行榜
```bash
# 檢視排行榜（預設前 10 名）
kaggle competitions leaderboard -c {competition_id}

# 檢視自己的排名
kaggle competitions leaderboard -c {competition_id} -s
```

### 查看提交記錄
```bash
kaggle competitions submissions -c {competition_id}
```

---

## 4) 資料集操作

### 下載資料集
```bash
# 依 owner/dataset 名稱下載
kaggle datasets download -d {owner}/{dataset}

# 指定輸出目錄
kaggle datasets download -d {owner}/{dataset} -p {output_path}

# 不解壓（保留 ZIP）
kaggle datasets download -d {owner}/{dataset} --unzip=false
```

**範例：**
```bash
kaggle datasets download -d raymondbacon/airbnb-predict-prices
kaggle datasets download -d raymondbacon/airbnb-predict-prices -p data/raw
```

### 上傳資料集
```bash
# 上傳新資料集
kaggle datasets create -p {dataset_folder}

# 更新現有資料集
kaggle datasets version -p {dataset_folder} -m "版本備註"
```

### 列出資料集
```bash
# 列出自己的資料集
kaggle datasets list --owner {your_kaggle_id}

# 依關鍵字搜尋
kaggle datasets list --search "{keyword}"
```

---

## 5) Notebook 操作

### 下載 Notebook
```bash
kaggle kernels pull {owner}/{kernel-name}
```

**範例：**
```bash
kaggle kernels pull raymondbacon/lgbm-baseline
```

### 上傳 Notebook
```bash
kaggle kernels push -p {notebook_folder}
```

### 列出 Notebook
```bash
kaggle kernels list --owner {owner}
kaggle kernels list --search "{keyword}"
```

---

## 6) API 驗證與設定

### 檢查 API 是否設定
```bash
kaggle --version
```

### API Token 位置
```
# 預設位置
C:\Users\{username}\.kaggle\kaggle.json
```

### kaggle.json 格式
```json
{
  "username": "your_kaggle_username",
  "key": "your_kaggle_api_key"
}
```

> 從 Kaggle 網站 Account → API → Create New API Token 下載。

---

## 7) 完整工作流程範例

### 新競賽完整流程
```bash
# Step 1: 建立目錄與下載資料
mkdir projects\{project_slug}\data\raw
cd projects\{project_slug}\data\raw
kaggle competitions download -c {competition_id}
tar -xf {competition_id}.zip

# Step 2: 切回專案根目錄
cd ..\..\..   # projects/{project_slug}

# Step 3: 訓練 + 集成推論
python src/train_multi_model.py --config configs/baseline.yaml --models "LightGBM,CatBoost,XGBoost" --run_id_prefix v1
python src/ensemble.py --run_ids "run_a,run_b" --output_run_id ensemble_v1_weighted
python src/infer_ensemble.py --ensemble_run_id ensemble_v1_weighted

# 可選：在 OOF 上重新優化權重
python src/infer_ensemble.py --ensemble_run_id ensemble_v1_weighted --optimize_weights

# Step 4: 提交結果
kaggle competitions submit -c {competition_id} -f runs/ensemble_v1_weighted/artifacts/submission.csv -m "Ensemble v1"

# Step 5: 查看排名
kaggle competitions leaderboard -c {competition_id} -s
```

---

## 8) 實用快捷命令備忘

| 動作 | 命令 |
|------|------|
| 下載競賽 | `kaggle competitions download -c {id}` |
| 提交 | `kaggle competitions submit -c {id} -f {file} -m "{msg}"` |
| 看提交記錄 | `kaggle competitions submissions -c {id}` |
| 檢查排名 | `kaggle competitions leaderboard -c {id} -s` |
| 下載資料集 | `kaggle datasets download -d {owner}/{dataset}` |
| 下載 Notebook | `kaggle kernels pull {owner}/{kernel}` |
| 列出競賽 | `kaggle competitions list` |
| 看 API 版本 | `kaggle --version` |