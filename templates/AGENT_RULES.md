# AGENT_RULES — Minimal Stable Policy (Auto-Extendable)

## 1) Data Leakage Zero Tolerance
- 任何 target proxy / post-event 欄位禁止使用
- 任何時間序列特徵（lag/rolling/expanding）必須：
  - 只用過去資料
  - 在每個 fold 的 train 區間內計算，再 apply 到 valid
- 任何 target encoding 必須 fold-safe（OOF encoding）

## 2) CV Authority & Compliance (Conditional, Lock-in Mode)

### 2.1 Authority Priority (highest → lowest)
P0) User prompt explicitly specifies CV and/or `group_key` and/or `time_col` and/or ranking `query_id`  
P1) Existing `docs/03_cv_strategy.md` (valid & non-empty)  
P2) Auto-infer from data understanding (EDA / df.info / schema scan) ONLY if P0 & P1 are absent

### 2.2 Lock-in Rule (Mode = (1))
- If P0: Agent MUST write the user-specified CV settings into `docs/03_cv_strategy.md` and lock it for all subsequent runs.
- Else if P1: Agent MUST follow `docs/03_cv_strategy.md` exactly. No silent changes.
- Else (P2): Agent MUST:
  1) infer a CV strategy using EDA evidence (columns, dtypes, duplication by group, time range, leakage risks)
  2) write the inferred strategy into `docs/03_cv_strategy.md` as the authoritative rule (include evidence/rationale section)
  3) from the very next run onward, treat it as P1 (locked unless user explicitly requests change)

### 2.3 Forbidden Silent Downgrades (After Lock-in)
Once `docs/03_cv_strategy.md` exists (P0/P1/P2), the agent MUST NOT:
- change GroupKFold → KFold
- change time-based split → random split
- break ranking query integrity (query_id must not cross folds)
unless the user explicitly requests AND `docs/03_cv_strategy.md` is updated with clear justification.

### 2.4 Documentation Requirement
Any CV change (only allowed via user request) MUST:
- update `docs/03_cv_strategy.md` (what changed + why + expected impact)
- create a new run with `runs/<run_id>/notes.md` recording the change rationale

## 3) Test Set Rule
- Test 不可用於：
  - 特徵選擇
  - 調參
  - 閾值搜尋
  - ensemble 權重學習
- Test 只可用於：
  - final report（一次）
  - 生成 submission / 推理輸出

## 4) Metric Rule
- primary metric 定義固定（可修 bug，不可改口徑）
- ranking 任務須明確：
  - query/group id
  - metric@k（如 NDCG@10）
- multi-class 須明確：
  - macro / micro / weighted averaging
  - probability calibration（如要做）

## 5) Change Size Rule (避免不可控)
- 每次 run 最多做 1~2 類改動：
  - (a) 一個 feature family
  - (b) 一組 hyperparameter tweak
  - (c) 一個 model class switch
  - (d) 一個 ensemble method
- 大改動必須拆成多次 run

## 6) Logging Rule
- 任何 run（keep 或 discard）都要寫入 `results.json`
- 每個 run 必須有 `runs/<run_id>/notes.md`，包含 hypothesis / change / outcome / next step
