# AGENT_RULES — Minimal Stable Policy (Auto-Extendable)

## 1) Data Leakage Zero Tolerance
- Any target proxy / post-event columns are forbidden
- Any time-series features (lag/rolling/expanding) must:
  - use past data only
  - be computed within each fold's train window, then applied to valid
- Any target encoding must be fold-safe (OOF encoding)

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
- Test must NOT be used for:
  - feature selection
  - hyperparameter tuning
  - threshold search
  - ensemble weight learning
- Test may ONLY be used for:
  - final report (once)
  - generating submission / inference output

## 4) Metric Rule
- primary metric definition is fixed (bug fixes allowed, not definition changes)
- ranking tasks must specify:
  - query/group id
  - metric@k (e.g. NDCG@10)
- multi-class must specify:
  - macro / micro / weighted averaging
  - probability calibration (if applicable)

## 5) Change Size Rule (avoid uncontrolled drift)
- Each run: at most 1–2 types of changes:
  - (a) one feature family
  - (b) one hyperparameter tweak set
  - (c) one model class switch
  - (d) one ensemble method
- Large changes must be split across multiple runs

## 6) Logging Rule
- Every run (keep or discard) must be written to `results.json`
- Each run must have `runs/<run_id>/notes.md` with hypothesis / change / outcome / next step

## 7) Memory & Debugging Rule
- **MEMORY.md**: Persistent conversation context
  - Store key decisions, user preferences, and project constraints
  - Maintain continuity across sessions
  - Update when significant context changes occur
- **debugging.md**: Debug and troubleshooting logs
  - Record error messages, stack traces, and debugging steps
  - Document failed attempts and solutions found
  - Include code snippets that caused issues and their fixes
- Agent responsibilities:
  - Read memory files at session start to restore context
  - Write important decisions and debugging info during problem-solving
  - Keep debugging.md focused on technical issues, MEMORY.md on context
