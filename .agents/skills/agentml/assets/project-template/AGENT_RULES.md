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
P1) Existing `doc/04_cv_strategy.md` (valid & non-empty)  
P2) Auto-infer from data understanding (EDA / df.info / schema scan) ONLY if P0 & P1 are absent

### 2.2 Lock-in Rule (Mode = (1))
- If P0: Agent MUST write the user-specified CV settings into `doc/04_cv_strategy.md` and lock it for all subsequent runs.
- Else if P1: Agent MUST follow `doc/04_cv_strategy.md` exactly. No silent changes.
- Else (P2): Agent MUST:
  1) infer a CV strategy using EDA evidence (columns, dtypes, duplication by group, time range, leakage risks)
  2) write the inferred strategy into `doc/04_cv_strategy.md` as the authoritative rule (include evidence/rationale section)
  3) from the very next run onward, treat it as P1 (locked unless user explicitly requests change)

### 2.3 Forbidden Silent Downgrades (After Lock-in)
Once `doc/04_cv_strategy.md` exists (P0/P1/P2), the agent MUST NOT:
- change GroupKFold → KFold
- change time-based split → random split
- break ranking query integrity (query_id must not cross folds)
unless the user explicitly requests AND `doc/04_cv_strategy.md` is updated with clear justification.

### 2.4 Documentation Requirement
Any CV change (only allowed via user request) MUST:
- update `doc/04_cv_strategy.md` (what changed + why + expected impact)
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

## 7) Memory Rule (HARD — cross-session state)

Six files under `memory/`. **Four are experiment state and are non-negotiable.**

| file | question it answers | write trigger | minimum content |
|---|---|---|---|
| `memory/NEXT.md` | what do I run next? — **live queue, authoritative over `doc/06` §G** | every round | immediate action + queue ≤5 (EV/cost) + stopping check |
| `memory/ITERATIONS.md` | what happened and what did it prove? | every round | `### Round <n>` block: run(s) · family · result · decision · **lesson** |
| `memory/WINS.md` | what already worked + current best | every `KEEP` | KEEP row + current best |
| `memory/FAILURES.md` | what was rejected + why | every `DISCARD` | row with closed-set reason (+ frozen family) |
| `memory/MEMORY.md` | context: goals, constraints, decisions, preferences | when context changes | concise — no experiment state |
| `memory/debugging.md` | program errors only | on program errors | error → cause → fix |

Agent responsibilities (**mandatory, not advisory**):
1. **Session start (read in order):** `NEXT.md` → `WINS.md` → `FAILURES.md` → `ITERATIONS.md` →
   `MEMORY.md` (`program.md` §1, `01_lifecycle.md` Step 0.5). Do not plan before reading them.
2. **After every round (write):** `ITERATIONS.md` + `NEXT.md`, then `WINS.md`/`FAILURES.md`
   according to the decision (`01_lifecycle.md` Step H, `program.md` Step E). A round that wrote
   `results.json` but no `ITERATIONS.md` block is incomplete.
3. **Never re-propose** an idea listed in `FAILURES.md` without new evidence; **never re-test** to
   rediscover a result already recorded in `WINS.md`.
4. **Never store** rounds/queue/keeps/rejects inside `MEMORY.md` or `debugging.md`.
5. **Append only** — correct forward, never rewrite history.
6. `doc/06` §G is the archived mirror of the round queue; `memory/NEXT.md` wins on disagreement.

Enforcement: `agentml guardrails` → G10 (presence + freshness vs `results.json`).
`agentml check-ssot` → R8 (`doc/06` §G must stay "Round Queue" and point at `memory/NEXT.md`).
