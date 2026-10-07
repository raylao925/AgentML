# 02 — Policy (Mechanism)

> This page owns the **mechanism** of the hard rules. The concrete *values* live in the
> project (`doc/04_cv_strategy.md`, `configs/search_space.yaml`, `project.yaml`), never here.

## 1) CV authority (lock-in)

Priority for choosing the split: **P0** user prompt > **P1** existing `doc/04_cv_strategy.md`
> **P2** auto-infer from EDA. Once written, the strategy is **locked**.

After lock-in the agent MUST NOT silently change:
- Group-based CV → random CV
- time-based split → random split
- ranking `query_id` integrity (a query must not cross folds)

Any change requires an explicit user request **and** an update to `doc/04_cv_strategy.md`
plus a `runs/<run_id>/notes.md` rationale. `doc/04_cv_strategy.md` carries a `locked_hash`;
`agentml guardrails` fails if it does not match the split implementation in `src/data.py`.

## 2) Leakage (zero tolerance)
- No target proxy / post-event columns. Honour `doc/01_data_card.md` leak list and
  `configs/baseline.yaml:data.drop_cols`.
- Time-series features use **past only**, computed inside the fold train window.
- Any target encoding must be **OOF / train-fold-only**.
- Every transformer is `fit` on the train fold only, then applied to valid/test.

## 3) Test-set rule
Test may be used **only** for the final report (once) and generating submission/inference.
It must never drive feature selection, hyperparameter tuning, threshold search, or ensemble
weight learning. `agentml guardrails` scans for test usage inside tuning paths.

## 4) Frozen / Writable zones

| Zone | Files | Who may change | Condition |
|---|---|---|---|
| **Frozen** | `doc/04_cv_strategy.md` lock block, `evaluate.py` metric semantics, `results.json` schema, `baseline.yaml` `task.target/primary_metric`, `project.yaml` versions | user + doc update only | keeps runs comparable |
| **Semi** | `configs/*.yaml`, `src/features.py`, `src/models.py` params, `src/eda.py` | agent | must log the diff in `runs/<run_id>/notes.md`; stay inside `search_space.yaml` |
| **Free** | `src/train*.py` flow, `src/deliver.py`, `plots/`, new feature functions | agent | this is the main work area |
| **Forbidden** | `data/raw/*` mutation, using test for tuning, `target` inside `feature_cols` | nobody | §2/§3 |

Frozen files may carry a header marker `# AGENTML:FROZEN v1 <sha256>`. `agentml doctor`
verifies these hashes before a run.

## 5) Metric rule
The primary-metric definition is fixed (bug fixes only, never redefinition). Ranking tasks
must declare query/group id and metric@k. Multi-class must declare macro/micro/weighted.

## 6) Change-size rule
At most 1–2 change families per run (feature family / hyperparam set / model class / ensemble
method). Split large changes across runs.

## 7) Logging & reproducibility
Every run (keep **and** discard) appends exactly one `results.json` record and writes
`runs/<run_id>/notes.md` (hypothesis / change / outcome / decision). Fix seeds and record
`data_version`, `feature_version`, `code_hash`.

## 8) Memory

Cross-session memory lives in `memory/` as six markdown files: two context files plus four
**experiment state** files — the four state files are what makes the loop survive a session break.

| file | question it answers | write trigger |
|---|---|---|
| `memory/NEXT.md` | what do I run next? (live queue, **authoritative over `doc/06` §G**) | every round (Step H) |
| `memory/ITERATIONS.md` | what happened and what did it prove? | every round (Step H) |
| `memory/WINS.md` | what already worked + current best | every `KEEP` |
| `memory/FAILURES.md` | what was rejected + why (frozen families) | every `DISCARD` |
| `memory/MEMORY.md` | goals, constraints, decisions, environment notes | when context changes |
| `memory/debugging.md` | program errors only (stack traces, fixes) | on program errors |

Lifecycle: **Step 0.5** (`01_lifecycle.md`) reads them in order NEXT → WINS → FAILURES →
ITERATIONS → MEMORY; **Step H** writes the four state files after every round and at session end.

Schemas live in the payload `memory/` (the contract of *what* each file holds). Values live in the
project. `agentml guardrails` enforces presence/freshness (G10); `agentml check-ssot` enforces that
`doc/06` §G keeps pointing at this memory (`R8`).

Hard rules:
- Never store experiment state (rounds, queue, keeps, rejects) in `MEMORY.md` or `debugging.md`.
- Never re-propose an idea listed in `FAILURES.md` without new evidence; never re-test to
  rediscover a result already in `WINS.md`.
- Append and correct forward — history is not rewritten.