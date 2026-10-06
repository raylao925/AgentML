# 01 — Lifecycle & Gates

The agent runs one **autonomous loop** per iteration. Gates below are hard: do not pass a
gate until its exit condition is met.

## Lifecycle

```
0. Bootstrap   → agentml new / project.yaml read
0.5 Restore    → read memory/NEXT, WINS, FAILURES, ITERATIONS, MEMORY  ← before planning
1. Discover    → 03_task_discovery  (mode=customer) | sources/kaggle.md (mode=kaggle)
2. Ingest      → 04_ingestion       (data/processed + reports/data_manifest.json)
3. EDA         → 02_eda.md          (agentml eda)
4. CV lock-in  → 04_cv_strategy.md written + locked_hash   ← GATE
5. Baseline    → agentml run        (first run in runs/ + results.json)
6. Iterate     → Plan → Execute → Log → Keep/Discard → Persist memory (1–2 change families/run)
7. Ensemble    → 07_ensemble
8. Deliver     → 08_delivery
H. Persist     → memory/ITERATIONS, WINS, NEXT, FAILURES  ← every round & session end
```

## Step 0.5 — Restore memory (mandatory, before any planning)

Every session (and every round after a break) starts by reading, **in this order**:

| # | file | why |
|---|---|---|
| 1 | `memory/NEXT.md` | the live queue — authoritative over `doc/06` §G; answers "what next" |
| 2 | `memory/WINS.md` | current best + every KEEP — do not re-test to rediscover |
| 3 | `memory/FAILURES.md` | every DISCARD + frozen families — do not re-run without new evidence |
| 4 | `memory/ITERATIONS.md` | round history and lessons |
| 5 | `memory/MEMORY.md` | goals, constraints, decisions, environment notes |
| 6 | `memory/debugging.md` | only when a known error resurfaces |

Skipping Step 0.5 is how a project silently loses its experimental memory: the next session
re-derives context and re-runs work that already failed. Files are schemas under the payload
`memory/`; see `references/02_policy.md` §8.

## Gate A — Task confirmed (customer mode)
- `project.yaml` `project.mode == customer` → `doc/00_problem_statement.md` must have
  `confirmed_by_user: true` before modeling. See `03_task_discovery.md`.

## Gate B — CV locked
- `doc/04_cv_strategy.md` exists, is non-empty, and contains `locked_hash:`.
- The hash matches the current split implementation (verified by `agentml guardrails`).
- Until then, only baseline/inference exploration is allowed — no hyperparameter search.

## Gate C — Data manifest
- `reports/data_manifest.json` exists (row count / schema / hash) for the current
  `data_version`. Re-run `agentml ingest` if the raw data changes.

## Gate D — Before delivery
- `agentml guardrails` passes (no leakage, no placeholder residue, ledger valid).
- Test set was **never** used for tuning.

## Iteration loop (Step A–D)

### Step A — Plan
Read `memory/NEXT.md` first, then find the best `keep` run in `results.json`; pick one
highest-expected-value change (1–2 families: one feature family / one hyperparam set / one model
class / one ensemble method). Write hypothesis + expected direction + file-level diff into
`runs/<run_id>/notes.md`. An idea already listed in `memory/FAILURES.md` needs new evidence to
be re-proposed.

### Step B — Execute
`run_id = YYYYMMDD_HHMM_<shortdesc>`. Produce `runs/<run_id>/{params,metrics,notes}` +
`artifacts/*`.

### Step C — Log
Append exactly one record to `results.json` (schema = `06_experiment_log.md`).

### Step D — Keep / Discard
Thresholds live in `configs/search_space.yaml:policy`. If discarded, still log a reason
(variance / speed / leakage risk / constraint violation).

### Steps E–G — Keep looping (see `10_iteration_loop.md`)
- **E — Queue refresh** (every round): re-rank `memory/NEXT.md` (authoritative), then copy the
  round block into the `doc/06_experiment_log.md` §G archive.
- **F — Divergence** (queue empty): invent/rank candidates from the idea bank
  (`10_iteration_loop.md` §3–4) before considering the work done — check `memory/FAILURES.md` first
  so frozen families are not regenerated.
- **G — Stop check**: plateau (3 rounds Δ < improve_threshold) / budget / deadline / exhaustion —
  stop only with a documented retrospective (`10_iteration_loop.md` §5).

### Step H — Persist memory (every round + session end)

After `results.json` is appended and the queue is refreshed, write the four state files **before**
ending the session. This is what makes cross-session continuity real.

| file | write when | minimum content |
|---|---|---|
| `memory/ITERATIONS.md` | every round | one `### Round <n>` block: run(s), family, result, decision, **lesson** |
| `memory/WINS.md` | every `KEEP` | a row in the KEEP table + **Current best** if it moved |
| `memory/FAILURES.md` | every `DISCARD` | a row with a closed-set `reason` (+ frozen family if exhausted) |
| `memory/NEXT.md` | every round | **Immediate next action** + re-ranked queue (≤5) + stopping check |

Also update `memory/MEMORY.md` when context/decisions changed, and `memory/debugging.md` when a
program error was resolved. Order: the four state files first, `MEMORY.md` last.

Hard rules:
1. A round that wrote `results.json` but no `ITERATIONS.md` block is **incomplete** —
   `agentml guardrails` flags it (G10).
2. `memory/NEXT.md` is the live queue; `doc/06` §G is the archived mirror of the same rounds.
3. Never delete history: append and correct forward.

## Reproducibility
- Fix seeds (`seed`, `seed_list`) and record `data_version`, `feature_version`, `code_hash`.
- Same data + config ⇒ same artifacts.