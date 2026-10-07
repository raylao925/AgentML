# WINS — Accepted Improvements (KEEP)

## Purpose
The compounding record: every `KEEP` that actually moved the metric, plus the **current best**.
Without it each session restarts from scratch and re-tests work that already won.

## How to Use
- **Read at session start** (`01_lifecycle.md` Step 0.5) before proposing anything new.
- **Write on every KEEP** (`01_lifecycle.md` Step H) — a KEEP is not finished until it is here.
- A `TIE` is recorded too (it proves the lever is exhausted) but never becomes "current best".

## Current best
```markdown
- Current best: <run_id> @ <primary metric> <value> (as of <YYYY-MM-DD>)
- Shipped / submitted: <run_id or "none">
- Rollback target: <run_id or "none">
- CV: <cv_type from doc/04_cv_strategy.md, locked_hash prefix>   <- all Δ below are only
                                                                    comparable within this CV
```

## KEEP table (chronological — never reorder)

```markdown
| date | run_id | family | Δ vs prev best | metric (mean ± std) | why it paid | evidence |
|------|--------|--------|----------------|---------------------|-------------|----------|
| YYYY-MM-DD | <run_id> | <family> | +0.000000 | 0.000000 ± 0.000000 | <one line> | runs/<run_id>/, results.json |
```

## Rules
- `Δ vs prev best` is computed against the row above it, using the primary metric direction
  (`doc/06_experiment_log.md` §D). Never recompute a historical row after a CV or metric change —
  note the invalidation in `FAILURES.md` instead.
- Keep this table short: it is the "what already works" index, not the full log (that is
  `ITERATIONS.md`).
- Current best must always equal the best `keep` record in `results.json`; if they disagree,
  `results.json` wins and this file is stale.

---
*No KEEPs yet. First baseline run becomes the initial current best.*
