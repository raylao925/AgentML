# FAILURES — Discarded Paths (do not repeat without new evidence)

## Purpose
The negative record: every path that was tried and rejected, with the reason and the evidence.
Reading this before generating ideas is what stops a session from re-running work that already
failed — the most expensive form of forgetting.

## How to Use
- **Read at session start** and **before every divergence pass** (`10_iteration_loop.md` §4).
- **Write on every DISCARD** (`01_lifecycle.md` Step H). A DISCARD is not complete until it is here.
- Scope: *experiment* failures (features / HPO / seeds / ensemble / data hypotheses). Program
  errors (stack traces, bugs, fixes) belong in `debugging.md`.

## Frozen families (rejected outright — need NEW evidence to reopen)

```markdown
- <family>: <why it is dead here> — <evidence link> — reopened only if: <condition>
```

## DISCARD table (chronological)

```markdown
| date | run_id / attempt | family | reason | detail | evidence |
|------|------------------|--------|--------|--------|----------|
| YYYY-MM-DD | <run_id> | <family> | overfit \| variance \| slow \| no-signal \| leakage-risk \| constraint | <one line> | runs/<run_id>/ |
```

## Rules
- `reason` must come from the closed set above so patterns are countable; `detail` carries the
  specifics (metric moved the wrong way, std blew up, exceeded `infer_seconds_max`, ...).
- Citing a row here is sufficient justification to reject a *proposed* idea in a later session —
  the proposal is then only reopened by new contradicting evidence.
- Never delete a row. If a discard was later proven wrong, add a new row with
  `reason = reopened` and link it.

---
*No failures recorded yet. First DISCARD (including a failed baseline) starts the table.*
