# ITERATIONS — Round Log (append-only)

## Purpose
One block per experiment round, newest at the **bottom**. This is the human-readable history of
*what was tried and what it proved*. The machine ledger stays `results.json`; this file carries the
reasoning that a JSON record cannot.

## How to Use
- **Read at session start** (`01_lifecycle.md` Step 0.5) — restore where the loop actually is.
- **Write every round** (`01_lifecycle.md` Step H) — one block, after `results.json` is appended.
- **Freshness rule** (`09_guardrails.md` G10): if `results.json` has records, this file must have a
  matching block. A run with no block here means the memory was not persisted.

## File Map
- `WINS.md` — accepted improvements (KEEP) + current best
- `NEXT.md` — live queue / what to run next (authoritative over `doc/06` §G)
- `FAILURES.md` — discarded paths, do not re-test without new evidence
- `MEMORY.md` — project context, decisions, user preferences
- `debugging.md` — program errors only (stack traces, fixes)

## Format (append one block per round)

```markdown
### Round <n> — <YYYY-MM-DD>
- Run(s): <run_id>
- Family: <features | HPO | seeds | diversity | ensemble | postproc | data | infra | fix>
- Change: <one or two lines — what exactly changed>
- Result: <primary metric> <mean> ± <std> (Δ vs best = <signed delta>)
- Decision: KEEP | DISCARD | TIE — <rule applied from doc/06 §D>
- Lesson: <one line — what this proves / disproves, reusable next time>
- Next: <which queue item this unlocks, or "queue unchanged">
```

Minimum fields: **Run(s) · Family · Result · Decision · Lesson**. A block without a `Lesson`
is an orphan record and will not help the next session.

---
*No rounds recorded yet. Append the first `### Round 1` block after the baseline run.*
