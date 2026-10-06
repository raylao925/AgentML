# Project Memory & Context

## Purpose
This file maintains persistent context across agent sessions to ensure continuity and consistency.

## How to Use
- **Read at session start**: Restore context from previous work
- **Update during work**: Record key decisions and user preferences
- **Keep concise**: Focus on essential context, not implementation details

## Session Start Protocol (read in this order)
1. `NEXT.md` — what is queued right now (authoritative over `doc/06` §G)
2. `WINS.md` — what already won + current best (never re-test to rediscover it)
3. `FAILURES.md` — what was rejected + why (never re-run without new evidence)
4. `ITERATIONS.md` — the round history and lessons
5. this file — goals, constraints, decisions, environment notes
6. `debugging.md` — only when a known error resurfaces

Reverse the order at session end: write `ITERATIONS/WINS/NEXT/FAILURES` first (`01_lifecycle.md`
Step H), then update this file if context or decisions changed.

## File Map
| file | owner question | write trigger |
|---|---|---|
| `NEXT.md` | what do I run next? | every round (Step H) |
| `WINS.md` | what already worked? | every KEEP |
| `FAILURES.md` | what must I not repeat? | every DISCARD |
| `ITERATIONS.md` | what happened, and what did it prove? | every round (Step H) |
| `MEMORY.md` | who/what/why is this project? | when context changes |
| `debugging.md` | what broke and how was it fixed? | on program errors |

## Content Guidelines
- Project goals and constraints
- User preferences and feedback history
- Key architectural decisions and rationale
- Important constraints or limitations
- Cross-session continuity notes

## Example Structure
```markdown
## Project Context
- Goal: [Main objective]
- Constraints: [Key limitations]
- User preferences: [Important preferences]

## Key Decisions
- [Decision 1]: [Rationale]
- [Decision 2]: [Rationale]

## Session Notes
- [Date]: [Key points from last session]
```

> Experiment state does **not** belong here. Rounds live in `ITERATIONS.md`, keeps in `WINS.md`,
> queue in `NEXT.md`, rejects in `FAILURES.md`. This file stays the context layer.

---
*This file should be updated whenever significant context changes occur.*