# 09 — Guardrails (executable)

> Code: `scripts/guardrails.py`. Run: `agentml guardrails [--project <slug>]`.
> `references/02_policy.md` owns the *rules*; this page owns the *checks*.

Guardrails are executable and CI-wired (`.github/workflows/ci.yml`). Exit code `0` = all
pass, `1` = at least one FAIL.

## Checks

| id | check | severity |
|---|---|---|
| G1 | `target` not present in effective `feature_cols` | FAIL |
| G2 | `doc/04_cv_strategy.md` has `locked_hash` matching `src/data.py` split code | FAIL |
| G3 | no test-set usage inside tuning paths (tune/optuna/hyperopt scripts) | FAIL |
| G4 | `results.json` is a valid ledger (schema per `06_experiment_log.md`) | FAIL |
| G5 | no unresolved `{{...}}` placeholders in doc/configs (project mode) | FAIL |
| G6 | GroupKFold / ranking: no group/query crossing train↔valid folds | FAIL |
| G7 | AST heuristic: no `.fit()`/`.fit_transform()` called on valid/test data | WARN |
| G8 | `project.yaml` present with `skill.version` + `policy.version` | FAIL |
| G9 | Frozen-file hashes (when `# AGENTML:FROZEN` marker present) unchanged | FAIL |

## Hash convention
`agentml cv-lock` writes into `doc/04_cv_strategy.md`:

```
locked_hash: <sha256 of the normalised split implementation in src/data.py>
```

`guardrails` recomputes the hash from the current `src/data.py` and compares. Any silent CV
change is caught here and in CI.

## CI
`.github/workflows/ci.yml` runs, on push/PR:
1. `python scripts/check_ssot.py` (structure/anti-drift)
2. `python scripts/guardrails.py --payload` (template-level G1/G4/G8/G9)
3. `python -m unittest discover -s tests` (smoke: train→evaluate→ensemble)

## Fixing a failure
Guardrails failures are **blocking**. Resolve by fixing the code/config/doc — never by
weakening the check. If a check is genuinely wrong, change it here in the SSoT and note why.