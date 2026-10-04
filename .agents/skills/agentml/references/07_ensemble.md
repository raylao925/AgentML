# 07 — Ensemble

> Code: `src/ensemble.py` (build) + `src/infer_ensemble.py` (apply). Design:
> `doc/05_ensemble.md`.

## Methods
- **weighted_average** (default): weighted blend of per-model OOF predictions.
- **blending**: hold-out blend (stack a simple meta-learner on OOF).
- **stacking** (implemented in `src/ensemble.py --method stacking`): fold-safe meta-learner —
  `classification_binary` → LogisticRegression, `regression` → Ridge; the meta model is fit on the
  OTHER folds' OOF rows and predicts each held-out fold (meta-OOF). The persisted
  `ensemble_metadata.json` gains a `meta` block (coef/intercept) which `infer_ensemble.py`
  applies at serving time (`sigmoid(coef · p + intercept)` for binary, linear for regression).
  Families outside binary/regression raise `NotImplementedError` (extend before use).
- **rank_average**: rank-normalise then average — probe it with an A/B (prob-avg vs rank-avg on
  OOF); it is a cheap test but frequently a no-op on already rank-aligned members.

All methods operate on **OOF predictions only** — the test set never participates in
weight learning.

## Weight optimization
- Weights are fit on OOF predictions via the project's CV (nested if possible), maximizing
  the primary metric (or minimizing it if lower-is-better).
- Constraints: weights ≥ 0 and sum to 1 (hill-climb / scipy SLSQP).
- Record weights into `runs/<ensemble_run_id>/params.json` and
  `runs/<ensemble_run_id>/ensemble_metadata.json` (members, metric, method, weights).

## Model diversity
- Prefer diversity across model families and `seed_list` values.
- Only add a member if it does not degrade the ensemble CV beyond `std_worsen_ratio_max`.

## Procedure
1. `agentml ensemble --run_ids a,b,c --method weighted_average`
2. Read each member's `runs/<id>/artifacts/oof_predictions.*` + `metrics.json`.
3. Optimize weights → write `ensemble_metadata.json` + a synthetic ledger record.
4. Inference: `agentml infer --ensemble --ensemble_run_id <id>`.

## Do not
- Do not select ensemble members using test scores.
- Do not blend models trained on different CV strategies without noting the mismatch.