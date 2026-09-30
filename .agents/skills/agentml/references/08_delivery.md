# 08 — Delivery

> Code: `src/deliver.py` (general) + `src/infer.py` / `src/infer_ensemble.py`.
> Spec: `doc/08_deployment_or_submission.md`.

`project.yaml` `project.mode` selects the default mode; `deliver --mode` can override.

## Modes

### `submission_csv` (kaggle)
- Columns exactly as `sample_submission.csv` (id + target), same row order.
- No thresholding; write raw predicted value / probability.
- Output: `deliverables/submission.csv`.

### `batch_scoring` (customer, default)
- Score a scored batch (test/append set) and emit a **decision** column:
  - classification: `score` + `predicted_label` via a configurable `threshold`
    (default `0.5`, or the F-beta-optimal threshold chosen on **OOF**, never test).
  - regression: `score` (and optional clipped value).
  - ranking: `score` per (query, item).
- Output: `deliverables/scored_<name>.csv` + `deliverables/scoring_manifest.json`.

### `api_contract`
- Emit a schema/spec (feature order + dtype + missing-value policy) for the serving layer:
  `deliverables/api_contract.json`. The agent does not host the endpoint.

## Rules
- Inference loads the saved pipeline (preprocessing + model) so train/serve preprocessing
  are identical. Missing handling matches `05_features`.
- Thresholds (if any) are chosen on OOF predictions, never on the test set.
- Record the producing `run_id` / `ensemble_run_id` and `data_version` in the manifest.

## Output
- `deliverables/*.csv` (or `api_contract.json`) + `deliverables/*_manifest.json`.

## Do not
- Do not re-fit any preprocessing at inference time.
- Do not leak test labels; scoring input must not contain the target.