# {{PROJECT_NAME}}

## 1) Problem
- Task type: (binary / multiclass / regression / ranking)
- Target: `{{TARGET_COLUMN}}`
- Primary metric: `{{PRIMARY_METRIC}}`
- Business goal: {{BUSINESS_GOAL}}

## 2) Data
- Data source: {{SOURCE}}
- Data version: {{DATA_VERSION}}
- Row count: {{N_ROWS}}
- Feature count: {{N_FEATURES}}
- Time range (if applicable): {{TIME_RANGE}}

See `docs/01_data_card.md` for details.

## 3) Approach Summary
- Baseline: {{BASELINE_MODEL}}
- CV strategy: {{CV_STRATEGY}}
- Best single model: {{BEST_SINGLE_MODEL}}
- Ensemble: {{ENSEMBLE_METHOD}}
- Notes: {{KEY_INSIGHTS}}

## 4) How to Run
- Train: `python src/train.py --config configs/baseline.yaml`
- Evaluate: `python src/evaluate.py --run_id <run_id>`
- Infer: `python src/infer.py --model runs/<run_id>/artifacts/model.pkl --input ...`

## 5) Results
- Ledger: `results.json` (append-only; schema in `docs/06_experiment_log.md`)
- Artifacts: `runs/<run_id>/`

---

## 6) Src Contract (Programmatic Interface for Agent)

After this template is copied to `projects/<project_slug>/`, the expected `src/` interface:

- `data.py`
  - Read `data` and `cv` blocks from `configs/baseline.yaml`.
  - Build folds per `docs/03_cv_strategy.md` (StratifiedKFold / GroupKFold / TimeSeriesSplit, etc.).
  - Split / schema exploration / profiling only here; do not change CV rules elsewhere.

- `features.py`
  - Build **fold-safe** feature pipeline from `features.flags` and `features.params`.
  - Flags: scaler / onehot / target encoding / featuretools (auto feature), etc.
  - Fit on train fold only; transform valid/test; no leakage.

- `train.py`
  - Entry: `python src/train.py --config configs/baseline.yaml [--run_id ...]`.
  - Flow: read baseline config → load data → build folds → run CV → write:
    - `runs/<run_id>/params.json`, `metrics.json`, `notes.md`
    - `runs/<run_id>/artifacts/*` (model, OOF, feature_list, dataset_profile_*, etc.)
  - Append one record to `results.json`.

- `evaluate.py`
  - Entry: `python src/evaluate.py --run_id <run_id>`.
  - Read OOF from `runs/<run_id>/artifacts/oof_predictions.*`; recompute primary/secondary metrics (metric definition fixed).

- `infer.py`
  - Entry: `python src/infer.py --run_id <run_id> [--output ...]`.
  - Load `artifacts/model.pkl` and config `data.id_cols`; run inference on test; produce submission (default `artifacts/submission.csv` with id + prediction).

- `ensemble.py` (*optional*)
  - Use as ensemble entry: read `runs/*/artifacts/oof_predictions.*` and `metrics.json`; per `docs/05_ensemble.md` produce weighted average or stacking; write to new `runs/<run_id_ensemble>/` and `results.json`.

> Recommendations for OpenClaw / Agent:  
> - Drive experiments via `configs/*.yaml`, `docs/*`, `src/train.py`, `src/ensemble.py` only.  
> - Avoid changing metric definition, CV implementation, or ledger format directly.
