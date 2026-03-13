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

詳見：`docs/01_data_card.md`

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
- Ledger: `results.csv`
- Artifacts: `runs/<run_id>/`