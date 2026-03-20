# 01 — Data Card

## Dataset Overview
- Source: {{SOURCE}}
- Source Path: {{SOURCE_PATH}}
- Owner / Contact: {{CONTACT}}
- Data versioning:
  - Raw snapshot: {{RAW_SNAPSHOT_ID}}
  - Extraction SQL / Pipeline ref: {{PIPELINE_REF}}
- Grain (data granularity): e.g. per-customer / per-order / per-day / per-room-night

## Schema
> Suggested: paste a column list (name / type / meaning / null% / notes)

## Target Definition
- Target column: `{{TARGET}}`
- Label window / horizon (if applicable):
- Positive class definition (classification):
- Business meaning / threshold (if any):

## Leakage Checklist (Must Pass)
- [ ] Features contain no post-event information
- [ ] No target synonym / proxy columns used (e.g. refund status)
- [ ] Time features do not leak future (training must not use future dates)
- [ ] Aggregation features computed per fold / per time window (not on full data)
- [ ] No ID leakage (e.g. customer_id directly mapping to target)

## Data Quality Notes
- Missing patterns:
- Outliers:
- Duplicates:
- Label noise / ambiguity:

## Splitting Requirements
- Rationale for Time-based / Group-based / Stratified (conclusion here; details in `03_cv_strategy.md`)