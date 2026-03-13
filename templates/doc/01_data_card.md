# 01 — Data Card

## Dataset Overview
- Source: {{SOURCE}}
- Owner / Contact: {{CONTACT}}
- Data versioning:
  - Raw snapshot: {{RAW_SNAPSHOT_ID}}
  - Extraction SQL / Pipeline ref: {{PIPELINE_REF}}
- Grain（資料粒度）: 例如「每客/每訂單/每日/每房晚」

## Schema
> 建議貼一段欄位清單（name / type / meaning / null% / notes）

## Target Definition
- Target column: `{{TARGET}}`
- Label window / horizon（如適用）:
- Positive class definition（分類）:
- Business meaning / threshold（如有）:

## Leakage Checklist (Must Pass)
- [ ] 特徵是否包含「事後才知道」的資訊（post-event）
- [ ] 是否使用 target 同義欄位 / proxy（例如退款狀態）
- [ ] 時間特徵是否穿越（training 時用到未來日期）
- [ ] 聚合特徵是否用全量資料算（必須按 fold / 按時間切）
- [ ] ID leakage（例如 customer_id 直接映射 target）

## Data Quality Notes
- Missing patterns:
- Outliers:
- Duplicates:
- Label noise / ambiguity:

## Splitting Requirements
- Time-based / Group-based / Stratified 的理由（先寫結論，細節放 `03_cv_strategy.md`）