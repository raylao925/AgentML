# 00 — Problem Statement

## Objective
用一句話定義：**要預測/分類/排序什麼？對誰？在什麼時間點？**

## Success Criteria
- Primary metric: {{PRIMARY_METRIC}}
- Minimum acceptable score: {{MIN_SCORE}}
- Constraints:
  - Latency: {{LATENCY_LIMIT}}
  - Model size: {{SIZE_LIMIT}}
  - Interpretability: {{INTERPRETABILITY_REQ}}

## Evaluation Setup (High Level)
- Train/Valid/Test 定義：
- Online/Offline 指標對齊：
- Leakage 風險假設：

## Risks & Non-goals
- 不做什麼（例如：不做 causal、唔做 real-time streaming）
- 已知風險（data drift / label noise / bias）