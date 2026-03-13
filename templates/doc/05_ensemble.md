# 05 — Ensemble

## 1) Why Ensemble
- 單模型 bias/variance 問題？
- 不同模型互補？
- 在 CV 上是否穩定提升？

## 2) Allowed Ensemble Types
- Averaging (simple / weighted)
- Bagging over seeds
- Stacking (meta-model)
- Blending (holdout)

## 3) OOF Design (No Leakage)
- 只用 OOF predictions 訓練 meta-model
- meta-model 的 CV 也要清晰定義（可用同 folds 或 nested CV）
- 禁止用 test set 產生 ensemble 權重

## 4) Candidate Ensembles
### Ensemble A: Weighted Average
- Inputs: [model_1, model_2, ...]
- Weights: [...]
- How weights chosen: (grid / optimization on OOF)
- Result: mean/std

### Ensemble B: Stacking
- Base models:
- Meta model:
- Features to meta:
- Regularization:
- Result: mean/std

## 5) Final Ensemble Decision
- Selected:
- Inference complexity:
- Deployment risk: