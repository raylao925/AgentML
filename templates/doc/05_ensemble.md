# 05 — Ensemble

## 1) Why Ensemble
- Single-model bias/variance issues?
- Do different models complement each other?
- Stable improvement on CV?

## 2) Allowed Ensemble Types
- Averaging (simple / weighted)
- Bagging over seeds
- Stacking (meta-model)
- Blending (holdout)

## 3) OOF Design (No Leakage)
- Train meta-model on OOF predictions only
- Meta-model CV must be clearly defined (same folds or nested CV)
- Do not use test set to derive ensemble weights

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