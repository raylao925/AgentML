"""
templates/src/features.py

Feature engineering template for AgentML projects.

Goal:
- Build a fold-safe preprocessing pipeline.
- Enable configurable feature families via `configs/baseline.yaml` flags.
- Support optional auto-feature generation (e.g. featuretools) in a fold-safe manner.

Agent development steps (commentary):
1) Read `config.features.flags` and `config.features.params`.
2) Determine feature columns:
   - exclude `target`
   - exclude `data.id_cols`
   - exclude `data.drop_cols`
3) Construct preprocessing as an sklearn pipeline / transformer that is fit on the TRAIN fold only.
4) Ensure OOF / fold-safe logic for any target encoding or fold aggregation.
5) Optionally support:
   - numeric: impute, optional scaling, transforms
   - categorical: impute + one-hot (or categorical-friendly handling)
   - target encoding: fold-safe mapping trained on train fold only
   - auto feature (featuretools): DFS features fit on train fold only, then applied to valid/test

This module should not train models or write ledger entries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_columns(
    df: pd.DataFrame,
    target_col: str,
    id_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
) -> list[str]:
    exclude = {target_col}
    if id_cols:
        exclude.update(c for c in id_cols if c in df.columns)
    if drop_cols:
        exclude.update(c for c in drop_cols if c in df.columns)
    return [c for c in df.columns if c not in exclude]


class FeaturetoolsTransformer:
    """
    Minimal placeholder transformer for fold-safe featuretools usage.

    IMPORTANT:
    - This is a template stub. Implement full sklearn-style fit/transform if you enable it.
    - In production, ensure:
      - fit DFS only on train fold
      - apply the same feature definitions to valid/test
      - no leakage of statistics computed on valid
    """

    def __init__(self, trans_primitives=None, agg_primitives=None, max_depth: int = 1) -> None:
        self.trans_primitives = trans_primitives or ["day", "month", "year"]
        self.agg_primitives = agg_primitives or ["mean", "std", "sum", "count"]
        self.max_depth = max_depth
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y=None):  # type: ignore[override]
        import featuretools as ft

        # Step A: create entity set on train fold only
        df = X.copy()
        df = df.reset_index(drop=True)
        df["__ft_index__"] = np.arange(len(df))
        es = ft.EntitySet(id="main").add_dataframe(
            dataframe_name="data",
            dataframe=df,
            index="__ft_index__",
        )

        # Step B: run DFS on train fold only
        self._fm, self._features = ft.dfs(
            entityset=es,
            target_dataframe_name="data",
            trans_primitives=self.trans_primitives,
            agg_primitives=self.agg_primitives,
            max_depth=self.max_depth,
        )
        self._feature_names = [f.get_name() for f in self._features]
        return self

    def transform(self, X: pd.DataFrame):  # type: ignore[override]
        import featuretools as ft

        if not hasattr(self, "_features"):
            raise RuntimeError("FeaturetoolsTransformer must be fit before transform.")
        df = X.copy()
        df = df.reset_index(drop=True)
        df["__ft_index__"] = np.arange(len(df))
        es = ft.EntitySet(id="main").add_dataframe(
            dataframe_name="data",
            dataframe=df,
            index="__ft_index__",
        )
        fm = ft.calculate_feature_matrix(self._features, entityset=es)
        fm = fm[self._feature_names]
        return fm.values

    def get_feature_names_out(self) -> list[str]:
        return list(self._feature_names)


def build_feature_pipeline(
    config: dict[str, Any],
    feature_cols: list[str],
    train_df: pd.DataFrame,
) -> tuple[Pipeline | ColumnTransformer, list[str]]:
    """
    Build and FIT a preprocessing pipeline on the train fold.

    Agent notes:
    - Caller (train.py) should pass train-fold slice as `train_df`.
    - This function must fit only on the provided train-fold data.
    """
    features_cfg = config.get("features", {})
    flags = features_cfg.get("flags", {})

    use_scaler = bool(flags.get("use_scaler", False))
    use_onehot = bool(flags.get("use_onehot", True))
    use_featuretools = bool(flags.get("use_featuretools", False))

    if use_featuretools:
        # Auto Feature (featuretools) branch
        params = features_cfg.get("params", {}).get("featuretools", {})
        ft_trans = params.get("trans_primitives", None)
        ft_agg = params.get("agg_primitives", None)
        max_depth = int(params.get("max_depth", 1))
        ft_t = FeaturetoolsTransformer(trans_primitives=ft_trans, agg_primitives=ft_agg, max_depth=max_depth)
        ft_t.fit(train_df[feature_cols])
        return Pipeline([("ft", ft_t)]), ft_t.get_feature_names_out()

    numeric_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in numeric_cols]

    transformers = []
    out_names: list[str] = []

    if numeric_cols:
        steps = [("impute", SimpleImputer(strategy="median"))]
        if use_scaler:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric_cols))
        out_names.extend(numeric_cols)

    if cat_cols:
        if use_onehot:
            cat_pipe = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value="__MISS__")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            )
            transformers.append(("cat", cat_pipe, cat_cols))
            # Note: actual output names are generated after fit; callers may infer them from get_feature_names_out.
        else:
            transformers.append(
                ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="__MISS__"))]), cat_cols)
            )
            out_names.extend(cat_cols)

    if not transformers:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), feature_cols))
        out_names = feature_cols[:]

    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    ct.set_output(transform="pandas")
    ct.fit(train_df[feature_cols])

    # Output feature names after fit (important for saving feature_list.json)
    try:
        out = ct.get_feature_names_out()
        if isinstance(out, np.ndarray):
            out_names = out.tolist()
        else:
            out_names = list(out)
    except Exception:
        pass

    return ct, out_names


def prepare_fold(
    config: dict[str, Any],
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    train_df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    pipeline_and_names=None,
):
    """
    Prepare fold-safe X/y for model training.

    Agent steps:
    - Split X/y using indices
    - Fit preprocessing on train fold only
    - Transform train and valid using the same fitted preprocessing
    """
    X_tr = train_df.iloc[train_idx][feature_cols]
    X_va = train_df.iloc[valid_idx][feature_cols]
    y_tr = train_df.iloc[train_idx][target_col].values
    y_va = train_df.iloc[valid_idx][target_col].values

    if pipeline_and_names is None:
        pipeline, feature_names = build_feature_pipeline(config, feature_cols, X_tr)
    else:
        pipeline, feature_names = pipeline_and_names

    X_tr_out = pipeline.transform(X_tr)
    X_va_out = pipeline.transform(X_va)
    if hasattr(X_tr_out, "values"):
        X_tr_out = X_tr_out.values
    if hasattr(X_va_out, "values"):
        X_va_out = X_va_out.values

    return X_tr_out, X_va_out, y_tr, y_va, pipeline, list(feature_names)

