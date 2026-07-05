from __future__ import annotations

import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.logger import get_logger


disable_broken_dask_autoload()
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


logger = get_logger(__name__)


def _balanced_training_frame(train: pd.DataFrame, label_col: str, random_state: int) -> pd.DataFrame:
    counts = train[label_col].astype(int).value_counts()
    if len(counts) < 2 or counts.min() == counts.max():
        return train
    majority = counts.idxmax()
    minority = counts.idxmin()
    majority_part = train[train[label_col].astype(int) == majority]
    minority_part = train[train[label_col].astype(int) == minority]
    sampled_minority = minority_part.sample(
        n=len(majority_part),
        replace=True,
        random_state=random_state,
    )
    return (
        pd.concat([majority_part, sampled_minority], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def _classifier(
    model_type: str,
    target: pd.Series,
    random_state: int,
    model_params: dict | None = None,
):
    params = model_params or {}
    if model_type == "lgbm":
        defaults = dict(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=random_state,
            class_weight="balanced",
            verbose=-1,
        )
        defaults.update(params)
        return LGBMClassifier(**defaults)
    if model_type == "xgboost":
        positives = int((target == 1).sum())
        negatives = int((target == 0).sum())
        scale_pos_weight = negatives / positives if positives else 1.0
        defaults = dict(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            scale_pos_weight=scale_pos_weight,
            n_jobs=1,
        )
        defaults.update(params)
        return XGBClassifier(**defaults)
    if model_type == "bp":
        defaults = dict(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=random_state,
        )
        defaults.update(params)
        return MLPClassifier(**defaults)
    raise ValueError(f"Unknown model_type: {model_type}")


def fit_probability_model(
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    random_state: int,
    model_type: str = "lgbm",
    model_params: dict | None = None,
) -> tuple[Pipeline, list[str]]:
    usable_features = [col for col in features if train[col].notna().any()]
    dropped = [col for col in features if col not in usable_features]
    if dropped:
        logger.warning("drop all-missing features label=%s features=%s", label_col, dropped)
    target = train[label_col].astype(int)
    if target.nunique() < 2:
        raise RuntimeError(f"{label_col} training label has only one class.")
    train_data = _balanced_training_frame(train, label_col, random_state) if model_type == "bp" else train
    target = train_data[label_col].astype(int)
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_type == "bp":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", _classifier(model_type, target, random_state, model_params)))
    model = Pipeline(
        steps
    )
    model.fit(train_data[usable_features], target)
    return model, usable_features


def positive_probability(model: Pipeline, data: pd.DataFrame, features: list[str]) -> pd.Series:
    classes = list(model.named_steps["model"].classes_)
    return pd.Series(model.predict_proba(data[features])[:, classes.index(1)], index=data.index)
