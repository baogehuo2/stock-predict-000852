from __future__ import annotations

import argparse
import json
from functools import partial

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, VarianceThreshold, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.svm import SVC

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.db import read_sql
from src.modeling.bottom_data import (
    assert_bottom_database,
    augment_bottom_features,
    load_bottom_dataset,
    load_bottom_features,
)
from src.modeling.bp_classifier import BPNeuralNetworkClassifier


disable_broken_dask_autoload()
from lightgbm import LGBMClassifier


MANUAL_META_COLUMNS = [
    "is_manual_bottom_region",
    "is_manual_top_region",
    "bottom_level_score",
    "top_level_score",
    "bottom_confidence",
    "top_confidence",
    "bottom_cycle_score",
    "top_cycle_score",
    "bottom_label_freq_score",
    "top_label_freq_score",
    "manual_bottom_region_ids",
    "manual_top_region_ids",
    "manual_bottom_cycles",
    "manual_top_cycles",
    "manual_bottom_label_freqs",
    "manual_top_label_freqs",
]


def _normalize_model_kind(model_kind: str) -> str:
    normalized = model_kind.lower().strip()
    if normalized == "xboost":
        return "xgboost"
    if normalized in {"bpnn", "bp_neural_network", "neural_network", "mlp"}:
        return "bp"
    if normalized in {"rf", "randomforest", "random-forest", "random_forest"}:
        return "random_forest"
    if normalized in {"gam", "spline_gam", "spline-gam"}:
        return "gam"
    if normalized in {"linear_discriminant_analysis", "linear-discriminant-analysis"}:
        return "lda"
    if normalized in {"quadratic_discriminant_analysis", "quadratic-discriminant-analysis"}:
        return "qda"
    if normalized in {"svc", "support_vector_machine", "support-vector-machine"}:
        return "svm"
    if normalized in {"l1-logistic", "l1_logit", "l1-logit"}:
        return "l1_logistic"
    if normalized in {"l1-logistic-var", "l1_logistic_variance", "l1-logistic-variance"}:
        return "l1_logistic_var"
    if normalized in {"l1-logistic-kbest", "l1_logistic_selectkbest", "l1-logistic-selectkbest"}:
        return "l1_logistic_kbest"
    if normalized in {"xgboost-small", "xgb_small", "xgb-small"}:
        return "xgboost_small"
    return normalized


FIXED_2025_PARAMS: dict[str, dict[str, dict]] = {
    "bp": {
        "bottom": {
            "batch_size": 128,
            "dropout": 0.10,
            "hidden_layers": (64, 32),
            "learning_rate": 0.001,
            "max_epochs": 100,
            "patience": 10,
            "weight_decay": 0.001,
        },
        "top": {
            "batch_size": 128,
            "dropout": 0.0,
            "hidden_layers": (32,),
            "learning_rate": 0.001,
            "max_epochs": 100,
            "patience": 10,
            "weight_decay": 0.001,
        },
    },
    "xgboost": {
        "bottom": {
            "colsample_bytree": 0.85,
            "learning_rate": 0.04,
            "max_depth": 2,
            "min_child_weight": 8,
            "n_estimators": 600,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "subsample": 0.8,
        },
        "top": {
            "colsample_bytree": 0.65,
            "learning_rate": 0.04,
            "max_depth": 4,
            "min_child_weight": 3,
            "n_estimators": 600,
            "reg_alpha": 0.0,
            "reg_lambda": 5.0,
            "subsample": 0.8,
        },
    },
}


def _default_params(model_kind: str, side: str) -> dict:
    params = FIXED_2025_PARAMS.get(model_kind, {}).get(side, {})
    return dict(params)


def _manual_labels(target_index: str | None = None) -> pd.DataFrame:
    cfg = get_config("bottom_model.yaml")
    table = str(cfg["outputs"]["manual_daily_table"])
    labels = read_sql(f"SELECT * FROM `{table}` ORDER BY trade_date")
    if labels.empty:
        raise RuntimeError("manual_turning_region_daily is empty. Run build_manual_turning_labels first.")
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    if target_index:
        labels = labels[labels["index_code"].astype(str) == str(target_index)].copy()
        if labels.empty:
            raise RuntimeError(f"manual_turning_region_daily has no rows for index_code={target_index}.")
    return labels


def load_manual_model_frame(target_index: str | None = None) -> pd.DataFrame:
    assert_bottom_database()
    features = load_bottom_dataset()
    if target_index:
        features = features[features["index_code"].astype(str) == str(target_index)].copy()
        if features.empty:
            raise RuntimeError(f"bottom_model_dataset_daily has no rows for index_code={target_index}.")
    manual = _manual_labels(target_index)
    data = features.merge(
        manual.drop(columns=["id", "created_at", "updated_at"], errors="ignore"),
        on=["trade_date", "index_code"],
        how="inner",
        suffixes=("", "_manual"),
    )
    if data.empty:
        raise RuntimeError("No rows matched between bottom dataset and manual labels.")
    data = augment_bottom_features(data)
    return data.sort_values("trade_date").reset_index(drop=True)


def _fit_model(
    model_kind: str,
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    random_state: int,
    side: str,
    tune: bool = False,
) -> tuple[Pipeline, list[str], dict]:
    model_kind = _normalize_model_kind(model_kind)
    usable = [feature for feature in features if train[feature].notna().any()]
    target = train[label_col].astype(int)
    if target.nunique() < 2:
        raise RuntimeError(f"{label_col} has only one class in training data.")

    selected_params = _select_hyperparams(
        model_kind,
        train,
        usable,
        label_col,
        random_state,
        side,
        tune,
    )
    pipeline = _make_pipeline(
        model_kind,
        selected_params["params"],
        random_state,
        _scale_pos_weight(target),
    )
    if model_kind in {"lda", "qda"}:
        pipeline.fit(train[usable], target)
    else:
        pipeline.fit(
            train[usable],
            target,
            model__sample_weight=_sample_weight(train, label_col, side),
        )
    return pipeline, usable, selected_params


def _scale_pos_weight(target: pd.Series) -> float:
    positive_count = int(target.sum())
    negative_count = int(len(target) - positive_count)
    return negative_count / positive_count if positive_count else 1.0


def _make_pipeline(
    model_kind: str,
    params: dict,
    random_state: int,
    scale_pos_weight: float,
) -> Pipeline:
    if model_kind in {"logistic", "l1_logistic", "l1_logistic_var", "l1_logistic_kbest"}:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if model_kind in {"l1_logistic_var", "l1_logistic_kbest"}:
            steps.append(("variance", VarianceThreshold(threshold=float(params.get("variance_threshold", 1e-10)))))
        if model_kind == "l1_logistic_kbest":
            score_func = partial(mutual_info_classif, random_state=random_state)
            steps.append(("select", SelectKBest(score_func=score_func, k=int(params.get("k", 50)))))
        penalty = str(params.get("penalty", "l2"))
        solver = str(params.get("solver", "lbfgs"))
        if model_kind.startswith("l1_"):
            penalty = "l1"
            solver = str(params.get("solver", "liblinear"))
        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=int(params.get("max_iter", 4000)),
                        C=float(params.get("C", 1.0)),
                        penalty=penalty,
                        solver=solver,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        pipeline = Pipeline(steps)
    elif model_kind == "gam":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "spline",
                    SplineTransformer(
                        n_knots=int(params.get("n_knots", 4)),
                        degree=int(params.get("degree", 3)),
                        knots="quantile",
                        include_bias=False,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=int(params.get("max_iter", 4000)),
                        C=float(params.get("C", 0.1)),
                        solver="lbfgs",
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_kind == "lda":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LinearDiscriminantAnalysis(
                        solver=str(params.get("solver", "lsqr")),
                        shrinkage=params.get("shrinkage", "auto"),
                    ),
                ),
            ]
        )
    elif model_kind == "qda":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    QuadraticDiscriminantAnalysis(
                        reg_param=float(params.get("reg_param", 0.5)),
                    ),
                ),
            ]
        )
    elif model_kind == "svm":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        C=float(params.get("C", 1.0)),
                        kernel=str(params.get("kernel", "rbf")),
                        gamma=params.get("gamma", "scale"),
                        degree=int(params.get("degree", 3)),
                        class_weight="balanced",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_kind == "lightgbm":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=int(params.get("n_estimators", 250)),
                        learning_rate=float(params.get("learning_rate", 0.03)),
                        num_leaves=int(params.get("num_leaves", 15)),
                        max_depth=int(params.get("max_depth", 5)),
                        min_child_samples=int(params.get("min_child_samples", 20)),
                        subsample=float(params.get("subsample", 0.85)),
                        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
                        reg_lambda=float(params.get("reg_lambda", 1.0)),
                        class_weight="balanced",
                        random_state=random_state,
                        verbose=-1,
                    ),
                ),
            ]
        )
    elif model_kind in {"xgboost", "xgboost_small"}:
        from xgboost import XGBClassifier

        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=int(params.get("n_estimators", 250)),
                        learning_rate=float(params.get("learning_rate", 0.03)),
                        max_depth=int(params.get("max_depth", 3)),
                        min_child_weight=float(params.get("min_child_weight", 5)),
                        subsample=float(params.get("subsample", 0.85)),
                        colsample_bytree=float(params.get("colsample_bytree", 0.85)),
                        reg_lambda=float(params.get("reg_lambda", 2.0)),
                        reg_alpha=float(params.get("reg_alpha", 0.1)),
                        objective="binary:logistic",
                        eval_metric="logloss",
                        tree_method="hist",
                        scale_pos_weight=scale_pos_weight,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    elif model_kind == "bp":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    BPNeuralNetworkClassifier(
                        hidden_layers=tuple(params.get("hidden_layers", (64, 32))),
                        dropout=float(params.get("dropout", 0.10)),
                        learning_rate=float(params.get("learning_rate", 0.001)),
                        weight_decay=float(params.get("weight_decay", 0.0001)),
                        max_epochs=int(params.get("max_epochs", 100)),
                        batch_size=int(params.get("batch_size", 128)),
                        patience=int(params.get("patience", 10)),
                        positive_weight=scale_pos_weight,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    elif model_kind == "random_forest":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(params.get("n_estimators", 500)),
                        max_depth=None
                        if params.get("max_depth", None) in {None, "none", "None"}
                        else int(params.get("max_depth")),
                        min_samples_leaf=int(params.get("min_samples_leaf", 5)),
                        min_samples_split=int(params.get("min_samples_split", 10)),
                        max_features=params.get("max_features", "sqrt"),
                        bootstrap=bool(params.get("bootstrap", True)),
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    return pipeline


def _candidate_params(model_kind: str) -> list[dict]:
    if model_kind == "logistic":
        return list(
            ParameterGrid(
                {
                    "C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
                    "max_iter": [4000],
                }
            )
        )
    if model_kind == "l1_logistic":
        return list(
            ParameterGrid(
                {
                    "C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    "penalty": ["l1"],
                    "solver": ["liblinear"],
                    "max_iter": [4000],
                }
            )
        )
    if model_kind == "l1_logistic_var":
        return list(
            ParameterGrid(
                {
                    "C": [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                    "penalty": ["l1"],
                    "solver": ["liblinear"],
                    "variance_threshold": [1e-10, 1e-6],
                    "max_iter": [4000],
                }
            )
        )
    if model_kind == "l1_logistic_kbest":
        return list(
            ParameterGrid(
                {
                    "C": [0.003, 0.01, 0.03, 0.1, 0.3],
                    "penalty": ["l1"],
                    "solver": ["liblinear"],
                    "variance_threshold": [1e-10],
                    "k": [30, 50, 80],
                    "max_iter": [4000],
                }
            )
        )
    if model_kind == "lightgbm":
        return list(
            ParameterGrid(
                {
                    "n_estimators": [300, 600],
                    "learning_rate": [0.02, 0.04],
                    "num_leaves": [7, 15, 31],
                    "max_depth": [3, 5],
                    "min_child_samples": [10, 30],
                    "subsample": [0.80],
                    "colsample_bytree": [0.65, 0.85],
                    "reg_lambda": [1.0, 5.0],
                }
            )
        )
    if model_kind == "gam":
        return list(
            ParameterGrid(
                {
                    "C": [0.03, 0.1, 0.3],
                    "n_knots": [3, 4],
                    "degree": [2, 3],
                    "max_iter": [4000],
                }
            )
        )
    if model_kind == "lda":
        return list(
            ParameterGrid(
                {
                    "solver": ["lsqr"],
                    "shrinkage": ["auto", 0.1, 0.3, 0.5, 0.7],
                }
            )
        )
    if model_kind == "qda":
        return list(
            ParameterGrid(
                {
                    "reg_param": [0.1, 0.3, 0.5, 0.7, 0.9],
                }
            )
        )
    if model_kind == "svm":
        return list(
            ParameterGrid(
                {
                    "C": [0.3, 1.0],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale"],
                }
            )
        )
    if model_kind == "xgboost":
        return list(
            ParameterGrid(
                {
                    "n_estimators": [300, 600],
                    "learning_rate": [0.02, 0.04],
                    "max_depth": [2, 3, 4],
                    "min_child_weight": [3, 8],
                    "subsample": [0.80],
                    "colsample_bytree": [0.65, 0.85],
                    "reg_lambda": [1.0, 5.0],
                    "reg_alpha": [0.0, 0.1],
                }
            )
        )
    if model_kind == "xgboost_small":
        return list(
            ParameterGrid(
                {
                    "n_estimators": [300, 600],
                    "learning_rate": [0.03, 0.05],
                    "max_depth": [2, 3],
                    "min_child_weight": [8, 15],
                    "subsample": [0.70],
                    "colsample_bytree": [0.50, 0.70],
                    "reg_lambda": [5.0, 10.0],
                    "reg_alpha": [0.0, 0.5],
                }
            )
        )
    if model_kind == "bp":
        return list(
            ParameterGrid(
                {
                    "hidden_layers": [(32,), (64, 32)],
                    "dropout": [0.0, 0.10],
                    "learning_rate": [0.001],
                    "weight_decay": [0.0001, 0.001],
                    "max_epochs": [100],
                    "batch_size": [128],
                    "patience": [10],
                }
            )
        )
    if model_kind == "random_forest":
        return list(
            ParameterGrid(
                {
                    "n_estimators": [300],
                    "max_depth": [4, 6, None],
                    "min_samples_leaf": [3, 8],
                    "min_samples_split": [10],
                    "max_features": ["sqrt", 0.5],
                    "bootstrap": [True],
                }
            )
        )
    raise ValueError(f"Unsupported model kind: {model_kind}")


def _sample_weight(data: pd.DataFrame, label_col: str, side: str) -> np.ndarray:
    target = data[label_col].astype(int)
    weights = np.ones(len(data), dtype=float)
    region_col = f"is_manual_{side}_region"
    cycle_col = f"{side}_cycle_score"
    freq_col = f"{side}_label_freq_score"
    if region_col not in data or cycle_col not in data or freq_col not in data:
        return weights

    region_mask = pd.to_numeric(data[region_col], errors="coerce").fillna(0).astype(int) == 1
    cycle_score = pd.to_numeric(data[cycle_col], errors="coerce").fillna(0).clip(lower=1)
    freq_score = pd.to_numeric(data[freq_col], errors="coerce").fillna(0).clip(lower=1)
    metadata_weight = (1.0 + 0.20 * (cycle_score - 1.0)) * (1.0 + 0.15 * (freq_score - 1.0))
    weights[region_mask.to_numpy()] *= metadata_weight[region_mask].to_numpy()
    positive_mask = target == 1
    weights[positive_mask.to_numpy()] *= 1.10
    return weights


def _inner_validation_split(train: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    train = train.sort_values("trade_date").copy()
    years = sorted(int(year) for year in train["trade_date"].dt.year.unique())
    if len(years) < 3:
        return None
    for window in [1, 2, 3]:
        valid_years = years[-window:]
        valid_start_year = valid_years[0]
        inner_train = train[train["trade_date"].dt.year < valid_start_year].copy()
        valid = train[train["trade_date"].dt.year.isin(valid_years)].copy()
        if inner_train.empty or valid.empty:
            continue
        if inner_train[label_col].astype(int).nunique() < 2:
            continue
        if valid[label_col].astype(int).nunique() < 2:
            continue
        return inner_train, valid
    return None


def _select_hyperparams(
    model_kind: str,
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    random_state: int,
    side: str,
    tune: bool,
) -> dict:
    default_params = _default_params(model_kind, side)
    if not tune:
        return {
            "params": default_params,
            "tuning_enabled": False,
            "candidate_count": 1,
            "param_source": "fixed_2025_walk_forward_fold" if default_params else "library_default",
            "inner_valid_start": "",
            "inner_valid_end": "",
            "inner_valid_average_precision": None,
            "inner_valid_roc_auc": None,
        }

    split = _inner_validation_split(train, label_col)
    if split is None:
        return {
            "params": default_params,
            "tuning_enabled": True,
            "candidate_count": 0,
            "param_source": "fixed_2025_walk_forward_fold" if default_params else "library_default",
            "inner_valid_start": "",
            "inner_valid_end": "",
            "inner_valid_average_precision": None,
            "inner_valid_roc_auc": None,
        }

    inner_train, valid = split
    y_train = inner_train[label_col].astype(int)
    y_valid = valid[label_col].astype(int)
    candidates = _candidate_params(model_kind)
    best_params: dict | None = None
    best_ap = -np.inf
    best_auc = -np.inf

    for params in candidates:
        try:
            pipeline = _make_pipeline(model_kind, params, random_state, _scale_pos_weight(y_train))
            if model_kind in {"lda", "qda"}:
                pipeline.fit(inner_train[features], y_train)
            else:
                pipeline.fit(
                    inner_train[features],
                    y_train,
                    model__sample_weight=_sample_weight(inner_train, label_col, side),
                )
            proba = _positive_probability(pipeline, valid, features)
            ap = _safe_ap(y_valid, proba)
            auc = _safe_auc(y_valid, proba)
        except Exception:
            continue
        ap_score = float(ap) if ap is not None and np.isfinite(ap) else -np.inf
        auc_score = float(auc) if auc is not None and np.isfinite(auc) else -np.inf
        if (ap_score > best_ap) or (np.isclose(ap_score, best_ap) and auc_score > best_auc):
            best_params = params
            best_ap = ap_score
            best_auc = auc_score

    return {
        "params": best_params or default_params,
        "tuning_enabled": True,
        "candidate_count": len(candidates),
        "param_source": "inner_validation" if best_params else (
            "fixed_2025_walk_forward_fold" if default_params else "library_default"
        ),
        "inner_valid_start": valid["trade_date"].min().strftime("%Y-%m-%d"),
        "inner_valid_end": valid["trade_date"].max().strftime("%Y-%m-%d"),
        "inner_valid_average_precision": None if best_ap == -np.inf else float(best_ap),
        "inner_valid_roc_auc": None if best_auc == -np.inf else float(best_auc),
    }


def _positive_probability(model: Pipeline, data: pd.DataFrame, features: list[str]) -> np.ndarray:
    classes = list(model.named_steps["model"].classes_)
    return model.predict_proba(data[features])[:, classes.index(1)]


def _kind_path(path_value: str, model_kind: str, output_tag: str | None = None) -> str:
    path = project_path(path_value)
    tag = f"_{output_tag}" if output_tag else ""
    return str(path.with_name(f"{path.stem}{tag}_{model_kind}{path.suffix}"))


def _wf_kind_path(path_value: str, model_kind: str, output_tag: str | None = None) -> str:
    path = project_path(path_value)
    tag = f"_{output_tag}" if output_tag else ""
    return str(path.with_name(f"{path.stem}{tag}_{model_kind}_wf{path.suffix}"))


def _split_name(data: pd.DataFrame) -> pd.Series:
    cfg = get_config("bottom_model.yaml")
    split_cfg = cfg["manual_models"]
    split = pd.Series("ignored", index=data.index, dtype="object")
    split.loc[
        (data["trade_date"] >= pd.Timestamp(split_cfg["train_start"]))
        & (data["trade_date"] <= pd.Timestamp(split_cfg["train_end"]))
    ] = "train"
    split.loc[
        (data["trade_date"] >= pd.Timestamp(split_cfg["valid_start"]))
        & (data["trade_date"] <= pd.Timestamp(split_cfg["valid_end"]))
    ] = "valid"
    split.loc[data["trade_date"] >= pd.Timestamp(split_cfg["oot_start"])] = "oot"
    return split


def _safe_auc(y_true: pd.Series, proba: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(roc_auc_score(y_true, proba))


def _safe_ap(y_true: pd.Series, proba: np.ndarray) -> float | None:
    if y_true.nunique() < 2:
        return None
    return float(average_precision_score(y_true, proba))


def _signal_classification_metrics(part: pd.DataFrame, signal: pd.DataFrame, label_col: str) -> dict:
    positives = int(part[label_col].sum()) if label_col in part else 0
    true_positives = int(signal[label_col].sum()) if len(signal) and label_col in signal else 0
    precision = float(true_positives / len(signal)) if len(signal) else None
    recall = float(true_positives / positives) if positives else None
    f2 = None
    if precision is not None and recall is not None and (4 * precision + recall) > 0:
        f2 = float((5 * precision * recall) / (4 * precision + recall))
    return {
        "true_positive_count": true_positives,
        "recall": recall,
        "f2_score": f2,
    }


def _signal_outcome_metrics(signal: pd.DataFrame) -> dict:
    return {
        "avg_future_ret_15d": float(signal["future_ret_15d"].mean()) if len(signal) else None,
        "avg_future_mfe_15d": float(signal["future_mfe_15d"].mean()) if len(signal) else None,
        "avg_future_mae_15d": float(signal["future_mae_15d"].mean()) if len(signal) else None,
        "quality_bottom_rate": float(signal["quality_bottom_label"].mean())
        if len(signal) and "quality_bottom_label" in signal
        else None,
        "continuation_risk_rate": float(signal["continuation_risk_label"].mean())
        if len(signal) and "continuation_risk_label" in signal
        else None,
    }


def _threshold_rows(
    part: pd.DataFrame,
    label_col: str,
    proba_col: str,
    thresholds: list[float],
    metric_prefix: str,
) -> list[dict]:
    rows = []
    for threshold in thresholds:
        signal = part[part[proba_col] >= threshold]
        rows.append(
            {
                "selection": "threshold",
                "threshold": threshold,
                "top_n": None,
                "signal_count": int(len(signal)),
                "coverage": float(len(signal) / len(part)) if len(part) else 0.0,
                f"{metric_prefix}_precision": float(signal[label_col].mean()) if len(signal) else None,
                f"{metric_prefix}_base_rate": float(part[label_col].mean()) if len(part) else None,
                **_signal_classification_metrics(part, signal, label_col),
                **_signal_outcome_metrics(signal),
            }
        )
    return rows


def _topn_rows(
    part: pd.DataFrame,
    label_col: str,
    proba_col: str,
    top_ns: list[int],
    metric_prefix: str,
) -> list[dict]:
    rows = []
    ranked = part.sort_values(proba_col, ascending=False)
    for top_n in top_ns:
        actual_n = min(int(top_n), len(ranked))
        signal = ranked.head(actual_n)
        rows.append(
            {
                "selection": "top_n",
                "threshold": None,
                "top_n": top_n,
                "signal_count": int(len(signal)),
                "coverage": float(len(signal) / len(part)) if len(part) else 0.0,
                f"{metric_prefix}_precision": float(signal[label_col].mean()) if len(signal) else None,
                f"{metric_prefix}_base_rate": float(part[label_col].mean()) if len(part) else None,
                **_signal_classification_metrics(part, signal, label_col),
                **_signal_outcome_metrics(signal),
            }
        )
    return rows


def _prediction_columns(data: pd.DataFrame, split_column: str | None = None) -> list[str]:
    columns = [
        "trade_date",
        "index_code",
        "is_manual_buy_window",
        "is_manual_sell_window",
        "manual_state",
        "label_mode",
        "future_ret_15d",
        "future_mfe_15d",
        "future_mae_15d",
        "quality_bottom_label",
        "continuation_risk_label",
    ]
    if split_column is not None:
        columns.insert(2, split_column)
    columns.extend([column for column in MANUAL_META_COLUMNS if column in data.columns and column not in columns])
    return columns


def _tuning_metric_fields(tuning_info: dict) -> dict:
    return {
        "tuning_enabled": bool(tuning_info.get("tuning_enabled", False)),
        "candidate_count": int(tuning_info.get("candidate_count", 0) or 0),
        "param_source": tuning_info.get("param_source", ""),
        "best_params": json.dumps(tuning_info.get("params", {}), ensure_ascii=False, sort_keys=True),
        "inner_valid_start": tuning_info.get("inner_valid_start", ""),
        "inner_valid_end": tuning_info.get("inner_valid_end", ""),
        "inner_valid_average_precision": tuning_info.get("inner_valid_average_precision"),
        "inner_valid_roc_auc": tuning_info.get("inner_valid_roc_auc"),
    }


def train_manual_turning_models(
    model_kind: str = "logistic",
    tune: bool = False,
    target_index: str | None = None,
    output_tag: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_kind = _normalize_model_kind(model_kind)
    cfg = get_config("bottom_model.yaml")
    manual_cfg = cfg["manual_models"]
    data = load_manual_model_frame(target_index)
    data["manual_split"] = _split_name(data)
    features = load_bottom_features(data)
    random_state = int(cfg["model"]["random_state"])
    train = data[data["manual_split"] == "train"].copy()
    if train.empty:
        raise RuntimeError("Manual model training split is empty.")

    specs = [
        ("bottom", "is_manual_buy_window", "manual_weak_bottom_proba", str(manual_cfg["bottom_model_file"])),
        ("top", "is_manual_sell_window", "manual_weak_top_proba", str(manual_cfg["top_model_file"])),
    ]
    prediction = data[_prediction_columns(data, "manual_split")].copy()
    metrics_rows = []

    for side, label_col, proba_col, model_file in specs:
        model, usable, tuning_info = _fit_model(
            model_kind,
            train,
            features,
            label_col,
            random_state,
            side,
            tune=tune,
        )
        prediction[proba_col] = _positive_probability(model, data, usable)
        bundle = {
            "classifier": model,
            "features": usable,
            "model_kind": model_kind,
            "side": side,
            "label_col": label_col,
            "proba_col": proba_col,
            "feature_version": str(cfg["model"]["feature_version"]),
            "model_version": f"manual-{side}-zone-v0.1",
            "label_mode": str(manual_cfg["label_mode"]),
            "trained_from": train["trade_date"].min().strftime("%Y-%m-%d"),
            "trained_through": train["trade_date"].max().strftime("%Y-%m-%d"),
            "train_rows": int(len(train)),
            "sample_weight_metadata": "cycle_label_freq",
            "tuning": tuning_info,
        }
        path = project_path(_kind_path(model_file, model_kind, output_tag))
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, path)

        for split_name, part in prediction.groupby("manual_split", sort=True):
            if split_name == "ignored":
                continue
            y_true = part[label_col].astype(int)
            proba = part[proba_col].to_numpy()
            base = {
                "side": side,
                "model_kind": model_kind,
                "split": split_name,
                "rows": int(len(part)),
                "positive_count": int(y_true.sum()),
                "positive_rate": float(y_true.mean()),
                "roc_auc": _safe_auc(y_true, proba),
                "average_precision": _safe_ap(y_true, proba),
                **_tuning_metric_fields(tuning_info),
            }
            metrics_rows.append(
                {**base, "selection": None, "threshold": None, "top_n": None, "signal_count": None, "coverage": None}
            )
            for threshold_row in _threshold_rows(
                part,
                label_col,
                proba_col,
                [float(value) for value in manual_cfg["probability_thresholds"]],
                side,
            ):
                metrics_rows.append({**base, **threshold_row})
            for topn_row in _topn_rows(part, label_col, proba_col, [5, 10, 20, 40], side):
                metrics_rows.append({**base, **topn_row})

    prediction["weak_combined_bottom_signal"] = (
        (prediction["manual_weak_bottom_proba"] >= 0.6)
        & (prediction["manual_weak_top_proba"] < 0.5)
    ).astype(int)
    prediction["weak_combined_top_signal"] = (
        (prediction["manual_weak_top_proba"] >= 0.6)
        & (prediction["manual_weak_bottom_proba"] < 0.5)
    ).astype(int)

    combined_rows = []
    for split_name, part in prediction.groupby("manual_split", sort=True):
        if split_name == "ignored":
            continue
        for signal_col in ["weak_combined_bottom_signal", "weak_combined_top_signal"]:
            signal = part[part[signal_col] == 1]
            combined_rows.append(
                {
                    "signal": signal_col,
                    "label_mode": str(manual_cfg["label_mode"]),
                    "split": split_name,
                    "rows": int(len(part)),
                    "signal_count": int(len(signal)),
                    "coverage": float(len(signal) / len(part)) if len(part) else 0.0,
                    "weak_manual_buy_window_rate": float(signal["is_manual_buy_window"].mean())
                    if len(signal)
                    else None,
                    "weak_manual_sell_window_rate": float(signal["is_manual_sell_window"].mean())
                    if len(signal)
                    else None,
                    "avg_future_ret_15d": float(signal["future_ret_15d"].mean()) if len(signal) else None,
                    "quality_bottom_rate": float(signal["quality_bottom_label"].mean()) if len(signal) else None,
                    "continuation_risk_rate": float(signal["continuation_risk_label"].mean()) if len(signal) else None,
                }
            )

    metrics = pd.DataFrame(metrics_rows)
    combined = pd.DataFrame(combined_rows)
    for frame, key in [
        (metrics, "manual_model_metrics"),
        (prediction, "manual_model_predictions"),
        (combined, "manual_combined_signals"),
    ]:
        path = project_path(_kind_path(str(cfg["outputs"][key]), model_kind, output_tag))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return metrics, prediction, combined


def _combined_summary(prediction: pd.DataFrame, split_col: str) -> pd.DataFrame:
    manual_cfg = get_config("bottom_model.yaml")["manual_models"]
    rows = []
    for split_name, part in prediction.groupby(split_col, sort=True):
        for signal_col in ["weak_combined_bottom_signal", "weak_combined_top_signal"]:
            signal = part[part[signal_col] == 1]
            rows.append(
                {
                    "signal": signal_col,
                    "label_mode": str(manual_cfg["label_mode"]),
                    "split": split_name,
                    "rows": int(len(part)),
                    "signal_count": int(len(signal)),
                    "coverage": float(len(signal) / len(part)) if len(part) else 0.0,
                    "weak_manual_buy_window_rate": float(signal["is_manual_buy_window"].mean())
                    if len(signal)
                    else None,
                    "weak_manual_sell_window_rate": float(signal["is_manual_sell_window"].mean())
                    if len(signal)
                    else None,
                    "avg_future_ret_15d": float(signal["future_ret_15d"].mean()) if len(signal) else None,
                    "quality_bottom_rate": float(signal["quality_bottom_label"].mean()) if len(signal) else None,
                    "continuation_risk_rate": float(signal["continuation_risk_label"].mean()) if len(signal) else None,
                }
            )
    return pd.DataFrame(rows)


def _fold_years(data: pd.DataFrame, start_year: int) -> list[int]:
    years = sorted(int(year) for year in data["trade_date"].dt.year.unique())
    return [year for year in years if year >= start_year]


def walk_forward_manual_turning_models(
    model_kind: str = "logistic",
    start_year: int = 2022,
    tune: bool = False,
    target_index: str | None = None,
    output_tag: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_kind = _normalize_model_kind(model_kind)
    cfg = get_config("bottom_model.yaml")
    manual_cfg = cfg["manual_models"]
    data = load_manual_model_frame(target_index)
    features = load_bottom_features(data)
    random_state = int(cfg["model"]["random_state"])
    train_start = pd.Timestamp(str(manual_cfg["train_start"]))
    specs = [
        ("bottom", "is_manual_buy_window", "manual_weak_bottom_proba"),
        ("top", "is_manual_sell_window", "manual_weak_top_proba"),
    ]
    prediction_parts = []
    metric_rows = []

    for test_year in _fold_years(data, start_year):
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        train = data[(data["trade_date"] >= train_start) & (data["trade_date"] < test_start)].copy()
        test = data[(data["trade_date"] >= test_start) & (data["trade_date"] <= test_end)].copy()
        if train.empty or test.empty:
            continue
        fold = test[_prediction_columns(test)].copy()
        fold["test_year"] = test_year
        fold["train_start"] = train["trade_date"].min().strftime("%Y-%m-%d")
        fold["train_end"] = train["trade_date"].max().strftime("%Y-%m-%d")
        fold["train_rows"] = int(len(train))

        for side, label_col, proba_col in specs:
            try:
                model, usable, tuning_info = _fit_model(
                    model_kind,
                    train,
                    features,
                    label_col,
                    random_state,
                    side,
                    tune=tune,
                )
            except RuntimeError:
                fold[proba_col] = np.nan
                continue
            fold[proba_col] = _positive_probability(model, test, usable)
            y_true = test[label_col].astype(int)
            proba = fold[proba_col].to_numpy()
            base = {
                "side": side,
                "model_kind": model_kind,
                "test_year": test_year,
                "train_start": train["trade_date"].min().strftime("%Y-%m-%d"),
                "train_end": train["trade_date"].max().strftime("%Y-%m-%d"),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "positive_count": int(y_true.sum()),
                "positive_rate": float(y_true.mean()),
                "roc_auc": _safe_auc(y_true, proba),
                "average_precision": _safe_ap(y_true, proba),
                **_tuning_metric_fields(tuning_info),
            }
            metric_rows.append(
                {**base, "selection": None, "threshold": None, "top_n": None, "signal_count": None, "coverage": None}
            )
            for threshold_row in _threshold_rows(
                pd.concat([test.reset_index(drop=True), fold[[proba_col]].reset_index(drop=True)], axis=1),
                label_col,
                proba_col,
                [float(value) for value in manual_cfg["probability_thresholds"]],
                side,
            ):
                metric_rows.append({**base, **threshold_row})
            eval_part = pd.concat([test.reset_index(drop=True), fold[[proba_col]].reset_index(drop=True)], axis=1)
            for topn_row in _topn_rows(eval_part, label_col, proba_col, [5, 10, 20, 40], side):
                metric_rows.append({**base, **topn_row})

        fold["weak_combined_bottom_signal"] = (
            (fold["manual_weak_bottom_proba"] >= 0.6)
            & (fold["manual_weak_top_proba"] < 0.5)
        ).astype(int)
        fold["weak_combined_top_signal"] = (
            (fold["manual_weak_top_proba"] >= 0.6)
            & (fold["manual_weak_bottom_proba"] < 0.5)
        ).astype(int)
        prediction_parts.append(fold)

    if not prediction_parts:
        raise RuntimeError("No manual walk-forward folds were produced.")
    prediction = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    combined = _combined_summary(prediction.rename(columns={"test_year": "split"}), "split")
    for frame, key in [
        (metrics, "manual_walk_forward_metrics"),
        (prediction, "manual_walk_forward_predictions"),
        (combined, "manual_walk_forward_combined"),
    ]:
        path = project_path(_wf_kind_path(str(cfg["outputs"][key]), model_kind, output_tag))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return metrics, prediction, combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Train independent manual bottom/top zone models.")
    parser.add_argument(
        "--model-kind",
        choices=[
            "logistic",
            "l1_logistic",
            "l1-logistic",
            "l1_logistic_var",
            "l1-logistic-var",
            "l1_logistic_kbest",
            "l1-logistic-kbest",
            "gam",
            "spline_gam",
            "spline-gam",
            "lda",
            "linear_discriminant_analysis",
            "linear-discriminant-analysis",
            "qda",
            "quadratic_discriminant_analysis",
            "quadratic-discriminant-analysis",
            "svm",
            "svc",
            "support_vector_machine",
            "support-vector-machine",
            "lightgbm",
            "xgboost",
            "xgboost_small",
            "xgboost-small",
            "xboost",
            "bp",
            "bpnn",
            "mlp",
            "neural_network",
            "random_forest",
            "randomforest",
            "random-forest",
            "rf",
        ],
        default="logistic",
    )
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--target-index")
    parser.add_argument("--output-tag")
    args = parser.parse_args()
    model_kind = _normalize_model_kind(args.model_kind)
    if args.walk_forward:
        metrics, _, combined = walk_forward_manual_turning_models(
            model_kind,
            args.start_year,
            tune=args.tune,
            target_index=args.target_index,
            output_tag=args.output_tag,
        )
    else:
        metrics, _, combined = train_manual_turning_models(
            model_kind,
            tune=args.tune,
            target_index=args.target_index,
            output_tag=args.output_tag,
        )
    print(metrics.to_string(index=False))
    print("\n[combined]")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
