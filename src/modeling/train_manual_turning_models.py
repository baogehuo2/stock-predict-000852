from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    return normalized


def _manual_labels() -> pd.DataFrame:
    cfg = get_config("bottom_model.yaml")
    table = str(cfg["outputs"]["manual_daily_table"])
    labels = read_sql(f"SELECT * FROM `{table}` ORDER BY trade_date")
    if labels.empty:
        raise RuntimeError("manual_turning_region_daily is empty. Run build_manual_turning_labels first.")
    labels["trade_date"] = pd.to_datetime(labels["trade_date"])
    return labels


def load_manual_model_frame() -> pd.DataFrame:
    assert_bottom_database()
    features = load_bottom_dataset()
    manual = _manual_labels()
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
    if model_kind == "logistic":
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=int(params.get("max_iter", 4000)),
                        C=float(params.get("C", 1.0)),
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
    elif model_kind == "xgboost":
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
    default_params = {}
    if not tune:
        return {
            "params": default_params,
            "tuning_enabled": False,
            "candidate_count": 1,
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
        "inner_valid_start": valid["trade_date"].min().strftime("%Y-%m-%d"),
        "inner_valid_end": valid["trade_date"].max().strftime("%Y-%m-%d"),
        "inner_valid_average_precision": None if best_ap == -np.inf else float(best_ap),
        "inner_valid_roc_auc": None if best_auc == -np.inf else float(best_auc),
    }


def _positive_probability(model: Pipeline, data: pd.DataFrame, features: list[str]) -> np.ndarray:
    classes = list(model.named_steps["model"].classes_)
    return model.predict_proba(data[features])[:, classes.index(1)]


def _kind_path(path_value: str, model_kind: str) -> str:
    path = project_path(path_value)
    return str(path.with_name(f"{path.stem}_{model_kind}{path.suffix}"))


def _wf_kind_path(path_value: str, model_kind: str) -> str:
    path = project_path(path_value)
    return str(path.with_name(f"{path.stem}_{model_kind}_wf{path.suffix}"))


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
                "threshold": threshold,
                "signal_count": int(len(signal)),
                "coverage": float(len(signal) / len(part)) if len(part) else 0.0,
                f"{metric_prefix}_precision": float(signal[label_col].mean()) if len(signal) else None,
                f"{metric_prefix}_base_rate": float(part[label_col].mean()) if len(part) else None,
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
        "best_params": json.dumps(tuning_info.get("params", {}), ensure_ascii=False, sort_keys=True),
        "inner_valid_start": tuning_info.get("inner_valid_start", ""),
        "inner_valid_end": tuning_info.get("inner_valid_end", ""),
        "inner_valid_average_precision": tuning_info.get("inner_valid_average_precision"),
        "inner_valid_roc_auc": tuning_info.get("inner_valid_roc_auc"),
    }


def train_manual_turning_models(
    model_kind: str = "logistic",
    tune: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_kind = _normalize_model_kind(model_kind)
    cfg = get_config("bottom_model.yaml")
    manual_cfg = cfg["manual_models"]
    data = load_manual_model_frame()
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
        path = project_path(_kind_path(model_file, model_kind))
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
            metrics_rows.append({**base, "threshold": None, "signal_count": None, "coverage": None})
            for threshold_row in _threshold_rows(
                part,
                label_col,
                proba_col,
                [float(value) for value in manual_cfg["probability_thresholds"]],
                side,
            ):
                metrics_rows.append({**base, **threshold_row})

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
        path = project_path(_kind_path(str(cfg["outputs"][key]), model_kind))
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_kind = _normalize_model_kind(model_kind)
    cfg = get_config("bottom_model.yaml")
    manual_cfg = cfg["manual_models"]
    data = load_manual_model_frame()
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
            metric_rows.append({**base, "threshold": None, "signal_count": None, "coverage": None})
            for threshold_row in _threshold_rows(
                pd.concat([test.reset_index(drop=True), fold[[proba_col]].reset_index(drop=True)], axis=1),
                label_col,
                proba_col,
                [float(value) for value in manual_cfg["probability_thresholds"]],
                side,
            ):
                metric_rows.append({**base, **threshold_row})

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
        path = project_path(_wf_kind_path(str(cfg["outputs"][key]), model_kind))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return metrics, prediction, combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Train independent manual bottom/top zone models.")
    parser.add_argument(
        "--model-kind",
        choices=["logistic", "lightgbm", "xgboost", "xboost", "bp", "bpnn", "mlp", "neural_network"],
        default="logistic",
    )
    parser.add_argument("--walk-forward", action="store_true")
    parser.add_argument("--start-year", type=int, default=2022)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    model_kind = _normalize_model_kind(args.model_kind)
    if args.walk_forward:
        metrics, _, combined = walk_forward_manual_turning_models(model_kind, args.start_year, tune=args.tune)
    else:
        metrics, _, combined = train_manual_turning_models(model_kind, tune=args.tune)
    print(metrics.to_string(index=False))
    print("\n[combined]")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
