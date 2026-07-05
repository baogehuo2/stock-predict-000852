from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.logger import get_logger
from src.modeling.bp_classifier import BPNeuralNetworkClassifier
from src.modeling.bottom_weekly_data import load_bottom_weekly_dataset, load_bottom_weekly_features


disable_broken_dask_autoload()
from lightgbm import LGBMClassifier


logger = get_logger(__name__)


SIDE_CONFIG = {
    "bottom": {
        "label": "manual_weak_bottom_label",
        "candidate": "is_bottom_candidate",
        "positive_return_metric": "actual_up_rate",
    },
    "top": {
        "label": "manual_weak_top_label",
        "candidate": "is_top_candidate",
        "positive_return_metric": "actual_down_rate",
    },
}


def _parse_floats(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def _imbalance_config() -> dict:
    cfg = get_config("bottom_weekly_model.yaml")
    imbalance = dict(cfg.get("model", {}).get("imbalance", {}))
    imbalance.setdefault("enabled", True)
    imbalance.setdefault("max_scale_pos_weight", 80.0)
    imbalance.setdefault("negative_sample_ratio", 3.0)
    imbalance.setdefault("lightgbm_is_unbalance", False)
    imbalance.setdefault("lightgbm_balanced_bagging", True)
    imbalance.setdefault("bp_use_focal_loss", True)
    imbalance.setdefault("bp_focal_gamma", 2.0)
    imbalance.setdefault("bp_focal_alpha", 0.75)
    return imbalance


def _positive_weight(target: pd.Series, imbalance: dict) -> float:
    if not bool(imbalance.get("enabled", True)):
        return 1.0
    positive_count = int((target == 1).sum())
    negative_count = int((target == 0).sum())
    if positive_count <= 0:
        return 1.0
    raw_weight = negative_count / positive_count
    max_weight = float(imbalance.get("max_scale_pos_weight", 80.0))
    return float(np.clip(raw_weight, 1.0, max_weight))


def _negative_bagging_fraction(target: pd.Series, imbalance: dict, params: dict) -> float:
    positive_count = int((target == 1).sum())
    negative_count = int((target == 0).sum())
    if positive_count <= 0 or negative_count <= 0:
        return 1.0
    ratio = float(params.get("negative_sample_ratio", imbalance.get("negative_sample_ratio", 3.0)))
    return float(np.clip((positive_count * ratio) / negative_count, 0.05, 1.0))


def _balanced_resample(
    train: pd.DataFrame,
    label_col: str,
    negative_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    target = train[label_col].astype(int)
    positive = train[target == 1]
    negative = train[target == 0]
    if positive.empty or negative.empty:
        return train
    negative_rows = min(len(negative), max(1, int(round(len(positive) * negative_ratio))))
    sampled_negative = negative.sample(
        n=negative_rows,
        replace=False,
        random_state=random_state,
    )
    return (
        pd.concat([positive, sampled_negative], axis=0)
        .sample(frac=1.0, random_state=random_state)
        .sort_values("week_pos")
        .reset_index(drop=True)
    )


def _fit_model(
    model_name: str,
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    random_state: int,
    params: dict | None = None,
) -> tuple[Pipeline, list[str]]:
    params = params or {}
    imbalance = _imbalance_config()
    fit_train = train
    if bool(imbalance.get("enabled", True)) and model_name == "mlp":
        ratio = float(params.get("negative_sample_ratio", imbalance.get("negative_sample_ratio", 3.0)))
        fit_train = _balanced_resample(train, label_col, ratio, random_state)
    usable = [feature for feature in features if fit_train[feature].notna().any()]
    target = fit_train[label_col].astype(int)
    if target.nunique() < 2:
        raise RuntimeError(f"{label_col} has only one class in training data.")
    positive_weight = _positive_weight(target, imbalance)
    if model_name == "logistic":
        estimator = LogisticRegression(
            C=float(params.get("C", 1.0)),
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    elif model_name == "lda":
        estimator = LinearDiscriminantAnalysis(
            solver=str(params.get("solver", "lsqr")),
            shrinkage=params.get("shrinkage", "auto"),
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    elif model_name == "qda":
        estimator = QuadraticDiscriminantAnalysis(
            reg_param=float(params.get("reg_param", 0.30)),
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    elif model_name == "bp":
        estimator = BPNeuralNetworkClassifier(
            hidden_layers=tuple(params.get("hidden_layers", (48, 24))),
            dropout=float(params.get("dropout", 0.10)),
            learning_rate=float(params.get("learning_rate", 0.001)),
            weight_decay=0.0001,
            max_epochs=int(params.get("max_epochs", 160)),
            batch_size=64,
            patience=int(params.get("patience", 18)),
            positive_weight=positive_weight,
            focal_gamma=(
                float(params.get("focal_gamma", imbalance.get("bp_focal_gamma", 2.0)))
                if bool(imbalance.get("bp_use_focal_loss", True))
                else 0.0
            ),
            focal_alpha=float(params.get("focal_alpha", imbalance.get("bp_focal_alpha", 0.75))),
            random_state=random_state,
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    elif model_name == "mlp":
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (48, 24))),
            activation=str(params.get("activation", "relu")),
            alpha=float(params.get("alpha", 0.0001)),
            learning_rate_init=float(params.get("learning_rate_init", 0.001)),
            max_iter=int(params.get("max_iter", 500)),
            early_stopping=True,
            validation_fraction=0.20,
            n_iter_no_change=20,
            random_state=random_state,
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    elif model_name == "svm":
        estimator = SVC(
            C=float(params.get("C", 1.0)),
            kernel="rbf",
            gamma=params.get("gamma", "scale"),
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    elif model_name == "lightgbm":
        use_is_unbalance = bool(params.get("is_unbalance", imbalance.get("lightgbm_is_unbalance", False)))
        lgbm_kwargs = {
            "n_estimators": int(params.get("n_estimators", 250)),
            "learning_rate": float(params.get("learning_rate", 0.03)),
            "num_leaves": int(params.get("num_leaves", 15)),
            "max_depth": int(params.get("max_depth", 4)),
            "min_child_samples": int(params.get("min_child_samples", 10)),
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 1.0,
            "random_state": random_state,
            "verbose": -1,
        }
        if bool(imbalance.get("enabled", True)):
            if use_is_unbalance:
                lgbm_kwargs["is_unbalance"] = True
            else:
                lgbm_kwargs["scale_pos_weight"] = positive_weight
            if bool(params.get("balanced_bagging", imbalance.get("lightgbm_balanced_bagging", True))):
                lgbm_kwargs["bagging_freq"] = 1
                lgbm_kwargs["pos_bagging_fraction"] = 1.0
                lgbm_kwargs["neg_bagging_fraction"] = _negative_bagging_fraction(target, imbalance, params)
        estimator = LGBMClassifier(
            **lgbm_kwargs
        )
        pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    pipeline.fit(fit_train[usable], target)
    return pipeline, usable


def _positive_probability(model: Pipeline, data: pd.DataFrame, features: list[str]) -> np.ndarray:
    classes = list(model.named_steps["model"].classes_)
    return model.predict_proba(data[features])[:, classes.index(1)]


def _param_grid(model_name: str) -> list[dict]:
    if model_name == "logistic":
        return [{"C": value} for value in [0.3, 1.0, 3.0]]
    if model_name == "lda":
        return [
            {"solver": "lsqr", "shrinkage": "auto"},
            {"solver": "lsqr", "shrinkage": 0.20},
            {"solver": "lsqr", "shrinkage": 0.50},
        ]
    if model_name == "qda":
        return [
            {"reg_param": 0.20},
            {"reg_param": 0.50},
            {"reg_param": 0.80},
        ]
    if model_name == "lightgbm":
        return [
            {
                "n_estimators": 180,
                "learning_rate": 0.04,
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 8,
                "is_unbalance": False,
                "balanced_bagging": True,
                "negative_sample_ratio": 3.0,
            },
            {
                "n_estimators": 250,
                "learning_rate": 0.03,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 10,
                "is_unbalance": False,
                "balanced_bagging": True,
                "negative_sample_ratio": 5.0,
            },
            {
                "n_estimators": 320,
                "learning_rate": 0.025,
                "num_leaves": 15,
                "max_depth": 5,
                "min_child_samples": 12,
                "is_unbalance": True,
                "balanced_bagging": True,
                "negative_sample_ratio": 3.0,
            },
        ]
    if model_name == "bp":
        return [
            {
                "hidden_layers": (32,),
                "dropout": 0.05,
                "learning_rate": 0.001,
                "max_epochs": 130,
                "patience": 15,
                "focal_gamma": 1.5,
                "focal_alpha": 0.70,
            },
            {
                "hidden_layers": (48, 24),
                "dropout": 0.10,
                "learning_rate": 0.001,
                "max_epochs": 160,
                "patience": 18,
                "focal_gamma": 2.0,
                "focal_alpha": 0.75,
            },
            {
                "hidden_layers": (64, 32),
                "dropout": 0.15,
                "learning_rate": 0.0007,
                "max_epochs": 180,
                "patience": 20,
                "focal_gamma": 2.5,
                "focal_alpha": 0.80,
            },
        ]
    if model_name == "mlp":
        return [
            {
                "hidden_layer_sizes": (32,),
                "alpha": 0.0001,
                "learning_rate_init": 0.001,
                "max_iter": 500,
                "negative_sample_ratio": 2.0,
            },
            {
                "hidden_layer_sizes": (48, 24),
                "alpha": 0.0001,
                "learning_rate_init": 0.001,
                "max_iter": 600,
                "negative_sample_ratio": 3.0,
            },
            {
                "hidden_layer_sizes": (64, 32),
                "alpha": 0.001,
                "learning_rate_init": 0.0007,
                "max_iter": 700,
                "negative_sample_ratio": 5.0,
            },
        ]
    if model_name == "svm":
        return [
            {"C": 0.5, "gamma": "scale"},
            {"C": 1.0, "gamma": "scale"},
            {"C": 2.0, "gamma": "scale"},
        ]
    raise ValueError(f"Unsupported model: {model_name}")


def _non_overlapping(signal: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    if signal.empty:
        return signal
    selected = []
    next_position = -1
    for idx, row in signal.sort_values("week_pos").iterrows():
        position = int(row["week_pos"])
        if position >= next_position:
            selected.append(idx)
            next_position = position + horizon_weeks
    return signal.loc[selected]


def _split_tuning_train_valid(
    train: pd.DataFrame,
    validation_fraction: float,
    min_validation_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = train.sort_values("week_pos").reset_index(drop=True)
    valid_rows = max(int(round(len(ordered) * validation_fraction)), min_validation_rows)
    valid_rows = min(valid_rows, max(1, len(ordered) // 3))
    fit_rows = len(ordered) - valid_rows
    if fit_rows < min_validation_rows:
        raise RuntimeError("Not enough rows for tuning split.")
    return ordered.iloc[:fit_rows].copy(), ordered.iloc[fit_rows:].copy()


def _score_validation(
    valid: pd.DataFrame,
    proba: np.ndarray,
    label_col: str,
    threshold: float,
    horizon_weeks: int,
    min_signal_count: int,
) -> dict:
    scored = valid[
        ["week_pos", label_col, "future_ret_3w", "future_mfe_3w", "future_mae_3w"]
    ].copy()
    scored["turning_proba"] = proba
    signal = _non_overlapping(scored[scored["turning_proba"] >= threshold], horizon_weeks)
    signal_count = len(signal)
    precision = float(signal[label_col].mean()) if signal_count else np.nan
    avg_ret = float(signal["future_ret_3w"].mean()) if signal_count else np.nan
    natural_precision = float(scored[label_col].mean())
    natural_ret = float(scored["future_ret_3w"].mean())
    return {
        "threshold": threshold,
        "validation_rows": len(scored),
        "validation_signal_count": signal_count,
        "validation_precision": precision,
        "validation_natural_precision": natural_precision,
        "validation_precision_lift": precision - natural_precision if signal_count else np.nan,
        "validation_avg_future_ret_3w": avg_ret,
        "validation_natural_avg_future_ret_3w": natural_ret,
        "validation_avg_return_lift": avg_ret - natural_ret if signal_count else np.nan,
        "valid_signal_floor": int(signal_count >= min_signal_count),
    }


def _tune_model(
    model_name: str,
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    random_state: int,
    thresholds: list[float],
    horizon_weeks: int,
    validation_fraction: float,
    min_validation_rows: int,
    min_signal_count: int,
) -> tuple[dict, dict]:
    try:
        fit_part, valid_part = _split_tuning_train_valid(
            train, validation_fraction, min_validation_rows
        )
    except RuntimeError:
        return {}, {
            "selected": 1,
            "tuning_status": "skipped_not_enough_rows",
            "params_json": "{}",
        }
    if fit_part[label_col].nunique() < 2 or valid_part[label_col].nunique() < 2:
        return {}, {
            "selected": 1,
            "tuning_status": "skipped_one_class_split",
            "params_json": "{}",
            "fit_rows": len(fit_part),
            "validation_rows": len(valid_part),
        }

    rows = []
    threshold_order = sorted(thresholds, reverse=True)
    for params in _param_grid(model_name):
        try:
            fitted, usable = _fit_model(model_name, fit_part, features, label_col, random_state, params)
            proba = _positive_probability(fitted, valid_part, usable)
        except Exception as exc:
            rows.append(
                {
                    "selected": 0,
                    "tuning_status": f"failed:{type(exc).__name__}",
                    "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "fit_rows": len(fit_part),
                    "validation_rows": len(valid_part),
                }
            )
            continue
        for threshold in threshold_order:
            row = _score_validation(
                valid_part, proba, label_col, threshold, horizon_weeks, min_signal_count
            )
            row.update(
                {
                    "selected": 0,
                    "tuning_status": "ok",
                    "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
                    "fit_rows": len(fit_part),
                    "usable_features": len(usable),
                }
            )
            rows.append(row)

    candidates = pd.DataFrame(rows)
    if candidates.empty or not (candidates["tuning_status"] == "ok").any():
        return {}, {
            "selected": 1,
            "tuning_status": "fallback_no_valid_candidate",
            "params_json": "{}",
            "fit_rows": len(fit_part),
            "validation_rows": len(valid_part),
        }
    ok = candidates[candidates["tuning_status"] == "ok"].copy()
    ok["rank_precision"] = ok["validation_precision"].fillna(-1.0)
    ok["rank_return"] = ok["validation_avg_return_lift"].fillna(-999.0)
    ok = ok.sort_values(
        [
            "valid_signal_floor",
            "rank_precision",
            "rank_return",
            "validation_signal_count",
            "threshold",
        ],
        ascending=[False, False, False, False, False],
    )
    selected = ok.iloc[0].copy()
    selected_params = json.loads(str(selected["params_json"]))
    selected_rows = []
    for row in rows:
        row = dict(row)
        row["selected"] = int(
            row.get("tuning_status") == "ok"
            and row.get("params_json") == selected["params_json"]
            and float(row.get("threshold", np.nan)) == float(selected["threshold"])
        )
        selected_rows.append(row)
    return selected_params, {
        "selected": 1,
        "tuning_status": "ok",
        "params_json": selected["params_json"],
        "selected_threshold": float(selected["threshold"]),
        "fit_rows": len(fit_part),
        "validation_rows": len(valid_part),
        "validation_signal_count": int(selected["validation_signal_count"]),
        "validation_precision": selected["validation_precision"],
        "validation_avg_return_lift": selected["validation_avg_return_lift"],
        "all_rows": selected_rows,
    }


def _metric_row(
    universe: pd.DataFrame,
    signal: pd.DataFrame,
    threshold: float,
    sample_mode: str,
) -> dict:
    label_col = str(universe["label_col"].iloc[0])
    side = str(universe["side"].iloc[0])
    signal_count = len(signal)
    return {
        "test_year": int(universe["test_year"].iloc[0]),
        "year_complete": bool(universe["year_complete"].iloc[0]),
        "model": universe["model"].iloc[0],
        "side": side,
        "label": label_col,
        "threshold": threshold,
        "sample_mode": sample_mode,
        "train_rows": int(universe["train_rows"].iloc[0]),
        "candidate_rows": len(universe),
        "signal_count": signal_count,
        "coverage": signal_count / len(universe),
        "precision": float(signal[label_col].mean()) if signal_count else None,
        "natural_precision": float(universe[label_col].mean()),
        "precision_lift": (
            float(signal[label_col].mean() - universe[label_col].mean()) if signal_count else None
        ),
        "actual_up_rate": float((signal["future_ret_3w"] > 0).mean()) if signal_count else None,
        "actual_down_rate": float((signal["future_ret_3w"] < 0).mean()) if signal_count else None,
        "avg_future_ret_3w": float(signal["future_ret_3w"].mean()) if signal_count else None,
        "median_future_ret_3w": float(signal["future_ret_3w"].median()) if signal_count else None,
        "avg_future_mfe_3w": float(signal["future_mfe_3w"].mean()) if signal_count else None,
        "avg_future_mae_3w": float(signal["future_mae_3w"].mean()) if signal_count else None,
        "natural_avg_future_ret_3w": float(universe["future_ret_3w"].mean()),
        "avg_return_lift": (
            float(signal["future_ret_3w"].mean() - universe["future_ret_3w"].mean())
            if signal_count
            else None
        ),
    }


def _summarize(details: pd.DataFrame, thresholds: list[float], horizon_weeks: int) -> pd.DataFrame:
    rows = []
    for (_, _, _), part in details.groupby(["test_year", "model", "side"], sort=True):
        for threshold in thresholds:
            signal = part[part["turning_proba"] >= threshold]
            rows.append(_metric_row(part, signal, threshold, "all_signals"))
            rows.append(
                _metric_row(part, _non_overlapping(signal, horizon_weeks), threshold, "non_overlapping")
            )
    return pd.DataFrame(rows)


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    rows = []
    group_cols = ["model", "side", "label", "threshold", "sample_mode"]
    for keys, part in summary.groupby(group_cols, sort=True):
        signal_count = int(part["signal_count"].sum())
        candidate_rows = int(part["candidate_rows"].sum())
        signal_weight = part["signal_count"].astype(float)
        candidate_weight = part["candidate_rows"].astype(float)

        def weighted(column: str, weights: pd.Series) -> float | None:
            valid = part[column].notna() & (weights > 0)
            if not valid.any():
                return None
            return float(np.average(part.loc[valid, column], weights=weights.loc[valid]))

        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "evaluated_years": int(part["test_year"].nunique()),
                "complete_years": int(part.loc[part["year_complete"], "test_year"].nunique()),
                "candidate_rows": candidate_rows,
                "signal_count": signal_count,
                "coverage": signal_count / candidate_rows if candidate_rows else None,
                "precision": weighted("precision", signal_weight),
                "natural_precision": weighted("natural_precision", candidate_weight),
                "actual_up_rate": weighted("actual_up_rate", signal_weight),
                "actual_down_rate": weighted("actual_down_rate", signal_weight),
                "avg_future_ret_3w": weighted("avg_future_ret_3w", signal_weight),
                "avg_future_mfe_3w": weighted("avg_future_mfe_3w", signal_weight),
                "avg_future_mae_3w": weighted("avg_future_mae_3w", signal_weight),
                "natural_avg_future_ret_3w": weighted("natural_avg_future_ret_3w", candidate_weight),
            }
        )
    result = pd.DataFrame(rows)
    result["precision_lift"] = result["precision"] - result["natural_precision"]
    result["avg_return_lift"] = result["avg_future_ret_3w"] - result["natural_avg_future_ret_3w"]
    return result


def walk_forward_bottom_weekly_evaluation(
    sides: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    thresholds: list[float] | None = None,
    models: list[str] | None = None,
    feature_manifest: str | None = None,
    use_candidates_only: bool = True,
    write_reports: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = get_config("bottom_weekly_model.yaml")
    model_cfg = cfg["model"]
    horizon_weeks = int(model_cfg["horizon_weeks"])
    start_year = start_year or int(model_cfg["evaluation_start_year"])
    data = load_bottom_weekly_dataset()
    if data.empty:
        raise RuntimeError("Weekly bottom dataset is empty. Run build_bottom_weekly_dataset first.")
    features = load_bottom_weekly_features(data, feature_manifest)
    complete = data.dropna(subset=["future_ret_3w"]).copy()
    end_year = end_year or int(complete["week_end_date"].dt.year.max())
    thresholds = thresholds or [float(value) for value in model_cfg["probability_thresholds"]]
    models = models or [str(value) for value in model_cfg.get("model_kinds", ["logistic", "lightgbm"])]
    sides = sides or ["bottom", "top"]
    invalid_sides = sorted(set(sides) - set(SIDE_CONFIG))
    if invalid_sides:
        raise ValueError(f"Unsupported sides: {invalid_sides}")
    min_train_rows = int(model_cfg["min_train_rows"])
    train_start = pd.Timestamp(str(model_cfg["train_start"]))
    random_state = int(model_cfg["random_state"])
    latest_complete_date = complete["week_end_date"].max()
    detail_parts = []
    tuning_rows = []
    enable_tuning = bool(model_cfg.get("enable_tuning", False))
    validation_fraction = float(model_cfg.get("tuning_validation_fraction", 0.20))
    min_validation_rows = int(model_cfg.get("tuning_min_validation_rows", 20))
    min_signal_count = int(model_cfg.get("tuning_min_signal_count", 2))

    for year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        year_rows = complete[
            (complete["week_end_date"] >= test_start) & (complete["week_end_date"] <= test_end)
        ]
        if year_rows.empty:
            continue
        first_test_position = int(year_rows["week_pos"].min())
        base_train = complete[
            (complete["week_end_date"] >= train_start)
            & (complete["week_pos"] <= first_test_position - horizon_weeks - 1)
        ].copy()
        for side in sides:
            label_col = SIDE_CONFIG[side]["label"]
            candidate_col = SIDE_CONFIG[side]["candidate"]
            train = base_train.copy()
            test = year_rows.copy()
            if use_candidates_only:
                train = train[train[candidate_col] == 1].copy()
                test = test[test[candidate_col] == 1].copy()
            if len(train) < min_train_rows or test.empty:
                logger.warning(
                    "skip weekly year=%s side=%s train_rows=%s test_rows=%s",
                    year, side, len(train), len(test),
                )
                continue
            for model_name in models:
                selected_params: dict = {}
                selected_tuning = {"tuning_status": "disabled", "params_json": "{}"}
                if enable_tuning:
                    selected_params, selected_tuning = _tune_model(
                        model_name,
                        train,
                        features,
                        label_col,
                        random_state,
                        thresholds,
                        horizon_weeks,
                        validation_fraction,
                        min_validation_rows,
                        min_signal_count,
                    )
                    all_tuning_rows = selected_tuning.pop("all_rows", None)
                    if all_tuning_rows:
                        for tuning_row in all_tuning_rows:
                            tuning_row.update(
                                {
                                    "test_year": year,
                                    "side": side,
                                    "model": model_name,
                                    "label_col": label_col,
                                    "train_rows": len(train),
                                }
                            )
                            tuning_rows.append(tuning_row)
                    else:
                        tuning_record = dict(selected_tuning)
                        tuning_record.update(
                            {
                                "test_year": year,
                                "side": side,
                                "model": model_name,
                                "label_col": label_col,
                                "train_rows": len(train),
                            }
                        )
                        tuning_rows.append(tuning_record)
                fitted, usable = _fit_model(model_name, train, features, label_col, random_state, selected_params)
                part = test[
                    [
                        "week_start_date", "week_end_date", "week_pos", label_col,
                        "future_ret_3w", "future_mfe_3w", "future_mae_3w",
                        "is_bottom_candidate", "is_top_candidate",
                    ]
                ].copy()
                part["turning_proba"] = _positive_probability(fitted, test, usable)
                part["test_year"] = year
                part["year_complete"] = latest_complete_date >= test_end
                part["model"] = model_name
                part["side"] = side
                part["label_col"] = label_col
                part["train_rows"] = len(train)
                part["tuned_params_json"] = json.dumps(selected_params, ensure_ascii=False, sort_keys=True)
                part["tuning_status"] = selected_tuning.get("tuning_status", "disabled")
                detail_parts.append(part)
                logger.info(
                    "weekly walk-forward year=%s side=%s model=%s train=%s test=%s features=%s tuning=%s",
                    year, side, model_name, len(train), len(test), len(usable),
                    selected_tuning.get("tuning_status", "disabled"),
                )

    if not detail_parts:
        raise RuntimeError("No weekly bottom walk-forward folds were produced.")
    details = pd.concat(detail_parts, ignore_index=True)
    summary = _summarize(details, thresholds, horizon_weeks)
    aggregate = _aggregate(summary)
    if write_reports:
        for frame, output_key in [
            (details, "walk_forward_detail"),
            (summary, "walk_forward_summary"),
            (aggregate, "walk_forward_aggregate"),
        ]:
            path = project_path(str(cfg["outputs"][output_key]))
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(path, index=False, encoding="utf-8-sig")
        if tuning_rows:
            path = project_path(str(cfg["outputs"]["tuning_report"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(tuning_rows).to_csv(path, index=False, encoding="utf-8-sig")
    return details, summary, aggregate


def main() -> None:
    cfg = get_config("bottom_weekly_model.yaml")
    parser = argparse.ArgumentParser(description="Purged yearly weekly top/bottom walk-forward evaluation.")
    parser.add_argument("--sides", default="bottom,top")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in cfg["model"]["probability_thresholds"]),
    )
    parser.add_argument("--models", default=",".join(str(value) for value in cfg["model"].get("model_kinds", ["logistic", "lightgbm"])))
    parser.add_argument("--feature-manifest")
    parser.add_argument("--all-weeks", action="store_true")
    args = parser.parse_args()
    _, _, aggregate = walk_forward_bottom_weekly_evaluation(
        sides=[item.strip() for item in args.sides.split(",") if item.strip()],
        start_year=args.start_year,
        end_year=args.end_year,
        thresholds=_parse_floats(args.thresholds),
        models=[item.strip() for item in args.models.split(",") if item.strip()],
        feature_manifest=args.feature_manifest,
        use_candidates_only=not args.all_weeks,
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
