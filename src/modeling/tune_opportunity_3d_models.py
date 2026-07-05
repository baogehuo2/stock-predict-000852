from __future__ import annotations

import argparse
import itertools
from typing import Any

import pandas as pd

from src.common.config import get_config, project_path
from src.modeling.probability_lgbm import fit_probability_model, positive_probability
from src.modeling.walk_forward_opportunity_3d_lgbm import (
    HORIZON,
    _attach_signal_scores,
    _metric,
    _non_overlapping,
    _rule_mask,
    load_opportunity_frame,
)


BALANCED_RULE = {
    "rule": "balanced_long",
    "opportunity_threshold": 0.4,
    "position_threshold": None,
    "risk_threshold": None,
    "risk_index_threshold": 50.0,
    "long_score_threshold": 50.0,
}

THREE_STAGE_RULE = {
    "rule": "three_stage",
    "opportunity_threshold": 0.2,
    "position_threshold": 0.2,
    "risk_threshold": 0.5,
    "risk_index_threshold": None,
    "long_score_threshold": None,
}


def _param_grid(model_type: str) -> list[dict[str, Any]]:
    if model_type == "lgbm":
        return [
            {
                "n_estimators": 200,
                "learning_rate": 0.04,
                "num_leaves": 15,
                "min_child_samples": 50,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            },
            {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            },
            {
                "n_estimators": 400,
                "learning_rate": 0.02,
                "num_leaves": 15,
                "min_child_samples": 20,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            },
            {
                "n_estimators": 400,
                "learning_rate": 0.02,
                "num_leaves": 31,
                "min_child_samples": 50,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
            }
        ]
    if model_type == "xgboost":
        return [
            {
                "n_estimators": 200,
                "learning_rate": 0.04,
                "max_depth": 2,
                "min_child_weight": 6,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 3.0,
            },
            {
                "n_estimators": 300,
                "learning_rate": 0.03,
                "max_depth": 3,
                "min_child_weight": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 2.0,
            },
            {
                "n_estimators": 400,
                "learning_rate": 0.02,
                "max_depth": 2,
                "min_child_weight": 3,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 400,
                "learning_rate": 0.02,
                "max_depth": 3,
                "min_child_weight": 6,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "reg_lambda": 3.0,
            }
        ]
    if model_type == "bp":
        return [
            {
                "hidden_layer_sizes": (32,),
                "alpha": 0.001,
                "learning_rate_init": 0.001,
                "batch_size": 64,
                "max_iter": 400,
                "early_stopping": True,
                "validation_fraction": 0.15,
                "n_iter_no_change": 20,
            },
            {
                "hidden_layer_sizes": (64, 32),
                "alpha": 0.001,
                "learning_rate_init": 0.001,
                "batch_size": 64,
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.15,
                "n_iter_no_change": 20,
            },
            {
                "hidden_layer_sizes": (64,),
                "alpha": 0.0005,
                "learning_rate_init": 0.0005,
                "batch_size": 64,
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.15,
                "n_iter_no_change": 20,
            },
            {
                "hidden_layer_sizes": (128, 64),
                "alpha": 0.002,
                "learning_rate_init": 0.0005,
                "batch_size": 64,
                "max_iter": 500,
                "early_stopping": True,
                "validation_fraction": 0.15,
                "n_iter_no_change": 20,
            }
        ]
    raise ValueError(f"Unknown model_type: {model_type}")


def _parse_model_types(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _score_row(row: dict[str, Any]) -> float:
    signal_count = float(row.get("signal_count") or 0)
    avg_ret = float(row.get("avg_future_ret") or 0)
    up_rate = float(row.get("future_up_3d_rate") or 0)
    worst_ret = float(row.get("worst_future_ret") or 0)
    high_risk = float(row.get("high_risk_signal_rate") or 0)
    count_penalty = 0.0 if signal_count >= 50 else (50.0 - signal_count) / 5000.0
    return avg_ret + 0.01 * (up_rate - 0.5) + 0.10 * min(worst_ret, 0) - 0.02 * high_risk - count_penalty


def _fit_fold_probabilities(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    random_state: int,
    model_type: str,
    model_params: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    result = test[
        [
            "trade_date",
            "index_code",
            "future_ret",
            "future_mfe",
            "future_mae",
            "opportunity_label",
            "risk_label",
            "manual_buy_label",
            "manual_sell_label",
        ]
    ].copy()
    feature_counts = {}
    for output_col, label_col, count_col in [
        ("opportunity_proba", "opportunity_label", "opportunity_feature_count"),
        ("position_proba", "manual_buy_label", "position_feature_count"),
        ("risk_proba", "risk_label", "risk_feature_count"),
    ]:
        model, usable = fit_probability_model(
            train,
            features,
            label_col,
            random_state,
            model_type=model_type,
            model_params=model_params,
        )
        result[output_col] = positive_probability(model, test, usable)
        feature_counts[count_col] = len(usable)
    return _attach_signal_scores(result), feature_counts


def _evaluate_params(
    data: pd.DataFrame,
    features: list[str],
    model_type: str,
    model_params: dict[str, Any],
    start_year: int,
    end_year: int,
    train_start: str,
    min_train_rows: int,
) -> list[dict[str, Any]]:
    random_state = int(get_config()["model"].get("random_state", 42))
    train_start_ts = pd.Timestamp(train_start)
    detail_parts = []
    for test_year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        train = data[
            (data["trade_date"] >= train_start_ts) & (data["trade_date"] < test_start)
        ].dropna(subset=["opportunity_label", "manual_buy_label", "risk_label"]).copy()
        test = data[
            (data["trade_date"] >= test_start) & (data["trade_date"] <= test_end)
        ].dropna(subset=["opportunity_label", "manual_buy_label", "risk_label"]).copy()
        train = train.sort_values("trade_date")
        test = test.sort_values("trade_date")
        if len(train) > HORIZON:
            train = train.iloc[:-HORIZON].copy()
        if len(train) < min_train_rows or test.empty:
            continue
        if train["opportunity_label"].nunique() < 2:
            continue
        if train["manual_buy_label"].nunique() < 2:
            continue
        if train["risk_label"].nunique() < 2:
            continue
        part, counts = _fit_fold_probabilities(
            train,
            test,
            features,
            random_state,
            model_type,
            model_params,
        )
        part["model_type"] = model_type
        part["test_year"] = test_year
        part["train_rows"] = len(train)
        for key, value in counts.items():
            part[key] = value
        part["trade_pos"] = range(len(part))
        detail_parts.append(part)
    if not detail_parts:
        return []
    details = pd.concat(detail_parts, ignore_index=True)
    rows = []
    for rule_spec in [BALANCED_RULE, THREE_STAGE_RULE]:
        yearly_signals = []
        yearly_universes = []
        for year, part0 in details.groupby("test_year", sort=True):
            universe = part0.dropna(subset=["future_ret", "future_mfe", "future_mae"]).copy()
            signal = _non_overlapping(universe[_rule_mask(universe, **rule_spec)])
            yearly_signals.append(signal)
            yearly_universes.append(universe)
        signal = pd.concat(yearly_signals, ignore_index=True) if yearly_signals else pd.DataFrame()
        universe = pd.concat(yearly_universes, ignore_index=True) if yearly_universes else pd.DataFrame()
        row = _metric(
            universe,
            signal,
            model_type,
            rule_spec["rule"],
            "buy_plus_bottom",
            rule_spec["opportunity_threshold"],
            rule_spec["position_threshold"],
            rule_spec["risk_threshold"],
            rule_spec["risk_index_threshold"],
            rule_spec["long_score_threshold"],
            "aggregate",
            None,
        )
        row["score"] = _score_row(row)
        for key, value in model_params.items():
            row[f"param_{key}"] = str(value)
        rows.append(row)
    return rows


def tune_models(
    model_types: list[str],
    start_year: int,
    end_year: int,
    train_start: str,
    min_train_rows: int,
    output_csv: str,
) -> pd.DataFrame:
    data, features = load_opportunity_frame("buy_plus_bottom", "data/manual_labels/market_turning_regions.csv")
    rows = []
    for model_type in model_types:
        for idx, params in enumerate(_param_grid(model_type), start=1):
            print(f"tuning model_type={model_type} params={idx}/{len(_param_grid(model_type))} {params}")
            rows.extend(
                _evaluate_params(
                    data,
                    features,
                    model_type,
                    params,
                    start_year,
                    end_year,
                    train_start,
                    min_train_rows,
                )
            )
    result = pd.DataFrame(rows)
    path = project_path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved tuning report path={path} rows={len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune 3-day opportunity model backends.")
    parser.add_argument("--model-types", default="lgbm,xgboost,bp")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--output-csv", default="data/reports/opportunity_3d_hyperparameter_tuning.csv")
    args = parser.parse_args()
    result = tune_models(
        model_types=_parse_model_types(args.model_types),
        start_year=args.start_year,
        end_year=args.end_year,
        train_start=args.train_start,
        min_train_rows=args.min_train_rows,
        output_csv=args.output_csv,
    )
    if result.empty:
        print("No tuning rows.")
        return
    columns = [
        "model_type",
        "rule",
        "signal_count",
        "future_up_3d_rate",
        "avg_future_ret",
        "worst_future_ret",
        "risk_rate",
        "avg_risk_index",
        "score",
    ]
    print(result.sort_values("score", ascending=False)[columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
