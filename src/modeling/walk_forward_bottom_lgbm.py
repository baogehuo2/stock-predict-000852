from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.logger import get_logger
from src.modeling.bottom_data import load_bottom_dataset, load_bottom_features


disable_broken_dask_autoload()
from lightgbm import LGBMClassifier


logger = get_logger(__name__)


def _parse_floats(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def _fit_model(
    model_name: str,
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    random_state: int,
) -> tuple[Pipeline, list[str]]:
    usable = [feature for feature in features if train[feature].notna().any()]
    target = train[label_col].astype(int)
    if target.nunique() < 2:
        raise RuntimeError(f"{label_col} has only one class in training data.")
    if model_name == "logistic":
        estimator = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=random_state
        )
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", estimator),
            ]
        )
    elif model_name == "lightgbm":
        estimator = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=5,
            min_child_samples=20,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
        pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("model", estimator)]
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    pipeline.fit(train[usable], target)
    return pipeline, usable


def _positive_probability(model: Pipeline, data: pd.DataFrame, features: list[str]) -> np.ndarray:
    classes = list(model.named_steps["model"].classes_)
    return model.predict_proba(data[features])[:, classes.index(1)]


def _non_overlapping(signal: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if signal.empty:
        return signal
    selected = []
    next_position = -1
    for idx, row in signal.sort_values("trade_pos").iterrows():
        position = int(row["trade_pos"])
        if position >= next_position:
            selected.append(idx)
            next_position = position + horizon
    return signal.loc[selected]


def _metric_row(
    universe: pd.DataFrame,
    signal: pd.DataFrame,
    threshold: float,
    sample_mode: str,
) -> dict:
    label_col = str(universe["label_col"].iloc[0])
    signal_count = len(signal)
    return {
        "test_year": int(universe["test_year"].iloc[0]),
        "year_complete": bool(universe["year_complete"].iloc[0]),
        "model": universe["model"].iloc[0],
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
        "actual_up_rate": float((signal["future_ret_15d"] > 0).mean()) if signal_count else None,
        "avg_future_ret_15d": float(signal["future_ret_15d"].mean()) if signal_count else None,
        "median_future_ret_15d": float(signal["future_ret_15d"].median()) if signal_count else None,
        "avg_future_mfe_15d": float(signal["future_mfe_15d"].mean()) if signal_count else None,
        "avg_future_mae_15d": float(signal["future_mae_15d"].mean()) if signal_count else None,
        "continuation_risk_rate": float(signal["continuation_risk_label"].mean()) if signal_count else None,
        "natural_avg_future_ret_15d": float(universe["future_ret_15d"].mean()),
        "avg_return_lift": (
            float(signal["future_ret_15d"].mean() - universe["future_ret_15d"].mean())
            if signal_count
            else None
        ),
    }


def _summarize(details: pd.DataFrame, thresholds: list[float], horizon: int) -> pd.DataFrame:
    rows = []
    for (_, _), part in details.groupby(["test_year", "model"], sort=True):
        for threshold in thresholds:
            signal = part[part["bottom_proba"] >= threshold]
            rows.append(_metric_row(part, signal, threshold, "all_signals"))
            rows.append(
                _metric_row(part, _non_overlapping(signal, horizon), threshold, "non_overlapping")
            )
    return pd.DataFrame(rows)


def _aggregate(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    rows = []
    group_cols = ["model", "label", "threshold", "sample_mode"]
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
                "avg_future_ret_15d": weighted("avg_future_ret_15d", signal_weight),
                "avg_future_mfe_15d": weighted("avg_future_mfe_15d", signal_weight),
                "avg_future_mae_15d": weighted("avg_future_mae_15d", signal_weight),
                "continuation_risk_rate": weighted("continuation_risk_rate", signal_weight),
                "natural_avg_future_ret_15d": weighted("natural_avg_future_ret_15d", candidate_weight),
            }
        )
    result = pd.DataFrame(rows)
    result["precision_lift"] = result["precision"] - result["natural_precision"]
    result["avg_return_lift"] = result["avg_future_ret_15d"] - result["natural_avg_future_ret_15d"]
    return result


def walk_forward_bottom_evaluation(
    label_col: str = "quality_bottom_label",
    start_year: int | None = None,
    end_year: int | None = None,
    thresholds: list[float] | None = None,
    models: list[str] | None = None,
    feature_manifest: str | None = None,
    write_reports: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = get_config("bottom_model.yaml")
    model_cfg = cfg["model"]
    horizon = int(model_cfg["horizon"])
    start_year = start_year or int(model_cfg["evaluation_start_year"])
    data = load_bottom_dataset()
    if data.empty:
        raise RuntimeError("Bottom dataset is empty. Run build_bottom_dataset first.")
    features = load_bottom_features(data, feature_manifest)
    if label_col not in data:
        raise ValueError(f"Unknown label column: {label_col}")
    complete = data.dropna(subset=[label_col, "future_ret_15d"]).copy()
    end_year = end_year or int(complete["trade_date"].dt.year.max())
    thresholds = thresholds or [float(value) for value in model_cfg["probability_thresholds"]]
    models = models or ["logistic", "lightgbm"]
    min_train_rows = int(model_cfg["min_train_rows"])
    train_start = pd.Timestamp(str(model_cfg["train_start"]))
    random_state = int(model_cfg["random_state"])
    latest_complete_date = complete["trade_date"].max()
    detail_parts = []

    for year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        year_rows = complete[
            (complete["trade_date"] >= test_start) & (complete["trade_date"] <= test_end)
        ]
        if year_rows.empty:
            continue
        first_test_position = int(year_rows["trade_pos"].min())
        train = complete[
            (complete["trade_date"] >= train_start)
            & (complete["trade_pos"] <= first_test_position - horizon - 1)
            & (complete["is_candidate"] == 1)
        ].copy()
        test = year_rows[year_rows["is_candidate"] == 1].copy()
        if len(train) < min_train_rows or test.empty:
            logger.warning("skip year=%s train_candidates=%s test_candidates=%s", year, len(train), len(test))
            continue
        for model_name in models:
            fitted, usable = _fit_model(model_name, train, features, label_col, random_state)
            part = test[
                [
                    "trade_date", "trade_pos", label_col, "future_ret_15d", "future_mfe_15d",
                    "future_mae_15d", "continuation_risk_label", "candidate_score",
                ]
            ].copy()
            part["bottom_proba"] = _positive_probability(fitted, test, usable)
            part["test_year"] = year
            part["year_complete"] = latest_complete_date >= test_end
            part["model"] = model_name
            part["label_col"] = label_col
            part["train_rows"] = len(train)
            detail_parts.append(part)
            logger.info(
                "bottom walk-forward year=%s model=%s train=%s test=%s features=%s",
                year, model_name, len(train), len(test), len(usable),
            )

    if not detail_parts:
        raise RuntimeError("No bottom walk-forward folds were produced.")
    details = pd.concat(detail_parts, ignore_index=True)
    summary = _summarize(details, thresholds, horizon)
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
    return details, summary, aggregate


def main() -> None:
    cfg = get_config("bottom_model.yaml")
    parser = argparse.ArgumentParser(description="Purged yearly bottom-fishing walk-forward evaluation.")
    parser.add_argument("--label", default="quality_bottom_label")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(value) for value in cfg["model"]["probability_thresholds"]),
    )
    parser.add_argument("--models", default="logistic,lightgbm")
    parser.add_argument("--feature-manifest")
    args = parser.parse_args()
    _, _, aggregate = walk_forward_bottom_evaluation(
        label_col=args.label,
        start_year=args.start_year,
        end_year=args.end_year,
        thresholds=_parse_floats(args.thresholds),
        models=[item.strip() for item in args.models.split(",") if item.strip()],
        feature_manifest=args.feature_manifest,
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
