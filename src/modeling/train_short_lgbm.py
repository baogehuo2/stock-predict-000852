from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.logger import get_logger
from src.modeling.data import feature_columns, load_dataset
from src.modeling.train_lgbm import GUBA_SENTIMENT_FEATURES


disable_broken_dask_autoload()
from lightgbm import LGBMClassifier


logger = get_logger(__name__)


def _parse_csv_floats(value: str) -> list[float]:
    result = []
    for item in value.split(","):
        item = item.strip()
        if item:
            result.append(float(item))
    return sorted(set(result))


def _split(df: pd.DataFrame, splits: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = splits or get_config()["model"]["splits"]
    train = df[(df["trade_date"] >= splits["train_start"]) & (df["trade_date"] <= splits["train_end"])]
    valid = df[(df["trade_date"] >= splits["valid_start"]) & (df["trade_date"] <= splits["valid_end"])]
    test = df[df["trade_date"] >= splits["test_start"]]
    return train, valid, test


def _model_path(model_dir: Path, horizon: int, model_tag: str | None) -> Path:
    tag_part = f"_{model_tag}" if model_tag else ""
    return model_dir / f"short_lgbm{tag_part}_{horizon}d.joblib"


def _signal_metrics(
    part: pd.DataFrame,
    signal: pd.DataFrame,
    ret_col: str,
    label_col: str,
    rule: str,
    threshold: float | None,
) -> dict:
    if signal.empty:
        return {
            "rule": rule,
            "threshold": threshold,
            "signal_count": 0,
            "coverage": 0.0,
            "short_precision": None,
            "short_recall": None,
            "negative_ret_rate": None,
            "avg_future_ret": None,
            "median_future_ret": None,
            "avg_short_strategy_ret": None,
            "median_short_strategy_ret": None,
            "avg_short_proba": None,
        }

    y_true = signal[label_col].astype(int)
    full_true = part[label_col].astype(int)
    full_pred = pd.Series(0, index=part.index)
    full_pred.loc[signal.index] = 1
    short_strategy_ret = -signal[ret_col]
    return {
        "rule": rule,
        "threshold": threshold,
        "signal_count": int(len(signal)),
        "coverage": float(len(signal) / len(part)) if len(part) else 0.0,
        "short_precision": float(precision_score(y_true, pd.Series(1, index=signal.index), zero_division=0)),
        "short_recall": float(recall_score(full_true, full_pred, zero_division=0)),
        "negative_ret_rate": float((signal[ret_col] < 0).mean()),
        "avg_future_ret": float(signal[ret_col].mean()),
        "median_future_ret": float(signal[ret_col].median()),
        "avg_short_strategy_ret": float(short_strategy_ret.mean()),
        "median_short_strategy_ret": float(short_strategy_ret.median()),
        "avg_short_proba": float(signal["short_proba"].mean()) if "short_proba" in signal else None,
    }


def _evaluate_part(
    part: pd.DataFrame,
    model: Pipeline,
    features: list[str],
    horizon: int,
    split_name: str,
    thresholds: list[float],
    model_tag: str | None,
) -> list[dict]:
    ret_col = f"future_ret_{horizon}d"
    label_col = f"short_label_{horizon}d"
    if part.empty:
        return []

    eval_part = part.copy()
    eval_part["pred_short"] = model.predict(eval_part[features]).astype(int)
    proba = model.predict_proba(eval_part[features])
    short_class_index = list(model.named_steps["model"].classes_).index(1)
    eval_part["short_proba"] = proba[:, short_class_index]

    rows = []
    baseline = _signal_metrics(eval_part, eval_part, ret_col, label_col, "baseline_all_rows_short", None)
    baseline.update({"horizon": horizon, "split": split_name, "model_tag": model_tag or "default"})
    rows.append(baseline)

    baseline_rules = [
        ("baseline_ret_5d_negative", "f_ret_5d"),
        ("baseline_ma20_gap_negative", "f_ma20_gap"),
    ]
    for rule_name, feature_col in baseline_rules:
        if feature_col in eval_part:
            signal = eval_part[pd.to_numeric(eval_part[feature_col], errors="coerce") < 0]
            metrics = _signal_metrics(eval_part, signal, ret_col, label_col, rule_name, None)
            metrics.update({"horizon": horizon, "split": split_name, "model_tag": model_tag or "default"})
            rows.append(metrics)

    predicted_short = eval_part[eval_part["pred_short"] == 1]
    pred_metrics = _signal_metrics(eval_part, predicted_short, ret_col, label_col, "predicted_short_label", None)
    pred_metrics.update({"horizon": horizon, "split": split_name, "model_tag": model_tag or "default"})
    rows.append(pred_metrics)

    for threshold in thresholds:
        signal = eval_part[eval_part["short_proba"] >= threshold]
        metrics = _signal_metrics(eval_part, signal, ret_col, label_col, "short_probability", threshold)
        metrics.update({"horizon": horizon, "split": split_name, "model_tag": model_tag or "default"})
        rows.append(metrics)

    for year, year_part in eval_part.groupby(eval_part["trade_date"].dt.year):
        year_signal = year_part[year_part["pred_short"] == 1]
        year_metrics = _signal_metrics(
            year_part,
            year_signal,
            ret_col,
            label_col,
            f"predicted_short_label_year_{year}",
            None,
        )
        year_metrics.update({"horizon": horizon, "split": split_name, "model_tag": model_tag or "default"})
        rows.append(year_metrics)

    return rows


def train_short_models(
    splits: dict | None = None,
    model_tag: str | None = None,
    min_train_rows: int = 200,
    thresholds: list[float] | None = None,
    output_csv: str | None = None,
) -> dict[int, list[dict]]:
    cfg = get_config()
    short_cfg = cfg.get("binary_short_model", {})
    horizons = [int(h) for h in short_cfg.get("horizons", [5, 7])]
    thresholds = thresholds or [
        float(x) for x in short_cfg.get("probability_thresholds", [0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    ]

    df = load_dataset()
    if df.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")

    features = feature_columns(df)
    if not cfg.get("features", {}).get("use_guba_sentiment", False):
        features = [col for col in features if col not in GUBA_SENTIMENT_FEATURES]
    if not features:
        raise RuntimeError("No feature columns found in dataset.")

    model_dir = project_path(cfg["paths"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    effective_splits = dict(splits or cfg["model"]["splits"])
    train, valid, test = _split(df, effective_splits)
    logger.info(
        "train binary short split rows train=%s valid=%s test=%s splits=%s model_tag=%s",
        len(train),
        len(valid),
        len(test),
        effective_splits,
        model_tag or "default",
    )

    all_rows = []
    output: dict[int, list[dict]] = {}
    for horizon in horizons:
        ret_col = f"future_ret_{horizon}d"
        label_col = f"short_label_{horizon}d"
        if label_col not in df.columns:
            raise RuntimeError(f"{label_col} is missing. Run build_dataset first to generate binary short labels.")

        train_h = train.dropna(subset=[ret_col, label_col]).copy()
        valid_h = valid.dropna(subset=[ret_col, label_col]).copy()
        test_h = test.dropna(subset=[ret_col, label_col]).copy()
        if len(train_h) < min_train_rows:
            raise RuntimeError(f"Not enough training rows for short {horizon}d: {len(train_h)}")

        usable_features = [col for col in features if train_h[col].notna().any()]
        dropped_features = [col for col in features if col not in usable_features]
        if dropped_features:
            logger.warning("drop all-missing training features horizon=%sd features=%s", horizon, dropped_features)

        y_train = train_h[label_col].astype(int)
        if y_train.nunique() < 2:
            raise RuntimeError(f"short {horizon}d training label has only one class.")

        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.03,
                        num_leaves=31,
                        random_state=cfg["model"].get("random_state", 42),
                        class_weight="balanced",
                        verbose=-1,
                    ),
                ),
            ]
        )
        model.fit(train_h[usable_features], y_train)
        bundle = {
            "classifier": model,
            "features": usable_features,
            "horizon": horizon,
            "label_col": label_col,
            "ret_col": ret_col,
            "model_type": "binary_short",
            "splits": effective_splits,
        }
        joblib.dump(bundle, _model_path(model_dir, horizon, model_tag))

        rows = []
        rows.extend(_evaluate_part(valid_h, model, usable_features, horizon, "valid", thresholds, model_tag))
        rows.extend(_evaluate_part(test_h, model, usable_features, horizon, "test", thresholds, model_tag))
        output[horizon] = rows
        all_rows.extend(rows)
        logger.info("trained binary short %sd rows=%s", horizon, len(rows))

    metrics_name = f"short_metrics_{model_tag}.joblib" if model_tag else "short_metrics.joblib"
    joblib.dump(output, model_dir / metrics_name)

    if output_csv:
        out_path = project_path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved short model report path={out_path} rows={len(all_rows)}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Train binary short LightGBM models and evaluate short signal quality.")
    parser.add_argument("--train-start")
    parser.add_argument("--train-end")
    parser.add_argument("--valid-start")
    parser.add_argument("--valid-end")
    parser.add_argument("--test-start")
    parser.add_argument("--model-tag")
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--thresholds", help="Comma-separated short probability thresholds.")
    parser.add_argument("--output-csv", help="Optional CSV output path, e.g. data/reports/short_signal_quality.csv.")
    args = parser.parse_args()
    split_values = {
        "train_start": args.train_start,
        "train_end": args.train_end,
        "valid_start": args.valid_start,
        "valid_end": args.valid_end,
        "test_start": args.test_start,
    }
    splits = {key: value for key, value in split_values.items() if value}
    if splits and set(splits) != set(split_values):
        missing = sorted(set(split_values) - set(splits))
        raise SystemExit(f"Near-term split override requires all split dates. Missing: {missing}")

    result = train_short_models(
        splits=splits or None,
        model_tag=args.model_tag,
        min_train_rows=args.min_train_rows,
        thresholds=_parse_csv_floats(args.thresholds) if args.thresholds else None,
        output_csv=args.output_csv,
    )
    for horizon, rows in result.items():
        print(f"\n=== short {horizon}d ===")
        df = pd.DataFrame(rows)
        cols = [
            "split",
            "rule",
            "threshold",
            "signal_count",
            "coverage",
            "short_precision",
            "short_recall",
            "negative_ret_rate",
            "avg_future_ret",
            "avg_short_strategy_ret",
            "median_short_strategy_ret",
            "avg_short_proba",
        ]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
