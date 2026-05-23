from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.logger import get_logger
from src.modeling.data import feature_columns, load_dataset


disable_broken_dask_autoload()
from lightgbm import LGBMClassifier, LGBMRegressor


logger = get_logger(__name__)

GUBA_SENTIMENT_FEATURES = {
    "f_heat_score",
    "f_heat_zscore_20d",
    "f_sentiment_score",
    "f_disagreement",
}


def _label_metrics(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    pred_counts = pd.Series(y_pred).value_counts()
    return {
        label: {
            "pred_accuracy": float(precision[i]),
            "actual_recall": float(recall[i]),
            "precision": float(precision[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "pred_count": int(pred_counts.get(label, 0)),
        }
        for i, label in enumerate(labels)
    }


def _split(df: pd.DataFrame, splits: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = splits or get_config()["model"]["splits"]
    train = df[(df["trade_date"] >= splits["train_start"]) & (df["trade_date"] <= splits["train_end"])]
    valid = df[(df["trade_date"] >= splits["valid_start"]) & (df["trade_date"] <= splits["valid_end"])]
    test = df[df["trade_date"] >= splits["test_start"]]
    return train, valid, test


def train_models(
    splits: dict | None = None,
    model_tag: str | None = None,
    min_train_rows: int = 200,
) -> dict:
    cfg = get_config()
    df = load_dataset()
    if df.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    features = feature_columns(df)
    if not cfg.get("features", {}).get("use_guba_sentiment", False):
        before = len(features)
        features = [col for col in features if col not in GUBA_SENTIMENT_FEATURES]
        removed = before - len(features)
        if removed:
            logger.info("guba sentiment training features disabled removed=%s", removed)
    if not features:
        raise RuntimeError("No feature columns found in dataset.")

    model_dir = project_path(cfg["paths"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    train, valid, test = _split(df, splits)
    logger.info(
        "train split rows train=%s valid=%s test=%s splits=%s model_tag=%s",
        len(train),
        len(valid),
        len(test),
        splits or cfg["model"]["splits"],
        model_tag or "default",
    )
    metrics: dict = {}
    for horizon in cfg["model"]["horizons"]:
        ret_col = f"future_ret_{horizon}d"
        label_col = f"label_{horizon}d"
        train_h = train.dropna(subset=[ret_col, label_col]).copy()
        valid_h = valid.dropna(subset=[ret_col, label_col]).copy()
        test_h = test.dropna(subset=[ret_col, label_col]).copy()
        if len(train_h) < min_train_rows:
            raise RuntimeError(f"Not enough training rows for {horizon}d: {len(train_h)}")
        usable_features = [col for col in features if train_h[col].notna().any()]
        dropped_features = [col for col in features if col not in usable_features]
        if dropped_features:
            logger.warning("drop all-missing training features horizon=%sd features=%s", horizon, dropped_features)

        encoder = LabelEncoder()
        y_cls = encoder.fit_transform(train_h[label_col])
        cls_model = Pipeline(
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
        reg_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=300,
                        learning_rate=0.03,
                        num_leaves=31,
                        random_state=cfg["model"].get("random_state", 42),
                        verbose=-1,
                    ),
                ),
            ]
        )
        cls_model.fit(train_h[usable_features], y_cls)
        reg_model.fit(train_h[usable_features], train_h[ret_col])

        bundle = {"classifier": cls_model, "regressor": reg_model, "label_encoder": encoder, "features": usable_features}
        tag_part = f"_{model_tag}" if model_tag else ""
        joblib.dump(bundle, model_dir / f"lgbm{tag_part}_{horizon}d.joblib")

        metrics[horizon] = {}
        for name, part in [("valid", valid_h), ("test", test_h)]:
            if part.empty:
                continue
            pred_cls = encoder.inverse_transform(cls_model.predict(part[usable_features]))
            pred_ret = reg_model.predict(part[usable_features])
            labels = list(encoder.classes_)
            metrics[horizon][name] = {
                "accuracy": float(accuracy_score(part[label_col], pred_cls)),
                "mae": float(mean_absolute_error(part[ret_col], pred_ret)),
                "by_label": _label_metrics(part[label_col], pd.Series(pred_cls), labels),
            }
        logger.info("trained %sd metrics=%s", horizon, metrics[horizon])

    metrics_name = f"metrics_{model_tag}.joblib" if model_tag else "metrics.joblib"
    joblib.dump(metrics, model_dir / metrics_name)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start")
    parser.add_argument("--train-end")
    parser.add_argument("--valid-start")
    parser.add_argument("--valid-end")
    parser.add_argument("--test-start")
    parser.add_argument("--model-tag")
    parser.add_argument("--min-train-rows", type=int, default=200)
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
    print(train_models(splits=splits or None, model_tag=args.model_tag, min_train_rows=args.min_train_rows))


if __name__ == "__main__":
    main()
