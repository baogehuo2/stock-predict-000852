from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.logger import get_logger
from src.modeling.data import feature_columns, load_dataset


disable_broken_dask_autoload()
from lightgbm import LGBMClassifier, LGBMRegressor


logger = get_logger(__name__)


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = get_config()["model"]["splits"]
    train = df[(df["trade_date"] >= splits["train_start"]) & (df["trade_date"] <= splits["train_end"])]
    valid = df[(df["trade_date"] >= splits["valid_start"]) & (df["trade_date"] <= splits["valid_end"])]
    test = df[df["trade_date"] >= splits["test_start"]]
    return train, valid, test


def train_models() -> dict:
    cfg = get_config()
    df = load_dataset()
    if df.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    features = feature_columns(df)
    if not features:
        raise RuntimeError("No feature columns found in dataset.")

    model_dir = project_path(cfg["paths"]["models"])
    model_dir.mkdir(parents=True, exist_ok=True)
    train, valid, test = _split(df)
    metrics: dict = {}
    for horizon in cfg["model"]["horizons"]:
        ret_col = f"future_ret_{horizon}d"
        label_col = f"label_{horizon}d"
        train_h = train.dropna(subset=[ret_col, label_col]).copy()
        valid_h = valid.dropna(subset=[ret_col, label_col]).copy()
        test_h = test.dropna(subset=[ret_col, label_col]).copy()
        if len(train_h) < 200:
            raise RuntimeError(f"Not enough training rows for {horizon}d: {len(train_h)}")

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
        cls_model.fit(train_h[features], y_cls)
        reg_model.fit(train_h[features], train_h[ret_col])

        bundle = {"classifier": cls_model, "regressor": reg_model, "label_encoder": encoder, "features": features}
        joblib.dump(bundle, model_dir / f"lgbm_{horizon}d.joblib")

        metrics[horizon] = {}
        for name, part in [("valid", valid_h), ("test", test_h)]:
            if part.empty:
                continue
            pred_cls = encoder.inverse_transform(cls_model.predict(part[features]))
            pred_ret = reg_model.predict(part[features])
            metrics[horizon][name] = {
                "accuracy": float(accuracy_score(part[label_col], pred_cls)),
                "mae": float(mean_absolute_error(part[ret_col], pred_ret)),
            }
        logger.info("trained %sd metrics=%s", horizon, metrics[horizon])

    joblib.dump(metrics, model_dir / "metrics.joblib")
    return metrics


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(train_models())


if __name__ == "__main__":
    main()
