from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.modeling.data import load_dataset


disable_broken_dask_autoload()


def _rank_ic(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank()))


def evaluate_models() -> dict:
    cfg = get_config()
    df = load_dataset()
    splits = cfg["model"]["splits"]
    test = df[df["trade_date"] >= splits["test_start"]].copy()
    model_dir = project_path(cfg["paths"]["models"])
    output = {}
    for horizon in cfg["model"]["horizons"]:
        bundle = joblib.load(model_dir / f"lgbm_{horizon}d.joblib")
        features = bundle["features"]
        ret_col = f"future_ret_{horizon}d"
        label_col = f"label_{horizon}d"
        part = test.dropna(subset=[ret_col, label_col])
        if part.empty:
            continue
        pred_cls = bundle["label_encoder"].inverse_transform(bundle["classifier"].predict(part[features]))
        pred_ret = pd.Series(bundle["regressor"].predict(part[features]), index=part.index)
        proba = bundle["classifier"].predict_proba(part[features])
        confidence = pd.Series(np.max(proba, axis=1), index=part.index)
        high = part[confidence >= cfg["model"].get("high_confidence_threshold", 0.6)]
        high_pred = pred_cls[confidence >= cfg["model"].get("high_confidence_threshold", 0.6)]
        output[horizon] = {
            "accuracy": float(accuracy_score(part[label_col], pred_cls)),
            "mae": float(mean_absolute_error(part[ret_col], pred_ret)),
            "rmse": float(mean_squared_error(part[ret_col], pred_ret) ** 0.5),
            "ic": float(pred_ret.corr(part[ret_col])),
            "rank_ic": _rank_ic(pred_ret, part[ret_col]),
            "high_confidence_count": int(len(high)),
            "high_confidence_accuracy": float(accuracy_score(high[label_col], high_pred)) if len(high) else None,
        }
    return output


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(evaluate_models())


if __name__ == "__main__":
    main()
