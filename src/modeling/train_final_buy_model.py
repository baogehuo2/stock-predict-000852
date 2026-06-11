from __future__ import annotations

import argparse

import joblib

from src.common.config import get_config, project_path
from src.common.logger import get_logger
from src.modeling.buy_feature_set import DEFAULT_BUY_FEATURE_VERSION, select_buy_features
from src.modeling.data import load_dataset
from src.modeling.walk_forward_buy_lgbm import _fit_model


logger = get_logger(__name__)


def train_final_buy_model(output_file: str | None = None, min_train_rows: int = 200) -> str:
    cfg = get_config()
    buy_cfg = cfg.get("binary_buy_model", {})
    horizons = [int(value) for value in buy_cfg.get("horizons", [7])]
    if horizons != [7]:
        raise RuntimeError(f"Model 1.0 final training requires only horizon 7, got {horizons}.")
    data = load_dataset()
    if data.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    features = select_buy_features(data, buy_cfg)
    train = data.dropna(subset=["future_ret_7d", "buy_label_7d"]).sort_values("trade_date").copy()
    if len(train) < min_train_rows:
        raise RuntimeError(f"Not enough rows for final Buy model: {len(train)}")
    random_state = int(cfg["model"].get("random_state", 42))
    model, usable_features = _fit_model(train, features, "buy_label_7d", random_state)
    model_version = str(buy_cfg.get("model_version", "buy-signal-v1.0"))
    feature_version = str(buy_cfg.get("feature_version", DEFAULT_BUY_FEATURE_VERSION))
    label_threshold = float(buy_cfg.get("label_thresholds", {}).get("7", 0.0))
    bundle = {
        "classifier": model,
        "features": usable_features,
        "horizon": 7,
        "label_col": "buy_label_7d",
        "ret_col": "future_ret_7d",
        "model_type": "binary_buy",
        "model_version": model_version,
        "feature_version": feature_version,
        "label_threshold": label_threshold,
        "trained_from": train["trade_date"].min().strftime("%Y-%m-%d"),
        "trained_through": train["trade_date"].max().strftime("%Y-%m-%d"),
        "train_rows": len(train),
    }
    output = output_file or str(
        buy_cfg.get("final_model_file", "models/buy_lgbm_buy_signal_v1_7d.joblib")
    )
    path = project_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    logger.info(
        "trained final Buy model path=%s rows=%s through=%s features=%s",
        path,
        len(train),
        bundle["trained_through"],
        len(usable_features),
    )
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final full-history 7-day Buy model.")
    parser.add_argument("--output-file")
    parser.add_argument("--min-train-rows", type=int, default=200)
    args = parser.parse_args()
    print(train_final_buy_model(args.output_file, args.min_train_rows))


if __name__ == "__main__":
    main()
