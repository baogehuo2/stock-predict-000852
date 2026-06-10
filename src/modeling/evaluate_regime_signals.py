from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.modeling.data import load_dataset
from src.modeling.market_regime import assign_evaluation_regime


disable_broken_dask_autoload()


def _parse_thresholds(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def _model_path(model_dir: Path, side: str, horizon: int, model_tag: str | None) -> Path:
    tag_part = f"_{model_tag}" if model_tag else ""
    return model_dir / f"{side}_lgbm{tag_part}_{horizon}d.joblib"


def _positive_proba(bundle: dict, part: pd.DataFrame) -> pd.Series:
    features = bundle["features"]
    data = part.copy()
    for col in features:
        if col not in data:
            data[col] = 0
    model = bundle["classifier"]
    classes = list(model.named_steps["model"].classes_)
    positive_index = classes.index(1)
    return pd.Series(model.predict_proba(data[features])[:, positive_index], index=data.index)


def _assign_sample_scope(trade_date: pd.Series, splits: dict) -> pd.Series:
    dates = pd.to_datetime(trade_date)
    scope = pd.Series("outside", index=trade_date.index, dtype="object")
    scope.loc[(dates >= pd.Timestamp(splits["train_start"])) & (dates <= pd.Timestamp(splits["train_end"]))] = "train"
    scope.loc[(dates >= pd.Timestamp(splits["valid_start"])) & (dates <= pd.Timestamp(splits["valid_end"]))] = "valid"
    scope.loc[dates >= pd.Timestamp(splits["test_start"])] = "test"
    return scope


def evaluate_regime_signals(
    model_tag: str | None = None,
    start_date: str = "2021-01-01",
    thresholds: list[float] | None = None,
    output_csv: str | None = None,
) -> list[dict]:
    cfg = get_config()
    thresholds = thresholds or [0.5, 0.55, 0.6]
    df = load_dataset()
    if df.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    part = df[df["trade_date"] >= pd.Timestamp(start_date)].copy()
    part["evaluation_regime"] = assign_evaluation_regime(part["trade_date"])
    part = part[part["evaluation_regime"].isin(["bull", "bear"])]

    model_dir = project_path(cfg["paths"]["models"])
    horizons = sorted(
        set(int(h) for h in cfg.get("binary_buy_model", {}).get("horizons", [5, 7]))
        & set(int(h) for h in cfg.get("binary_short_model", {}).get("horizons", [5, 7]))
    )
    rows = []
    for horizon in horizons:
        ret_col = f"future_ret_{horizon}d"
        buy_label_col = f"buy_label_{horizon}d"
        short_label_col = f"short_label_{horizon}d"
        horizon_part = part.dropna(subset=[ret_col, buy_label_col, short_label_col]).copy()
        buy_bundle = joblib.load(_model_path(model_dir, "buy", horizon, model_tag))
        short_bundle = joblib.load(_model_path(model_dir, "short", horizon, model_tag))
        buy_splits = buy_bundle.get("splits")
        short_splits = short_bundle.get("splits")
        if not buy_splits or not short_splits:
            raise RuntimeError("Model bundle has no split metadata. Retrain buy and short models first.")
        if buy_splits != short_splits:
            raise RuntimeError("Buy and short model split metadata do not match.")
        horizon_part["sample_scope"] = _assign_sample_scope(horizon_part["trade_date"], buy_splits)
        horizon_part["buy_proba"] = _positive_proba(buy_bundle, horizon_part)
        horizon_part["short_proba"] = _positive_proba(short_bundle, horizon_part)

        for (sample_scope, regime), regime_part in horizon_part.groupby(["sample_scope", "evaluation_regime"]):
            for threshold in thresholds:
                for side in ["buy", "short"]:
                    signal = regime_part[regime_part[f"{side}_proba"] >= threshold]
                    if side == "buy":
                        correct = signal[ret_col] > 0
                        strategy_ret = signal[ret_col]
                    else:
                        correct = signal[ret_col] < 0
                        strategy_ret = -signal[ret_col]
                    rows.append(
                        {
                            "sample_scope": sample_scope,
                            "regime": regime,
                            "horizon": horizon,
                            "side": side,
                            "threshold": threshold,
                            "regime_rows": int(len(regime_part)),
                            "signal_count": int(len(signal)),
                            "coverage": float(len(signal) / len(regime_part)) if len(regime_part) else 0.0,
                            "precision": float(correct.mean()) if len(signal) else None,
                            "avg_strategy_ret": float(strategy_ret.mean()) if len(signal) else None,
                            "median_strategy_ret": float(strategy_ret.median()) if len(signal) else None,
                            "avg_proba": float(signal[f"{side}_proba"].mean()) if len(signal) else None,
                        }
                    )

    if output_csv:
        path = project_path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved regime signal report path={path} rows={len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate buy and short signals by configured bull/bear periods.")
    parser.add_argument("--model-tag")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--thresholds", default="0.50,0.55,0.60")
    parser.add_argument("--output-csv")
    args = parser.parse_args()
    rows = evaluate_regime_signals(
        model_tag=args.model_tag,
        start_date=args.start_date,
        thresholds=_parse_thresholds(args.thresholds),
        output_csv=args.output_csv,
    )
    cols = [
        "sample_scope",
        "regime",
        "horizon",
        "side",
        "threshold",
        "regime_rows",
        "signal_count",
        "coverage",
        "precision",
        "avg_strategy_ret",
        "median_strategy_ret",
    ]
    print(pd.DataFrame(rows)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
