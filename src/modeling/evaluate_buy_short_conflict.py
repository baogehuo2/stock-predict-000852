from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.modeling.data import load_dataset


disable_broken_dask_autoload()


def _parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if item:
            thresholds.append(float(item))
    return sorted(set(thresholds))


def _binary_model_path(model_dir: Path, side: str, horizon: int, model_tag: str | None) -> Path:
    tag_part = f"_{model_tag}" if model_tag else ""
    return model_dir / f"{side}_lgbm{tag_part}_{horizon}d.joblib"


def _predict_positive_proba(bundle: dict, part: pd.DataFrame) -> pd.Series:
    features = bundle["features"]
    missing_features = [col for col in features if col not in part.columns]
    if missing_features:
        part = part.copy()
        for col in missing_features:
            part[col] = 0
    model = bundle["classifier"]
    classes = list(model.named_steps["model"].classes_)
    positive_index = classes.index(1)
    return pd.Series(model.predict_proba(part[features])[:, positive_index], index=part.index)


def evaluate_buy_short_conflict(
    model_tag: str | None = None,
    test_start: str | None = None,
    thresholds: list[float] | None = None,
    output_csv: str | None = None,
    detail_csv: str | None = None,
) -> list[dict]:
    cfg = get_config()
    thresholds = thresholds or [0.5, 0.55, 0.6]
    horizons = sorted(
        set(int(h) for h in cfg.get("binary_buy_model", {}).get("horizons", [5, 7]))
        & set(int(h) for h in cfg.get("binary_short_model", {}).get("horizons", [5, 7]))
    )
    df = load_dataset()
    if df.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")

    split_test_start = test_start or cfg["model"]["splits"]["test_start"]
    test = df[df["trade_date"] >= split_test_start].copy()
    if test.empty:
        raise RuntimeError(f"No test rows found since {split_test_start}.")

    model_dir = project_path(cfg["paths"]["models"])
    rows = []
    detail_rows = []
    for horizon in horizons:
        buy_path = _binary_model_path(model_dir, "buy", horizon, model_tag)
        short_path = _binary_model_path(model_dir, "short", horizon, model_tag)
        if not buy_path.exists():
            print(f"WARNING: buy model missing horizon={horizon}d path={buy_path}", file=sys.stderr)
            continue
        if not short_path.exists():
            print(f"WARNING: short model missing horizon={horizon}d path={short_path}", file=sys.stderr)
            continue

        buy_bundle = joblib.load(buy_path)
        short_bundle = joblib.load(short_path)
        ret_col = f"future_ret_{horizon}d"
        buy_label_col = f"buy_label_{horizon}d"
        short_label_col = f"short_label_{horizon}d"
        part = test.dropna(subset=[ret_col, buy_label_col, short_label_col]).copy()
        if part.empty:
            continue

        part["buy_proba"] = _predict_positive_proba(buy_bundle, part)
        part["short_proba"] = _predict_positive_proba(short_bundle, part)
        for threshold in thresholds:
            buy_signal = part["buy_proba"] >= threshold
            short_signal = part["short_proba"] >= threshold
            conflict = part[buy_signal & short_signal].copy()
            rows.append(
                {
                    "horizon": horizon,
                    "model_tag": model_tag or "default",
                    "test_start": split_test_start,
                    "threshold": threshold,
                    "test_rows": int(len(part)),
                    "buy_signal_count": int(buy_signal.sum()),
                    "short_signal_count": int(short_signal.sum()),
                    "conflict_count": int(len(conflict)),
                    "conflict_rate_in_rows": float(len(conflict) / len(part)) if len(part) else 0.0,
                    "conflict_rate_in_buy_signals": float(len(conflict) / buy_signal.sum()) if buy_signal.sum() else None,
                    "conflict_rate_in_short_signals": float(len(conflict) / short_signal.sum()) if short_signal.sum() else None,
                    "conflict_avg_future_ret": float(conflict[ret_col].mean()) if len(conflict) else None,
                    "conflict_median_future_ret": float(conflict[ret_col].median()) if len(conflict) else None,
                    "conflict_actual_up_rate": float((conflict[ret_col] > 0).mean()) if len(conflict) else None,
                    "conflict_actual_down_rate": float((conflict[ret_col] < 0).mean()) if len(conflict) else None,
                }
            )
            for item in conflict.itertuples(index=False):
                detail_rows.append(
                    {
                        "horizon": horizon,
                        "model_tag": model_tag or "default",
                        "threshold": threshold,
                        "trade_date": item.trade_date,
                        "buy_proba": item.buy_proba,
                        "short_proba": item.short_proba,
                        ret_col: getattr(item, ret_col),
                    }
                )

    if output_csv:
        out_path = project_path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved buy/short conflict summary path={out_path} rows={len(rows)}")
    if detail_csv:
        detail_path = project_path(detail_csv)
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(detail_rows).to_csv(detail_path, index=False, encoding="utf-8-sig")
        print(f"saved buy/short conflict details path={detail_path} rows={len(detail_rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate same-day conflicts between binary buy and short models.")
    parser.add_argument("--model-tag")
    parser.add_argument("--test-start")
    parser.add_argument("--thresholds", default="0.50,0.55,0.60")
    parser.add_argument("--output-csv")
    parser.add_argument("--detail-csv")
    args = parser.parse_args()
    rows = evaluate_buy_short_conflict(
        model_tag=args.model_tag,
        test_start=args.test_start,
        thresholds=_parse_thresholds(args.thresholds),
        output_csv=args.output_csv,
        detail_csv=args.detail_csv,
    )
    if not rows:
        print("No conflict rows. Check that buy and short models exist.")
        return
    cols = [
        "horizon",
        "threshold",
        "test_rows",
        "buy_signal_count",
        "short_signal_count",
        "conflict_count",
        "conflict_rate_in_rows",
        "conflict_avg_future_ret",
        "conflict_actual_up_rate",
        "conflict_actual_down_rate",
    ]
    print(pd.DataFrame(rows)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
