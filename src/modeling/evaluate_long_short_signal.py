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


def _model_path(model_dir: Path, horizon: int, model_tag: str | None) -> Path:
    tag_part = f"_{model_tag}" if model_tag else ""
    return model_dir / f"lgbm{tag_part}_{horizon}d.joblib"


def _infer_direction_labels(part: pd.DataFrame, label_col: str, ret_col: str, classes: list[str]) -> tuple[str, str]:
    means = part.groupby(label_col)[ret_col].mean().reindex(classes).dropna()
    if len(means) < 2:
        raise RuntimeError(f"Cannot infer long/short labels from {label_col}.")
    return str(means.idxmax()), str(means.idxmin())


def _safe_mean(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.mean())


def _safe_median(series: pd.Series) -> float | None:
    if series.empty:
        return None
    return float(series.median())


def _long_short_metrics(
    part: pd.DataFrame,
    signal: pd.DataFrame,
    ret_col: str,
    rule: str,
    threshold: float | None,
) -> dict:
    if signal.empty:
        return {
            "rule": rule,
            "threshold": threshold,
            "signal_count": 0,
            "coverage": 0.0,
            "long_count": 0,
            "short_count": 0,
            "direction_accuracy": None,
            "long_precision": None,
            "short_precision": None,
            "avg_strategy_ret": None,
            "median_strategy_ret": None,
            "avg_future_ret_when_long": None,
            "avg_future_ret_when_short": None,
            "avg_direction_proba": None,
        }

    strategy_ret = signal["position"] * signal[ret_col]
    long_signal = signal[signal["position"] == 1]
    short_signal = signal[signal["position"] == -1]
    return {
        "rule": rule,
        "threshold": threshold,
        "signal_count": int(len(signal)),
        "coverage": float(len(signal) / len(part)) if len(part) else 0.0,
        "long_count": int(len(long_signal)),
        "short_count": int(len(short_signal)),
        "direction_accuracy": float((strategy_ret > 0).mean()),
        "long_precision": _safe_mean(long_signal[ret_col] > 0),
        "short_precision": _safe_mean(short_signal[ret_col] < 0),
        "avg_strategy_ret": float(strategy_ret.mean()),
        "median_strategy_ret": float(strategy_ret.median()),
        "avg_future_ret_when_long": _safe_mean(long_signal[ret_col]),
        "avg_future_ret_when_short": _safe_mean(short_signal[ret_col]),
        "avg_direction_proba": _safe_mean(signal["direction_proba"]) if "direction_proba" in signal else None,
    }


def evaluate_long_short_signal(
    model_tag: str | None = None,
    test_start: str | None = None,
    thresholds: list[float] | None = None,
    output_csv: str | None = None,
) -> dict[int, list[dict]]:
    cfg = get_config()
    thresholds = thresholds or [0.4, 0.45, 0.5, 0.55, 0.6]
    df = load_dataset()
    if df.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")

    split_test_start = test_start or cfg["model"]["splits"]["test_start"]
    test = df[df["trade_date"] >= split_test_start].copy()
    if test.empty:
        raise RuntimeError(f"No test rows found since {split_test_start}.")

    model_dir = project_path(cfg["paths"]["models"])
    output: dict[int, list[dict]] = {}
    all_rows = []
    for horizon in cfg["model"]["horizons"]:
        path = _model_path(model_dir, horizon, model_tag)
        if not path.exists():
            print(f"WARNING: model file missing horizon={horizon}d path={path}", file=sys.stderr)
            continue

        bundle = joblib.load(path)
        features = bundle["features"]
        ret_col = f"future_ret_{horizon}d"
        label_col = f"label_{horizon}d"
        part = test.dropna(subset=[ret_col, label_col]).copy()
        if part.empty:
            continue

        missing_features = [col for col in features if col not in part.columns]
        for col in missing_features:
            part[col] = 0

        classes = [str(item) for item in bundle["label_encoder"].classes_]
        long_label, short_label = _infer_direction_labels(part, label_col, ret_col, classes)
        class_index = {label: index for index, label in enumerate(classes)}

        proba = bundle["classifier"].predict_proba(part[features])
        pred_cls = bundle["label_encoder"].inverse_transform(bundle["classifier"].predict(part[features]))
        pred_ret = bundle["regressor"].predict(part[features])
        part["pred_label"] = pred_cls
        part["pred_ret"] = pred_ret
        part["long_proba"] = proba[:, class_index[long_label]]
        part["short_proba"] = proba[:, class_index[short_label]]
        part["position"] = 0
        part.loc[part["pred_label"] == long_label, "position"] = 1
        part.loc[part["pred_label"] == short_label, "position"] = -1
        part["direction_proba"] = part[["long_proba", "short_proba"]].max(axis=1)

        rows = []
        baseline_long = part.copy()
        baseline_long["position"] = 1
        baseline_long["direction_proba"] = None
        rows.append(_long_short_metrics(part, baseline_long, ret_col, "baseline_always_long", None))

        baseline_short = part.copy()
        baseline_short["position"] = -1
        baseline_short["direction_proba"] = None
        rows.append(_long_short_metrics(part, baseline_short, ret_col, "baseline_always_short", None))

        predicted_direction = part[part["position"] != 0].copy()
        rows.append(_long_short_metrics(part, predicted_direction, ret_col, "predicted_long_short_label", None))

        for threshold in thresholds:
            signal = part[(part["position"] != 0) & (part["direction_proba"] >= threshold)].copy()
            rows.append(_long_short_metrics(part, signal, ret_col, "long_short_probability", threshold))

        for year, year_part in part.groupby(part["trade_date"].dt.year):
            year_signal = year_part[year_part["position"] != 0].copy()
            rows.append(
                _long_short_metrics(
                    year_part,
                    year_signal,
                    ret_col,
                    f"predicted_long_short_label_year_{year}",
                    None,
                )
            )

        for row in rows:
            row.update(
                {
                    "horizon": horizon,
                    "model_tag": model_tag or "default",
                    "test_start": split_test_start,
                    "long_label": long_label,
                    "short_label": short_label,
                }
            )
        output[horizon] = rows
        all_rows.extend(rows)

    if output_csv:
        out_path = project_path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved long-short signal report path={out_path} rows={len(all_rows)}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate long-short signal quality from three-class models.")
    parser.add_argument("--model-tag", help="Model tag used by train_lgbm, e.g. news2016_202505.")
    parser.add_argument("--test-start", help="Override test start date, YYYY-MM-DD.")
    parser.add_argument(
        "--thresholds",
        default="0.40,0.45,0.50,0.55,0.60",
        help="Comma-separated long/short direction probability thresholds.",
    )
    parser.add_argument("--output-csv", help="Optional CSV output path.")
    args = parser.parse_args()
    result = evaluate_long_short_signal(
        model_tag=args.model_tag,
        test_start=args.test_start,
        thresholds=_parse_thresholds(args.thresholds),
        output_csv=args.output_csv,
    )
    for horizon, rows in result.items():
        print(f"\n=== long-short {horizon}d ===")
        df = pd.DataFrame(rows)
        cols = [
            "rule",
            "threshold",
            "signal_count",
            "coverage",
            "long_count",
            "short_count",
            "direction_accuracy",
            "long_precision",
            "short_precision",
            "avg_strategy_ret",
            "median_strategy_ret",
            "avg_direction_proba",
        ]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
