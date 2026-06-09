from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import read_sql
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.modeling.data import load_dataset


disable_broken_dask_autoload()


def _parse_thresholds(value: str) -> list[float]:
    thresholds = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        thresholds.append(float(item))
    return sorted(set(thresholds))


def _model_path(model_dir: Path, horizon: int, model_tag: str | None) -> Path:
    tag_part = f"_{model_tag}" if model_tag else ""
    return model_dir / f"lgbm{tag_part}_{horizon}d.joblib"


def _infer_up_label(part: pd.DataFrame, label_col: str, ret_col: str, classes: list[str]) -> str:
    means = part.groupby(label_col)[ret_col].mean()
    available = means.reindex(classes).dropna()
    if available.empty:
        return classes[-1]
    return str(available.idxmax())


def _safe_rate(mask: pd.Series) -> float | None:
    if len(mask) == 0:
        return None
    return float(mask.mean())


def _attach_price_baseline_features(df: pd.DataFrame, target_index: str) -> pd.DataFrame:
    market = read_sql(
        "SELECT trade_date, close FROM market_index_daily "
        "WHERE index_code=:target ORDER BY trade_date",
        {"target": target_index},
    )
    if market.empty:
        return df

    market["trade_date"] = pd.to_datetime(market["trade_date"])
    close = pd.to_numeric(market["close"], errors="coerce")
    market["baseline_ret_20d"] = close.pct_change(20)
    return df.merge(market[["trade_date", "baseline_ret_20d"]], on="trade_date", how="left")


def _signal_metrics(
    part: pd.DataFrame,
    signal: pd.DataFrame,
    ret_col: str,
    label_col: str,
    up_label: str,
) -> dict:
    if signal.empty:
        return {
            "signal_count": 0,
            "coverage": 0.0,
            "actual_up_rate": None,
            "positive_ret_rate": None,
            "avg_future_ret": None,
            "median_future_ret": None,
            "avg_pred_ret": None,
            "avg_up_proba": None,
        }
    return {
        "signal_count": int(len(signal)),
        "coverage": float(len(signal) / len(part)) if len(part) else 0.0,
        "actual_up_rate": _safe_rate(signal[label_col] == up_label),
        "positive_ret_rate": _safe_rate(signal[ret_col] > 0),
        "avg_future_ret": float(signal[ret_col].mean()),
        "median_future_ret": float(signal[ret_col].median()),
        "avg_pred_ret": float(signal["pred_ret"].mean()),
        "avg_up_proba": float(signal["up_proba"].mean()),
    }


def _append_rule_metrics(
    rows: list[dict],
    part: pd.DataFrame,
    signal: pd.DataFrame,
    ret_col: str,
    label_col: str,
    up_label: str,
    horizon: int,
    model_tag: str | None,
    split_test_start: str,
    rule: str,
    threshold: float | None = None,
) -> None:
    metrics = _signal_metrics(part, signal, ret_col, label_col, up_label)
    metrics.update(
        {
            "horizon": horizon,
            "model_tag": model_tag or "default",
            "test_start": split_test_start,
            "up_label": up_label,
            "rule": rule,
            "threshold": threshold,
        }
    )
    rows.append(metrics)


def _simple_baseline_signals(part: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    rules = [("always_up", part)]

    if "f_ma20_gap" in part.columns:
        rules.append(("ma20_trend_up", part[part["f_ma20_gap"] > 0]))
    if "f_ret_5d" in part.columns:
        rules.append(("momentum_5d_positive", part[part["f_ret_5d"] > 0]))
    if "baseline_ret_20d" in part.columns:
        rules.append(("ret_20d_positive", part[part["baseline_ret_20d"] > 0]))

    return rules


def evaluate_signal_quality(
    model_tag: str | None = None,
    test_start: str | None = None,
    thresholds: list[float] | None = None,
    output_csv: str | None = None,
) -> dict[int, list[dict]]:
    cfg = get_config()
    thresholds = thresholds or [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    df = load_dataset()
    if df.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    df = _attach_price_baseline_features(df, cfg["project"]["target_index"])

    split_test_start = test_start or cfg["model"]["splits"]["test_start"]
    test = df[df["trade_date"] >= split_test_start].copy()
    if test.empty:
        raise RuntimeError(f"No test rows found since {split_test_start}.")

    model_dir = project_path(cfg["paths"]["models"])
    all_rows = []
    output: dict[int, list[dict]] = {}
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
        if missing_features:
            print(f"WARNING: missing features horizon={horizon}d features={missing_features}", file=sys.stderr)
            for col in missing_features:
                part[col] = 0

        classes = [str(item) for item in bundle["label_encoder"].classes_]
        up_label = _infer_up_label(part, label_col, ret_col, classes)
        up_index = classes.index(up_label)

        proba = bundle["classifier"].predict_proba(part[features])
        pred_cls = bundle["label_encoder"].inverse_transform(bundle["classifier"].predict(part[features]))
        pred_ret = bundle["regressor"].predict(part[features])
        part["pred_label"] = pred_cls
        part["pred_ret"] = pred_ret
        part["up_proba"] = proba[:, up_index]

        rows = []
        baseline = _signal_metrics(part, part, ret_col, label_col, up_label)
        baseline.update(
            {
                "horizon": horizon,
                "model_tag": model_tag or "default",
                "test_start": split_test_start,
                "up_label": up_label,
                "rule": "baseline_all_rows",
                "threshold": None,
            }
        )
        rows.append(baseline)

        for rule, signal in _simple_baseline_signals(part):
            _append_rule_metrics(
                rows,
                part,
                signal,
                ret_col,
                label_col,
                up_label,
                horizon,
                model_tag,
                split_test_start,
                rule,
            )

        predicted_up = part[part["pred_label"] == up_label]
        pred_up_metrics = _signal_metrics(part, predicted_up, ret_col, label_col, up_label)
        pred_up_metrics.update(
            {
                "horizon": horizon,
                "model_tag": model_tag or "default",
                "test_start": split_test_start,
                "up_label": up_label,
                "rule": "predicted_up_label",
                "threshold": None,
            }
        )
        rows.append(pred_up_metrics)

        for threshold in thresholds:
            signal = part[part["up_proba"] >= threshold]
            metrics = _signal_metrics(part, signal, ret_col, label_col, up_label)
            metrics.update(
                {
                    "horizon": horizon,
                    "model_tag": model_tag or "default",
                    "test_start": split_test_start,
                    "up_label": up_label,
                    "rule": "up_probability",
                    "threshold": threshold,
                }
            )
            rows.append(metrics)

        for year, year_part in part.groupby(part["trade_date"].dt.year):
            predicted_up_year = year_part[year_part["pred_label"] == up_label]
            year_metrics = _signal_metrics(year_part, predicted_up_year, ret_col, label_col, up_label)
            year_metrics.update(
                {
                    "horizon": horizon,
                    "model_tag": model_tag or "default",
                    "test_start": split_test_start,
                    "up_label": up_label,
                    "rule": f"predicted_up_label_year_{year}",
                    "threshold": None,
                }
            )
            rows.append(year_metrics)

        output[horizon] = rows
        all_rows.extend(rows)

    if output_csv:
        out_path = project_path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved signal quality report path={out_path} rows={len(all_rows)}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model signal quality by upward probability thresholds.")
    parser.add_argument("--model-tag", help="Model tag used by train_lgbm, e.g. news2016_202505.")
    parser.add_argument("--test-start", help="Override test start date, YYYY-MM-DD.")
    parser.add_argument(
        "--thresholds",
        default="0.35,0.40,0.45,0.50,0.55,0.60",
        help="Comma-separated upward probability thresholds.",
    )
    parser.add_argument("--output-csv", help="Optional CSV output path, e.g. data/reports/signal_quality.csv.")
    args = parser.parse_args()
    result = evaluate_signal_quality(
        model_tag=args.model_tag,
        test_start=args.test_start,
        thresholds=_parse_thresholds(args.thresholds),
        output_csv=args.output_csv,
    )
    for horizon, rows in result.items():
        print(f"\n=== {horizon}d ===")
        df = pd.DataFrame(rows)
        cols = [
            "rule",
            "threshold",
            "signal_count",
            "coverage",
            "actual_up_rate",
            "positive_ret_rate",
            "avg_future_ret",
            "median_future_ret",
            "avg_up_proba",
        ]
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
