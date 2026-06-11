from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import get_config, project_path
from src.common.logger import get_logger
from src.modeling.data import feature_columns, load_dataset
from src.modeling.train_lgbm import GUBA_SENTIMENT_FEATURES
from src.modeling.walk_forward_buy_lgbm import _fit_model, _positive_probability


logger = get_logger(__name__)

PROBABILITY_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 1.000001]
PROBABILITY_LABELS = ["<0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", ">=0.80"]


def _parse_floats(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def _non_overlapping(signal: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    if signal.empty:
        return signal
    selected = []
    next_allowed_position = -1
    for idx, row in signal.sort_values("trade_date").iterrows():
        position = int(row["trade_pos"])
        if position >= next_allowed_position:
            selected.append(idx)
            next_allowed_position = position + horizon
    return signal.loc[selected]


def _signal_metrics(signal: pd.DataFrame, label_threshold: float, probability_threshold: float) -> dict:
    label_correct = signal["future_ret"] > label_threshold
    direction_correct = signal["future_ret"] > 0
    return {
        "label_threshold": label_threshold,
        "probability_threshold": probability_threshold,
        "signal_count": int(len(signal)),
        "label_precision": float(label_correct.mean()) if len(signal) else None,
        "actual_up_rate": float(direction_correct.mean()) if len(signal) else None,
        "avg_future_ret": float(signal["future_ret"].mean()) if len(signal) else None,
        "median_future_ret": float(signal["future_ret"].median()) if len(signal) else None,
        "avg_buy_proba": float(signal["buy_proba"].mean()) if len(signal) else None,
    }


def _build_reports(details: pd.DataFrame, probability_thresholds: list[float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparison_rows = []
    bucket_rows = []
    for (label_threshold, test_year), year_part in details.groupby(
        ["label_threshold", "test_year"], sort=True
    ):
        hard_gate = year_part[year_part["f_is_bull_trend"] == 1].copy()
        for probability_threshold in probability_thresholds:
            signal = hard_gate[hard_gate["buy_proba"] >= probability_threshold]
            independent = _non_overlapping(signal)
            row = _signal_metrics(independent, label_threshold, probability_threshold)
            row.update({"scope": "year", "test_year": int(test_year)})
            comparison_rows.append(row)

        hard_gate["probability_bucket"] = pd.cut(
            hard_gate["buy_proba"],
            bins=PROBABILITY_BINS,
            labels=PROBABILITY_LABELS,
            right=False,
        )
        for bucket in PROBABILITY_LABELS:
            bucket_signal = _non_overlapping(hard_gate[hard_gate["probability_bucket"] == bucket])
            row = _signal_metrics(bucket_signal, label_threshold, probability_threshold=float("nan"))
            row.update({"scope": "year", "test_year": int(test_year), "probability_bucket": bucket})
            bucket_rows.append(row)

    for label_threshold, threshold_part in details.groupby("label_threshold", sort=True):
        for probability_threshold in probability_thresholds:
            yearly_signals = []
            for _, year_part in threshold_part.groupby("test_year", sort=True):
                signal = year_part[
                    (year_part["f_is_bull_trend"] == 1)
                    & (year_part["buy_proba"] >= probability_threshold)
                ]
                yearly_signals.append(_non_overlapping(signal))
            independent = pd.concat(yearly_signals, ignore_index=True) if yearly_signals else threshold_part.iloc[0:0]
            row = _signal_metrics(independent, label_threshold, probability_threshold)
            row.update({"scope": "aggregate", "test_year": None})
            comparison_rows.append(row)

        hard_gate = threshold_part[threshold_part["f_is_bull_trend"] == 1].copy()
        hard_gate["probability_bucket"] = pd.cut(
            hard_gate["buy_proba"],
            bins=PROBABILITY_BINS,
            labels=PROBABILITY_LABELS,
            right=False,
        )
        for bucket in PROBABILITY_LABELS:
            yearly_signals = []
            for _, year_part in hard_gate.groupby("test_year", sort=True):
                yearly_signals.append(
                    _non_overlapping(year_part[year_part["probability_bucket"] == bucket])
                )
            independent = pd.concat(yearly_signals, ignore_index=True) if yearly_signals else hard_gate.iloc[0:0]
            row = _signal_metrics(independent, label_threshold, probability_threshold=float("nan"))
            row.update({"scope": "aggregate", "test_year": None, "probability_bucket": bucket})
            bucket_rows.append(row)
    return pd.DataFrame(comparison_rows), pd.DataFrame(bucket_rows)


def evaluate_buy_label_thresholds(
    start_year: int = 2021,
    end_year: int = 2026,
    train_start: str = "2016-01-01",
    label_thresholds: list[float] | None = None,
    probability_thresholds: list[float] | None = None,
    min_train_rows: int = 200,
    comparison_csv: str = "data/reports/buy_7d_label_threshold_comparison.csv",
    bucket_csv: str = "data/reports/buy_7d_probability_buckets.csv",
    detail_csv: str = "data/reports/buy_7d_label_threshold_details.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = get_config()
    label_thresholds = label_thresholds or [0.0, 0.005, 0.01]
    probability_thresholds = probability_thresholds or [0.5, 0.6, 0.7, 0.8]
    data = load_dataset()
    if data.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    required = ["future_ret_7d", "f_is_bull_trend"]
    missing = [col for col in required if col not in data]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}. Build dataset first.")

    features = feature_columns(data)
    if not cfg.get("features", {}).get("use_guba_sentiment", False):
        features = [col for col in features if col not in GUBA_SENTIMENT_FEATURES]
    random_state = int(cfg["model"].get("random_state", 42))
    train_window_start = pd.Timestamp(train_start)
    detail_parts = []

    for label_threshold in label_thresholds:
        for test_year in range(start_year, end_year + 1):
            test_start = pd.Timestamp(f"{test_year}-01-01")
            test_end = pd.Timestamp(f"{test_year}-12-31")
            train = data[
                (data["trade_date"] >= train_window_start)
                & (data["trade_date"] < test_start)
            ].dropna(subset=["future_ret_7d"]).sort_values("trade_date").copy()
            test = data[
                (data["trade_date"] >= test_start)
                & (data["trade_date"] <= test_end)
            ].dropna(subset=["future_ret_7d"]).sort_values("trade_date").copy()
            if len(train) > 7:
                train = train.iloc[:-7].copy()
            if len(train) < min_train_rows or test.empty:
                logger.warning(
                    "skip label_threshold=%s test_year=%s train_rows=%s test_rows=%s",
                    label_threshold,
                    test_year,
                    len(train),
                    len(test),
                )
                continue

            train["experiment_label"] = (train["future_ret_7d"] > label_threshold).astype(int)
            model, usable_features = _fit_model(train, features, "experiment_label", random_state)
            part = test[
                ["trade_date", "future_ret_7d", "f_is_bull_trend", "f_is_bear_trend"]
            ].copy()
            part = part.rename(columns={"future_ret_7d": "future_ret"})
            part["buy_proba"] = _positive_probability(model, test, usable_features)
            part["label_threshold"] = label_threshold
            part["test_year"] = test_year
            part["trade_pos"] = range(len(part))
            detail_parts.append(part)
            logger.info(
                "label threshold walk-forward threshold=%.4f year=%s train_rows=%s test_rows=%s positive_rate=%.4f",
                label_threshold,
                test_year,
                len(train),
                len(test),
                train["experiment_label"].mean(),
            )

    if not detail_parts:
        raise RuntimeError("No label threshold evaluation rows generated.")
    details = pd.concat(detail_parts, ignore_index=True)
    comparison, buckets = _build_reports(details, probability_thresholds)
    for frame, output in [
        (comparison, comparison_csv),
        (buckets, bucket_csv),
        (details.drop(columns=["trade_pos"]), detail_csv),
    ]:
        path = project_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved report path={path} rows={len(frame)}")
    return comparison, buckets


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare 7-day buy label thresholds and probability buckets.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--label-thresholds", default="0,0.005,0.01")
    parser.add_argument("--probability-thresholds", default="0.50,0.60,0.70,0.80")
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--comparison-csv", default="data/reports/buy_7d_label_threshold_comparison.csv")
    parser.add_argument("--bucket-csv", default="data/reports/buy_7d_probability_buckets.csv")
    parser.add_argument("--detail-csv", default="data/reports/buy_7d_label_threshold_details.csv")
    args = parser.parse_args()
    comparison, buckets = evaluate_buy_label_thresholds(
        start_year=args.start_year,
        end_year=args.end_year,
        train_start=args.train_start,
        label_thresholds=_parse_floats(args.label_thresholds),
        probability_thresholds=_parse_floats(args.probability_thresholds),
        min_train_rows=args.min_train_rows,
        comparison_csv=args.comparison_csv,
        bucket_csv=args.bucket_csv,
        detail_csv=args.detail_csv,
    )
    print("\n=== aggregate label comparison: hard gate + non-overlapping ===")
    aggregate = comparison[comparison["scope"] == "aggregate"]
    print(aggregate.to_string(index=False))
    print("\n=== aggregate probability buckets: hard gate + non-overlapping ===")
    aggregate_buckets = buckets[buckets["scope"] == "aggregate"]
    print(aggregate_buckets.to_string(index=False))


if __name__ == "__main__":
    main()
