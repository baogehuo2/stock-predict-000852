from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

from src.common.config import get_config, project_path
from src.common.logger import get_logger
from src.modeling.data import feature_columns, load_dataset
from src.modeling.train_lgbm import GUBA_SENTIMENT_FEATURES
from src.modeling.walk_forward_buy_lgbm import _fit_model, _positive_probability


logger = get_logger(__name__)

CALIBRATION_METHODS = ["raw", "sigmoid", "isotonic"]
PROBABILITY_BINS = [0.0, 0.5, 0.6, 0.7, 0.8, 1.000001]
PROBABILITY_LABELS = ["<0.50", "0.50-0.60", "0.60-0.70", "0.70-0.80", ">=0.80"]


def _parse_floats(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def _logit(probability: pd.Series | np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


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


def _oof_probabilities(
    train: pd.DataFrame,
    features: list[str],
    random_state: int,
    n_splits: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    probabilities = []
    labels = []
    for fold, (fit_idx, calibration_idx) in enumerate(splitter.split(train), start=1):
        fit = train.iloc[fit_idx].copy()
        calibration = train.iloc[calibration_idx].copy()
        if len(fit) > 7:
            fit = fit.iloc[:-7].copy()
        if len(fit) < 200 or fit["experiment_label"].nunique() < 2:
            logger.warning("skip calibration fold=%s fit_rows=%s", fold, len(fit))
            continue
        model, usable_features = _fit_model(fit, features, "experiment_label", random_state + fold)
        probabilities.extend(_positive_probability(model, calibration, usable_features).tolist())
        labels.extend(calibration["experiment_label"].astype(int).tolist())
    if not probabilities or len(set(labels)) < 2:
        raise RuntimeError("Not enough out-of-fold predictions for probability calibration.")
    return np.asarray(probabilities), np.asarray(labels)


def _fit_calibrators(raw_probability: np.ndarray, labels: np.ndarray) -> tuple[LogisticRegression, IsotonicRegression]:
    sigmoid = LogisticRegression(random_state=42)
    sigmoid.fit(_logit(raw_probability), labels)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_probability, labels)
    return sigmoid, isotonic


def _apply_calibration(
    raw_probability: pd.Series,
    sigmoid: LogisticRegression,
    isotonic: IsotonicRegression,
) -> dict[str, np.ndarray]:
    values = raw_probability.to_numpy(dtype=float)
    return {
        "raw": values,
        "sigmoid": sigmoid.predict_proba(_logit(values))[:, 1],
        "isotonic": isotonic.predict(values),
    }


def _build_reports(
    details: pd.DataFrame,
    probability_thresholds: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_rows = []
    bucket_rows = []
    calibration_rows = []
    for (method, test_year), year_part in details.groupby(["calibration_method", "test_year"], sort=True):
        hard_gate = year_part[year_part["f_is_bull_trend"] == 1].copy()
        actual = hard_gate["actual_label"].astype(int)
        calibration_rows.append(
            {
                "scope": "year",
                "test_year": int(test_year),
                "calibration_method": method,
                "sample_count": int(len(hard_gate)),
                "avg_predicted_probability": float(hard_gate["buy_proba"].mean()),
                "actual_target_rate": float(actual.mean()),
                "calibration_gap": float(hard_gate["buy_proba"].mean() - actual.mean()),
                "brier_score": float(brier_score_loss(actual, hard_gate["buy_proba"])),
            }
        )
        for threshold in probability_thresholds:
            signal = _non_overlapping(hard_gate[hard_gate["buy_proba"] >= threshold])
            signal_rows.append(
                {
                    "scope": "year",
                    "test_year": int(test_year),
                    "calibration_method": method,
                    "probability_threshold": threshold,
                    "signal_count": int(len(signal)),
                    "target_precision": float(signal["actual_label"].mean()) if len(signal) else None,
                    "actual_up_rate": float((signal["future_ret"] > 0).mean()) if len(signal) else None,
                    "avg_future_ret": float(signal["future_ret"].mean()) if len(signal) else None,
                    "median_future_ret": float(signal["future_ret"].median()) if len(signal) else None,
                    "avg_buy_proba": float(signal["buy_proba"].mean()) if len(signal) else None,
                }
            )
        hard_gate["probability_bucket"] = pd.cut(
            hard_gate["buy_proba"], PROBABILITY_BINS, labels=PROBABILITY_LABELS, right=False
        )
        for bucket in PROBABILITY_LABELS:
            signal = _non_overlapping(hard_gate[hard_gate["probability_bucket"] == bucket])
            bucket_rows.append(
                {
                    "scope": "year",
                    "test_year": int(test_year),
                    "calibration_method": method,
                    "probability_bucket": bucket,
                    "signal_count": int(len(signal)),
                    "avg_predicted_probability": float(signal["buy_proba"].mean()) if len(signal) else None,
                    "target_precision": float(signal["actual_label"].mean()) if len(signal) else None,
                    "actual_up_rate": float((signal["future_ret"] > 0).mean()) if len(signal) else None,
                    "avg_future_ret": float(signal["future_ret"].mean()) if len(signal) else None,
                }
            )

    for method, method_part in details.groupby("calibration_method", sort=True):
        hard_gate = method_part[method_part["f_is_bull_trend"] == 1].copy()
        actual = hard_gate["actual_label"].astype(int)
        calibration_rows.append(
            {
                "scope": "aggregate",
                "test_year": None,
                "calibration_method": method,
                "sample_count": int(len(hard_gate)),
                "avg_predicted_probability": float(hard_gate["buy_proba"].mean()),
                "actual_target_rate": float(actual.mean()),
                "calibration_gap": float(hard_gate["buy_proba"].mean() - actual.mean()),
                "brier_score": float(brier_score_loss(actual, hard_gate["buy_proba"])),
            }
        )
        for threshold in probability_thresholds:
            signals = []
            for _, year_part in hard_gate.groupby("test_year", sort=True):
                signals.append(_non_overlapping(year_part[year_part["buy_proba"] >= threshold]))
            signal = pd.concat(signals, ignore_index=True) if signals else hard_gate.iloc[0:0]
            signal_rows.append(
                {
                    "scope": "aggregate",
                    "test_year": None,
                    "calibration_method": method,
                    "probability_threshold": threshold,
                    "signal_count": int(len(signal)),
                    "target_precision": float(signal["actual_label"].mean()) if len(signal) else None,
                    "actual_up_rate": float((signal["future_ret"] > 0).mean()) if len(signal) else None,
                    "avg_future_ret": float(signal["future_ret"].mean()) if len(signal) else None,
                    "median_future_ret": float(signal["future_ret"].median()) if len(signal) else None,
                    "avg_buy_proba": float(signal["buy_proba"].mean()) if len(signal) else None,
                }
            )
        hard_gate["probability_bucket"] = pd.cut(
            hard_gate["buy_proba"], PROBABILITY_BINS, labels=PROBABILITY_LABELS, right=False
        )
        for bucket in PROBABILITY_LABELS:
            signals = []
            for _, year_part in hard_gate.groupby("test_year", sort=True):
                signals.append(_non_overlapping(year_part[year_part["probability_bucket"] == bucket]))
            signal = pd.concat(signals, ignore_index=True) if signals else hard_gate.iloc[0:0]
            bucket_rows.append(
                {
                    "scope": "aggregate",
                    "test_year": None,
                    "calibration_method": method,
                    "probability_bucket": bucket,
                    "signal_count": int(len(signal)),
                    "avg_predicted_probability": float(signal["buy_proba"].mean()) if len(signal) else None,
                    "target_precision": float(signal["actual_label"].mean()) if len(signal) else None,
                    "actual_up_rate": float((signal["future_ret"] > 0).mean()) if len(signal) else None,
                    "avg_future_ret": float(signal["future_ret"].mean()) if len(signal) else None,
                }
            )
    return pd.DataFrame(signal_rows), pd.DataFrame(bucket_rows), pd.DataFrame(calibration_rows)


def evaluate_buy_probability_calibration(
    start_year: int = 2021,
    end_year: int = 2026,
    train_start: str = "2016-01-01",
    label_threshold: float = 0.005,
    probability_thresholds: list[float] | None = None,
    calibration_splits: int = 3,
    signal_csv: str = "data/reports/buy_7d_calibration_signal_quality.csv",
    bucket_csv: str = "data/reports/buy_7d_calibration_buckets.csv",
    calibration_csv: str = "data/reports/buy_7d_calibration_quality.csv",
    detail_csv: str = "data/reports/buy_7d_calibration_details.csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = get_config()
    thresholds = probability_thresholds or [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8]
    data = load_dataset()
    if data.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    features = feature_columns(data)
    if not cfg.get("features", {}).get("use_guba_sentiment", False):
        features = [col for col in features if col not in GUBA_SENTIMENT_FEATURES]
    random_state = int(cfg["model"].get("random_state", 42))
    train_window_start = pd.Timestamp(train_start)
    detail_parts = []

    for test_year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        train = data[
            (data["trade_date"] >= train_window_start) & (data["trade_date"] < test_start)
        ].dropna(subset=["future_ret_7d"]).sort_values("trade_date").copy()
        test = data[
            (data["trade_date"] >= test_start) & (data["trade_date"] <= test_end)
        ].dropna(subset=["future_ret_7d"]).sort_values("trade_date").copy()
        if len(train) > 7:
            train = train.iloc[:-7].copy()
        if test.empty:
            continue
        train["experiment_label"] = (train["future_ret_7d"] > label_threshold).astype(int)
        oof_probability, oof_label = _oof_probabilities(
            train, features, random_state, calibration_splits
        )
        sigmoid, isotonic = _fit_calibrators(oof_probability, oof_label)
        final_model, usable_features = _fit_model(train, features, "experiment_label", random_state)
        raw_probability = _positive_probability(final_model, test, usable_features)
        calibrated = _apply_calibration(raw_probability, sigmoid, isotonic)

        base = test[["trade_date", "future_ret_7d", "f_is_bull_trend", "f_is_bear_trend"]].copy()
        base = base.rename(columns={"future_ret_7d": "future_ret"})
        base["actual_label"] = (base["future_ret"] > label_threshold).astype(int)
        base["test_year"] = test_year
        base["trade_pos"] = range(len(base))
        for method in CALIBRATION_METHODS:
            part = base.copy()
            part["calibration_method"] = method
            part["buy_proba"] = calibrated[method]
            detail_parts.append(part)
        logger.info(
            "probability calibration year=%s train_rows=%s test_rows=%s oof_rows=%s",
            test_year,
            len(train),
            len(test),
            len(oof_probability),
        )

    details = pd.concat(detail_parts, ignore_index=True)
    signals, buckets, calibration = _build_reports(details, thresholds)
    for frame, output in [
        (signals, signal_csv),
        (buckets, bucket_csv),
        (calibration, calibration_csv),
        (details.drop(columns=["trade_pos"]), detail_csv),
    ]:
        path = project_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved report path={path} rows={len(frame)}")
    return signals, buckets, calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="Time-series probability calibration for the 7-day buy model.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--label-threshold", type=float, default=0.005)
    parser.add_argument("--probability-thresholds", default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.80")
    parser.add_argument("--calibration-splits", type=int, default=3)
    args = parser.parse_args()
    signals, buckets, calibration = evaluate_buy_probability_calibration(
        start_year=args.start_year,
        end_year=args.end_year,
        train_start=args.train_start,
        label_threshold=args.label_threshold,
        probability_thresholds=_parse_floats(args.probability_thresholds),
        calibration_splits=args.calibration_splits,
    )
    print("\n=== aggregate calibration quality ===")
    print(calibration[calibration["scope"] == "aggregate"].to_string(index=False))
    print("\n=== aggregate signal quality: hard gate + non-overlapping ===")
    print(signals[signals["scope"] == "aggregate"].to_string(index=False))
    print("\n=== aggregate probability buckets ===")
    print(buckets[buckets["scope"] == "aggregate"].to_string(index=False))


if __name__ == "__main__":
    main()
