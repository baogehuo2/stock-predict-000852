from __future__ import annotations

import argparse

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.logger import get_logger
from src.modeling.buy_feature_set import select_buy_features
from src.modeling.data import load_dataset
from src.modeling.market_regime import assign_evaluation_regime


disable_broken_dask_autoload()
from lightgbm import LGBMClassifier


logger = get_logger(__name__)


def _parse_ints(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def _parse_floats(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def _fit_model(
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    random_state: int,
) -> tuple[Pipeline, list[str]]:
    usable_features = [col for col in features if train[col].notna().any()]
    dropped = [col for col in features if col not in usable_features]
    if dropped:
        logger.warning("drop all-missing features label=%s features=%s", label_col, dropped)
    target = train[label_col].astype(int)
    if target.nunique() < 2:
        raise RuntimeError(f"{label_col} training label has only one class.")
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=31,
                    random_state=random_state,
                    class_weight="balanced",
                    verbose=-1,
                ),
            ),
        ]
    )
    model.fit(train[usable_features], target)
    return model, usable_features


def _positive_probability(model: Pipeline, data: pd.DataFrame, features: list[str]) -> pd.Series:
    classes = list(model.named_steps["model"].classes_)
    return pd.Series(model.predict_proba(data[features])[:, classes.index(1)], index=data.index)


def _non_overlapping(signal: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if signal.empty:
        return signal
    selected = []
    next_allowed_pos = -1
    for idx, row in signal.sort_values("trade_date").iterrows():
        position = int(row["trade_pos"])
        if position >= next_allowed_pos:
            selected.append(idx)
            next_allowed_pos = position + horizon
    return signal.loc[selected]


def _metric_row(
    universe: pd.DataFrame,
    signal: pd.DataFrame,
    threshold: float,
    sample_mode: str,
    regime: str,
) -> dict:
    strategy_ret = signal["future_ret"]
    label_threshold = float(universe["label_threshold"].iloc[0])
    correct = signal["future_ret"] > label_threshold
    actual_up = signal["future_ret"] > 0
    natural_correct = universe["future_ret"] > label_threshold
    natural_actual_up = universe["future_ret"] > 0
    natural_strategy_ret = universe["future_ret"]
    return {
        "test_year": int(universe["test_year"].iloc[0]),
        "horizon": int(universe["horizon"].iloc[0]),
        "regime": regime,
        "side": "buy",
        "threshold": threshold,
        "label_threshold": label_threshold,
        "sample_mode": sample_mode,
        "train_start": universe["train_start"].iloc[0],
        "train_end": universe["train_end"].iloc[0],
        "train_rows": int(universe["train_rows"].iloc[0]),
        "regime_rows": int(len(universe)),
        "signal_count": int(len(signal)),
        "coverage": float(len(signal) / len(universe)) if len(universe) else 0.0,
        "precision": float(correct.mean()) if len(signal) else None,
        "actual_up_rate": float(actual_up.mean()) if len(signal) else None,
        "avg_strategy_ret": float(strategy_ret.mean()) if len(signal) else None,
        "median_strategy_ret": float(strategy_ret.median()) if len(signal) else None,
        "avg_proba": float(signal["buy_proba"].mean()) if len(signal) else None,
        "natural_precision": float(natural_correct.mean()) if len(universe) else None,
        "natural_actual_up_rate": float(natural_actual_up.mean()) if len(universe) else None,
        "natural_avg_strategy_ret": float(natural_strategy_ret.mean()) if len(universe) else None,
        "precision_lift": float(correct.mean() - natural_correct.mean()) if len(signal) else None,
        "avg_strategy_ret_lift": (
            float(strategy_ret.mean() - natural_strategy_ret.mean()) if len(signal) else None
        ),
    }


def _summarize(details: pd.DataFrame, thresholds: list[float]) -> list[dict]:
    rows = []
    for (test_year, horizon), year_part in details.groupby(["test_year", "horizon"], sort=True):
        for regime in ["all", "bull", "bear"]:
            universe = year_part if regime == "all" else year_part[year_part["evaluation_regime"] == regime]
            if universe.empty:
                continue
            for threshold in thresholds:
                signal = universe[universe["buy_proba"] >= threshold]
                rows.append(_metric_row(universe, signal, threshold, "all_signals", regime))
                independent = _non_overlapping(signal, int(horizon))
                rows.append(_metric_row(universe, independent, threshold, "non_overlapping", regime))
    return rows


def _aggregate_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    data = summary.copy()
    data["correct_count"] = data["precision"] * data["signal_count"]
    data["actual_up_count"] = data["actual_up_rate"] * data["signal_count"]
    data["strategy_ret_sum"] = data["avg_strategy_ret"] * data["signal_count"]
    data["natural_correct_count"] = data["natural_precision"] * data["regime_rows"]
    data["natural_actual_up_count"] = data["natural_actual_up_rate"] * data["regime_rows"]
    data["natural_strategy_ret_sum"] = data["natural_avg_strategy_ret"] * data["regime_rows"]
    group_cols = ["horizon", "regime", "side", "threshold", "label_threshold", "sample_mode"]
    result = (
        data.groupby(group_cols, as_index=False)
        .agg(
            evaluated_years=("test_year", "nunique"),
            regime_rows=("regime_rows", "sum"),
            signal_count=("signal_count", "sum"),
            correct_count=("correct_count", "sum"),
            actual_up_count=("actual_up_count", "sum"),
            strategy_ret_sum=("strategy_ret_sum", "sum"),
            natural_correct_count=("natural_correct_count", "sum"),
            natural_actual_up_count=("natural_actual_up_count", "sum"),
            natural_strategy_ret_sum=("natural_strategy_ret_sum", "sum"),
        )
    )
    result["coverage"] = result["signal_count"] / result["regime_rows"]
    result["precision"] = result["correct_count"] / result["signal_count"]
    result["actual_up_rate"] = result["actual_up_count"] / result["signal_count"]
    result["avg_strategy_ret"] = result["strategy_ret_sum"] / result["signal_count"]
    result["natural_precision"] = result["natural_correct_count"] / result["regime_rows"]
    result["natural_actual_up_rate"] = result["natural_actual_up_count"] / result["regime_rows"]
    result["natural_avg_strategy_ret"] = result["natural_strategy_ret_sum"] / result["regime_rows"]
    result["precision_lift"] = result["precision"] - result["natural_precision"]
    result["avg_strategy_ret_lift"] = result["avg_strategy_ret"] - result["natural_avg_strategy_ret"]
    return result.drop(
        columns=[
            "correct_count",
            "actual_up_count",
            "strategy_ret_sum",
            "natural_correct_count",
            "natural_actual_up_count",
            "natural_strategy_ret_sum",
        ]
    )


def walk_forward_buy_evaluation(
    start_year: int = 2021,
    end_year: int = 2026,
    train_start: str = "2016-01-01",
    horizons: list[int] | None = None,
    thresholds: list[float] | None = None,
    min_train_rows: int = 200,
    output_csv: str | None = None,
    detail_csv: str | None = None,
    aggregate_csv: str | None = None,
) -> list[dict]:
    cfg = get_config()
    buy_cfg = cfg.get("binary_buy_model", {})
    horizons = horizons or [int(h) for h in buy_cfg.get("horizons", [7])]
    label_thresholds = {
        int(horizon): float(value)
        for horizon, value in buy_cfg.get("label_thresholds", {}).items()
    }
    thresholds = thresholds or [0.6, 0.7, 0.8]
    df = load_dataset()
    if df.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    features = select_buy_features(df, buy_cfg)
    if not features:
        raise RuntimeError("No feature columns found in dataset.")

    detail_parts = []
    random_state = int(cfg["model"].get("random_state", 42))
    train_window_start = pd.Timestamp(train_start)
    for test_year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        train_end = test_start - pd.Timedelta(days=1)
        train_base = df[(df["trade_date"] >= train_window_start) & (df["trade_date"] <= train_end)].copy()
        test_base = df[(df["trade_date"] >= test_start) & (df["trade_date"] <= test_end)].copy()
        if test_base.empty:
            logger.warning("skip test_year=%s because no rows", test_year)
            continue

        for horizon in horizons:
            ret_col = f"future_ret_{horizon}d"
            buy_label = f"buy_label_{horizon}d"
            required = [ret_col, buy_label]
            missing = [col for col in required if col not in df]
            if missing:
                raise RuntimeError(f"Missing columns {missing}. Run build_dataset first.")

            train_h = train_base.dropna(subset=required).sort_values("trade_date").copy()
            if len(train_h) > horizon:
                train_h = train_h.iloc[:-horizon].copy()
            test_h = test_base.dropna(subset=required).sort_values("trade_date").copy()
            if len(train_h) < min_train_rows or test_h.empty:
                logger.warning(
                    "skip year=%s horizon=%sd train_rows=%s test_rows=%s",
                    test_year,
                    horizon,
                    len(train_h),
                    len(test_h),
                )
                continue

            buy_model, buy_features = _fit_model(train_h, features, buy_label, random_state)
            causal_cols = [
                col
                for col in [
                    "f_regime_trend_score",
                    "f_is_bull_trend",
                    "f_is_bear_trend",
                    "f_regime_state_days",
                ]
                if col in test_h
            ]
            part = test_h[["trade_date", ret_col, buy_label, *causal_cols]].copy()
            part = part.rename(columns={ret_col: "future_ret"})
            part["buy_proba"] = _positive_probability(buy_model, test_h, buy_features)
            part["evaluation_regime"] = assign_evaluation_regime(part["trade_date"])
            part["test_year"] = test_year
            part["horizon"] = horizon
            part["label_threshold"] = label_thresholds.get(horizon, 0.0)
            part["train_start"] = train_start
            part["train_end"] = train_h["trade_date"].max().strftime("%Y-%m-%d")
            part["train_rows"] = len(train_h)
            part["trade_pos"] = range(len(part))
            detail_parts.append(part)
            logger.info(
                "walk-forward year=%s horizon=%sd train_rows=%s test_rows=%s purged_rows=%s",
                test_year,
                horizon,
                len(train_h),
                len(test_h),
                horizon,
            )

    if not detail_parts:
        return []
    details = pd.concat(detail_parts, ignore_index=True)
    rows = _summarize(details, thresholds)
    summary = pd.DataFrame(rows)
    if output_csv:
        path = project_path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved walk-forward summary path={path} rows={len(rows)}")
    if detail_csv:
        path = project_path(detail_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        details.drop(columns=["trade_pos"]).to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved walk-forward details path={path} rows={len(details)}")
    if aggregate_csv:
        path = project_path(aggregate_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        aggregate = _aggregate_summary(summary)
        aggregate.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved walk-forward aggregate path={path} rows={len(aggregate)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Purged yearly walk-forward evaluation for the binary buy model.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--horizons", default="7")
    parser.add_argument("--thresholds", default="0.60,0.70,0.80")
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--output-csv", default="data/reports/walk_forward_buy_7d_summary.csv")
    parser.add_argument("--detail-csv", default="data/reports/walk_forward_buy_7d_details.csv")
    parser.add_argument("--aggregate-csv", default="data/reports/walk_forward_buy_7d_aggregate.csv")
    args = parser.parse_args()
    rows = walk_forward_buy_evaluation(
        start_year=args.start_year,
        end_year=args.end_year,
        train_start=args.train_start,
        horizons=_parse_ints(args.horizons),
        thresholds=_parse_floats(args.thresholds),
        min_train_rows=args.min_train_rows,
        output_csv=args.output_csv,
        detail_csv=args.detail_csv,
        aggregate_csv=args.aggregate_csv,
    )
    report = pd.DataFrame(rows)
    if report.empty:
        print("No walk-forward rows.")
        return
    focus = report[(report["regime"] == "bull") & (report["side"] == "buy")]
    cols = [
        "test_year",
        "horizon",
        "regime",
        "side",
        "threshold",
        "sample_mode",
        "signal_count",
        "precision",
        "actual_up_rate",
        "natural_precision",
        "natural_actual_up_rate",
        "precision_lift",
        "avg_strategy_ret",
        "natural_avg_strategy_ret",
        "avg_strategy_ret_lift",
    ]
    print(focus[cols].to_string(index=False))


if __name__ == "__main__":
    main()
