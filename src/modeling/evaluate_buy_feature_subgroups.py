from __future__ import annotations

import argparse
import re

import pandas as pd

from src.common.config import get_config, project_path
from src.common.logger import get_logger
from src.modeling.data import feature_columns, load_dataset
from src.modeling.evaluate_buy_feature_ablation import (
    BASIC_MARKET_FEATURES,
    HORIZON,
    RISK_FEATURES,
    STYLE_MARKET_FEATURES,
    TECHNICAL_FEATURES,
    _metrics,
    _non_overlapping,
    _ordered_subset,
)
from src.modeling.train_lgbm import GUBA_SENTIMENT_FEATURES
from src.modeling.walk_forward_buy_lgbm import _fit_model, _positive_probability


logger = get_logger(__name__)

TREND_FEATURES = {
    "f_ma20_slope_5d",
    "f_ma60_slope_10d",
    "f_ma20_slope_5d_negative",
    "f_ma60_slope_10d_negative",
    "f_close_above_ma20",
    "f_close_above_ma60",
    "f_above_ma20_days",
    "f_below_ma20_days",
    "f_above_ma60_days",
    "f_below_ma60_days",
    "f_ma20_ma60_gap",
    "f_drawdown_20d",
    "f_drawdown_60d",
    "f_regime_trend_score",
    "f_is_bull_trend",
    "f_is_bear_trend",
    "f_regime_state_days",
}

VOLUME_FEATURES = {
    "f_amount_zscore_20d",
    "f_amount_percentile_60d",
    "f_volume_zscore_20d",
    "f_volume_ratio_5d_20d",
    "f_volume_percentile_60d",
    "f_down_with_volume",
    "f_up_with_volume",
    "f_shrink_rebound",
}

VOLATILITY_FEATURES = {
    "f_volatility_5d",
    "f_volatility_10d",
    "f_volatility_20d",
    "f_volatility_expand",
    "f_atr_expand",
    "f_boll_lower_break",
    "f_boll_upper_break",
}


def _event_window(feature: str) -> str:
    match = re.search(r"_roll_(3|5|10)d$", feature)
    return match.group(1) if match else "raw"


def _available_features(data: pd.DataFrame) -> list[str]:
    cfg = get_config()
    features = feature_columns(data)
    if not cfg.get("features", {}).get("use_guba_sentiment", False):
        features = [feature for feature in features if feature not in GUBA_SENTIMENT_FEATURES]
    return features


def _groups(available: list[str]) -> dict[str, set[str]]:
    available_set = set(available)
    event_features = {
        feature
        for feature in available
        if feature == "f_event_score" or feature.startswith("f_event_")
    }
    groups = {
        "basic": BASIC_MARKET_FEATURES & available_set,
        "technical": TECHNICAL_FEATURES & available_set,
        "style": STYLE_MARKET_FEATURES & available_set,
        "trend": TREND_FEATURES & available_set,
        "volume": VOLUME_FEATURES & available_set,
        "volatility": VOLATILITY_FEATURES & available_set,
        "event_raw": {feature for feature in event_features if _event_window(feature) == "raw"},
        "event_roll_3d": {feature for feature in event_features if _event_window(feature) == "3"},
        "event_roll_5d": {feature for feature in event_features if _event_window(feature) == "5"},
        "event_roll_10d": {feature for feature in event_features if _event_window(feature) == "10"},
    }
    risk_parts = groups["trend"] | groups["volume"] | groups["volatility"]
    missing_risk = (RISK_FEATURES & available_set) - risk_parts
    if missing_risk:
        raise RuntimeError(f"Unassigned risk feature columns: {sorted(missing_risk)}")
    return groups


def _variants(available: list[str], groups: dict[str, set[str]]) -> dict[str, list[str]]:
    market_base = groups["basic"] | groups["technical"] | groups["style"]
    risk_all = groups["trend"] | groups["volume"] | groups["volatility"]
    event_all = (
        groups["event_raw"]
        | groups["event_roll_3d"]
        | groups["event_roll_5d"]
        | groups["event_roll_10d"]
    )
    event_base = market_base | risk_all
    selected = {
        "risk_0_none": market_base,
        "risk_1_trend": market_base | groups["trend"],
        "risk_2_volume": market_base | groups["volume"],
        "risk_3_volatility": market_base | groups["volatility"],
        "risk_4_trend_volume": market_base | groups["trend"] | groups["volume"],
        "risk_5_trend_volatility": market_base | groups["trend"] | groups["volatility"],
        "risk_6_volume_volatility": market_base | groups["volume"] | groups["volatility"],
        "risk_7_all": event_base,
        "event_0_none": event_base,
        "event_1_raw": event_base | groups["event_raw"],
        "event_2_raw_3d": event_base | groups["event_raw"] | groups["event_roll_3d"],
        "event_3_raw_5d": event_base | groups["event_raw"] | groups["event_roll_5d"],
        "event_4_raw_10d": event_base | groups["event_raw"] | groups["event_roll_10d"],
        "event_5_roll_all_no_raw": event_base | (event_all - groups["event_raw"]),
        "event_6_raw_3d_5d": (
            event_base | groups["event_raw"] | groups["event_roll_3d"] | groups["event_roll_5d"]
        ),
        "event_7_raw_3d_10d": (
            event_base | groups["event_raw"] | groups["event_roll_3d"] | groups["event_roll_10d"]
        ),
        "event_8_raw_5d_10d": (
            event_base | groups["event_raw"] | groups["event_roll_5d"] | groups["event_roll_10d"]
        ),
        "event_9_all": event_base | event_all,
        "final_0_no_risk": market_base | event_all,
        "final_1_trend": market_base | groups["trend"] | event_all,
        "final_2_volume": market_base | groups["volume"] | event_all,
        "final_3_volatility": market_base | groups["volatility"] | event_all,
        "final_4_trend_volume": (
            market_base | groups["trend"] | groups["volume"] | event_all
        ),
        "final_5_trend_volatility": (
            market_base | groups["trend"] | groups["volatility"] | event_all
        ),
        "final_6_volume_volatility": (
            market_base | groups["volume"] | groups["volatility"] | event_all
        ),
        "final_7_all_risk": event_base | event_all,
    }
    return {name: _ordered_subset(available, features) for name, features in selected.items()}


def evaluate_buy_feature_subgroups(
    start_year: int = 2021,
    end_year: int = 2026,
    train_start: str = "2016-01-01",
    probability_threshold: float = 0.60,
    min_train_rows: int = 200,
    summary_csv: str = "data/reports/buy_7d_feature_subgroups.csv",
    detail_csv: str = "data/reports/buy_7d_feature_subgroup_details.csv",
    group_csv: str = "data/reports/buy_7d_feature_subgroup_members.csv",
    variant_prefix: str | None = None,
) -> pd.DataFrame:
    cfg = get_config()
    label_threshold = float(
        cfg.get("binary_buy_model", {}).get("label_thresholds", {}).get(str(HORIZON), 0.0)
    )
    random_state = int(cfg["model"].get("random_state", 42))
    data = load_dataset()
    if data.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    available = _available_features(data)
    groups = _groups(available)
    variants = _variants(available, groups)
    if variant_prefix:
        variants = {
            name: features for name, features in variants.items() if name.startswith(variant_prefix)
        }
        if not variants:
            raise RuntimeError(f"No variants matched prefix: {variant_prefix}")

    group_rows = [
        {
            "group": group,
            "feature": feature,
            "non_null_rows": int(data[feature].notna().sum()),
            "non_zero_rows": int((pd.to_numeric(data[feature], errors="coerce").fillna(0) != 0).sum()),
        }
        for group, features in groups.items()
        for feature in _ordered_subset(available, features)
    ]
    group_path = project_path(group_csv)
    group_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(group_rows).to_csv(group_path, index=False, encoding="utf-8-sig")

    train_window_start = pd.Timestamp(train_start)
    detail_parts = []
    metric_rows = []
    for variant, features in variants.items():
        for test_year in range(start_year, end_year + 1):
            test_start = pd.Timestamp(f"{test_year}-01-01")
            test_end = pd.Timestamp(f"{test_year}-12-31")
            train = data[
                (data["trade_date"] >= train_window_start) & (data["trade_date"] < test_start)
            ].dropna(subset=["future_ret_7d", "buy_label_7d"]).sort_values("trade_date").copy()
            test = data[
                (data["trade_date"] >= test_start) & (data["trade_date"] <= test_end)
            ].dropna(subset=["future_ret_7d", "buy_label_7d"]).sort_values("trade_date").copy()
            if len(train) > HORIZON:
                train = train.iloc[:-HORIZON].copy()
            if len(train) < min_train_rows or test.empty:
                continue

            model, usable_features = _fit_model(train, features, "buy_label_7d", random_state)
            part = test[
                ["trade_date", "future_ret_7d", "f_is_bull_trend", "f_is_bear_trend"]
            ].copy()
            part = part.rename(columns={"future_ret_7d": "future_ret"})
            part["buy_proba"] = _positive_probability(model, test, usable_features)
            part["variant"] = variant
            part["test_year"] = test_year
            part["trade_pos"] = range(len(part))
            part["feature_count"] = len(usable_features)
            detail_parts.append(part)

            universe = part[part["f_is_bull_trend"] == 1]
            signal = _non_overlapping(universe[universe["buy_proba"] >= probability_threshold])
            metric_rows.append(
                _metrics(
                    signal,
                    universe,
                    variant,
                    len(usable_features),
                    label_threshold,
                    probability_threshold,
                    "year",
                    test_year,
                )
            )
            logger.info(
                "feature subgroup variant=%s year=%s features=%s signals=%s",
                variant,
                test_year,
                len(usable_features),
                len(signal),
            )

    details = pd.concat(detail_parts, ignore_index=True)
    for variant, variant_part in details.groupby("variant", sort=True):
        universe = variant_part[variant_part["f_is_bull_trend"] == 1]
        yearly_signals = [
            _non_overlapping(year_part[year_part["buy_proba"] >= probability_threshold])
            for _, year_part in universe.groupby("test_year", sort=True)
        ]
        signal = pd.concat(yearly_signals, ignore_index=True) if yearly_signals else universe.iloc[0:0]
        metric_rows.append(
            _metrics(
                signal,
                universe,
                variant,
                int(variant_part["feature_count"].max()),
                label_threshold,
                probability_threshold,
                "aggregate",
                None,
            )
        )

    summary = pd.DataFrame(metric_rows)
    for frame, output in [
        (summary, summary_csv),
        (details.drop(columns=["trade_pos"]), detail_csv),
    ]:
        path = project_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved report path={path} rows={len(frame)}")
    print(f"saved feature subgroup members path={group_path} rows={len(group_rows)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Risk and news-window feature subgroup ablation.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--probability-threshold", type=float, default=0.60)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--summary-csv", default="data/reports/buy_7d_feature_subgroups.csv")
    parser.add_argument("--detail-csv", default="data/reports/buy_7d_feature_subgroup_details.csv")
    parser.add_argument("--group-csv", default="data/reports/buy_7d_feature_subgroup_members.csv")
    parser.add_argument("--variant-prefix")
    args = parser.parse_args()
    summary = evaluate_buy_feature_subgroups(
        start_year=args.start_year,
        end_year=args.end_year,
        train_start=args.train_start,
        probability_threshold=args.probability_threshold,
        min_train_rows=args.min_train_rows,
        summary_csv=args.summary_csv,
        detail_csv=args.detail_csv,
        group_csv=args.group_csv,
        variant_prefix=args.variant_prefix,
    )
    columns = [
        "variant",
        "feature_count",
        "signal_count",
        "precision",
        "actual_up_rate",
        "avg_future_ret",
        "median_future_ret",
        "coverage",
    ]
    print("\n=== aggregate subgroup ablation ===")
    print(summary[summary["scope"] == "aggregate"][columns].to_string(index=False))


if __name__ == "__main__":
    main()
