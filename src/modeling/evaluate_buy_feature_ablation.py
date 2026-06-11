from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import get_config, project_path
from src.common.logger import get_logger
from src.modeling.data import feature_columns, load_dataset
from src.modeling.train_lgbm import GUBA_SENTIMENT_FEATURES
from src.modeling.walk_forward_buy_lgbm import _fit_model, _positive_probability


logger = get_logger(__name__)

HORIZON = 7

BASIC_MARKET_FEATURES = {
    "f_ret_1d",
    "f_ret_3d",
    "f_ret_5d",
    "f_ret_10d",
    "f_upper_shadow_ratio",
    "f_lower_shadow_ratio",
    "f_body_ratio",
    "f_intraday_range",
    "f_upper_probe",
    "f_lower_probe",
    "f_doji",
    "f_long_upper_shadow",
    "f_long_lower_shadow",
    "f_heavy_upper_shadow",
    "f_heavy_lower_shadow",
    "f_gap_up",
    "f_gap_down",
    "f_large_range_day",
}

TECHNICAL_FEATURES = {
    "f_ma5_gap",
    "f_ma10_gap",
    "f_ma20_gap",
    "f_macd",
    "f_macd_signal",
    "f_macd_hist",
    "f_macd_golden_cross",
    "f_macd_dead_cross",
    "f_macd_golden_cross_days",
    "f_macd_dead_cross_days",
    "f_macd_hist_turn_positive",
    "f_macd_hist_turn_negative",
    "f_rsi6",
    "f_rsi14",
    "f_kdj_k",
    "f_kdj_d",
    "f_kdj_j",
    "f_kdj_k_minus_d",
    "f_kdj_golden_cross",
    "f_kdj_dead_cross",
    "f_kdj_golden_cross_days",
    "f_kdj_dead_cross_days",
    "f_kdj_j_overbought",
    "f_kdj_j_oversold",
    "f_kdj_j_overbought_turn_down",
    "f_expma12",
    "f_expma50",
    "f_expma12_gap",
    "f_expma_golden_cross",
    "f_expma_dead_cross",
    "f_expma_golden_cross_days",
    "f_expma_dead_cross_days",
    "f_cci14",
    "f_cci14_overbought",
    "f_cci14_turn_down",
    "f_cci14_overbought_turn_down",
    "f_wr14",
    "f_atr14",
    "f_boll_width",
    "f_boll_band_position",
}

RISK_FEATURES = {
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
    "f_amount_zscore_20d",
    "f_amount_percentile_60d",
    "f_volume_zscore_20d",
    "f_volume_ratio_5d_20d",
    "f_volume_percentile_60d",
    "f_down_with_volume",
    "f_up_with_volume",
    "f_shrink_rebound",
    "f_volatility_5d",
    "f_volatility_10d",
    "f_volatility_20d",
    "f_volatility_expand",
    "f_atr_expand",
    "f_boll_lower_break",
    "f_boll_upper_break",
    "f_regime_trend_score",
    "f_is_bull_trend",
    "f_is_bear_trend",
    "f_regime_state_days",
}

STYLE_MARKET_FEATURES = {
    "f_relative_hs300",
    "f_relative_zz500",
    "f_relative_cyb",
}


def _parse_ints(value: str) -> list[int]:
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def _event_features(features: list[str]) -> set[str]:
    return {feature for feature in features if feature.startswith("f_event_") or feature == "f_event_score"}


def _feature_groups(data: pd.DataFrame) -> dict[str, list[str]]:
    cfg = get_config()
    available = feature_columns(data)
    if not cfg.get("features", {}).get("use_guba_sentiment", False):
        available = [feature for feature in available if feature not in GUBA_SENTIMENT_FEATURES]
    event_features = _event_features(available)
    groups = {
        "basic_market": [feature for feature in available if feature in BASIC_MARKET_FEATURES],
        "technical": [feature for feature in available if feature in TECHNICAL_FEATURES],
        "trend_volume_risk": [feature for feature in available if feature in RISK_FEATURES],
        "style_market": [feature for feature in available if feature in STYLE_MARKET_FEATURES],
        "news_event": [feature for feature in available if feature in event_features],
    }
    assigned = set().union(*map(set, groups.values()))
    unassigned = sorted(set(available) - assigned)
    if unassigned:
        raise RuntimeError(f"Unassigned feature columns: {unassigned}")
    return groups


def _ordered_subset(available: list[str], selected: set[str]) -> list[str]:
    return [feature for feature in available if feature in selected]


def _variants(groups: dict[str, list[str]], available: list[str]) -> dict[str, list[str]]:
    order = ["basic_market", "technical", "trend_volume_risk", "style_market", "news_event"]
    variants = {}
    selected: set[str] = set()
    for index, group in enumerate(order, start=1):
        selected.update(groups[group])
        variants[f"{index}_{group}"] = _ordered_subset(available, selected)
    variants["6_basic_plus_risk"] = _ordered_subset(
        available, set(groups["basic_market"]) | set(groups["trend_volume_risk"])
    )
    variants["7_core_no_style"] = _ordered_subset(
        available, set(available) - set(groups["style_market"])
    )
    variants["8_full_no_technical"] = _ordered_subset(
        available, set(available) - set(groups["technical"])
    )
    variants["9_full_no_risk"] = _ordered_subset(
        available, set(available) - set(groups["trend_volume_risk"])
    )
    return variants


def _non_overlapping(signal: pd.DataFrame) -> pd.DataFrame:
    if signal.empty:
        return signal
    selected = []
    next_allowed_position = -1
    for idx, row in signal.sort_values("trade_date").iterrows():
        position = int(row["trade_pos"])
        if position >= next_allowed_position:
            selected.append(idx)
            next_allowed_position = position + HORIZON
    return signal.loc[selected]


def _metrics(
    signal: pd.DataFrame,
    universe: pd.DataFrame,
    variant: str,
    feature_count: int,
    label_threshold: float,
    probability_threshold: float,
    scope: str,
    test_year: int | None,
) -> dict:
    target_correct = signal["future_ret"] > label_threshold
    actual_up = signal["future_ret"] > 0
    natural_target = universe["future_ret"] > label_threshold
    natural_up = universe["future_ret"] > 0
    return {
        "scope": scope,
        "test_year": test_year,
        "variant": variant,
        "feature_count": feature_count,
        "label_threshold": label_threshold,
        "probability_threshold": probability_threshold,
        "universe_rows": int(len(universe)),
        "signal_count": int(len(signal)),
        "coverage": float(len(signal) / len(universe)) if len(universe) else 0.0,
        "precision": float(target_correct.mean()) if len(signal) else None,
        "actual_up_rate": float(actual_up.mean()) if len(signal) else None,
        "avg_future_ret": float(signal["future_ret"].mean()) if len(signal) else None,
        "median_future_ret": float(signal["future_ret"].median()) if len(signal) else None,
        "avg_buy_proba": float(signal["buy_proba"].mean()) if len(signal) else None,
        "natural_precision": float(natural_target.mean()) if len(universe) else None,
        "natural_actual_up_rate": float(natural_up.mean()) if len(universe) else None,
        "natural_avg_future_ret": float(universe["future_ret"].mean()) if len(universe) else None,
    }


def evaluate_buy_feature_ablation(
    start_year: int = 2021,
    end_year: int = 2026,
    train_start: str = "2016-01-01",
    probability_threshold: float = 0.60,
    min_train_rows: int = 200,
    summary_csv: str = "data/reports/buy_7d_feature_ablation.csv",
    detail_csv: str = "data/reports/buy_7d_feature_ablation_details.csv",
    group_csv: str = "data/reports/buy_7d_feature_groups.csv",
) -> pd.DataFrame:
    cfg = get_config()
    label_threshold = float(
        cfg.get("binary_buy_model", {}).get("label_thresholds", {}).get(str(HORIZON), 0.0)
    )
    random_state = int(cfg["model"].get("random_state", 42))
    data = load_dataset()
    if data.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    groups = _feature_groups(data)
    available = feature_columns(data)
    if not cfg.get("features", {}).get("use_guba_sentiment", False):
        available = [feature for feature in available if feature not in GUBA_SENTIMENT_FEATURES]
    variants = _variants(groups, available)
    group_rows = [
        {"group": group, "feature": feature, "non_null_rows": int(data[feature].notna().sum())}
        for group, features in groups.items()
        for feature in features
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
            model, usable_features = _fit_model(
                train, features, "buy_label_7d", random_state
            )
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
                "feature ablation variant=%s year=%s features=%s train_rows=%s test_rows=%s signals=%s",
                variant,
                test_year,
                len(usable_features),
                len(train),
                len(test),
                len(signal),
            )

    details = pd.concat(detail_parts, ignore_index=True)
    for variant, variant_part in details.groupby("variant", sort=True):
        universe = variant_part[variant_part["f_is_bull_trend"] == 1]
        yearly_signals = []
        for _, year_part in universe.groupby("test_year", sort=True):
            yearly_signals.append(
                _non_overlapping(year_part[year_part["buy_proba"] >= probability_threshold])
            )
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
    print(f"saved feature groups path={group_path} rows={len(group_rows)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Cumulative feature ablation for the 7-day buy model.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--probability-threshold", type=float, default=0.60)
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--summary-csv", default="data/reports/buy_7d_feature_ablation.csv")
    parser.add_argument("--detail-csv", default="data/reports/buy_7d_feature_ablation_details.csv")
    parser.add_argument("--group-csv", default="data/reports/buy_7d_feature_groups.csv")
    args = parser.parse_args()
    summary = evaluate_buy_feature_ablation(
        start_year=args.start_year,
        end_year=args.end_year,
        train_start=args.train_start,
        probability_threshold=args.probability_threshold,
        min_train_rows=args.min_train_rows,
        summary_csv=args.summary_csv,
        detail_csv=args.detail_csv,
        group_csv=args.group_csv,
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
    print("\n=== aggregate hard-gate non-overlapping feature ablation ===")
    print(summary[summary["scope"] == "aggregate"][columns].to_string(index=False))


if __name__ == "__main__":
    main()
