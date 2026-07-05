from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import read_sql
from src.features.build_manual_bottom_features import (
    bottom_feature_columns,
    build_manual_bottom_feature_frame,
)
from src.modeling.buy_feature_set import select_buy_features
from src.modeling.data import load_dataset
from src.modeling.market_regime import assign_evaluation_regime
from src.modeling.probability_lgbm import fit_probability_model, positive_probability


HORIZON = 20
REQUIRED_REGION_COLUMNS = [
    "region_id",
    "start_date",
    "end_date",
    "region_type",
    "cycle",
    "level",
    "label_freq",
    "confidence",
    "usable_for_signal",
    "entry_start",
    "entry_end",
    "exit_start",
    "exit_end",
    "reason",
    "notes",
]


def _parse_floats(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def _manual_regions(path: str | Path) -> pd.DataFrame:
    regions = pd.read_csv(path, dtype=str).fillna("")
    missing = [column for column in REQUIRED_REGION_COLUMNS if column not in regions.columns]
    if missing:
        raise RuntimeError(f"manual region file is missing columns: {missing}")
    regions = regions[REQUIRED_REGION_COLUMNS].copy()
    for column in ["start_date", "end_date", "entry_start", "entry_end", "exit_start", "exit_end"]:
        regions[column] = pd.to_datetime(regions[column], errors="coerce")
    regions["confidence"] = pd.to_numeric(regions["confidence"], errors="coerce").fillna(0).astype(int)
    regions["usable_for_signal"] = pd.to_numeric(
        regions["usable_for_signal"], errors="coerce"
    ).fillna(0).astype(int)
    return regions


def _expand_manual_labels(dates: pd.DataFrame, regions: pd.DataFrame) -> pd.DataFrame:
    labels = dates[["trade_date", "index_code"]].copy()
    labels["manual_buy_label"] = 0
    labels["manual_sell_label"] = 0
    labels["manual_bottom_region"] = 0
    labels["manual_top_region"] = 0
    labels["manual_bottom_confidence"] = 0
    labels["manual_top_confidence"] = 0

    for region in regions.itertuples(index=False):
        region_mask = (labels["trade_date"] >= region.start_date) & (labels["trade_date"] <= region.end_date)
        if region.region_type == "bottom":
            labels.loc[region_mask, "manual_bottom_region"] = 1
            labels.loc[region_mask, "manual_bottom_confidence"] = labels.loc[
                region_mask, "manual_bottom_confidence"
            ].clip(lower=int(region.confidence))
            if region.usable_for_signal == 1 and pd.notna(region.entry_start) and pd.notna(region.entry_end):
                entry_mask = (
                    (labels["trade_date"] >= region.entry_start)
                    & (labels["trade_date"] <= region.entry_end)
                )
                labels.loc[entry_mask, "manual_buy_label"] = 1
        elif region.region_type == "top":
            labels.loc[region_mask, "manual_top_region"] = 1
            labels.loc[region_mask, "manual_top_confidence"] = labels.loc[
                region_mask, "manual_top_confidence"
            ].clip(lower=int(region.confidence))
            if region.usable_for_signal == 1 and pd.notna(region.exit_start) and pd.notna(region.exit_end):
                exit_mask = (
                    (labels["trade_date"] >= region.exit_start)
                    & (labels["trade_date"] <= region.exit_end)
                )
                labels.loc[exit_mask, "manual_sell_label"] = 1
    return labels


def _future_ret_20d() -> pd.DataFrame:
    cfg = get_config()
    target = str(cfg["project"]["target_index"])
    raw = read_sql(
        "SELECT trade_date,index_code,close FROM market_index_daily "
        "WHERE index_code=:target ORDER BY trade_date",
        {"target": target},
    )
    if raw.empty:
        raise RuntimeError(f"No market_index_daily rows found for {target}.")
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw["future_ret_20d"] = raw["close"].shift(-HORIZON) / raw["close"] - 1
    return raw[["trade_date", "index_code", "future_ret_20d"]]


def load_manual_buy_frame(
    manual_region_csv: str = "data/manual_labels/market_turning_regions.csv",
    include_bottom_features: bool = False,
) -> pd.DataFrame:
    data = load_dataset()
    if data.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    data = data.sort_values("trade_date").copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    regions = _manual_regions(project_path(manual_region_csv))
    labels = _expand_manual_labels(data[["trade_date", "index_code"]], regions)
    data = data.merge(labels, on=["trade_date", "index_code"], how="left", validate="one_to_one")
    data = data.merge(_future_ret_20d(), on=["trade_date", "index_code"], how="left", validate="one_to_one")
    if include_bottom_features:
        bottom_features = build_manual_bottom_feature_frame()
        data = data.merge(bottom_features, on=["trade_date", "index_code"], how="left", validate="one_to_one")
    for column in [
        "manual_buy_label",
        "manual_sell_label",
        "manual_bottom_region",
        "manual_top_region",
        "manual_bottom_confidence",
        "manual_top_confidence",
    ]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int)
    return data


def _non_overlapping(signal: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    selected = []
    next_allowed_pos = -1
    for idx, row in signal.sort_values("trade_date").iterrows():
        position = int(row["trade_pos"])
        if position >= next_allowed_pos:
            selected.append(idx)
            next_allowed_pos = position + horizon
    return signal.loc[selected]


def _metric(
    universe: pd.DataFrame,
    signal: pd.DataFrame,
    threshold: float,
    scope: str,
    year: int | None,
    variant: str,
    top_risk_threshold: float | None,
) -> dict:
    if top_risk_threshold is None or "manual_top_proba" not in signal:
        gated_signal = signal
    else:
        gated_signal = signal[signal["manual_top_proba"] < top_risk_threshold]
    return {
        "scope": scope,
        "test_year": year,
        "variant": variant,
        "threshold": threshold,
        "top_risk_threshold": top_risk_threshold,
        "universe_rows": int(len(universe)),
        "signal_count": int(len(signal)),
        "gated_signal_count": int(len(gated_signal)),
        "coverage": float(len(signal) / len(universe)) if len(universe) else 0.0,
        "manual_buy_precision": float(signal["manual_buy_label"].mean()) if len(signal) else None,
        "gated_manual_buy_precision": float(gated_signal["manual_buy_label"].mean()) if len(gated_signal) else None,
        "manual_buy_base_rate": float(universe["manual_buy_label"].mean()) if len(universe) else None,
        "manual_precision_lift": (
            float(signal["manual_buy_label"].mean() - universe["manual_buy_label"].mean()) if len(signal) else None
        ),
        "manual_sell_rate": float(signal["manual_sell_label"].mean()) if len(signal) else None,
        "gated_manual_sell_rate": float(gated_signal["manual_sell_label"].mean()) if len(gated_signal) else None,
        "future_up_20d_rate": float((signal["future_ret_20d"] > 0).mean()) if len(signal) else None,
        "gated_future_up_20d_rate": float((gated_signal["future_ret_20d"] > 0).mean()) if len(gated_signal) else None,
        "avg_future_ret_20d": float(signal["future_ret_20d"].mean()) if len(signal) else None,
        "gated_avg_future_ret_20d": float(gated_signal["future_ret_20d"].mean()) if len(gated_signal) else None,
        "median_future_ret_20d": float(signal["future_ret_20d"].median()) if len(signal) else None,
        "worst_future_ret_20d": float(signal["future_ret_20d"].min()) if len(signal) else None,
        "gated_worst_future_ret_20d": float(gated_signal["future_ret_20d"].min()) if len(gated_signal) else None,
        "natural_future_up_20d_rate": float((universe["future_ret_20d"] > 0).mean()) if len(universe) else None,
        "natural_avg_future_ret_20d": float(universe["future_ret_20d"].mean()) if len(universe) else None,
        "avg_manual_buy_proba": float(signal["manual_buy_proba"].mean()) if len(signal) else None,
        "avg_manual_top_proba": float(signal["manual_top_proba"].mean()) if len(signal) and "manual_top_proba" in signal else None,
    }


def _summarize(
    details: pd.DataFrame,
    thresholds: list[float],
    variant: str,
    top_risk_thresholds: list[float | None],
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        for top_risk_threshold in top_risk_thresholds:
            yearly_signals = []
            yearly_universes = []
            for year, part0 in details.groupby("test_year", sort=True):
                part = part0.sort_values("trade_date").copy()
                universe = part.dropna(subset=["future_ret_20d"]).copy()
                signal = _non_overlapping(universe[universe["manual_buy_proba"] >= threshold])
                rows.append(_metric(universe, signal, threshold, "year", int(year), variant, top_risk_threshold))
                yearly_signals.append(signal)
                yearly_universes.append(universe)
            all_signal = pd.concat(yearly_signals, ignore_index=True) if yearly_signals else pd.DataFrame()
            all_universe = pd.concat(yearly_universes, ignore_index=True) if yearly_universes else pd.DataFrame()
            rows.append(_metric(all_universe, all_signal, threshold, "aggregate", None, variant, top_risk_threshold))
    return pd.DataFrame(rows)


def walk_forward_manual_label_buy_evaluation(
    start_year: int = 2021,
    end_year: int = 2026,
    train_start: str = "2016-01-01",
    thresholds: list[float] | None = None,
    top_risk_thresholds: list[float | None] | None = None,
    min_train_rows: int = 200,
    manual_region_csv: str = "data/manual_labels/market_turning_regions.csv",
    feature_variant: str = "buy_v1",
    summary_csv: str = "data/reports/manual_label_buy_walk_forward_summary.csv",
    detail_csv: str = "data/reports/manual_label_buy_walk_forward_details.csv",
) -> pd.DataFrame:
    thresholds = thresholds or [0.40, 0.50, 0.60, 0.70]
    top_risk_thresholds = top_risk_thresholds or [None]
    cfg = get_config()
    buy_cfg = cfg.get("binary_buy_model", {})
    include_bottom = feature_variant in {"bottom_only", "buy_plus_bottom", "buy_plus_bottom_daily"}
    data = load_manual_buy_frame(manual_region_csv, include_bottom_features=include_bottom)
    buy_features = select_buy_features(data, buy_cfg)
    bottom_features = bottom_feature_columns(
        data,
        include_weekly=feature_variant != "buy_plus_bottom_daily",
    )
    if feature_variant == "buy_v1":
        features = buy_features
    elif feature_variant == "bottom_only":
        features = bottom_features
    elif feature_variant in {"buy_plus_bottom", "buy_plus_bottom_daily"}:
        features = list(dict.fromkeys([*buy_features, *bottom_features]))
    else:
        raise ValueError(f"Unknown feature_variant: {feature_variant}")
    if not features:
        raise RuntimeError("No feature columns found for manual label training.")
    random_state = int(cfg["model"].get("random_state", 42))
    train_window_start = pd.Timestamp(train_start)
    detail_parts = []

    for test_year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        train = data[
            (data["trade_date"] >= train_window_start)
            & (data["trade_date"] < test_start)
        ].dropna(subset=["manual_buy_label"]).sort_values("trade_date").copy()
        test = data[
            (data["trade_date"] >= test_start)
            & (data["trade_date"] <= test_end)
        ].dropna(subset=["manual_buy_label"]).sort_values("trade_date").copy()
        if len(train) < min_train_rows or test.empty:
            continue
        if train["manual_buy_label"].nunique() < 2:
            continue
        buy_model, usable = fit_probability_model(train, features, "manual_buy_label", random_state)
        top_model, top_usable = fit_probability_model(train, features, "manual_sell_label", random_state)
        part = test[
            [
                "trade_date",
                "index_code",
                "manual_buy_label",
                "manual_sell_label",
                "manual_bottom_region",
                "manual_top_region",
                "future_ret_7d",
                "future_ret_20d",
            ]
        ].copy()
        part["manual_buy_proba"] = positive_probability(buy_model, test, usable)
        part["manual_top_proba"] = positive_probability(top_model, test, top_usable)
        part["feature_variant"] = feature_variant
        part["test_year"] = test_year
        part["train_start"] = train["trade_date"].min().strftime("%Y-%m-%d")
        part["train_end"] = train["trade_date"].max().strftime("%Y-%m-%d")
        part["train_rows"] = len(train)
        part["positive_train_rows"] = int(train["manual_buy_label"].sum())
        part["feature_count"] = len(usable)
        part["top_feature_count"] = len(top_usable)
        part["evaluation_regime"] = assign_evaluation_regime(part["trade_date"])
        part["trade_pos"] = range(len(part))
        detail_parts.append(part)

    if not detail_parts:
        raise RuntimeError("No manual label walk-forward folds were produced.")
    details = pd.concat(detail_parts, ignore_index=True)
    summary = _summarize(details, thresholds, feature_variant, top_risk_thresholds)
    for frame, output in [
        (summary, summary_csv),
        (details.drop(columns=["trade_pos"]), detail_csv),
    ]:
        path = project_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved report path={path} rows={len(frame)}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward Buy model trained on manual turning labels.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--thresholds", default="0.40,0.50,0.60,0.70")
    parser.add_argument("--top-risk-thresholds", default="")
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--manual-region-csv", default="data/manual_labels/market_turning_regions.csv")
    parser.add_argument(
        "--feature-variant",
        choices=["buy_v1", "bottom_only", "buy_plus_bottom", "buy_plus_bottom_daily"],
        default="buy_v1",
    )
    parser.add_argument("--summary-csv", default="data/reports/manual_label_buy_walk_forward_summary.csv")
    parser.add_argument("--detail-csv", default="data/reports/manual_label_buy_walk_forward_details.csv")
    args = parser.parse_args()
    summary = walk_forward_manual_label_buy_evaluation(
        start_year=args.start_year,
        end_year=args.end_year,
        train_start=args.train_start,
        thresholds=_parse_floats(args.thresholds),
        top_risk_thresholds=[None] + _parse_floats(args.top_risk_thresholds) if args.top_risk_thresholds else [None],
        min_train_rows=args.min_train_rows,
        manual_region_csv=args.manual_region_csv,
        feature_variant=args.feature_variant,
        summary_csv=args.summary_csv,
        detail_csv=args.detail_csv,
    )
    columns = [
        "scope",
        "test_year",
        "threshold",
        "top_risk_threshold",
        "signal_count",
        "gated_signal_count",
        "manual_buy_precision",
        "gated_manual_buy_precision",
        "manual_precision_lift",
        "manual_sell_rate",
        "gated_manual_sell_rate",
        "future_up_20d_rate",
        "gated_future_up_20d_rate",
        "avg_future_ret_20d",
        "gated_avg_future_ret_20d",
        "worst_future_ret_20d",
        "gated_worst_future_ret_20d",
    ]
    print(summary[summary["scope"] == "aggregate"][columns].to_string(index=False))


if __name__ == "__main__":
    main()
