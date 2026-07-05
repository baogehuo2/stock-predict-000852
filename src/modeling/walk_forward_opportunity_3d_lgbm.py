from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import read_sql
from src.modeling.market_regime import assign_evaluation_regime
from src.modeling.probability_lgbm import fit_probability_model, positive_probability
from src.modeling.walk_forward_manual_label_buy_lgbm import load_manual_buy_frame
from src.modeling.buy_feature_set import select_buy_features
from src.features.build_manual_bottom_features import bottom_feature_columns


HORIZON = 3
OPPORTUNITY_MFE_THRESHOLD = 0.01
OPPORTUNITY_MAE_LIMIT = -0.01
RISK_MAE_THRESHOLD = -0.01
RISK_INDEX_HIGH = 70.0
RISK_INDEX_MEDIUM = 50.0


def _parse_floats(value: str) -> list[float]:
    return sorted({float(item.strip()) for item in value.split(",") if item.strip()})


def _future_path() -> pd.DataFrame:
    target = str(get_config()["project"]["target_index"])
    raw = read_sql(
        "SELECT trade_date,index_code,close,high,low FROM market_index_daily "
        "WHERE index_code=:target ORDER BY trade_date",
        {"target": target},
    )
    if raw.empty:
        raise RuntimeError(f"No market_index_daily rows found for {target}.")
    raw["trade_date"] = pd.to_datetime(raw["trade_date"])
    for column in ["close", "high", "low"]:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    high_paths = pd.concat(
        [raw["high"].shift(-step) / raw["close"] - 1 for step in range(1, HORIZON + 1)],
        axis=1,
    )
    low_paths = pd.concat(
        [raw["low"].shift(-step) / raw["close"] - 1 for step in range(1, HORIZON + 1)],
        axis=1,
    )
    out = raw[["trade_date", "index_code"]].copy()
    out["future_ret"] = raw["close"].shift(-HORIZON) / raw["close"] - 1
    out["future_mfe"] = high_paths.max(axis=1)
    out["future_mae"] = low_paths.min(axis=1)
    out["opportunity_label"] = (
        (out["future_mfe"] >= OPPORTUNITY_MFE_THRESHOLD)
        & (out["future_mae"] > OPPORTUNITY_MAE_LIMIT)
        & out["future_ret"].notna()
    ).astype(int)
    out["risk_label"] = (
        (out["future_mae"] <= RISK_MAE_THRESHOLD)
        & out["future_ret"].notna()
    ).astype(int)
    return out


def load_opportunity_frame(
    feature_variant: str = "buy_plus_bottom",
    manual_region_csv: str = "data/manual_labels/market_turning_regions.csv",
) -> tuple[pd.DataFrame, list[str]]:
    include_bottom = feature_variant in {"bottom_only", "buy_plus_bottom", "buy_plus_bottom_daily"}
    data = load_manual_buy_frame(manual_region_csv, include_bottom_features=include_bottom)
    data = data.drop(columns=["future_ret"], errors="ignore")
    data = data.merge(_future_path(), on=["trade_date", "index_code"], how="left", validate="one_to_one")
    buy_features = select_buy_features(data, get_config().get("binary_buy_model", {}))
    bottom_features = bottom_feature_columns(data, include_weekly=feature_variant != "buy_plus_bottom_daily")
    if feature_variant == "buy_v1":
        features = buy_features
    elif feature_variant == "bottom_only":
        features = bottom_features
    elif feature_variant in {"buy_plus_bottom", "buy_plus_bottom_daily"}:
        features = list(dict.fromkeys([*buy_features, *bottom_features]))
    else:
        raise ValueError(f"Unknown feature_variant: {feature_variant}")
    if not features:
        raise RuntimeError(f"No features found for variant={feature_variant}.")
    return data.sort_values("trade_date").reset_index(drop=True), features


def _non_overlapping(signal: pd.DataFrame) -> pd.DataFrame:
    selected = []
    next_allowed = -1
    for idx, row in signal.sort_values("trade_date").iterrows():
        pos = int(row["trade_pos"])
        if pos >= next_allowed:
            selected.append(idx)
            next_allowed = pos + HORIZON
    return signal.loc[selected]


def _fit_probability(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    label_col: str,
    random_state: int,
    model_type: str,
) -> tuple[pd.Series, int]:
    model, usable = fit_probability_model(train, features, label_col, random_state, model_type=model_type)
    return positive_probability(model, test, usable), len(usable)


def _attach_signal_scores(part: pd.DataFrame) -> pd.DataFrame:
    scored = part.copy()
    scored["risk_index"] = (
        100.0
        * (
            0.70 * scored["risk_proba"].clip(0, 1)
            + 0.20 * (1.0 - scored["position_proba"].clip(0, 1))
            + 0.10 * (1.0 - scored["opportunity_proba"].clip(0, 1))
        )
    ).clip(0, 100)
    scored["long_score"] = (
        100.0
        * (
            0.55 * scored["opportunity_proba"].clip(0, 1)
            + 0.30 * scored["position_proba"].clip(0, 1)
            + 0.15 * (1.0 - scored["risk_proba"].clip(0, 1))
        )
    ).clip(0, 100)
    scored["risk_level"] = "low"
    scored.loc[scored["risk_index"] >= RISK_INDEX_MEDIUM, "risk_level"] = "medium"
    scored.loc[scored["risk_index"] >= RISK_INDEX_HIGH, "risk_level"] = "high"
    return scored


def _metric(
    universe: pd.DataFrame,
    signal: pd.DataFrame,
    model_type: str,
    rule: str,
    feature_variant: str,
    opportunity_threshold: float,
    position_threshold: float | None,
    risk_threshold: float | None,
    risk_index_threshold: float | None,
    long_score_threshold: float | None,
    scope: str,
    year: int | None,
) -> dict:
    return {
        "scope": scope,
        "test_year": year,
        "model_type": model_type,
        "feature_variant": feature_variant,
        "rule": rule,
        "opportunity_threshold": opportunity_threshold,
        "position_threshold": position_threshold,
        "risk_threshold": risk_threshold,
        "risk_index_threshold": risk_index_threshold,
        "long_score_threshold": long_score_threshold,
        "universe_rows": int(len(universe)),
        "signal_count": int(len(signal)),
        "coverage": float(len(signal) / len(universe)) if len(universe) else 0.0,
        "opportunity_precision": float(signal["opportunity_label"].mean()) if len(signal) else None,
        "opportunity_base_rate": float(universe["opportunity_label"].mean()) if len(universe) else None,
        "opportunity_lift": (
            float(signal["opportunity_label"].mean() - universe["opportunity_label"].mean())
            if len(signal)
            else None
        ),
        "position_hit_rate": float(signal["manual_buy_label"].mean()) if len(signal) else None,
        "top_hit_rate": float(signal["manual_sell_label"].mean()) if len(signal) else None,
        "risk_rate": float(signal["risk_label"].mean()) if len(signal) else None,
        "future_up_3d_rate": float((signal["future_ret"] > 0).mean()) if len(signal) else None,
        "avg_future_ret": float(signal["future_ret"].mean()) if len(signal) else None,
        "median_future_ret": float(signal["future_ret"].median()) if len(signal) else None,
        "worst_future_ret": float(signal["future_ret"].min()) if len(signal) else None,
        "avg_future_mfe": float(signal["future_mfe"].mean()) if len(signal) else None,
        "avg_future_mae": float(signal["future_mae"].mean()) if len(signal) else None,
        "worst_future_mae": float(signal["future_mae"].min()) if len(signal) else None,
        "natural_avg_future_ret": float(universe["future_ret"].mean()) if len(universe) else None,
        "avg_opportunity_proba": float(signal["opportunity_proba"].mean()) if len(signal) else None,
        "avg_position_proba": float(signal["position_proba"].mean()) if len(signal) else None,
        "avg_risk_proba": float(signal["risk_proba"].mean()) if len(signal) else None,
        "avg_risk_index": float(signal["risk_index"].mean()) if len(signal) else None,
        "max_risk_index": float(signal["risk_index"].max()) if len(signal) else None,
        "high_risk_signal_rate": float((signal["risk_level"] == "high").mean()) if len(signal) else None,
        "avg_long_score": float(signal["long_score"].mean()) if len(signal) else None,
    }


def _rule_mask(
    part: pd.DataFrame,
    rule: str,
    opportunity_threshold: float,
    position_threshold: float | None,
    risk_threshold: float | None,
    risk_index_threshold: float | None,
    long_score_threshold: float | None,
) -> pd.Series:
    if rule == "balanced_long":
        if risk_index_threshold is None or long_score_threshold is None:
            raise ValueError("balanced_long requires risk_index_threshold and long_score_threshold.")
        return (
            (part["opportunity_proba"] >= opportunity_threshold)
            & (part["long_score"] >= long_score_threshold)
            & (part["risk_index"] < risk_index_threshold)
        )
    mask = part["opportunity_proba"] >= opportunity_threshold
    if position_threshold is not None:
        mask = mask & (part["position_proba"] >= position_threshold)
    if risk_threshold is not None:
        mask = mask & (part["risk_proba"] < risk_threshold)
    return mask


def _summarize(
    details: pd.DataFrame,
    model_type: str,
    feature_variant: str,
    opportunity_thresholds: list[float],
    position_thresholds: list[float | None],
    risk_thresholds: list[float | None],
    risk_index_thresholds: list[float],
    long_score_thresholds: list[float],
) -> pd.DataFrame:
    rows = []
    rule_specs = []
    for opp in opportunity_thresholds:
        rule_specs.append(("opportunity_only", opp, None, None))
        for pos in position_thresholds:
            if pos is not None:
                rule_specs.append(("opportunity_plus_position", opp, pos, None))
        for risk in risk_thresholds:
            if risk is not None:
                rule_specs.append(("opportunity_risk_filtered", opp, None, risk))
        for pos in position_thresholds:
            if pos is None:
                continue
            for risk in risk_thresholds:
                if risk is None:
                    continue
                rule_specs.append(("three_stage", opp, pos, risk, None, None))
        for risk_index in risk_index_thresholds:
            for long_score in long_score_thresholds:
                rule_specs.append(("balanced_long", opp, None, None, risk_index, long_score))

    expanded_specs = []
    for spec in rule_specs:
        if len(spec) == 4:
            rule, opp, pos, risk = spec
            expanded_specs.append((rule, opp, pos, risk, None, None))
        else:
            expanded_specs.append(spec)

    seen = set()
    unique_specs = []
    for spec in expanded_specs:
        if spec not in seen:
            seen.add(spec)
            unique_specs.append(spec)

    for rule, opp, pos, risk, risk_index, long_score in unique_specs:
        yearly_signals = []
        yearly_universes = []
        for year, part0 in details.groupby("test_year", sort=True):
            universe = part0.dropna(subset=["future_ret", "future_mfe", "future_mae"]).copy()
            signal = _non_overlapping(universe[_rule_mask(universe, rule, opp, pos, risk, risk_index, long_score)])
            rows.append(
                _metric(
                    universe,
                    signal,
                    model_type,
                    rule,
                    feature_variant,
                    opp,
                    pos,
                    risk,
                    risk_index,
                    long_score,
                    "year",
                    int(year),
                )
            )
            yearly_signals.append(signal)
            yearly_universes.append(universe)
        all_signal = pd.concat(yearly_signals, ignore_index=True) if yearly_signals else pd.DataFrame()
        all_universe = pd.concat(yearly_universes, ignore_index=True) if yearly_universes else pd.DataFrame()
        rows.append(
            _metric(
                all_universe,
                all_signal,
                model_type,
                rule,
                feature_variant,
                opp,
                pos,
                risk,
                risk_index,
                long_score,
                "aggregate",
                None,
            )
        )
    return pd.DataFrame(rows)


def walk_forward_opportunity_3d_evaluation(
    model_type: str = "lgbm",
    feature_variant: str = "buy_plus_bottom",
    start_year: int = 2021,
    end_year: int = 2026,
    train_start: str = "2016-01-01",
    opportunity_thresholds: list[float] | None = None,
    position_thresholds: list[float | None] | None = None,
    risk_thresholds: list[float | None] | None = None,
    risk_index_thresholds: list[float] | None = None,
    long_score_thresholds: list[float] | None = None,
    min_train_rows: int = 200,
    manual_region_csv: str = "data/manual_labels/market_turning_regions.csv",
    summary_csv: str = "data/reports/opportunity_3d_walk_forward_summary.csv",
    detail_csv: str = "data/reports/opportunity_3d_walk_forward_details.csv",
) -> pd.DataFrame:
    opportunity_thresholds = opportunity_thresholds or [0.20, 0.30, 0.40, 0.50, 0.60]
    position_thresholds = position_thresholds or [None, 0.20, 0.30]
    risk_thresholds = risk_thresholds or [None, 0.40, 0.50]
    risk_index_thresholds = risk_index_thresholds or [50.0, 60.0, 70.0]
    long_score_thresholds = long_score_thresholds or [45.0, 50.0, 55.0]
    data, features = load_opportunity_frame(feature_variant, manual_region_csv)
    random_state = int(get_config()["model"].get("random_state", 42))
    train_start_ts = pd.Timestamp(train_start)
    detail_parts = []

    for test_year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(f"{test_year}-01-01")
        test_end = pd.Timestamp(f"{test_year}-12-31")
        train = data[
            (data["trade_date"] >= train_start_ts) & (data["trade_date"] < test_start)
        ].dropna(subset=["opportunity_label", "manual_buy_label", "risk_label"]).copy()
        test = data[
            (data["trade_date"] >= test_start) & (data["trade_date"] <= test_end)
        ].dropna(subset=["opportunity_label", "manual_buy_label", "risk_label"]).copy()
        train = train.sort_values("trade_date")
        test = test.sort_values("trade_date")
        if len(train) > HORIZON:
            train = train.iloc[:-HORIZON].copy()
        if len(train) < min_train_rows or test.empty:
            continue
        if train["opportunity_label"].nunique() < 2:
            continue
        if train["manual_buy_label"].nunique() < 2:
            continue
        if train["risk_label"].nunique() < 2:
            continue

        opportunity_proba, opportunity_feature_count = _fit_probability(
            train, test, features, "opportunity_label", random_state, model_type
        )
        position_proba, position_feature_count = _fit_probability(
            train, test, features, "manual_buy_label", random_state, model_type
        )
        risk_proba, risk_feature_count = _fit_probability(
            train, test, features, "risk_label", random_state, model_type
        )

        part = test[
            [
                "trade_date",
                "index_code",
                "future_ret",
                "future_mfe",
                "future_mae",
                "opportunity_label",
                "risk_label",
                "manual_buy_label",
                "manual_sell_label",
            ]
        ].copy()
        part["opportunity_proba"] = opportunity_proba
        part["position_proba"] = position_proba
        part["risk_proba"] = risk_proba
        part = _attach_signal_scores(part)
        part["model_type"] = model_type
        part["feature_variant"] = feature_variant
        part["test_year"] = test_year
        part["train_start"] = train["trade_date"].min().strftime("%Y-%m-%d")
        part["train_end"] = train["trade_date"].max().strftime("%Y-%m-%d")
        part["train_rows"] = len(train)
        part["opportunity_feature_count"] = opportunity_feature_count
        part["position_feature_count"] = position_feature_count
        part["risk_feature_count"] = risk_feature_count
        part["evaluation_regime"] = assign_evaluation_regime(part["trade_date"])
        part["trade_pos"] = range(len(part))
        detail_parts.append(part)

    if not detail_parts:
        raise RuntimeError("No opportunity walk-forward folds were produced.")
    details = pd.concat(detail_parts, ignore_index=True)
    summary = _summarize(
        details,
        model_type,
        feature_variant,
        opportunity_thresholds,
        position_thresholds,
        risk_thresholds,
        risk_index_thresholds,
        long_score_thresholds,
    )
    for frame, output in [(summary, summary_csv), (details.drop(columns=["trade_pos"]), detail_csv)]:
        path = project_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"saved report path={path} rows={len(frame)}")
    return summary


def _parse_optional_floats(value: str) -> list[float | None]:
    if not value:
        return [None]
    return [None, *_parse_floats(value)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward 3-day opportunity/position/risk model.")
    parser.add_argument("--model-type", choices=["lgbm", "xgboost", "bp"], default="lgbm")
    parser.add_argument(
        "--feature-variant",
        choices=["buy_v1", "bottom_only", "buy_plus_bottom", "buy_plus_bottom_daily"],
        default="buy_plus_bottom",
    )
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--opportunity-thresholds", default="0.30,0.40,0.50,0.60")
    parser.add_argument("--position-thresholds", default="0.20,0.30")
    parser.add_argument("--risk-thresholds", default="0.40,0.50")
    parser.add_argument("--risk-index-thresholds", default="50,60,70")
    parser.add_argument("--long-score-thresholds", default="45,50,55")
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--manual-region-csv", default="data/manual_labels/market_turning_regions.csv")
    parser.add_argument("--summary-csv", default="data/reports/opportunity_3d_walk_forward_summary.csv")
    parser.add_argument("--detail-csv", default="data/reports/opportunity_3d_walk_forward_details.csv")
    args = parser.parse_args()
    summary = walk_forward_opportunity_3d_evaluation(
        model_type=args.model_type,
        feature_variant=args.feature_variant,
        start_year=args.start_year,
        end_year=args.end_year,
        train_start=args.train_start,
        opportunity_thresholds=_parse_floats(args.opportunity_thresholds),
        position_thresholds=_parse_optional_floats(args.position_thresholds),
        risk_thresholds=_parse_optional_floats(args.risk_thresholds),
        risk_index_thresholds=_parse_floats(args.risk_index_thresholds),
        long_score_thresholds=_parse_floats(args.long_score_thresholds),
        min_train_rows=args.min_train_rows,
        manual_region_csv=args.manual_region_csv,
        summary_csv=args.summary_csv,
        detail_csv=args.detail_csv,
    )
    columns = [
        "feature_variant",
        "model_type",
        "rule",
        "opportunity_threshold",
        "position_threshold",
        "risk_threshold",
        "risk_index_threshold",
        "long_score_threshold",
        "signal_count",
        "opportunity_precision",
        "position_hit_rate",
        "top_hit_rate",
        "risk_rate",
        "future_up_3d_rate",
        "avg_future_ret",
        "avg_future_mfe",
        "avg_future_mae",
        "worst_future_ret",
        "avg_risk_index",
        "max_risk_index",
        "high_risk_signal_rate",
        "avg_long_score",
    ]
    print(summary[summary["scope"] == "aggregate"][columns].to_string(index=False))


if __name__ == "__main__":
    main()
