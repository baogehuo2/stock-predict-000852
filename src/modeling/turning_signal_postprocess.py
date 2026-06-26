from __future__ import annotations

import pandas as pd


def assign_bottom_signal_layers(
    prediction: pd.DataFrame,
    watch_threshold: float,
    action_threshold: float,
    top_veto_threshold: float,
    risk_threshold: float | None = None,
    week_kdj_min: float | None = None,
) -> pd.DataFrame:
    result = prediction.copy()
    watch = result["manual_weak_bottom_proba"] >= watch_threshold
    candidate = watch & (result["manual_weak_top_proba"] < top_veto_threshold)
    action = candidate & (result["manual_weak_bottom_proba"] >= action_threshold)
    if risk_threshold is not None and "manual_weak_risk_proba" in result.columns:
        action = action & (result["manual_weak_risk_proba"] <= risk_threshold)
    if week_kdj_min is not None and "f_week_kdj_k_minus_d" in result.columns:
        action = action & (result["f_week_kdj_k_minus_d"] >= week_kdj_min)

    result["signal_layer"] = ""
    result.loc[watch, "signal_layer"] = "bottom_watch"
    result.loc[candidate, "signal_layer"] = "bottom_candidate"
    result.loc[action, "signal_layer"] = "bottom_action"
    return result[result["signal_layer"] != ""].copy()


def compress_top_signal_regions(
    signals: pd.DataFrame,
    max_gap_days: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse dense daily top signals into risk regions plus one representative point."""
    if signals.empty:
        empty_regions = pd.DataFrame(
            columns=[
                "top_region_id",
                "region_start",
                "region_end",
                "signal_count",
                "representative_date",
                "representative_top_proba",
                "avg_future_ret_15d",
                "min_future_ret_15d",
                "max_future_ret_15d",
                "down_rate_15d",
                "continuation_risk_rate",
            ]
        )
        return signals.copy(), empty_regions

    result = signals.sort_values("trade_date").copy()
    gaps = result["trade_date"].diff().dt.days.fillna(max_gap_days + 1)
    result["top_region_id"] = (gaps > max_gap_days).cumsum().astype(int)
    region_rows = []
    representative_indexes = []

    for region_id, part in result.groupby("top_region_id", sort=True):
        representative_idx = part["manual_weak_top_proba"].idxmax()
        representative = result.loc[representative_idx]
        representative_indexes.append(representative_idx)
        complete = part[part["future_ret_15d"].notna()]
        count = int(len(complete))
        region_rows.append(
            {
                "top_region_id": int(region_id),
                "region_start": part["trade_date"].min().date(),
                "region_end": part["trade_date"].max().date(),
                "signal_count": int(len(part)),
                "representative_date": representative["trade_date"].date(),
                "representative_top_proba": float(representative["manual_weak_top_proba"]),
                "avg_future_ret_15d": float(complete["future_ret_15d"].mean()) if count else None,
                "min_future_ret_15d": float(complete["future_ret_15d"].min()) if count else None,
                "max_future_ret_15d": float(complete["future_ret_15d"].max()) if count else None,
                "down_rate_15d": float((complete["future_ret_15d"] < 0).mean()) if count else None,
                "continuation_risk_rate": float(complete["continuation_risk_label"].mean())
                if count
                else None,
            }
        )

    representatives = result.loc[representative_indexes].sort_values("trade_date").copy()
    representatives["is_top_region_representative"] = 1
    regions = pd.DataFrame(region_rows)
    return representatives, regions
