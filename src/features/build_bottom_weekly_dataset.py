from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import execute_sql, get_database_name, read_sql, upsert_dataframe
from src.common.logger import get_logger
from src.features.build_manual_turning_labels import (
    CYCLE_SCORE,
    LABEL_FREQ_SCORE,
    LEVEL_SCORE,
    load_manual_regions,
)
from src.features.technical_indicators import atr, boll, cci, kdj, macd, rsi_cn, wr
from src.modeling.bottom_weekly_data import assert_bottom_weekly_database


logger = get_logger(__name__)

EVENT_WEEKLY_WINDOWS = [4, 8, 12]
EVENT_WEEKLY_GROUPS = {
    "policy": ("event_type", "政策"),
    "macro": ("event_type", "宏观"),
    "liquidity": ("event_type", "流动性"),
    "regulation": ("event_type", "监管"),
    "industry": ("event_type", "产业"),
    "diplomacy": ("event_type", "外交"),
    "small_cap": ("affected_style", "小盘"),
    "growth": ("affected_style", "成长"),
}


def _consecutive_periods(mask: pd.Series) -> pd.Series:
    values = mask.fillna(False).astype(bool)
    groups = values.ne(values.shift(fill_value=False)).cumsum()
    periods = values.groupby(groups).cumcount() + 1
    return periods.where(values, 0).astype(float)


def _join_ids(values: list[str]) -> str:
    return "|".join(sorted(set(value for value in values if value)))


def _join_ordered(values: list[str]) -> str:
    seen = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return "|".join(seen)


def _overlap_mask(
    week_start: pd.Series,
    week_end: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    return (week_start <= end_date) & (week_end >= start_date)


def _contains(value: object, keyword: str) -> bool:
    if value is None or pd.isna(value):
        return False
    return keyword in str(value)


def _direction_score(events: pd.DataFrame) -> pd.Series:
    direction = events["impact_direction"].fillna("").astype(str)
    strength = pd.to_numeric(events["impact_strength"], errors="coerce").fillna(1.0).clip(lower=0)
    score = pd.Series(0.0, index=events.index)
    score.loc[direction.str.contains("利多|利好|多头", regex=True)] = strength
    score.loc[direction.str.contains("利空|空头", regex=True)] = -strength
    return score


def _aggregate_event_part(part: pd.DataFrame, prefix: str) -> dict[str, float]:
    score = pd.to_numeric(part["event_score"], errors="coerce").fillna(0.0)
    strength = pd.to_numeric(part["impact_strength"], errors="coerce").fillna(0.0)
    direction = pd.to_numeric(part["direction_score"], errors="coerce").fillna(0.0)
    return {
        f"{prefix}_count": float(len(part)),
        f"{prefix}_score_sum": float(score.sum()),
        f"{prefix}_score_mean": float(score.mean()) if len(part) else 0.0,
        f"{prefix}_positive_count": float((direction > 0).sum()),
        f"{prefix}_negative_count": float((direction < 0).sum()),
        f"{prefix}_direction_sum": float(direction.sum()),
        f"{prefix}_max_strength": float(strength.max()) if len(part) else 0.0,
    }


def build_weekly_market_bars(daily: pd.DataFrame) -> pd.DataFrame:
    data = daily.sort_values("trade_date").reset_index(drop=True).copy()
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    data["week_period"] = data["trade_date"].dt.to_period("W-FRI")
    weekly = (
        data.groupby("week_period", sort=True)
        .agg(
            week_start_date=("trade_date", "min"),
            week_end_date=("trade_date", "max"),
            index_code=("index_code", "last"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trade_days=("trade_date", "count"),
        )
        .reset_index(drop=True)
    )
    weekly["avg_daily_volume"] = weekly["volume"] / weekly["trade_days"].replace(0, np.nan)
    weekly["week_pos"] = np.arange(len(weekly))
    return weekly


def _build_monthly_bars_until(daily: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    data = daily[daily["trade_date"] <= end_date].sort_values("trade_date").copy()
    data["month_period"] = data["trade_date"].dt.to_period("M")
    return (
        data.groupby("month_period", sort=True)
        .agg(
            month_start_date=("trade_date", "min"),
            month_end_date=("trade_date", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            trade_days=("trade_date", "count"),
        )
        .reset_index(drop=True)
    )


def _monthly_state_record(monthly: pd.DataFrame, week_end_date: pd.Timestamp) -> dict[str, float | pd.Timestamp]:
    close = pd.to_numeric(monthly["close"], errors="coerce")
    high = pd.to_numeric(monthly["high"], errors="coerce")
    low = pd.to_numeric(monthly["low"], errors="coerce")
    open_ = pd.to_numeric(monthly["open"], errors="coerce")
    last = len(monthly) - 1
    record: dict[str, float | pd.Timestamp] = {"week_end_date": week_end_date}

    for window in [1, 3, 6]:
        record[f"month_ret_{window}m"] = float(close.pct_change(window).iloc[last])
    for window in [6, 12]:
        record[f"month_drawdown_{window}m"] = float(close.iloc[last] / high.rolling(window).max().iloc[last] - 1)
        record[f"month_runup_{window}m"] = float(close.iloc[last] / low.rolling(window).min().iloc[last] - 1)
    for window in [3, 6, 12]:
        average = close.rolling(window).mean()
        record[f"month_ma{window}_gap"] = float(close.iloc[last] / average.iloc[last] - 1)

    macd_frame = macd(close)
    record["month_macd"] = float(macd_frame["macd_dif"].iloc[last])
    record["month_macd_signal"] = float(macd_frame["macd_dea"].iloc[last])
    record["month_macd_hist"] = float(macd_frame["macd_hist"].iloc[last])
    record["month_rsi6"] = float(rsi_cn(close, 6).iloc[last])
    record["month_rsi12"] = float(rsi_cn(close, 12).iloc[last])

    month_range = high.iloc[last] - low.iloc[last]
    record["month_close_position"] = float((close.iloc[last] - low.iloc[last]) / month_range) if month_range else np.nan
    record["month_body_return"] = float(close.iloc[last] / open_.iloc[last] - 1)
    record["month_high_to_close"] = float(high.iloc[last] / close.iloc[last] - 1)
    record["month_close_to_low"] = float(close.iloc[last] / low.iloc[last] - 1)

    ma3 = close.rolling(3).mean()
    ma6 = close.rolling(6).mean()
    ma12 = close.rolling(12).mean()
    votes = pd.concat(
        [
            (close > ma6).map({True: 1, False: -1}),
            (close > ma12).map({True: 1, False: -1}),
            (ma3 > ma6).map({True: 1, False: -1}),
            (ma6 / ma6.shift(3) - 1 > 0).map({True: 1, False: -1}),
            (ma12 / ma12.shift(3) - 1 > 0).map({True: 1, False: -1}),
        ],
        axis=1,
    )
    unavailable = ma12.isna() | ma12.shift(3).isna()
    score = votes.sum(axis=1).where(~unavailable)
    state = pd.Series(0, index=score.index)
    state.loc[score >= 2] = 1
    state.loc[score <= -2] = -1
    groups = state.ne(state.shift(fill_value=0)).cumsum()
    state_months = (state.groupby(groups).cumcount() + 1).where(state != 0, 0)
    record["month_regime_trend_score"] = float(score.iloc[last])
    record["month_regime_state"] = float(state.iloc[last])
    record["month_regime_state_months"] = float(state_months.iloc[last])
    return record


def build_monthly_state_features(target_daily: pd.DataFrame, target_weekly: pd.DataFrame) -> pd.DataFrame:
    daily = target_daily.sort_values("trade_date").copy()
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    rows = []
    for week in target_weekly[["week_end_date"]].itertuples(index=False):
        monthly = _build_monthly_bars_until(daily, pd.Timestamp(week.week_end_date))
        rows.append(_monthly_state_record(monthly, pd.Timestamp(week.week_end_date)))
    result = pd.DataFrame(rows)
    monthly_cols = [column for column in result.columns if column.startswith("month_")]
    result[monthly_cols] = result[monthly_cols].replace([np.inf, -np.inf], np.nan)
    return result


def build_weekly_feature_frame(
    target_weekly: pd.DataFrame,
    all_weekly: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    out = target_weekly[["week_start_date", "week_end_date", "index_code", "week_pos"]].copy()
    open_ = pd.to_numeric(target_weekly["open"], errors="coerce")
    high = pd.to_numeric(target_weekly["high"], errors="coerce")
    low = pd.to_numeric(target_weekly["low"], errors="coerce")
    close = pd.to_numeric(target_weekly["close"], errors="coerce")
    avg_daily_volume = pd.to_numeric(target_weekly["avg_daily_volume"], errors="coerce")
    trade_days = pd.to_numeric(target_weekly["trade_days"], errors="coerce")
    ret_1w = close.pct_change()

    for window in [1, 2, 4]:
        out[f"week_ret_{window}w"] = close.pct_change(window)
    out["week_downside_ret_4w"] = ret_1w.clip(upper=0).rolling(4).sum()
    for window in [13, 26]:
        out[f"week_drawdown_{window}w"] = close / high.rolling(window).max() - 1
        out[f"week_runup_{window}w"] = close / low.rolling(window).min() - 1
    for window in [4, 13]:
        out[f"week_distance_low_{window}w"] = close / low.rolling(window).min() - 1

    moving_averages = {window: close.rolling(window).mean() for window in [5, 10, 20, 40]}
    for window, average in moving_averages.items():
        out[f"week_ma{window}_gap"] = close / average - 1
    out["week_ma20_slope_4w"] = moving_averages[20] / moving_averages[20].shift(4) - 1
    out["week_ma40_slope_8w"] = moving_averages[40] / moving_averages[40].shift(8) - 1
    out["week_below_ma20_weeks"] = _consecutive_periods(close < moving_averages[20])
    out["week_above_ma20_weeks"] = _consecutive_periods(close > moving_averages[20])

    out["week_rsi6"] = rsi_cn(close, 6)
    out["week_rsi14"] = rsi_cn(close, 14)
    out["week_wr14"] = wr(high, low, close, window=14, sign="negative")
    out["week_cci14"] = cci(high, low, close, window=14)

    boll_frame = boll(close, window=20, multiplier=2.0, ddof=0)
    out["week_boll_width"] = boll_frame["boll_width"]
    out["week_boll_position"] = boll_frame["boll_position"]
    out["week_boll_lower_break"] = (close < boll_frame["boll_lower"]).astype(float)
    out["week_boll_upper_break"] = (close > boll_frame["boll_upper"]).astype(float)

    macd_frame = macd(close)
    out["week_macd"] = macd_frame["macd_dif"]
    out["week_macd_signal"] = macd_frame["macd_dea"]
    out["week_macd_hist"] = macd_frame["macd_hist"]
    out["week_macd_hist_delta_1w"] = out["week_macd_hist"].diff()
    out["week_macd_hist_delta_3w"] = out["week_macd_hist"].diff(3)

    kdj_frame = kdj(high, low, close)
    out["week_kdj_k"] = kdj_frame["kdj_k"]
    out["week_kdj_d"] = kdj_frame["kdj_d"]
    out["week_kdj_j"] = kdj_frame["kdj_j"]
    out["week_kdj_k_minus_d"] = kdj_frame["kdj_k_minus_d"]
    out["week_kdj_golden_cross"] = (
        (out["week_kdj_k_minus_d"] > 0) & (out["week_kdj_k_minus_d"].shift(1) <= 0)
    ).astype(float)
    out["week_kdj_dead_cross"] = (
        (out["week_kdj_k_minus_d"] < 0) & (out["week_kdj_k_minus_d"].shift(1) >= 0)
    ).astype(float)

    previous_close = close.shift(1)
    out["week_atr14"] = atr(high, low, close, window=14, normalize=True)
    for window in [4, 8, 13]:
        out[f"week_volatility_{window}w"] = ret_1w.rolling(window).std()
    out["week_volatility_expand"] = out["week_volatility_4w"] / out["week_volatility_13w"].replace(0, np.nan)

    week_range = (high - low).replace(0, np.nan)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    out["week_range"] = week_range / previous_close
    out["week_body_return"] = close / open_ - 1
    out["week_close_position"] = (close - low) / week_range
    out["week_open_position"] = (open_ - low) / week_range
    out["week_close_to_low"] = close / low - 1
    out["week_high_to_close"] = high / close - 1
    out["week_lower_shadow_ratio"] = lower_shadow / week_range
    out["week_upper_shadow_ratio"] = upper_shadow / week_range
    out["week_long_lower_shadow"] = (out["week_lower_shadow_ratio"] >= 0.45).astype(float)
    out["week_long_upper_shadow"] = (out["week_upper_shadow_ratio"] >= 0.45).astype(float)
    out["week_gap_down"] = (open_ < previous_close * 0.995).astype(float)
    out["week_gap_up"] = (open_ > previous_close * 1.005).astype(float)
    out["week_down_streak"] = _consecutive_periods(ret_1w < 0)
    out["week_up_streak"] = _consecutive_periods(ret_1w > 0)

    out["week_trade_days"] = trade_days
    out["week_avg_daily_volume"] = avg_daily_volume
    out["week_avg_daily_volume_zscore_20w"] = (
        (avg_daily_volume - avg_daily_volume.rolling(20).mean())
        / avg_daily_volume.rolling(20).std()
    )
    out["week_avg_daily_volume_ratio_5w_20w"] = (
        avg_daily_volume.rolling(5).mean() / avg_daily_volume.rolling(20).mean().replace(0, np.nan)
    )
    out["week_avg_daily_volume_ratio_5w"] = (
        avg_daily_volume / avg_daily_volume.shift(1).rolling(5).mean().replace(0, np.nan)
    )
    volume_expand_ratio = float(cfg["candidate"]["volume_expand_ratio"])
    out["week_down_volume_expand"] = (
        (ret_1w < 0) & (out["week_avg_daily_volume_ratio_5w"] >= volume_expand_ratio)
    ).astype(float)
    out["week_up_volume_expand"] = (
        (ret_1w > 0) & (out["week_avg_daily_volume_ratio_5w"] >= volume_expand_ratio)
    ).astype(float)

    close_pivot = all_weekly.pivot(index="week_end_date", columns="index_code", values="close").sort_index()
    relative = pd.DataFrame(index=close_pivot.index)
    target_index = str(target_weekly["index_code"].iloc[0])
    for code, name in [("000300", "hs300"), ("000905", "zz500"), ("399006", "cyb")]:
        if target_index in close_pivot and code in close_pivot:
            relative[f"relative_{name}_1w"] = close_pivot[target_index].pct_change() - close_pivot[code].pct_change()
            relative[f"relative_{name}_4w"] = close_pivot[target_index].pct_change(4) - close_pivot[code].pct_change(4)
    out = out.merge(relative.reset_index(), on="week_end_date", how="left")

    trend_votes = pd.concat(
        [
            (close > moving_averages[20]).map({True: 1, False: -1}),
            (close > moving_averages[40]).map({True: 1, False: -1}),
            (out["week_ma20_slope_4w"] > 0).map({True: 1, False: -1}),
            (out["week_ma40_slope_8w"] > 0).map({True: 1, False: -1}),
        ],
        axis=1,
    )
    unavailable = moving_averages[40].isna() | out["week_ma40_slope_8w"].isna()
    out["week_regime_trend_score"] = trend_votes.sum(axis=1).where(~unavailable)
    regime = pd.Series(0, index=out.index)
    regime.loc[out["week_regime_trend_score"] >= 2] = 1
    regime.loc[out["week_regime_trend_score"] <= -2] = -1
    groups = regime.ne(regime.shift(fill_value=0)).cumsum()
    out["week_regime_state_weeks"] = (regime.groupby(groups).cumcount() + 1).where(regime != 0, 0)

    candidate_cfg = cfg["candidate"]
    bottom_conditions = pd.concat(
        [
            out["week_drawdown_13w"] <= float(candidate_cfg["bottom_drawdown_13w"]),
            out["week_boll_position"] <= float(candidate_cfg["bottom_boll_position"]),
            out["week_rsi6"] <= float(candidate_cfg["bottom_rsi6"]),
            out["week_kdj_j"] <= float(candidate_cfg["bottom_kdj_j"]),
            out["week_long_lower_shadow"] == 1,
            out["week_down_volume_expand"] == 1,
        ],
        axis=1,
    )
    top_conditions = pd.concat(
        [
            out["week_runup_13w"] >= float(candidate_cfg["top_runup_13w"]),
            out["week_boll_position"] >= float(candidate_cfg["top_boll_position"]),
            out["week_rsi6"] >= float(candidate_cfg["top_rsi6"]),
            out["week_kdj_j"] >= float(candidate_cfg["top_kdj_j"]),
            out["week_long_upper_shadow"] == 1,
            out["week_up_volume_expand"] == 1,
        ],
        axis=1,
    )
    out["week_bottom_candidate_score"] = bottom_conditions.sum(axis=1).astype(float)
    out["week_top_candidate_score"] = top_conditions.sum(axis=1).astype(float)
    out["is_bottom_candidate"] = (
        out["week_bottom_candidate_score"] >= int(candidate_cfg["bottom_min_conditions"])
    ).astype(int)
    out["is_top_candidate"] = (
        out["week_top_candidate_score"] >= int(candidate_cfg["top_min_conditions"])
    ).astype(int)
    out["is_candidate"] = ((out["is_bottom_candidate"] == 1) | (out["is_top_candidate"] == 1)).astype(int)
    return out.replace([np.inf, -np.inf], np.nan)


def add_weekly_path_labels(frame: pd.DataFrame, target_weekly: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    result = frame.copy()
    horizon = int(cfg["model"]["horizon_weeks"])
    close = pd.to_numeric(target_weekly["close"], errors="coerce").reset_index(drop=True)
    high = pd.to_numeric(target_weekly["high"], errors="coerce").reset_index(drop=True)
    low = pd.to_numeric(target_weekly["low"], errors="coerce").reset_index(drop=True)
    high_paths = pd.concat([high.shift(-step) / close - 1 for step in range(1, horizon + 1)], axis=1)
    low_paths = pd.concat([low.shift(-step) / close - 1 for step in range(1, horizon + 1)], axis=1)
    result[f"future_ret_{horizon}w"] = close.shift(-horizon) / close - 1
    result[f"future_mfe_{horizon}w"] = high_paths.max(axis=1)
    result[f"future_mae_{horizon}w"] = low_paths.min(axis=1)
    return result


def apply_auto_future_return_labels(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    result = frame.copy()
    label_cfg = cfg["manual_labels"]
    if str(label_cfg.get("label_source", "manual")) != "auto_future_return":
        return result
    horizon = int(cfg["model"]["horizon_weeks"])
    ret_col = f"future_ret_{horizon}w"
    bottom_threshold = float(label_cfg["auto_bottom_return_threshold"])
    top_threshold = float(label_cfg["auto_top_return_threshold"])
    future_ret = pd.to_numeric(result[ret_col], errors="coerce")
    valid = future_ret.notna()
    result["manual_weak_bottom_label"] = ((future_ret > bottom_threshold) & valid).astype(int)
    result["manual_weak_top_label"] = ((future_ret < top_threshold) & valid).astype(int)
    result.loc[~valid, ["manual_weak_bottom_label", "manual_weak_top_label"]] = 0
    result["label_mode"] = str(label_cfg["label_mode"])
    result["manual_state"] = "neutral"
    result.loc[result["manual_weak_bottom_label"] == 1, "manual_state"] = "bottom"
    result.loc[result["manual_weak_top_label"] == 1, "manual_state"] = "top"
    return result


def add_manual_weekly_labels(frame: pd.DataFrame, regions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    result = frame.copy()
    result["label_mode"] = str(cfg["manual_labels"]["label_mode"])
    result["is_manual_bottom_region"] = 0
    result["is_manual_top_region"] = 0
    result["is_manual_buy_window"] = 0
    result["is_manual_sell_window"] = 0
    result["bottom_level_score"] = 0.0
    result["top_level_score"] = 0.0
    result["bottom_confidence"] = 0
    result["top_confidence"] = 0
    result["bottom_cycle_score"] = 0
    result["top_cycle_score"] = 0
    result["bottom_label_freq_score"] = 0
    result["top_label_freq_score"] = 0
    bottom_ids: list[list[str]] = [[] for _ in range(len(result))]
    top_ids: list[list[str]] = [[] for _ in range(len(result))]

    for region in regions.itertuples(index=False):
        region_mask = _overlap_mask(
            result["week_start_date"], result["week_end_date"], region.start_date, region.end_date
        )
        if not region_mask.any():
            logger.warning("weekly manual region has no matching weeks region_id=%s", region.region_id)
            continue
        idxs = result.index[region_mask].tolist()
        level_score = float(LEVEL_SCORE[region.level] * region.confidence)
        cycle_score = int(CYCLE_SCORE[region.cycle])
        label_freq_score = int(LABEL_FREQ_SCORE[region.label_freq])
        if region.region_type == "bottom":
            result.loc[region_mask, "is_manual_bottom_region"] = 1
            result.loc[region_mask, "bottom_level_score"] = result.loc[
                region_mask, "bottom_level_score"
            ].clip(lower=level_score)
            result.loc[region_mask, "bottom_confidence"] = result.loc[
                region_mask, "bottom_confidence"
            ].clip(lower=int(region.confidence))
            result.loc[region_mask, "bottom_cycle_score"] = result.loc[
                region_mask, "bottom_cycle_score"
            ].clip(lower=cycle_score)
            result.loc[region_mask, "bottom_label_freq_score"] = result.loc[
                region_mask, "bottom_label_freq_score"
            ].clip(lower=label_freq_score)
            for idx in idxs:
                bottom_ids[idx].append(region.region_id)
            if region.usable_for_signal == 1 and pd.notna(region.entry_start):
                entry_mask = _overlap_mask(
                    result["week_start_date"], result["week_end_date"], region.entry_start, region.entry_end
                )
                result.loc[entry_mask, "is_manual_buy_window"] = 1
        else:
            result.loc[region_mask, "is_manual_top_region"] = 1
            result.loc[region_mask, "top_level_score"] = result.loc[
                region_mask, "top_level_score"
            ].clip(lower=level_score)
            result.loc[region_mask, "top_confidence"] = result.loc[
                region_mask, "top_confidence"
            ].clip(lower=int(region.confidence))
            result.loc[region_mask, "top_cycle_score"] = result.loc[
                region_mask, "top_cycle_score"
            ].clip(lower=cycle_score)
            result.loc[region_mask, "top_label_freq_score"] = result.loc[
                region_mask, "top_label_freq_score"
            ].clip(lower=label_freq_score)
            for idx in idxs:
                top_ids[idx].append(region.region_id)
            if region.usable_for_signal == 1 and pd.notna(region.exit_start):
                exit_mask = _overlap_mask(
                    result["week_start_date"], result["week_end_date"], region.exit_start, region.exit_end
                )
                result.loc[exit_mask, "is_manual_sell_window"] = 1

    result["manual_bottom_region_ids"] = [_join_ids(values) for values in bottom_ids]
    result["manual_top_region_ids"] = [_join_ids(values) for values in top_ids]
    result["manual_state"] = "neutral"
    result.loc[result["is_manual_bottom_region"] == 1, "manual_state"] = "bottom"
    result.loc[result["is_manual_top_region"] == 1, "manual_state"] = "top"
    both = (result["is_manual_bottom_region"] == 1) & (result["is_manual_top_region"] == 1)
    result.loc[both, "manual_state"] = "overlap"
    result["manual_weak_bottom_label"] = result["is_manual_buy_window"].astype(int)
    result["manual_weak_top_label"] = result["is_manual_sell_window"].astype(int)
    return result


def build_weekly_event_features(weeks: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    result = weeks[["week_start_date", "week_end_date"]].copy()
    if events.empty:
        return result
    events = events.copy()
    events["trade_date"] = pd.to_datetime(events["trade_date"])
    events["event_score"] = pd.to_numeric(events["event_score"], errors="coerce").fillna(0.0)
    events["impact_strength"] = pd.to_numeric(events["impact_strength"], errors="coerce").fillna(0.0)
    events["direction_score"] = _direction_score(events)

    rows = []
    for week in result.itertuples(index=False):
        part = events[(events["trade_date"] >= week.week_start_date) & (events["trade_date"] <= week.week_end_date)]
        record = {"week_end_date": week.week_end_date}
        record.update(_aggregate_event_part(part, "f_event_week_all"))
        for group, (column, keyword) in EVENT_WEEKLY_GROUPS.items():
            group_mask = part[column].map(lambda value, key=keyword: _contains(value, key)).astype(bool)
            group_part = part.loc[group_mask]
            record.update(_aggregate_event_part(group_part, f"f_event_week_{group}"))
        rows.append(record)
    weekly_events = pd.DataFrame(rows)
    feature_cols = [column for column in weekly_events.columns if column.startswith("f_event_week_")]
    weekly_events = weekly_events.sort_values("week_end_date").reset_index(drop=True)
    rolling_features: dict[str, pd.Series] = {}
    for window in EVENT_WEEKLY_WINDOWS:
        for column in feature_cols:
            values = pd.to_numeric(weekly_events[column], errors="coerce").fillna(0.0)
            if column.endswith("_max_strength"):
                rolling_features[f"{column}_roll_{window}w"] = values.rolling(window, min_periods=1).max()
            else:
                rolling_features[f"{column}_roll_{window}w"] = values.rolling(window, min_periods=1).sum()
    if rolling_features:
        weekly_events = pd.concat([weekly_events, pd.DataFrame(rolling_features, index=weekly_events.index)], axis=1)
    return result.merge(weekly_events, on="week_end_date", how="left").fillna(0.0)


def _event_frame() -> pd.DataFrame:
    try:
        return read_sql(
            "SELECT trade_date,event_type,event_stage,expectation_level,surprise_level,"
            "affected_style,impact_direction,impact_strength,event_score "
            "FROM event_daily ORDER BY trade_date"
        )
    except Exception:
        logger.warning("event_daily is unavailable; weekly event features will be skipped", exc_info=True)
        return pd.DataFrame()


def _market_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    data = read_sql(
        "SELECT trade_date, index_code, open, high, low, close, volume "
        "FROM market_index_daily ORDER BY trade_date, index_code"
    )
    if data.empty:
        raise RuntimeError("market_index_daily is empty.")
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    weekly_parts = [build_weekly_market_bars(part) for _, part in data.groupby("index_code", sort=False)]
    return data, pd.concat(weekly_parts, ignore_index=True)


def _manual_label_path(cfg: dict, target_index: str) -> Path:
    files = cfg["manual_labels"]["files"]
    if target_index not in files:
        raise RuntimeError(f"No weekly manual label file configured for target_index={target_index}.")
    return project_path(str(files[target_index]))


def _ensure_table(table: str) -> None:
    execute_sql(
        f"""
        CREATE TABLE IF NOT EXISTS `{table}` (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            week_start_date DATE NOT NULL,
            week_end_date DATE NOT NULL,
            index_code VARCHAR(20) NOT NULL,
            week_pos INT NOT NULL,
            is_candidate TINYINT NOT NULL,
            is_bottom_candidate TINYINT NOT NULL,
            is_top_candidate TINYINT NOT NULL,
            manual_state VARCHAR(16) NOT NULL,
            manual_weak_bottom_label TINYINT NOT NULL,
            manual_weak_top_label TINYINT NOT NULL,
            is_manual_bottom_region TINYINT NOT NULL,
            is_manual_top_region TINYINT NOT NULL,
            is_manual_buy_window TINYINT NOT NULL,
            is_manual_sell_window TINYINT NOT NULL,
            label_mode VARCHAR(32) NOT NULL,
            future_ret_3w DECIMAL(12,8) NULL,
            future_mfe_3w DECIMAL(12,8) NULL,
            future_mae_3w DECIMAL(12,8) NULL,
            bottom_level_score DECIMAL(12,6) NOT NULL,
            top_level_score DECIMAL(12,6) NOT NULL,
            bottom_confidence TINYINT NOT NULL,
            top_confidence TINYINT NOT NULL,
            bottom_cycle_score TINYINT NOT NULL,
            top_cycle_score TINYINT NOT NULL,
            bottom_label_freq_score TINYINT NOT NULL,
            top_label_freq_score TINYINT NOT NULL,
            manual_bottom_region_ids VARCHAR(255) NOT NULL,
            manual_top_region_ids VARCHAR(255) NOT NULL,
            feature_json JSON NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uk_bottom_weekly_dataset (week_end_date, index_code),
            KEY idx_bottom_weekly_candidate (is_candidate, week_end_date),
            KEY idx_bottom_weekly_labels (manual_weak_bottom_label, manual_weak_top_label, week_end_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )
    database = get_database_name()
    existing = set(
        read_sql(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=:database AND TABLE_NAME=:table",
            {"database": database, "table": table},
        )["COLUMN_NAME"].tolist()
    )
    for column in ["future_ret_3w", "future_mfe_3w", "future_mae_3w"]:
        if column not in existing:
            execute_sql(f"ALTER TABLE `{table}` ADD COLUMN `{column}` DECIMAL(12,8) NULL")


def build_bottom_weekly_dataset(target_index: str | None = None) -> int:
    assert_bottom_weekly_database()
    cfg = get_config("bottom_weekly_model.yaml")
    target_index = str(target_index or cfg["model"]["target_index"])
    daily, all_weekly = _market_frame()
    target_weekly = all_weekly[all_weekly["index_code"] == target_index].sort_values("week_end_date").reset_index(drop=True)
    if target_weekly.empty:
        raise RuntimeError(f"No weekly market rows found for target index {target_index}.")
    feature_frame = build_weekly_feature_frame(target_weekly, all_weekly, cfg)
    target_daily = daily[daily["index_code"] == target_index].copy()
    monthly_features = build_monthly_state_features(target_daily, target_weekly)
    monthly_cols = [column for column in monthly_features.columns if column.startswith("month_")]
    feature_frame = feature_frame.merge(
        monthly_features[["week_end_date", *monthly_cols]],
        on="week_end_date",
        how="left",
    )
    data = add_weekly_path_labels(feature_frame, target_weekly, cfg)
    regions = load_manual_regions(_manual_label_path(cfg, target_index))
    data = add_manual_weekly_labels(data, regions, cfg)
    data = apply_auto_future_return_labels(data, cfg)
    if bool(cfg.get("feature_augments", {}).get("use_event_weekly_features", True)):
        event_features = build_weekly_event_features(data[["week_start_date", "week_end_date"]], _event_frame())
        event_cols = [column for column in event_features.columns if column.startswith("f_event_week_")]
        if event_cols:
            data = data.merge(event_features[["week_end_date", *event_cols]], on="week_end_date", how="left")

    horizon = int(cfg["model"]["horizon_weeks"])
    ret_col = f"future_ret_{horizon}w"
    mfe_col = f"future_mfe_{horizon}w"
    mae_col = f"future_mae_{horizon}w"
    excluded = {
        "week_start_date", "week_end_date", "index_code", "week_pos",
        "is_candidate", "is_bottom_candidate", "is_top_candidate",
        "manual_state", "manual_weak_bottom_label", "manual_weak_top_label",
        "is_manual_bottom_region", "is_manual_top_region",
        "is_manual_buy_window", "is_manual_sell_window", "label_mode",
        ret_col, mfe_col, mae_col,
        "bottom_level_score", "top_level_score", "bottom_confidence", "top_confidence",
        "bottom_cycle_score", "top_cycle_score", "bottom_label_freq_score", "top_label_freq_score",
        "manual_bottom_region_ids", "manual_top_region_ids",
    }
    feature_cols = [col for col in data.columns if col not in excluded]
    records = []
    for _, row in data.iterrows():
        features = {
            (col if col.startswith("f_") else f"f_{col}"): None if pd.isna(row[col]) else float(row[col])
            for col in feature_cols
        }
        records.append(
            {
                "week_start_date": row["week_start_date"].date(),
                "week_end_date": row["week_end_date"].date(),
                "index_code": row["index_code"],
                "week_pos": int(row["week_pos"]),
                "is_candidate": int(row["is_candidate"]),
                "is_bottom_candidate": int(row["is_bottom_candidate"]),
                "is_top_candidate": int(row["is_top_candidate"]),
                "manual_state": row["manual_state"],
                "manual_weak_bottom_label": int(row["manual_weak_bottom_label"]),
                "manual_weak_top_label": int(row["manual_weak_top_label"]),
                "is_manual_bottom_region": int(row["is_manual_bottom_region"]),
                "is_manual_top_region": int(row["is_manual_top_region"]),
                "is_manual_buy_window": int(row["is_manual_buy_window"]),
                "is_manual_sell_window": int(row["is_manual_sell_window"]),
                "label_mode": row["label_mode"],
                "future_ret_3w": row[ret_col],
                "future_mfe_3w": row[mfe_col],
                "future_mae_3w": row[mae_col],
                "bottom_level_score": row["bottom_level_score"],
                "top_level_score": row["top_level_score"],
                "bottom_confidence": int(row["bottom_confidence"]),
                "top_confidence": int(row["top_confidence"]),
                "bottom_cycle_score": int(row["bottom_cycle_score"]),
                "top_cycle_score": int(row["top_cycle_score"]),
                "bottom_label_freq_score": int(row["bottom_label_freq_score"]),
                "top_label_freq_score": int(row["top_label_freq_score"]),
                "manual_bottom_region_ids": row["manual_bottom_region_ids"],
                "manual_top_region_ids": row["manual_top_region_ids"],
                "feature_json": json.dumps(features, ensure_ascii=False),
            }
        )
    table = str(cfg["outputs"]["dataset_table"])
    _ensure_table(table)
    count = upsert_dataframe(pd.DataFrame(records), table, ["week_end_date", "index_code"])
    logger.info("built weekly bottom dataset rows=%s table=%s", count, table)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weekly top/bottom research dataset.")
    parser.add_argument("--target-index")
    args = parser.parse_args()
    print(build_bottom_weekly_dataset(target_index=args.target_index))


if __name__ == "__main__":
    main()
