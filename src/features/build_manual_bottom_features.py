from __future__ import annotations

import numpy as np
import pandas as pd

from src.common.config import get_config
from src.common.db import read_sql
from src.features.technical_indicators import atr, boll, cci, kdj, macd, rsi_cn, wr


BOTTOM_CANDIDATE_CONFIG = {
    "min_conditions": 2,
    "drawdown_20d": -0.04,
    "drawdown_60d": -0.07,
    "rsi6": 30.0,
    "boll_position": 0.15,
    "wr14": -80.0,
    "below_ma20_days": 5,
    "ma20_gap": -0.03,
    "ret_3d": -0.025,
    "volatility_expand": 1.20,
}


def _consecutive_days(mask: pd.Series) -> pd.Series:
    values = mask.fillna(False).astype(bool)
    groups = values.ne(values.shift(fill_value=False)).cumsum()
    days = values.groupby(groups).cumcount() + 1
    return days.where(values, 0).astype(float)


def _weekly_kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple[float, float, float, float]:
    frame = kdj(pd.Series(high), pd.Series(low), pd.Series(close))
    k_value = float(frame["kdj_k"].iloc[-1])
    d_value = float(frame["kdj_d"].iloc[-1])
    j_value = float(frame["kdj_j"].iloc[-1])
    previous_diff = float(frame["kdj_k_minus_d"].iloc[-2]) if len(frame) >= 2 else 0.0
    return k_value, d_value, j_value, previous_diff


def build_causal_weekly_bottom_features(target: pd.DataFrame) -> pd.DataFrame:
    data = target.sort_values("trade_date").reset_index(drop=True).copy()
    week_keys = data["trade_date"].dt.to_period("W-FRI")
    completed: list[dict[str, float]] = []
    rows: list[dict[str, float]] = []
    current_key = None
    current: dict[str, float] = {}

    for idx, row in data.iterrows():
        week_key = week_keys.iloc[idx]
        if current_key is None or week_key != current_key:
            if current_key is not None:
                completed.append(current.copy())
            current_key = week_key
            current = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "days": 1.0,
            }
        else:
            current["high"] = max(current["high"], float(row["high"]))
            current["low"] = min(current["low"], float(row["low"]))
            current["close"] = float(row["close"])
            current["volume"] += float(row["volume"])
            current["days"] += 1.0

        bars = [*completed, current]
        highs = np.array([bar["high"] for bar in bars], dtype=float)
        lows = np.array([bar["low"] for bar in bars], dtype=float)
        closes = np.array([bar["close"] for bar in bars], dtype=float)
        volumes = np.array([bar["volume"] for bar in bars], dtype=float)

        def gap(window: int) -> float:
            return float(closes[-1] / closes[-window:].mean() - 1) if len(closes) >= window else np.nan

        def ret(window: int) -> float:
            return float(closes[-1] / closes[-window - 1] - 1) if len(closes) > window else np.nan

        ma20 = float(closes[-20:].mean()) if len(closes) >= 20 else np.nan
        std20 = float(closes[-20:].std(ddof=0)) if len(closes) >= 20 else np.nan
        boll_lower = ma20 - 2 * std20 if np.isfinite(ma20) and np.isfinite(std20) else np.nan
        boll_upper = ma20 + 2 * std20 if np.isfinite(ma20) and np.isfinite(std20) else np.nan
        boll_range = boll_upper - boll_lower if np.isfinite(boll_lower) else np.nan
        k_value, d_value, j_value, previous_kdj_diff = _weekly_kdj(highs, lows, closes)
        kdj_diff = k_value - d_value

        prior_uptrend = 0.0
        if len(completed) >= 24:
            prior_closes = np.array([bar["close"] for bar in completed], dtype=float)
            prior_ma20 = float(prior_closes[-20:].mean())
            prior_ma20_four_weeks_ago = float(prior_closes[-24:-4].mean())
            prior_uptrend = float(prior_closes[-1] > prior_ma20 and prior_ma20 > prior_ma20_four_weeks_ago)

        previous_completed = completed[-5:]
        previous_weekly_volume = np.array([bar["volume"] for bar in previous_completed], dtype=float)
        previous_daily_volume = np.array([bar["volume"] / bar["days"] for bar in previous_completed], dtype=float)
        volume_ratio = (
            float(current["volume"] / previous_weekly_volume.mean())
            if len(previous_weekly_volume) == 5 and previous_weekly_volume.mean() != 0
            else np.nan
        )
        volume_pace_ratio = (
            float((current["volume"] / current["days"]) / previous_daily_volume.mean())
            if len(previous_daily_volume) == 5 and previous_daily_volume.mean() != 0
            else np.nan
        )

        week_range = current["high"] - current["low"]
        lower_shadow = min(current["open"], current["close"]) - current["low"]
        close_below_lower = float(np.isfinite(boll_lower) and current["close"] < boll_lower)
        low_below_lower = float(np.isfinite(boll_lower) and current["low"] < boll_lower)
        lower_reclaim = float(low_below_lower == 1 and current["close"] >= boll_lower)
        close_below_mid = float(np.isfinite(ma20) and current["close"] < ma20)
        uptrend_pullback = float(prior_uptrend == 1 and close_below_mid == 1)
        golden_cross = float(kdj_diff > 0 and previous_kdj_diff <= 0)
        kdj_oversold = float(j_value < 0 or k_value < 25)
        oversold_golden_cross = float(golden_cross == 1 and kdj_oversold == 1)
        long_lower_shadow = float(week_range > 0 and lower_shadow / week_range >= 0.45)
        down_volume_expand = float(
            np.isfinite(ret(1)) and ret(1) < 0 and np.isfinite(volume_pace_ratio) and volume_pace_ratio >= 1.2
        )

        rows.append(
            {
                "trade_date": row["trade_date"],
                "bf_week_progress": current["days"] / 5.0,
                "bf_week_ret_1w": ret(1),
                "bf_week_ret_2w": ret(2),
                "bf_week_ret_4w": ret(4),
                "bf_week_range": week_range / current["close"],
                "bf_week_body_return": current["close"] / current["open"] - 1,
                "bf_week_lower_shadow_ratio": lower_shadow / week_range if week_range > 0 else 0.0,
                "bf_week_drawdown_13w": closes[-1] / highs[-13:].max() - 1 if len(highs) >= 13 else np.nan,
                "bf_week_drawdown_26w": closes[-1] / highs[-26:].max() - 1 if len(highs) >= 26 else np.nan,
                "bf_week_ma5_gap": gap(5),
                "bf_week_ma10_gap": gap(10),
                "bf_week_ma20_gap": gap(20),
                "bf_week_ma20_slope_4w": ma20 / closes[-24:-4].mean() - 1 if len(closes) >= 24 else np.nan,
                "bf_week_boll_width": boll_range / ma20 if np.isfinite(boll_range) else np.nan,
                "bf_week_boll_position": (
                    (current["close"] - boll_lower) / boll_range
                    if np.isfinite(boll_range) and boll_range != 0
                    else np.nan
                ),
                "bf_week_close_below_boll_lower": close_below_lower,
                "bf_week_low_below_boll_lower": low_below_lower,
                "bf_week_boll_lower_reclaim": lower_reclaim,
                "bf_week_close_below_boll_mid": close_below_mid,
                "bf_week_prior_uptrend": prior_uptrend,
                "bf_week_uptrend_pullback_below_mid": uptrend_pullback,
                "bf_week_kdj_k": k_value,
                "bf_week_kdj_d": d_value,
                "bf_week_kdj_j": j_value,
                "bf_week_kdj_k_minus_d": kdj_diff,
                "bf_week_kdj_golden_cross": golden_cross,
                "bf_week_kdj_oversold": kdj_oversold,
                "bf_week_oversold_kdj_golden_cross": oversold_golden_cross,
                "bf_week_volume_pace_ratio_5w": volume_pace_ratio,
                "bf_week_volume_ratio_5w": volume_ratio,
                "bf_week_down_volume_expand": down_volume_expand,
                "bf_week_bottom_confirmation_score": float(
                    close_below_lower
                    + lower_reclaim
                    + uptrend_pullback
                    + golden_cross
                    + oversold_golden_cross
                    + long_lower_shadow
                ),
            }
        )
    return pd.DataFrame(rows)


def load_market_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    target = str(get_config()["project"]["target_index"])
    data = read_sql(
        "SELECT trade_date,index_code,open,high,low,close,volume "
        "FROM market_index_daily ORDER BY trade_date,index_code"
    )
    if data.empty:
        raise RuntimeError("market_index_daily is empty.")
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    target_data = data[data["index_code"] == target].sort_values("trade_date").reset_index(drop=True).copy()
    if target_data.empty:
        raise RuntimeError(f"No market rows found for target index {target}.")
    return target_data, data


def build_manual_bottom_feature_frame() -> pd.DataFrame:
    target, all_indices = load_market_frame()
    out = target[["trade_date", "index_code"]].copy()
    open_ = target["open"]
    high = target["high"]
    low = target["low"]
    close = target["close"]
    volume = target["volume"]
    ret_1d = close.pct_change()

    for window in [1, 3, 5, 10]:
        out[f"bf_ret_{window}d"] = close.pct_change(window)
    out["bf_downside_ret_3d"] = ret_1d.clip(upper=0).rolling(3).sum()
    out["bf_downside_ret_5d"] = ret_1d.clip(upper=0).rolling(5).sum()
    for window in [20, 60, 120]:
        out[f"bf_drawdown_{window}d"] = close / close.rolling(window).max() - 1
    for window in [5, 10, 20]:
        out[f"bf_distance_low_{window}d"] = close / low.rolling(window).min() - 1

    moving_averages = {window: close.rolling(window).mean() for window in [5, 10, 20, 60]}
    for window, average in moving_averages.items():
        out[f"bf_ma{window}_gap"] = close / average - 1
    out["bf_ma20_slope_5d"] = moving_averages[20] / moving_averages[20].shift(5) - 1
    out["bf_ma60_slope_10d"] = moving_averages[60] / moving_averages[60].shift(10) - 1
    out["bf_below_ma20_days"] = _consecutive_days(close < moving_averages[20])
    out["bf_below_ma60_days"] = _consecutive_days(close < moving_averages[60])

    out["bf_rsi6"] = rsi_cn(close, 6)
    out["bf_rsi14"] = rsi_cn(close, 14)
    out["bf_wr14"] = wr(high, low, close, 14, sign="negative")
    out["bf_cci14"] = cci(high, low, close, 14)

    boll_frame = boll(close, 20, 2, ddof=0)
    boll_lower = boll_frame["boll_lower"]
    out["bf_boll_width"] = boll_frame["boll_width"]
    out["bf_boll_position"] = boll_frame["boll_band_position"]
    out["bf_boll_lower_break"] = (close < boll_lower).astype(float)

    macd_frame = macd(close)
    out["bf_macd"] = macd_frame["macd_dif"]
    out["bf_macd_signal"] = macd_frame["macd_dea"]
    out["bf_macd_hist"] = macd_frame["macd_hist"]
    out["bf_macd_hist_delta_1d"] = out["bf_macd_hist"].diff()
    out["bf_macd_hist_delta_3d"] = out["bf_macd_hist"].diff(3)

    kdj_frame = kdj(high, low, close)
    out["bf_kdj_k"] = kdj_frame["kdj_k"]
    out["bf_kdj_d"] = kdj_frame["kdj_d"]
    out["bf_kdj_j"] = kdj_frame["kdj_j"]
    out["bf_kdj_k_minus_d"] = kdj_frame["kdj_k_minus_d"]
    out["bf_kdj_golden_cross"] = (
        (out["bf_kdj_k_minus_d"] > 0) & (out["bf_kdj_k_minus_d"].shift(1) <= 0)
    ).astype(float)

    previous_close = close.shift(1)
    out["bf_atr14"] = atr(high, low, close, 14, normalize=True)
    for window in [5, 10, 20]:
        out[f"bf_volatility_{window}d"] = ret_1d.rolling(window).std()
    out["bf_volatility_expand"] = out["bf_volatility_5d"] / out["bf_volatility_20d"].replace(0, np.nan)

    day_range = (high - low).replace(0, np.nan)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
    out["bf_intraday_range"] = day_range / close
    out["bf_body_return"] = close / open_ - 1
    out["bf_lower_shadow_ratio"] = lower_shadow / day_range
    out["bf_long_lower_shadow"] = (out["bf_lower_shadow_ratio"] >= 0.5).astype(float)
    out["bf_gap_down"] = (open_ < previous_close * 0.995).astype(float)
    out["bf_down_streak"] = _consecutive_days(ret_1d < 0)
    out["bf_up_streak"] = _consecutive_days(ret_1d > 0)
    out["bf_volume_zscore_20d"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    out["bf_volume_ratio_5d_20d"] = volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)

    close_pivot = all_indices.pivot(index="trade_date", columns="index_code", values="close").sort_index()
    relative = pd.DataFrame(index=close_pivot.index)
    for code, name in [("000300", "hs300"), ("000905", "zz500"), ("399006", "cyb")]:
        if code in close_pivot and target["index_code"].iloc[0] in close_pivot:
            relative[f"bf_relative_{name}_1d"] = (
                close_pivot[target["index_code"].iloc[0]].pct_change() - close_pivot[code].pct_change()
            )
            relative[f"bf_relative_{name}_5d"] = (
                close_pivot[target["index_code"].iloc[0]].pct_change(5) - close_pivot[code].pct_change(5)
            )
    out = out.merge(relative.reset_index(), on="trade_date", how="left")

    trend_votes = pd.concat(
        [
            (close > moving_averages[20]).map({True: 1, False: -1}),
            (close > moving_averages[60]).map({True: 1, False: -1}),
            (out["bf_ma20_slope_5d"] > 0).map({True: 1, False: -1}),
            (out["bf_ma60_slope_10d"] > 0).map({True: 1, False: -1}),
        ],
        axis=1,
    )
    unavailable = moving_averages[60].isna() | out["bf_ma60_slope_10d"].isna()
    out["bf_regime_trend_score"] = trend_votes.sum(axis=1).where(~unavailable)
    regime = pd.Series(0, index=out.index)
    regime.loc[out["bf_regime_trend_score"] >= 2] = 1
    regime.loc[out["bf_regime_trend_score"] <= -2] = -1
    groups = regime.ne(regime.shift(fill_value=0)).cumsum()
    out["bf_regime_state_days"] = (regime.groupby(groups).cumcount() + 1).where(regime != 0, 0)

    cfg = BOTTOM_CANDIDATE_CONFIG
    out["bf_candidate_depth"] = (
        (out["bf_drawdown_20d"] <= cfg["drawdown_20d"]) | (out["bf_drawdown_60d"] <= cfg["drawdown_60d"])
    ).astype(float)
    out["bf_candidate_oversold"] = (
        (out["bf_rsi6"] <= cfg["rsi6"])
        | (out["bf_boll_position"] <= cfg["boll_position"])
        | (out["bf_wr14"] <= cfg["wr14"])
    ).astype(float)
    out["bf_candidate_trend"] = (
        (out["bf_below_ma20_days"] >= cfg["below_ma20_days"]) | (out["bf_ma20_gap"] <= cfg["ma20_gap"])
    ).astype(float)
    out["bf_candidate_stress"] = (
        (out["bf_ret_3d"] <= cfg["ret_3d"])
        | (out["bf_volatility_expand"] >= cfg["volatility_expand"])
        | (out["bf_long_lower_shadow"] == 1)
    ).astype(float)
    out["bf_candidate_score"] = out[
        ["bf_candidate_depth", "bf_candidate_oversold", "bf_candidate_trend", "bf_candidate_stress"]
    ].sum(axis=1)
    out["bf_is_candidate"] = (out["bf_candidate_score"] >= cfg["min_conditions"]).astype(int)

    weekly = build_causal_weekly_bottom_features(target)
    return out.merge(weekly, on="trade_date", how="left")


def bottom_feature_columns(data: pd.DataFrame, include_weekly: bool = True) -> list[str]:
    columns = [column for column in data.columns if column.startswith("bf_")]
    if include_weekly:
        return columns
    return [column for column in columns if not column.startswith("bf_week_")]
