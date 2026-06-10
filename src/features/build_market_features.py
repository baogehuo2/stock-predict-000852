from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import get_config
from src.common.db import execute_sql, get_database_name, read_sql, upsert_dataframe
from src.common.logger import get_logger


logger = get_logger(__name__)


EXTRA_MARKET_FEATURE_COLS = [
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "body_ratio",
    "intraday_range",
    "upper_probe",
    "lower_probe",
    "doji",
    "long_upper_shadow",
    "long_lower_shadow",
    "heavy_upper_shadow",
    "heavy_lower_shadow",
    "gap_up",
    "gap_down",
    "large_range_day",
    "macd_golden_cross",
    "macd_dead_cross",
    "macd_golden_cross_days",
    "macd_dead_cross_days",
    "macd_hist_turn_positive",
    "macd_hist_turn_negative",
    "kdj_k",
    "kdj_d",
    "kdj_j",
    "kdj_k_minus_d",
    "kdj_golden_cross",
    "kdj_dead_cross",
    "kdj_golden_cross_days",
    "kdj_dead_cross_days",
    "kdj_j_overbought",
    "kdj_j_oversold",
    "kdj_j_overbought_turn_down",
    "expma12",
    "expma50",
    "expma12_gap",
    "expma_golden_cross",
    "expma_dead_cross",
    "expma_golden_cross_days",
    "expma_dead_cross_days",
    "cci14",
    "cci14_overbought",
    "cci14_turn_down",
    "cci14_overbought_turn_down",
    "wr14",
    "ma20_slope_5d",
    "ma60_slope_10d",
    "ma20_slope_5d_negative",
    "ma60_slope_10d_negative",
    "close_above_ma20",
    "close_above_ma60",
    "above_ma20_days",
    "below_ma20_days",
    "above_ma60_days",
    "below_ma60_days",
    "ma20_ma60_gap",
    "drawdown_20d",
    "drawdown_60d",
    "volume_zscore_20d",
    "volume_ratio_5d_20d",
    "amount_percentile_60d",
    "volume_percentile_60d",
    "down_with_volume",
    "up_with_volume",
    "shrink_rebound",
    "volatility_5d",
    "volatility_expand",
    "atr_expand",
    "boll_lower_break",
    "boll_upper_break",
    "boll_band_position",
    "regime_trend_score",
    "is_bull_trend",
    "is_bear_trend",
    "regime_state_days",
]


def _consecutive_days(mask: pd.Series) -> pd.Series:
    mask = mask.fillna(False).astype(bool)
    groups = mask.ne(mask.shift(fill_value=False)).cumsum()
    days = mask.groupby(groups).cumcount() + 1
    return days.where(mask, 0).astype(float)


def _rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).apply(lambda values: pd.Series(values).rank(pct=True).iloc[-1], raw=False)


def _ensure_market_feature_columns(cols: list[str]) -> None:
    database = get_database_name()
    existing = set(
        read_sql(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=:database AND TABLE_NAME='market_feature_daily'",
            {"database": database},
        )["COLUMN_NAME"].tolist()
    )
    for col in cols:
        if col not in existing:
            execute_sql(f"ALTER TABLE market_feature_daily ADD COLUMN `{col}` DECIMAL(12,6) NULL")


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_one(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("trade_date").copy()
    open_ = pd.to_numeric(df["open"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df["volume"], errors="coerce")
    amount = pd.to_numeric(df["amount"], errors="coerce")

    out = pd.DataFrame({"trade_date": df["trade_date"], "index_code": df["index_code"]})
    for n in [1, 3, 5, 10]:
        out[f"ret_{n}d"] = close.pct_change(n)
    for n in [5, 10, 20]:
        ma = close.rolling(n).mean()
        out[f"ma{n}_gap"] = close / ma - 1
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    macd_diff = out["macd"] - out["macd_signal"]
    out["macd_golden_cross"] = ((macd_diff > 0) & (macd_diff.shift(1) <= 0)).astype(int)
    out["macd_dead_cross"] = ((macd_diff < 0) & (macd_diff.shift(1) >= 0)).astype(int)
    out["macd_golden_cross_days"] = _consecutive_days(macd_diff > 0)
    out["macd_dead_cross_days"] = _consecutive_days(macd_diff < 0)
    out["macd_hist_turn_positive"] = ((out["macd_hist"] > 0) & (out["macd_hist"].shift(1) <= 0)).astype(int)
    out["macd_hist_turn_negative"] = ((out["macd_hist"] < 0) & (out["macd_hist"].shift(1) >= 0)).astype(int)
    out["rsi6"] = _rsi(close, 6)
    out["rsi14"] = _rsi(close, 14)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean() / close
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    boll_upper = mid + 2 * std
    boll_lower = mid - 2 * std
    out["boll_width"] = (4 * std) / mid
    out["boll_lower_break"] = (close < boll_lower).astype(int)
    out["boll_upper_break"] = (close > boll_upper).astype(int)
    out["boll_band_position"] = (close - boll_lower) / (boll_upper - boll_lower).replace(0, np.nan)
    out["amount_zscore_20d"] = (amount - amount.rolling(20).mean()) / amount.rolling(20).std()
    out["volume_zscore_20d"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    out["volatility_10d"] = close.pct_change().rolling(10).std()
    out["volatility_20d"] = close.pct_change().rolling(20).std()
    out["volatility_5d"] = close.pct_change().rolling(5).std()
    out["volatility_expand"] = out["volatility_5d"] / out["volatility_20d"].replace(0, np.nan)
    out["atr_expand"] = out["atr14"] / out["atr14"].rolling(20).mean().replace(0, np.nan)

    price_range = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    upper_shadow = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
    out["upper_shadow_ratio"] = upper_shadow / price_range
    out["lower_shadow_ratio"] = lower_shadow / price_range
    out["body_ratio"] = body / price_range
    out["intraday_range"] = price_range / close
    out["upper_probe"] = ((out["upper_shadow_ratio"] > 0.55) & (out["body_ratio"] < 0.35)).astype(int)
    out["lower_probe"] = ((out["lower_shadow_ratio"] > 0.55) & (out["body_ratio"] < 0.35)).astype(int)
    out["doji"] = (out["body_ratio"] < 0.1).astype(int)
    out["long_upper_shadow"] = (out["upper_shadow_ratio"] > 0.5).astype(int)
    out["long_lower_shadow"] = (out["lower_shadow_ratio"] > 0.5).astype(int)
    out["heavy_upper_shadow"] = ((out["long_upper_shadow"] == 1) & (out["volume_zscore_20d"] > 1)).astype(int)
    out["heavy_lower_shadow"] = ((out["long_lower_shadow"] == 1) & (out["volume_zscore_20d"] > 1)).astype(int)
    out["gap_up"] = (open_ > prev_close * 1.005).astype(int)
    out["gap_down"] = (open_ < prev_close * 0.995).astype(int)
    out["large_range_day"] = (out["intraday_range"] > out["intraday_range"].rolling(60).quantile(0.8)).astype(int)

    low_9 = low.rolling(9, min_periods=1).min()
    high_9 = high.rolling(9, min_periods=1).max()
    rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
    out["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    out["kdj_d"] = out["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]
    out["kdj_k_minus_d"] = out["kdj_k"] - out["kdj_d"]
    out["kdj_golden_cross"] = ((out["kdj_k_minus_d"] > 0) & (out["kdj_k_minus_d"].shift(1) <= 0)).astype(int)
    out["kdj_dead_cross"] = ((out["kdj_k_minus_d"] < 0) & (out["kdj_k_minus_d"].shift(1) >= 0)).astype(int)
    out["kdj_golden_cross_days"] = _consecutive_days(out["kdj_k_minus_d"] > 0)
    out["kdj_dead_cross_days"] = _consecutive_days(out["kdj_k_minus_d"] < 0)
    out["kdj_j_overbought"] = (out["kdj_j"] > 100).astype(int)
    out["kdj_j_oversold"] = (out["kdj_j"] < 0).astype(int)
    out["kdj_j_overbought_turn_down"] = ((out["kdj_j"].shift(1) > 100) & (out["kdj_j"] < out["kdj_j"].shift(1))).astype(int)

    out["expma12"] = close.ewm(span=12, adjust=False).mean()
    out["expma50"] = close.ewm(span=50, adjust=False).mean()
    out["expma12_gap"] = out["expma12"] / out["expma50"] - 1
    out["expma_golden_cross"] = ((out["expma12_gap"] > 0) & (out["expma12_gap"].shift(1) <= 0)).astype(int)
    out["expma_dead_cross"] = ((out["expma12_gap"] < 0) & (out["expma12_gap"].shift(1) >= 0)).astype(int)
    out["expma_golden_cross_days"] = _consecutive_days(out["expma12_gap"] > 0)
    out["expma_dead_cross_days"] = _consecutive_days(out["expma12_gap"] < 0)

    tp = (high + low + close) / 3
    tp_ma = tp.rolling(14).mean()
    tp_md = (tp - tp_ma).abs().rolling(14).mean()
    out["cci14"] = (tp - tp_ma) / (0.015 * tp_md.replace(0, np.nan))
    out["cci14_overbought"] = (out["cci14"] > 100).astype(int)
    out["cci14_turn_down"] = (out["cci14"] < out["cci14"].shift(1)).astype(int)
    out["cci14_overbought_turn_down"] = ((out["cci14"].shift(1) > 100) & (out["cci14"] < out["cci14"].shift(1))).astype(int)
    high_14 = high.rolling(14).max()
    low_14 = low.rolling(14).min()
    out["wr14"] = (high_14 - close) / (high_14 - low_14).replace(0, np.nan) * -100

    out["ma20_slope_5d"] = ma20 / ma20.shift(5) - 1
    out["ma60_slope_10d"] = ma60 / ma60.shift(10) - 1
    out["ma20_slope_5d_negative"] = (out["ma20_slope_5d"] < 0).astype(int)
    out["ma60_slope_10d_negative"] = (out["ma60_slope_10d"] < 0).astype(int)
    out["close_above_ma20"] = (close > ma20).astype(int)
    out["close_above_ma60"] = (close > ma60).astype(int)
    out["above_ma20_days"] = _consecutive_days(close > ma20)
    out["below_ma20_days"] = _consecutive_days(close < ma20)
    out["above_ma60_days"] = _consecutive_days(close > ma60)
    out["below_ma60_days"] = _consecutive_days(close < ma60)
    out["ma20_ma60_gap"] = ma20 / ma60 - 1
    out["drawdown_20d"] = close / close.rolling(20).max() - 1
    out["drawdown_60d"] = close / close.rolling(60).max() - 1

    regime_cfg = get_config().get("market_regime", {})
    bull_threshold = float(regime_cfg.get("causal_bull_score_threshold", 2))
    bear_threshold = float(regime_cfg.get("causal_bear_score_threshold", -2))
    trend_votes = pd.concat(
        [
            pd.Series(np.where(close > ma20, 1, -1), index=close.index),
            pd.Series(np.where(close > ma60, 1, -1), index=close.index),
            pd.Series(np.where(out["ma20_slope_5d"] > 0, 1, -1), index=close.index),
            pd.Series(np.where(out["ma60_slope_10d"] > 0, 1, -1), index=close.index),
        ],
        axis=1,
    )
    unavailable = ma20.isna() | ma60.isna() | out["ma20_slope_5d"].isna() | out["ma60_slope_10d"].isna()
    out["regime_trend_score"] = trend_votes.sum(axis=1).where(~unavailable)
    out["is_bull_trend"] = (out["regime_trend_score"] >= bull_threshold).astype(int)
    out["is_bear_trend"] = (out["regime_trend_score"] <= bear_threshold).astype(int)
    regime_state = pd.Series(0, index=out.index, dtype=int)
    regime_state.loc[out["is_bull_trend"] == 1] = 1
    regime_state.loc[out["is_bear_trend"] == 1] = -1
    state_groups = regime_state.ne(regime_state.shift(fill_value=0)).cumsum()
    out["regime_state_days"] = regime_state.groupby(state_groups).cumcount() + 1
    out.loc[regime_state == 0, "regime_state_days"] = 0

    out["volume_ratio_5d_20d"] = volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)
    out["amount_percentile_60d"] = _rolling_percentile(amount, 60)
    out["volume_percentile_60d"] = _rolling_percentile(volume, 60)
    ret_1d = close.pct_change()
    out["down_with_volume"] = ((ret_1d < 0) & (out["volume_zscore_20d"] > 1)).astype(int)
    out["up_with_volume"] = ((ret_1d > 0) & (out["volume_zscore_20d"] > 1)).astype(int)
    out["shrink_rebound"] = ((ret_1d > 0) & (out["volume_ratio_5d_20d"] < 0.8)).astype(int)
    return out


def build_market_features() -> int:
    raw = read_sql("SELECT * FROM market_index_daily ORDER BY index_code, trade_date")
    if raw.empty:
        logger.warning("no index data available for market features")
        return 0
    raw["trade_date"] = pd.to_datetime(raw["trade_date"]).dt.date
    features = pd.concat([_build_one(g) for _, g in raw.groupby("index_code")], ignore_index=True)

    pivot_ret = features.pivot(index="trade_date", columns="index_code", values="ret_1d")
    target = get_config()["project"]["target_index"]
    rel = pd.DataFrame(index=pivot_ret.index)
    rel["relative_hs300"] = pivot_ret.get(target) - pivot_ret.get("000300")
    rel["relative_zz500"] = pivot_ret.get(target) - pivot_ret.get("000905")
    rel["relative_cyb"] = pivot_ret.get(target) - pivot_ret.get("399006")
    rel = rel.reset_index().rename(columns={"index": "trade_date"})
    features = features.merge(rel, on="trade_date", how="left")

    cols = [
        "trade_date",
        "index_code",
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ma5_gap",
        "ma10_gap",
        "ma20_gap",
        "macd",
        "macd_signal",
        "macd_hist",
        "rsi6",
        "rsi14",
        "atr14",
        "boll_width",
        "amount_zscore_20d",
        "volatility_10d",
        "volatility_20d",
        "relative_hs300",
        "relative_zz500",
        "relative_cyb",
    ]
    cols.extend(EXTRA_MARKET_FEATURE_COLS)
    _ensure_market_feature_columns(cols)
    count = upsert_dataframe(features[cols], "market_feature_daily", ["trade_date", "index_code"])
    logger.info("built market features rows=%s", count)
    return count


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(build_market_features())


if __name__ == "__main__":
    main()
