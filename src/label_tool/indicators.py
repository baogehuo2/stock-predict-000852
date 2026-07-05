from __future__ import annotations

import numpy as np
import pandas as pd


def _sma_cn(values: pd.Series, window: int, weight: int = 1, initial: float | None = None) -> pd.Series:
    """Chinese trading software SMA: (M*X + (N-M)*REF(SMA,1)) / N."""
    src = pd.to_numeric(values, errors="coerce")
    result: list[float] = []
    prev = np.nan if initial is None else float(initial)
    for value in src:
        if pd.isna(value):
            result.append(prev)
            continue
        if pd.isna(prev):
            prev = float(value)
        else:
            prev = (weight * float(value) + (window - weight) * prev) / window
        result.append(prev)
    return pd.Series(result, index=src.index, dtype="float64")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.replace(0, np.nan)
    return numerator / den


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    close = pd.to_numeric(data["close"], errors="coerce")
    high = pd.to_numeric(data["high"], errors="coerce")
    low = pd.to_numeric(data["low"], errors="coerce")
    open_ = pd.to_numeric(data["open"], errors="coerce")
    volume = pd.to_numeric(data.get("volume", pd.Series(np.nan, index=data.index)), errors="coerce")

    for window in (5, 10, 20, 60, 120, 250):
        data[f"ma{window}"] = close.rolling(window, min_periods=1).mean()

    mid = close.rolling(20, min_periods=1).mean()
    std = close.rolling(20, min_periods=1).std(ddof=0)
    data["boll_mid"] = mid
    data["boll_upper"] = mid + 2 * std
    data["boll_lower"] = mid - 2 * std
    data["boll_upper_2"] = data["boll_upper"]
    data["boll_lower_2"] = data["boll_lower"]
    data["boll_upper_3"] = mid + 3 * std
    data["boll_lower_3"] = mid - 3 * std

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    data["macd_dif"] = ema12 - ema26
    data["macd_dea"] = data["macd_dif"].ewm(span=9, adjust=False).mean()
    data["macd_bar"] = (data["macd_dif"] - data["macd_dea"]) * 2

    low9 = low.rolling(9, min_periods=1).min()
    high9 = high.rolling(9, min_periods=1).max()
    rsv = _safe_divide(close - low9, high9 - low9) * 100
    rsv = rsv.fillna(50)
    data["kdj_k"] = _sma_cn(rsv, 3, 1, initial=50)
    data["kdj_d"] = _sma_cn(data["kdj_k"], 3, 1, initial=50)
    data["kdj_j"] = 3 * data["kdj_k"] - 2 * data["kdj_d"]

    diff = close.diff()
    gain = diff.clip(lower=0)
    abs_diff = diff.abs()
    for window in (6, 12, 24):
        up_sma = _sma_cn(gain, window, 1)
        abs_sma = _sma_cn(abs_diff, window, 1)
        data[f"rsi{window}"] = _safe_divide(up_sma, abs_sma) * 100

    typ = (high + low + close) / 3
    typ_ma14 = typ.rolling(14, min_periods=1).mean()
    avedev14 = (typ - typ_ma14).abs().rolling(14, min_periods=1).mean()
    data["cci14"] = _safe_divide(typ - typ_ma14, 0.015 * avedev14)

    high14 = high.rolling(14, min_periods=1).max()
    low14 = low.rolling(14, min_periods=1).min()
    data["wr14"] = _safe_divide(high14 - close, high14 - low14) * -100

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    data["atr14"] = tr.rolling(14, min_periods=1).mean()

    day_range = high - low
    body = (close - open_).abs()
    data["upper_shadow_ratio"] = _safe_divide(high - pd.concat([open_, close], axis=1).max(axis=1), day_range)
    data["lower_shadow_ratio"] = _safe_divide(pd.concat([open_, close], axis=1).min(axis=1) - low, day_range)
    data["body_ratio"] = _safe_divide(body, day_range)

    data["vol_ma5"] = volume.rolling(5, min_periods=1).mean()
    data["vol_ma10"] = volume.rolling(10, min_periods=1).mean()
    return data
