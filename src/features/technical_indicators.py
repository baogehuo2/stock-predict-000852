from __future__ import annotations

import numpy as np
import pandas as pd


def sma_cn(
    values: pd.Series,
    window: int,
    weight: int = 1,
    initial: float | None = None,
) -> pd.Series:
    """Chinese market SMA: (M * X + (N - M) * REF(SMA, 1)) / N."""
    series = pd.to_numeric(values, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    previous = initial
    has_previous = initial is not None
    for index, value in series.items():
        if pd.isna(value):
            result.loc[index] = previous if has_previous and initial is not None else np.nan
            continue
        current = float(value) if not has_previous else (weight * float(value) + (window - weight) * float(previous)) / window
        result.loc[index] = current
        previous = current
        has_previous = True
    return result


def rsi_cn(close: pd.Series, window: int) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    lc = close.shift(1)
    diff = close - lc
    up = diff.clip(lower=0)
    abs_diff = diff.abs()
    up_sma = sma_cn(up, window, 1)
    abs_sma = sma_cn(abs_diff, window, 1)
    return up_sma / abs_sma.replace(0, np.nan) * 100


def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    bar_multiplier: float = 2.0,
) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    dif = ema(close, fast) - ema(close, slow)
    dea = ema(dif, signal)
    hist = dif - dea
    return pd.DataFrame(
        {
            "macd_dif": dif,
            "macd_dea": dea,
            "macd_hist": hist,
            "macd_bar": hist * bar_multiplier,
        },
        index=close.index,
    )


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n: int = 9,
    m1: int = 3,
    m2: int = 3,
    initial: float = 50.0,
) -> pd.DataFrame:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    lowest = low.rolling(n, min_periods=1).min()
    highest = high.rolling(n, min_periods=1).max()
    denominator = highest - lowest
    rsv = (close - lowest) / denominator.replace(0, np.nan) * 100
    k_value = sma_cn(rsv, m1, 1, initial=initial)
    d_value = sma_cn(k_value, m2, 1, initial=initial)
    j_value = 3 * k_value - 2 * d_value
    return pd.DataFrame(
        {
            "kdj_k": k_value,
            "kdj_d": d_value,
            "kdj_j": j_value,
            "kdj_k_minus_d": k_value - d_value,
        },
        index=close.index,
    )


def boll(
    close: pd.Series,
    window: int = 20,
    multiplier: float = 2.0,
    ddof: int = 0,
) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=ddof)
    upper = mid + multiplier * std
    lower = mid - multiplier * std
    width = (upper - lower) / mid
    position = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {
            "boll_mid": mid,
            "boll_upper": upper,
            "boll_lower": lower,
            "boll_width": width,
            "boll_position": position,
        },
        index=close.index,
    )


def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    typical_price = (high + low + close) / 3
    typical_mean = typical_price.rolling(window).mean()
    mean_deviation = (typical_price - typical_mean).abs().rolling(window).mean()
    return (typical_price - typical_mean) / (0.015 * mean_deviation.replace(0, np.nan))


def wr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    sign: str = "negative",
) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    highest = high.rolling(window).max()
    lowest = low.rolling(window).min()
    value = (highest - close) / (highest - lowest).replace(0, np.nan) * 100
    if sign == "negative":
        return value * -1
    if sign == "positive":
        return value
    raise ValueError("sign must be 'negative' or 'positive'")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
    normalize: bool = True,
) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce")
    low = pd.to_numeric(low, errors="coerce")
    close = pd.to_numeric(close, errors="coerce")
    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    value = true_range.rolling(window).mean()
    return value / close if normalize else value
