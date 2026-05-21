from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.common.config import get_config
from src.common.db import read_sql, upsert_dataframe
from src.common.logger import get_logger


logger = get_logger(__name__)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _build_one(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("trade_date").copy()
    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    amount = pd.to_numeric(df["amount"], errors="coerce")

    out = pd.DataFrame({"trade_date": df["trade_date"], "index_code": df["index_code"]})
    for n in [1, 3, 5, 10]:
        out[f"ret_{n}d"] = close.pct_change(n)
    for n in [5, 10, 20]:
        ma = close.rolling(n).mean()
        out[f"ma{n}_gap"] = close / ma - 1

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["rsi6"] = _rsi(close, 6)
    out["rsi14"] = _rsi(close, 14)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean() / close
    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    out["boll_width"] = (4 * std) / mid
    out["amount_zscore_20d"] = (amount - amount.rolling(20).mean()) / amount.rolling(20).std()
    out["volatility_10d"] = close.pct_change().rolling(10).std()
    out["volatility_20d"] = close.pct_change().rolling(20).std()
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
    count = upsert_dataframe(features[cols], "market_feature_daily", ["trade_date", "index_code"])
    logger.info("built market features rows=%s", count)
    return count


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(build_market_features())


if __name__ == "__main__":
    main()
