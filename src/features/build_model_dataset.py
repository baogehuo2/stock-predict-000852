from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

from src.common.config import get_config
from src.common.db import read_sql, upsert_dataframe
from src.common.logger import get_logger


logger = get_logger(__name__)


def _label(ret: float | None, vol: float | None, horizon: int) -> str | None:
    if pd.isna(ret) or ret is None:
        return None
    if pd.isna(vol) or vol is None or vol == 0:
        threshold = 0.008
    else:
        threshold = 0.4 * float(vol) * math.sqrt(horizon)
    if ret > threshold:
        return "上涨"
    if ret < -threshold:
        return "下跌"
    return "震荡"


def build_model_dataset() -> int:
    cfg = get_config()
    target = cfg["project"]["target_index"]
    horizons = cfg["model"]["horizons"]
    market = read_sql(
        "SELECT f.*, i.close FROM market_feature_daily f "
        "JOIN market_index_daily i ON f.trade_date=i.trade_date AND f.index_code=i.index_code "
        "WHERE f.index_code=:target ORDER BY f.trade_date",
        {"target": target},
    )
    if market.empty:
        logger.warning("no market features available for dataset")
        return 0
    market["trade_date"] = pd.to_datetime(market["trade_date"]).dt.date
    close = pd.to_numeric(market["close"], errors="coerce")
    for h in horizons:
        market[f"future_ret_{h}d"] = close.shift(-h) / close - 1
        market[f"label_{h}d"] = [
            _label(ret, vol, h)
            for ret, vol in zip(market[f"future_ret_{h}d"], market["volatility_20d"])
        ]

    sentiment = read_sql(
        "SELECT trade_date, AVG(heat_score) heat_score, AVG(heat_zscore_20d) heat_zscore_20d, "
        "AVG(sentiment_score) sentiment_score, AVG(disagreement) disagreement "
        "FROM sentiment_feature_daily GROUP BY trade_date"
    )
    if not sentiment.empty:
        sentiment["trade_date"] = pd.to_datetime(sentiment["trade_date"]).dt.date
        market = market.merge(sentiment, on="trade_date", how="left", suffixes=("", "_sent"))
    else:
        for col in ["heat_score", "heat_zscore_20d", "sentiment_score", "disagreement"]:
            market[col] = 0

    events = read_sql("SELECT trade_date, AVG(event_score) event_score FROM event_daily GROUP BY trade_date")
    if not events.empty:
        events["trade_date"] = pd.to_datetime(events["trade_date"]).dt.date
        market = market.merge(events, on="trade_date", how="left")
    else:
        market["event_score"] = 0
    market["event_score"] = market["event_score"].fillna(0)

    feature_cols = [
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
        "heat_score",
        "heat_zscore_20d",
        "sentiment_score",
        "disagreement",
        "event_score",
    ]
    for col in feature_cols:
        if col not in market:
            market[col] = 0
    market[feature_cols] = market[feature_cols].replace([np.inf, -np.inf], np.nan)

    records = []
    for _, row in market.iterrows():
        features = {col: (None if pd.isna(row[col]) else float(row[col])) for col in feature_cols}
        record = {
            "trade_date": row["trade_date"],
            "index_code": target,
            "feature_json": json.dumps(features, ensure_ascii=False),
        }
        for h in horizons:
            record[f"future_ret_{h}d"] = row.get(f"future_ret_{h}d")
            record[f"label_{h}d"] = row.get(f"label_{h}d")
        records.append(record)
    count = upsert_dataframe(pd.DataFrame(records), "model_dataset_daily", ["trade_date", "index_code"])
    logger.info("built model dataset rows=%s", count)
    return count


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(build_model_dataset())


if __name__ == "__main__":
    main()

