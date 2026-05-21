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

EVENT_GROUPS = ["index_style", "liquidity", "policy_market", "growth_industry", "macro"]


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


def _json_list(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    try:
        data = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if str(item)]


def _event_features() -> pd.DataFrame:
    events = read_sql(
        "SELECT e.trade_date, e.event_score, e.impact_strength, "
        "COALESCE(n.matched_groups, '[]') matched_groups "
        "FROM event_daily e "
        "LEFT JOIN news_raw n ON e.source_ids=n.news_id"
    )
    base_cols = [
        "event_score",
        "event_count",
        "event_positive_count",
        "event_negative_count",
        "event_max_strength",
    ]
    group_cols = [
        f"event_{group}_{suffix}"
        for group in EVENT_GROUPS
        for suffix in ["score", "count", "positive_count", "negative_count", "max_strength"]
    ]
    all_cols = ["trade_date", *base_cols, *group_cols]
    if events.empty:
        return pd.DataFrame(columns=all_cols)

    events["trade_date"] = pd.to_datetime(events["trade_date"]).dt.date
    events["event_score"] = pd.to_numeric(events["event_score"], errors="coerce").fillna(0.0)
    events["impact_strength"] = pd.to_numeric(events["impact_strength"], errors="coerce").fillna(1.0)
    events["matched_groups"] = events["matched_groups"].apply(_json_list)

    rows = []
    for trade_date, part in events.groupby("trade_date"):
        record = {
            "trade_date": trade_date,
            "event_score": float(part["event_score"].mean()),
            "event_count": int(len(part)),
            "event_positive_count": int((part["event_score"] > 0).sum()),
            "event_negative_count": int((part["event_score"] < 0).sum()),
            "event_max_strength": float(part["impact_strength"].max()),
        }
        for group in EVENT_GROUPS:
            group_mask = part["matched_groups"].apply(lambda groups, name=group: name in groups)
            group_part = part[group_mask]
            prefix = f"event_{group}"
            record[f"{prefix}_score"] = float(group_part["event_score"].sum()) if not group_part.empty else 0.0
            record[f"{prefix}_count"] = int(len(group_part))
            record[f"{prefix}_positive_count"] = int((group_part["event_score"] > 0).sum()) if not group_part.empty else 0
            record[f"{prefix}_negative_count"] = int((group_part["event_score"] < 0).sum()) if not group_part.empty else 0
            record[f"{prefix}_max_strength"] = float(group_part["impact_strength"].max()) if not group_part.empty else 0.0
        rows.append(record)
    return pd.DataFrame(rows, columns=all_cols)


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

    events = _event_features()
    if not events.empty:
        market = market.merge(events, on="trade_date", how="left")
    else:
        market["event_score"] = 0
    event_feature_cols = [col for col in events.columns if col != "trade_date"] if not events.empty else ["event_score"]
    for col in event_feature_cols:
        if col not in market:
            market[col] = 0
        market[col] = market[col].fillna(0)

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
        "event_count",
        "event_positive_count",
        "event_negative_count",
        "event_max_strength",
        "event_index_style_score",
        "event_index_style_count",
        "event_index_style_positive_count",
        "event_index_style_negative_count",
        "event_index_style_max_strength",
        "event_liquidity_score",
        "event_liquidity_count",
        "event_liquidity_positive_count",
        "event_liquidity_negative_count",
        "event_liquidity_max_strength",
        "event_policy_market_score",
        "event_policy_market_count",
        "event_policy_market_positive_count",
        "event_policy_market_negative_count",
        "event_policy_market_max_strength",
        "event_growth_industry_score",
        "event_growth_industry_count",
        "event_growth_industry_positive_count",
        "event_growth_industry_negative_count",
        "event_growth_industry_max_strength",
        "event_macro_score",
        "event_macro_count",
        "event_macro_positive_count",
        "event_macro_negative_count",
        "event_macro_max_strength",
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
