from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from src.common.db import read_sql, upsert_dataframe
from src.common.logger import get_logger


logger = get_logger(__name__)
POS_WORDS = ["涨", "牛", "利好", "反弹", "突破", "看多", "机会", "上涨", "拉升"]
NEG_WORDS = ["跌", "熊", "利空", "回调", "破位", "看空", "风险", "下跌", "兑现"]


def _score_text(text: str) -> int:
    pos = sum(w in text for w in POS_WORDS)
    neg = sum(w in text for w in NEG_WORDS)
    return int(pos) - int(neg)


def _stage(row: pd.Series) -> str:
    z = row.get("heat_zscore_20d")
    change = row.get("heat_change_3d")
    disagreement = row.get("disagreement")
    if pd.isna(z):
        return "冷启动"
    if z > 2 and change > 0:
        return "高潮"
    if z > 1:
        return "发酵"
    if disagreement and disagreement > 0.45 and z > 0.8:
        return "分歧"
    if change and change < -0.2:
        return "退潮"
    if z > 0:
        return "预热"
    return "冷启动"


def build_sentiment_features() -> int:
    posts = read_sql("SELECT * FROM sentiment_guba_raw")
    if posts.empty:
        logger.warning("no guba data available for sentiment features")
        return 0
    posts["trade_date"] = pd.to_datetime(posts["trade_date"]).dt.date
    posts["topic"] = posts["bar_name"]
    posts["score"] = posts["title"].fillna("").map(_score_text)
    posts["pos"] = (posts["score"] > 0).astype(int)
    posts["neg"] = (posts["score"] < 0).astype(int)
    posts["comment_count"] = pd.to_numeric(posts["comment_count"], errors="coerce").fillna(0)
    posts["read_count"] = pd.to_numeric(posts["read_count"], errors="coerce").fillna(0)

    post_grouped = posts.groupby(["trade_date", "topic"], as_index=False).agg(
        post_count=("post_id", "nunique"),
        declared_comment_count=("comment_count", "sum"),
        read_count=("read_count", "sum"),
    )

    score_parts = [posts[["trade_date", "topic", "score", "pos", "neg"]]]
    comments = read_sql("SELECT * FROM sentiment_guba_comment_raw")
    if not comments.empty:
        comments["trade_date"] = pd.to_datetime(comments["trade_date"]).dt.date
        comments["topic"] = comments["bar_name"]
        comments["score"] = comments["content"].fillna("").map(_score_text)
        comments["pos"] = (comments["score"] > 0).astype(int)
        comments["neg"] = (comments["score"] < 0).astype(int)
        score_parts.append(comments[["trade_date", "topic", "score", "pos", "neg"]])
        comment_grouped = comments.groupby(["trade_date", "topic"], as_index=False).agg(
            observed_comment_count=("comment_id", "nunique")
        )
    else:
        comment_grouped = pd.DataFrame(columns=["trade_date", "topic", "observed_comment_count"])

    scored = pd.concat(score_parts, ignore_index=True)
    score_grouped = scored.groupby(["trade_date", "topic"], as_index=False).agg(
        sentiment_score=("score", "mean"),
        positive_ratio=("pos", "mean"),
        negative_ratio=("neg", "mean"),
    )

    grouped = post_grouped.merge(comment_grouped, on=["trade_date", "topic"], how="left")
    grouped = grouped.merge(score_grouped, on=["trade_date", "topic"], how="left")
    grouped["observed_comment_count"] = pd.to_numeric(
        grouped["observed_comment_count"], errors="coerce"
    ).fillna(0)
    grouped["comment_count"] = grouped[["declared_comment_count", "observed_comment_count"]].max(axis=1)
    grouped["source"] = "eastmoney_guba"
    grouped["heat_score"] = (
        np.log1p(grouped["post_count"])
        + 0.5 * np.log1p(grouped["comment_count"])
        + 0.2 * np.log1p(grouped["read_count"])
    )
    grouped["disagreement"] = 1 - (grouped["positive_ratio"] - grouped["negative_ratio"]).abs()

    outputs = []
    for _, g in grouped.sort_values("trade_date").groupby("topic"):
        g = g.copy()
        rolling20 = g["heat_score"].rolling(20)
        rolling60 = g["heat_score"].rolling(60)
        g["heat_zscore_20d"] = (g["heat_score"] - rolling20.mean()) / rolling20.std()
        g["heat_zscore_60d"] = (g["heat_score"] - rolling60.mean()) / rolling60.std()
        g["heat_change_3d"] = g["heat_score"].pct_change(3).replace([math.inf, -math.inf], np.nan)
        g["emotion_stage"] = g.apply(_stage, axis=1)
        g["top_keywords"] = ""
        outputs.append(g)
    result = pd.concat(outputs, ignore_index=True)
    cols = [
        "trade_date",
        "topic",
        "source",
        "post_count",
        "comment_count",
        "read_count",
        "heat_score",
        "heat_zscore_20d",
        "heat_zscore_60d",
        "heat_change_3d",
        "sentiment_score",
        "positive_ratio",
        "negative_ratio",
        "disagreement",
        "emotion_stage",
        "top_keywords",
    ]
    count = upsert_dataframe(result[cols], "sentiment_feature_daily", ["trade_date", "topic", "source"])
    logger.info("built sentiment features rows=%s", count)
    return count


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(build_sentiment_features())


if __name__ == "__main__":
    main()
