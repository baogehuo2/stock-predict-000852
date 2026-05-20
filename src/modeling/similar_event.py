from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from src.modeling.data import feature_columns, load_dataset


def find_similar_dates(trade_date: str | pd.Timestamp, top_n: int = 20) -> pd.DataFrame:
    df = load_dataset()
    if df.empty:
        return pd.DataFrame()
    features = feature_columns(df)
    df = df.dropna(subset=features, how="all").copy()
    target_date = pd.to_datetime(trade_date)
    current = df[df["trade_date"] == target_date]
    history = df[df["trade_date"] < target_date]
    if current.empty or history.empty:
        return pd.DataFrame()
    x = pd.concat([history[features], current[features]], axis=0).fillna(0)
    scaled = StandardScaler().fit_transform(x)
    sims = cosine_similarity(scaled[-1:], scaled[:-1]).ravel()
    result = history[["trade_date", "future_ret_3d", "future_ret_5d", "future_ret_7d"]].copy()
    result["similarity"] = sims
    return result.sort_values("similarity", ascending=False).head(top_n)


def summarize_similar_dates(trade_date: str | pd.Timestamp) -> dict:
    sims = find_similar_dates(trade_date)
    if sims.empty:
        return {"count": 0, "summary": "历史相似样本不足。", "score": 0.0}
    summary = {"count": int(len(sims)), "score": float(sims["similarity"].mean())}
    for horizon in [3, 5, 7]:
        col = f"future_ret_{horizon}d"
        summary[f"avg_ret_{horizon}d"] = float(sims[col].mean())
        summary[f"up_prob_{horizon}d"] = float((sims[col] > 0).mean())
    summary["dates"] = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(sims["trade_date"]).head(5)]
    summary["summary"] = (
        f"相似日期Top{len(sims)}中，未来5日平均收益{summary.get('avg_ret_5d', 0):.2%}，"
        f"上涨概率{summary.get('up_prob_5d', 0):.1%}。"
    )
    return summary

