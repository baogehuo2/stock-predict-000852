from __future__ import annotations

import json

import pandas as pd

from src.common.db import read_sql


def load_dataset() -> pd.DataFrame:
    df = read_sql("SELECT * FROM model_dataset_daily ORDER BY trade_date")
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    features = df["feature_json"].map(lambda x: json.loads(x) if isinstance(x, str) else (x or {}))
    feat_df = pd.json_normalize(features).add_prefix("f_")
    return pd.concat([df.drop(columns=["feature_json"]), feat_df], axis=1)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("f_")]

