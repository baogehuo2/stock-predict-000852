from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import read_sql, upsert_dataframe
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.logger import get_logger
from src.modeling.data import feature_columns, load_dataset
from src.modeling.similar_event import summarize_similar_dates


disable_broken_dask_autoload()


logger = get_logger(__name__)


def _latest_or_date(df: pd.DataFrame, trade_date: str | None) -> pd.Series:
    if trade_date:
        row = df[df["trade_date"] == pd.to_datetime(trade_date)]
        if row.empty:
            raise RuntimeError(f"No dataset row found for {trade_date}")
        return row.iloc[-1]
    return df.sort_values("trade_date").iloc[-1]


def predict(trade_date: str | None = None) -> int:
    cfg = get_config()
    df = load_dataset()
    if df.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    row = _latest_or_date(df, trade_date)
    model_dir = project_path(cfg["paths"]["models"])
    feature_cols = feature_columns(df)
    x = pd.DataFrame([row[feature_cols]], columns=feature_cols)
    similar = summarize_similar_dates(row["trade_date"])
    sent_score = float(row.get("f_sentiment_score") or 0)
    event_score = float(row.get("f_event_score") or 0)
    tech_score = float(np.nanmean([row.get("f_ret_1d"), row.get("f_ma5_gap"), row.get("f_relative_hs300")]))

    records = []
    for horizon in cfg["model"]["horizons"]:
        bundle = joblib.load(model_dir / f"lgbm_{horizon}d.joblib")
        xh = x.reindex(columns=bundle["features"], fill_value=0)
        pred_label = bundle["label_encoder"].inverse_transform(bundle["classifier"].predict(xh))[0]
        proba = bundle["classifier"].predict_proba(xh)[0]
        pred_ret = float(bundle["regressor"].predict(xh)[0])
        confidence = float(np.max(proba))
        band = max(abs(pred_ret) * 0.5, 0.005)
        explanation = (
            f"模型预测{horizon}日方向为{pred_label}，预测涨跌幅{pred_ret:.2%}，置信度{confidence:.1%}。"
            f"技术得分{tech_score:.4f}，舆情得分{sent_score:.4f}，事件得分{event_score:.4f}。"
            f"{similar['summary']}"
        )
        risk = "免费数据源和大模型抽取可能存在延迟或缺失；本报告仅用于量化研究与信息整理，不构成投资建议。"
        records.append(
            {
                "trade_date": row["trade_date"].date(),
                "horizon": horizon,
                "direction_pred": pred_label,
                "return_pred": pred_ret,
                "return_low": pred_ret - band,
                "return_high": pred_ret + band,
                "confidence": confidence,
                "tech_score": tech_score,
                "sentiment_score": sent_score,
                "event_score": event_score,
                "similar_event_score": similar["score"],
                "explanation": explanation,
                "risk_warning": risk,
            }
        )
    count = upsert_dataframe(pd.DataFrame(records), "prediction_result", ["trade_date", "horizon"])
    logger.info("prediction rows=%s trade_date=%s", count, row["trade_date"])
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date")
    args = parser.parse_args()
    print(predict(args.trade_date))


if __name__ == "__main__":
    main()
