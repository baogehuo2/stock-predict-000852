from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import text

from src.common.config import get_config, load_yaml, project_path
from src.common.db import get_engine, read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.quality.check_collection_data import DatasetSpec, evaluate_date_coverage, evaluate_dataframe, persist_metrics


logger = get_logger(__name__)
CSINDEX_URL = "https://www.csindex.com.cn/csindex-home/perf/index-perf"
CNI_URL = "http://hq.cnindex.com.cn/market/market/getIndexDailyDataWithDataFormat"


def _trade_calendar(start_date: str, end_date: str) -> pd.Series:
    calendar = read_sql(
        """
        SELECT DISTINCT trade_date
        FROM market_index_daily
        WHERE index_code = '000852'
          AND trade_date BETWEEN :start_date AND :end_date
        ORDER BY trade_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if calendar.empty:
        raise RuntimeError("market_index_daily缺少000852交易日历，不能安全过滤官方接口休市伪记录")
    return calendar["trade_date"]


def _available_time(trade_dates: pd.Series) -> pd.Series:
    return pd.to_datetime(trade_dates).map(lambda value: datetime.combine(value.date(), time(15, 30)))


def _normalize_csindex(raw: pd.DataFrame, code: str, name: str) -> pd.DataFrame:
    mapping = {
        "日期": "trade_date", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "涨跌": "change", "涨跌幅": "pct_chg",
        "成交量": "volume", "成交金额": "amount",
    }
    frame = raw.rename(columns=mapping).copy()
    required = ["trade_date", "open", "high", "low", "close", "pct_chg", "volume", "amount"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"中证指数接口缺少字段: {missing}; got={list(raw.columns)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["change"] = pd.to_numeric(frame["change"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "open", "high", "low", "close"])
    frame["amount"] = frame["amount"] * 100_000_000
    calculated_pre_close = frame["close"] / (1 + frame["pct_chg"] / 100)
    frame["pre_close"] = (frame["close"] - frame["change"]).combine_first(calculated_pre_close)
    frame["index_code"] = code
    frame["index_name"] = name
    frame["currency"] = "CNY"
    frame["pct_chg_unit"] = "percent"
    frame["volume_unit"] = "share"
    frame["amount_unit"] = "CNY_yuan"
    frame["data_source"] = "akshare:stock_zh_index_hist_csindex"
    frame["source_url"] = CSINDEX_URL
    return frame


def _normalize_cni(raw: pd.DataFrame, code: str, name: str) -> pd.DataFrame:
    mapping = {
        "日期": "trade_date", "开盘价": "open", "最高价": "high", "最低价": "low",
        "收盘价": "close", "涨跌幅": "pct_chg", "成交量": "volume", "成交额": "amount",
    }
    frame = raw.rename(columns=mapping).copy()
    required = ["trade_date", "open", "high", "low", "close", "pct_chg", "volume", "amount"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"国证指数接口缺少字段: {missing}; got={list(raw.columns)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "open", "high", "low", "close"])
    frame["pct_chg"] = frame["pct_chg"] * 100
    frame["volume"] = frame["volume"] * 1_000_000
    frame["amount"] = frame["amount"] * 100_000_000
    frame["pre_close"] = frame["close"] / (1 + frame["pct_chg"] / 100)
    frame["index_code"] = code
    frame["index_name"] = name
    frame["currency"] = "CNY"
    frame["pct_chg_unit"] = "percent"
    frame["volume_unit"] = "share"
    frame["amount_unit"] = "CNY_yuan"
    frame["data_source"] = "akshare:index_hist_cni"
    frame["source_url"] = CNI_URL
    return frame


def fetch_style_index(item: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    disable_env_proxies()
    import akshare as ak

    compact_start = start_date.replace("-", "")
    compact_end = end_date.replace("-", "")
    provider = item["provider"]
    if provider == "csindex":
        raw = ak.stock_zh_index_hist_csindex(
            symbol=item["code"], start_date=compact_start, end_date=compact_end
        )
        frame = _normalize_csindex(raw, item["code"], item["name"])
    elif provider == "cni":
        raw = ak.index_hist_cni(symbol=item["code"], start_date=compact_start, end_date=compact_end)
        frame = _normalize_cni(raw, item["code"], item["name"])
    else:
        raise ValueError(f"未知风格指数provider: {provider}")

    calendar = set(pd.to_datetime(_trade_calendar(start_date, end_date)).dt.date)
    frame = frame[frame["trade_date"].isin(calendar)].sort_values("trade_date").drop_duplicates("trade_date")
    if frame.empty:
        raise ValueError(f"风格指数{item['code']}过滤交易日后为空")
    frame["available_time"] = _available_time(frame["trade_date"])
    frame["crawl_time"] = datetime.now()
    columns = [
        "trade_date", "index_code", "index_name", "open", "high", "low", "close",
        "pre_close", "pct_chg", "volume", "amount", "currency", "pct_chg_unit",
        "volume_unit", "amount_unit", "data_source", "source_url", "available_time", "crawl_time",
    ]
    return frame[columns].reset_index(drop=True)


def _existing_count(code: str, start_date: str, end_date: str) -> int:
    result = read_sql(
        """
        SELECT COUNT(*) AS row_count FROM market_index_daily
        WHERE index_code = :code AND trade_date BETWEEN :start_date AND :end_date
        """,
        {"code": code, "start_date": start_date, "end_date": end_date},
    )
    return int(result.iloc[0]["row_count"])


def _quality_report(code: str, frame: pd.DataFrame, calendar: pd.Series) -> dict[str, Any]:
    spec = DatasetSpec(
        table="market_index_daily",
        date_column="trade_date",
        unique_columns=("trade_date", "index_code"),
        required_columns=(
            "trade_date", "index_code", "open", "high", "low", "close", "data_source",
            "source_url", "available_time", "pct_chg_unit", "volume_unit", "amount_unit",
        ),
        non_negative_columns=("volume", "amount"),
        ohlc=True,
    )
    metrics = evaluate_dataframe(frame, spec) + evaluate_date_coverage(frame, spec, calendar)
    persist_metrics(f"market_index_daily:{code}", metrics, date.today())
    return {"index_code": code, "metrics": metrics}


def collect_style_indices(start_date: str | None = None, end_date: str | None = None) -> int:
    config = get_config()
    items = load_yaml(project_path("config", "symbols.yaml"))["style_indices_v2"]
    start_date = start_date or config["project"]["start_date"]
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    calendar = _trade_calendar(start_date, end_date)
    total = 0
    reports: list[dict[str, Any]] = []
    for item in items:
        run = IngestionRun(
            dataset_name=f"market_index_daily:{item['code']}",
            data_source=f"akshare:{item['provider']}",
            requested_start_date=pd.Timestamp(start_date).date(),
            requested_end_date=pd.Timestamp(end_date).date(),
        )
        try:
            frame = fetch_style_index(item, start_date, end_date)
            existing = _existing_count(item["code"], start_date, end_date)
            upserted = upsert_dataframe(frame, "market_index_daily", ["trade_date", "index_code"])
            run.fetched_rows = len(frame)
            run.inserted_rows = max(0, len(frame) - existing)
            run.updated_rows = min(len(frame), existing)
            run.finish("success")
            reports.append(_quality_report(item["code"], frame, calendar))
            total += upserted
            logger.info("collected style index code=%s rows=%s", item["code"], len(frame))
        except Exception as exc:
            run.error_rows = 1
            run.finish("failed", str(exc))
            raise
    output = project_path("data", "reports", "style_indices_v2_quality.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"generated_at": datetime.now().isoformat(), "reports": reports}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="采集模型2.0风格指数行情。")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    print(collect_style_indices(args.start_date, args.end_date))


if __name__ == "__main__":
    main()
