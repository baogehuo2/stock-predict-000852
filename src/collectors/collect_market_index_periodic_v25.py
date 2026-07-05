from __future__ import annotations

import argparse
import hashlib
import json
import time as sleep_time
from datetime import date, datetime, time
from typing import Any

import pandas as pd
import requests

from src.common.config import get_config, load_yaml, project_path
from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.network import disable_env_proxies
from src.common.secrets import get_secret
from src.quality.check_collection_data import DatasetSpec, evaluate_dataframe, persist_metrics


DATA_SOURCE = "tushare:index_weekly_monthly"
SOURCE_URL = "https://tushare.pro/"
EASTMONEY_KLINE_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
TUSHARE_URL = "https://api.tushare.pro"
RETRY_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 2.0


def _raw_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compact(value: str) -> str:
    return value.replace("-", "")


def _available_time(trade_dates: pd.Series) -> pd.Series:
    return pd.to_datetime(trade_dates).map(lambda value: datetime.combine(value.date(), time(15, 30)))


def _period_start(trade_dates: pd.Series, period: str) -> pd.Series:
    timestamps = pd.to_datetime(trade_dates)
    if period == "weekly":
        return (timestamps - pd.to_timedelta(timestamps.dt.weekday, unit="D")).dt.date
    if period == "monthly":
        return timestamps.dt.to_period("M").dt.start_time.dt.date
    raise ValueError(f"unsupported period: {period}")


def normalize_index_periodic(
    raw: pd.DataFrame,
    item: dict[str, Any],
    period: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    mapping = {
        "日期": "trade_date",
        "date": "trade_date",
        "trade_date": "trade_date",
        "开盘": "open",
        "open": "open",
        "最高": "high",
        "high": "high",
        "最低": "low",
        "low": "low",
        "收盘": "close",
        "close": "close",
        "pre_close": "pre_close",
        "昨收": "pre_close",
        "涨跌幅": "pct_chg",
        "pct_chg": "pct_chg",
        "成交量": "volume",
        "volume": "volume",
        "vol": "volume",
        "成交额": "amount",
        "amount": "amount",
    }
    frame = raw.rename(columns=mapping).copy()
    required = ["trade_date", "open", "high", "low", "close", "pct_chg", "volume", "amount"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{item['code']} {period} missing columns {missing}; got={list(raw.columns)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "pre_close" in frame.columns:
        frame["pre_close"] = pd.to_numeric(frame["pre_close"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "open", "high", "low", "close"])
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()
    if frame.empty:
        return frame
    ohlc = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    frame["high"] = ohlc.max(axis=1)
    frame["low"] = ohlc.min(axis=1)
    frame["pct_chg"] = frame["pct_chg"].where(frame["pct_chg"].abs().le(1), frame["pct_chg"] / 100)
    calculated_pre_close = frame["close"] / (1 + frame["pct_chg"])
    if "pre_close" in frame.columns:
        frame["pre_close"] = frame["pre_close"].combine_first(calculated_pre_close)
    else:
        frame["pre_close"] = calculated_pre_close
    if "ts_code" in frame.columns:
        frame["amount"] = frame["amount"] * 1000
    frame["index_code"] = item["code"]
    frame["index_name"] = item["name"]
    frame["currency"] = "CNY"
    frame["pct_chg_unit"] = "decimal"
    frame["volume_unit"] = "share"
    frame["amount_unit"] = "CNY_yuan"
    frame["period_start_date"] = _period_start(pd.to_datetime(frame["trade_date"]), period)
    frame["period_end_date"] = frame["trade_date"]
    frame["data_source"] = DATA_SOURCE
    frame["source_url"] = SOURCE_URL
    frame["available_time"] = _available_time(frame["trade_date"])
    frame["crawl_time"] = datetime.now()
    frame["raw_hash"] = frame.apply(lambda row: _raw_hash(row.to_dict()), axis=1)
    columns = [
        "trade_date", "index_code", "index_name", "open", "high", "low", "close",
        "pre_close", "pct_chg", "volume", "amount", "currency", "pct_chg_unit",
        "volume_unit", "amount_unit", "period_start_date", "period_end_date",
        "data_source", "source_url", "available_time", "crawl_time", "raw_hash",
    ]
    return frame[columns].sort_values("trade_date").drop_duplicates(["trade_date", "index_code"], keep="last")


def _eastmoney_secid(index_code: str) -> str:
    market = "0" if index_code.startswith("399") else "1"
    return f"{market}.{index_code}"


def _eastmoney_klt(period: str) -> str:
    mapping = {"weekly": "102", "monthly": "103"}
    if period not in mapping:
        raise ValueError(f"unsupported period: {period}")
    return mapping[period]


def _tushare_api_name(period: str) -> str:
    mapping = {"weekly": "index_weekly", "monthly": "index_monthly"}
    if period not in mapping:
        raise ValueError(f"unsupported period: {period}")
    return mapping[period]


def _tushare_ts_code(index_code: str) -> str:
    suffix = "SZ" if index_code.startswith("399") else "SH"
    return f"{index_code}.{suffix}"


def fetch_tushare_index_periodic(item: dict[str, Any], period: str, start_date: str, end_date: str) -> pd.DataFrame:
    disable_env_proxies()
    token = get_secret("tushare.token")
    if not token:
        raise RuntimeError("missing tushare.token in encrypted secrets")
    payload = {
        "api_name": _tushare_api_name(period),
        "token": token,
        "params": {
            "ts_code": _tushare_ts_code(item["code"]),
            "start_date": _compact(start_date),
            "end_date": _compact(end_date),
        },
        "fields": "",
    }
    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(TUSHARE_URL, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            if data.get("code") != 0:
                raise RuntimeError(f"tushare {_tushare_api_name(period)} request failed: {data}")
            block = data.get("data", {})
            return pd.DataFrame(block.get("items", []), columns=block.get("fields", []))
        except Exception as exc:
            last_error = exc
            if attempt < RETRY_ATTEMPTS:
                sleep_time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"tushare:{_tushare_api_name(period)}:{item['code']} failed after {RETRY_ATTEMPTS} attempts: {last_error}")


def fetch_eastmoney_index_periodic(item: dict[str, Any], period: str, start_date: str, end_date: str) -> pd.DataFrame:
    disable_env_proxies()
    params = {
        "secid": _eastmoney_secid(item["code"]),
        "klt": _eastmoney_klt(period),
        "fqt": "0",
        "beg": _compact(start_date),
        "end": _compact(end_date),
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
    }
    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            response = requests.get(
                EASTMONEY_KLINE_URL,
                params=params,
                headers={"User-Agent": "Mozilla/5.0 data-collectors/2.5"},
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data") or {}
            klines = data.get("klines") or []
            if not klines:
                return pd.DataFrame()
            rows = [str(line).split(",") for line in klines]
            return pd.DataFrame(
                rows,
                columns=["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"],
            )
        except Exception as exc:
            last_error = exc
            if attempt < RETRY_ATTEMPTS:
                sleep_time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"eastmoney:index_kline:{item['code']}:{period} failed after {RETRY_ATTEMPTS} attempts: {last_error}")


def fetch_akshare_index_periodic(item: dict[str, Any], period: str, start_date: str, end_date: str) -> pd.DataFrame:
    disable_env_proxies()
    import akshare as ak

    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            return ak.index_zh_a_hist(
                symbol=item["code"],
                period=period,
                start_date=_compact(start_date),
                end_date=_compact(end_date),
            )
        except Exception as exc:
            last_error = exc
            if attempt < RETRY_ATTEMPTS:
                sleep_time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(f"{DATA_SOURCE}:{item['code']}:{period} failed after {RETRY_ATTEMPTS} attempts: {last_error}")


def fetch_index_periodic(item: dict[str, Any], period: str, start_date: str, end_date: str) -> pd.DataFrame:
    errors: list[str] = []
    try:
        return fetch_tushare_index_periodic(item, period, start_date, end_date)
    except Exception as exc:
        errors.append(f"tushare:{type(exc).__name__}:{exc}")
    try:
        return fetch_eastmoney_index_periodic(item, period, start_date, end_date)
    except Exception as exc:
        errors.append(f"eastmoney:{type(exc).__name__}:{exc}")
    try:
        return fetch_akshare_index_periodic(item, period, start_date, end_date)
    except Exception as exc:
        errors.append(f"akshare:{type(exc).__name__}:{exc}")
    raise RuntimeError("; ".join(errors))


def _existing_count(table: str, code: str, start_date: str, end_date: str) -> int:
    result = read_sql(
        f"""
        SELECT COUNT(*) AS row_count FROM {table}
        WHERE index_code = :code AND trade_date BETWEEN :start_date AND :end_date
        """,
        {"code": code, "start_date": start_date, "end_date": end_date},
    )
    return int(result.iloc[0]["row_count"])


def _quality_report(table: str, frame: pd.DataFrame) -> dict[str, Any]:
    spec = DatasetSpec(
        table=table,
        date_column="trade_date",
        unique_columns=("trade_date", "index_code"),
        required_columns=("trade_date", "index_code", "open", "high", "low", "close", "data_source", "available_time"),
        non_negative_columns=("volume", "amount"),
        ohlc=True,
    )
    metrics = evaluate_dataframe(frame, spec)
    persist_metrics(table, metrics, date.today())
    return {"metrics": metrics}


def _load_cfg() -> dict[str, Any]:
    return load_yaml(project_path("config", "data_sources_v2.yaml"))["market_index_periodic"]


def collect_market_index_periodic(
    start_date: str | None = None,
    end_date: str | None = None,
    period: str | None = None,
) -> dict[str, Any]:
    config = get_config()
    cfg = _load_cfg()
    start_date = start_date or config["project"]["start_date"]
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    period_specs = [item for item in cfg["periods"] if period is None or item["period"] == period]
    if not period_specs:
        raise ValueError(f"unsupported period: {period}")
    result: dict[str, Any] = {"periods": {}, "failed": []}
    for period_spec in period_specs:
        current_period = period_spec["period"]
        table = period_spec["table"]
        frames: list[pd.DataFrame] = []
        run = IngestionRun(
            dataset_name=table,
            data_source=DATA_SOURCE,
            requested_start_date=pd.Timestamp(start_date).date(),
            requested_end_date=pd.Timestamp(end_date).date(),
        )
        try:
            inserted = 0
            updated = 0
            for item in cfg["indices"]:
                raw = fetch_index_periodic(item, current_period, start_date, end_date)
                frame = normalize_index_periodic(raw, item, current_period, start_date, end_date)
                if frame.empty:
                    result["failed"].append(f"{table}:{item['code']}:empty")
                    continue
                existing = _existing_count(table, item["code"], start_date, end_date)
                upsert_dataframe(frame, table, ["trade_date", "index_code"])
                inserted += max(0, len(frame) - existing)
                updated += min(len(frame), existing)
                frames.append(frame)
            combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            if combined.empty:
                raise RuntimeError(f"{table} collected no rows")
            run.fetched_rows = len(combined)
            run.inserted_rows = inserted
            run.updated_rows = updated
            run.finish("success")
            result["periods"][current_period] = {
                "table": table,
                "rows": len(combined),
                "index_count": int(combined["index_code"].nunique()),
                "trade_date_min": str(combined["trade_date"].min()),
                "trade_date_max": str(combined["trade_date"].max()),
                "quality": _quality_report(table, combined),
            }
        except Exception as exc:
            run.error_rows = 1
            run.finish("failed", f"{type(exc).__name__}: {exc}")
            raise
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="采集2.5批指数周K/月K，直接使用AKShare周期接口。")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--period", choices=["weekly", "monthly"])
    args = parser.parse_args()
    print(collect_market_index_periodic(args.start_date, args.end_date, args.period))


if __name__ == "__main__":
    main()
