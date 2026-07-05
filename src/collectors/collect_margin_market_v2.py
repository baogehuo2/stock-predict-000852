from __future__ import annotations

import argparse
import json
import time as time_module
from datetime import date, datetime, time
from typing import Any

import pandas as pd
import requests

from src.common.config import project_path
from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.quality.check_collection_data import (
    DATASETS,
    evaluate_dataframe,
    evaluate_margin_relationships,
    persist_metrics,
)


logger = get_logger(__name__)
SSE_SOURCE = "akshare:stock_margin_sse"
SZSE_SOURCE = "akshare:stock_margin_szse"
SZSE_HISTORY_SOURCE = "akshare:macro_china_market_margin_sz"
SZSE_EASTMONEY_SOURCE = "eastmoney:RPTA_WEB_RZRQ_LSSH"
SSE_SOURCE_URL = "https://www.sse.com.cn/market/othersdata/margin/sum/"
SZSE_SOURCE_URL = "https://www.szse.cn/disclosure/margin/margin/index.html"
SZSE_HISTORY_SOURCE_URL = "https://datacenter.jin10.com/reportType/dc_market_margin_sz"
SZSE_EASTMONEY_SOURCE_URL = "https://data.eastmoney.com/rzrq/"
SZSE_UNIT_MULTIPLIER = 100_000_000.0


def _available_time(trade_date: date) -> datetime:
    return datetime.combine(trade_date, time(18, 0))


def _fallback_available_time(trade_date: date) -> datetime:
    return datetime.combine(
        (pd.Timestamp(trade_date) + pd.Timedelta(days=1)).date(), time(0, 0)
    )


def normalize_margin_sse(raw: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "信用交易日期": "trade_date",
        "融资余额": "financing_balance",
        "融资买入额": "financing_buy_amount",
        "融券余量": "securities_lending_remaining_volume",
        "融券余量金额": "securities_lending_balance",
        "融券卖出量": "securities_lending_sell_volume",
        "融资融券余额": "margin_balance",
    }
    missing = [column for column in columns if column not in raw.columns]
    if missing:
        raise ValueError(f"SSE margin data missing columns: {missing}")
    frame = raw.rename(columns=columns)[list(columns.values())].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    numeric_columns = [column for column in frame.columns if column != "trade_date"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["trade_date"]).drop_duplicates("trade_date")
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["exchange"] = "SSE"
    frame["currency"] = "CNY"
    frame["data_source"] = SSE_SOURCE
    frame["source_url"] = SSE_SOURCE_URL
    frame["release_time"] = frame["trade_date"].map(_available_time)
    frame["available_time"] = frame["release_time"]
    frame["crawl_time"] = datetime.now()
    return _ordered_columns(frame)


def normalize_margin_szse(raw: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    columns = {
        "融资余额": "financing_balance",
        "融资买入额": "financing_buy_amount",
        "融券余量": "securities_lending_remaining_volume",
        "融券余额": "securities_lending_balance",
        "融券卖出量": "securities_lending_sell_volume",
        "融资融券余额": "margin_balance",
    }
    missing = [column for column in columns if column not in raw.columns]
    if missing:
        raise ValueError(f"SZSE margin data missing columns: {missing}")
    if len(raw) != 1:
        raise ValueError(f"SZSE margin data expected one summary row, got {len(raw)}")
    frame = raw.rename(columns=columns)[list(columns.values())].copy()
    numeric_columns = list(columns.values())
    frame[numeric_columns] = (
        frame[numeric_columns].apply(pd.to_numeric, errors="coerce") * SZSE_UNIT_MULTIPLIER
    ).round(0)
    parsed_date = pd.Timestamp(trade_date).date()
    frame["trade_date"] = parsed_date
    frame["exchange"] = "SZSE"
    frame["currency"] = "CNY"
    frame["data_source"] = SZSE_SOURCE
    frame["source_url"] = SZSE_SOURCE_URL
    frame["release_time"] = _available_time(parsed_date)
    frame["available_time"] = frame["release_time"]
    frame["crawl_time"] = datetime.now()
    return _ordered_columns(frame)


def normalize_margin_szse_history(
    raw: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    columns = {
        "日期": "trade_date",
        "融资余额": "financing_balance",
        "融资买入额": "financing_buy_amount",
        "融券余量": "securities_lending_remaining_volume",
        "融券余额": "securities_lending_balance",
        "融券卖出量": "securities_lending_sell_volume",
        "融资融券余额": "margin_balance",
    }
    missing = [column for column in columns if column not in raw.columns]
    if missing:
        raise ValueError(f"SZSE historical margin data missing columns: {missing}")
    frame = raw.rename(columns=columns)[list(columns.values())].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    numeric_columns = [column for column in frame.columns if column != "trade_date"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").round(0)
    frame = frame[
        frame["trade_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].copy()
    frame = frame.dropna(subset=["trade_date"]).drop_duplicates("trade_date")
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["exchange"] = "SZSE"
    frame["currency"] = "CNY"
    frame["data_source"] = SZSE_HISTORY_SOURCE
    frame["source_url"] = SZSE_HISTORY_SOURCE_URL
    frame["release_time"] = None
    frame["available_time"] = frame["trade_date"].map(_fallback_available_time)
    frame["crawl_time"] = datetime.now()
    return _ordered_columns(frame)


def normalize_margin_szse_eastmoney(
    raw: pd.DataFrame, start_date: str, end_date: str
) -> pd.DataFrame:
    columns = {
        "DIM_DATE": "trade_date",
        "RZYE": "financing_balance",
        "RZMRE": "financing_buy_amount",
        "RQYL": "securities_lending_remaining_volume",
        "RQYE": "securities_lending_balance",
        "RQMCL": "securities_lending_sell_volume",
        "RZRQYE": "margin_balance",
    }
    missing = [column for column in columns if column not in raw.columns]
    if missing:
        raise ValueError(f"Eastmoney SZSE margin data missing columns: {missing}")
    frame = raw.rename(columns=columns)[list(columns.values())].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    numeric_columns = [column for column in frame.columns if column != "trade_date"]
    frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce").round(0)
    frame = frame[
        frame["trade_date"].between(pd.Timestamp(start_date), pd.Timestamp(end_date))
    ].copy()
    frame = frame.dropna(subset=["trade_date"]).drop_duplicates("trade_date")
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["exchange"] = "SZSE"
    frame["currency"] = "CNY"
    frame["data_source"] = SZSE_EASTMONEY_SOURCE
    frame["source_url"] = SZSE_EASTMONEY_SOURCE_URL
    frame["release_time"] = None
    frame["available_time"] = frame["trade_date"].map(_fallback_available_time)
    frame["crawl_time"] = datetime.now()
    return _ordered_columns(frame)


def _ordered_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[
        [
            "trade_date",
            "exchange",
            "financing_balance",
            "financing_buy_amount",
            "securities_lending_balance",
            "securities_lending_sell_volume",
            "securities_lending_remaining_volume",
            "margin_balance",
            "currency",
            "data_source",
            "source_url",
            "release_time",
            "available_time",
            "crawl_time",
        ]
    ].reset_index(drop=True)


def fetch_margin_sse(start_date: str, end_date: str) -> pd.DataFrame:
    disable_env_proxies()
    import akshare as ak

    raw = ak.stock_margin_sse(
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
    )
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    return normalize_margin_sse(raw)


def fetch_margin_szse(trade_date: str) -> pd.DataFrame:
    disable_env_proxies()
    import akshare as ak

    raw = ak.stock_margin_szse(date=trade_date.replace("-", ""))
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    return normalize_margin_szse(raw, trade_date)


def fetch_margin_szse_history(start_date: str, end_date: str) -> pd.DataFrame:
    disable_env_proxies()
    import akshare as ak

    raw = ak.macro_china_market_margin_sz()
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    return normalize_margin_szse_history(raw, start_date, end_date)


def fetch_margin_szse_eastmoney_history(
    start_date: str = "2015-01-05", end_date: str = "2026-06-10"
) -> pd.DataFrame:
    disable_env_proxies()
    url = "https://datacenter-web.eastmoney.com/api/data/v1/get"
    session = requests.Session()
    session.trust_env = False
    rows: list[dict[str, Any]] = []
    page = 1
    pages = 1
    while page <= pages:
        response = session.get(
            url,
            params={
                "reportName": "RPTA_WEB_RZRQ_LSSH",
                "columns": "ALL",
                "source": "WEB",
                "sortColumns": "DIM_DATE",
                "sortTypes": "-1",
                "pageNumber": page,
                "pageSize": 500,
                "filter": "(SCDM=001)",
            },
            headers={
                "User-Agent": "Mozilla/5.0",
                "Referer": SZSE_EASTMONEY_SOURCE_URL,
            },
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"Eastmoney margin API error: {payload.get('message')}")
        result = payload.get("result") or {}
        pages = int(result.get("pages") or 0)
        rows.extend(result.get("data") or [])
        page += 1
    if not rows:
        return pd.DataFrame()
    return normalize_margin_szse_eastmoney(pd.DataFrame(rows), start_date, end_date)


def _reference_trade_dates(start_date: str, end_date: str) -> list[str]:
    frame = read_sql(
        """
        SELECT DISTINCT trade_date
        FROM market_index_daily
        WHERE index_code='000852'
          AND trade_date BETWEEN :start_date AND :end_date
        ORDER BY trade_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        raise RuntimeError("market_index_daily 中缺少 000852 基准交易日")
    return pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d").tolist()


def _existing_dates(exchange: str, start_date: str, end_date: str) -> set[str]:
    frame = read_sql(
        """
        SELECT trade_date
        FROM margin_market_daily
        WHERE exchange=:exchange
          AND trade_date BETWEEN :start_date AND :end_date
        """,
        {"exchange": exchange, "start_date": start_date, "end_date": end_date},
    )
    return set(pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")) if not frame.empty else set()


def _retry_fetch(function: Any, retries: int, *args: Any) -> pd.DataFrame:
    errors: list[str] = []
    for attempt in range(retries):
        try:
            return function(*args)
        except Exception as exc:
            errors.append(f"attempt={attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt + 1 < retries:
                time_module.sleep(1 + attempt)
    raise RuntimeError("; ".join(errors))


def _year_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    ranges: list[tuple[str, str]] = []
    year = start.year
    while year <= end.year:
        range_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        range_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        ranges.append((range_start.strftime("%Y-%m-%d"), range_end.strftime("%Y-%m-%d")))
        year += 1
    return ranges


def collect_margin_market(
    start_date: str,
    end_date: str,
    exchanges: list[str] | None = None,
    resume: bool = False,
    sleep_seconds: float = 0.1,
    retries: int = 3,
    fill_szse_official_gaps: bool = True,
) -> dict[str, int]:
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("start_date must not be later than end_date")
    if retries <= 0:
        raise ValueError("retries must be positive")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be non-negative")
    selected = exchanges or ["SSE", "SZSE"]
    unknown = sorted(set(selected) - {"SSE", "SZSE"})
    if unknown:
        raise ValueError(f"unknown exchanges: {unknown}")

    run = IngestionRun(
        dataset_name="margin_market_daily",
        data_source="akshare:stock_margin_sse|akshare:stock_margin_szse",
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    summary = {
        "exchanges": len(selected),
        "success": 0,
        "empty": 0,
        "failed": 0,
        "rows": 0,
        "inserted_rows": 0,
        "updated_rows": 0,
    }
    collected: list[pd.DataFrame] = []
    try:
        if "SSE" in selected:
            existing = _existing_dates("SSE", start_date, end_date)
            if resume and existing:
                missing_dates = [
                    item for item in _reference_trade_dates(start_date, end_date)
                    if item not in existing
                ]
                ranges = [(item, item) for item in missing_dates]
            else:
                ranges = _year_ranges(start_date, end_date)
            for range_start, range_end in ranges:
                try:
                    frame = _retry_fetch(fetch_margin_sse, retries, range_start, range_end)
                    if frame.empty:
                        summary["empty"] += 1
                        continue
                    before = len(existing & set(frame["trade_date"].astype(str)))
                    upsert_dataframe(frame, "margin_market_daily", ["trade_date", "exchange"])
                    collected.append(frame)
                    summary["success"] += 1
                    summary["rows"] += len(frame)
                    summary["updated_rows"] += before
                    summary["inserted_rows"] += len(frame) - before
                except Exception:
                    summary["failed"] += 1
                    logger.exception("failed margin exchange=SSE range=%s:%s", range_start, range_end)

        if "SZSE" in selected:
            reference_dates = _reference_trade_dates(start_date, end_date)
            existing = _existing_dates("SZSE", start_date, end_date)
            try:
                history = _retry_fetch(fetch_margin_szse_history, retries, start_date, end_date)
                if not history.empty:
                    history = history[~history["trade_date"].astype(str).isin(existing)].copy()
                    if not history.empty:
                        upsert_dataframe(
                            history, "margin_market_daily", ["trade_date", "exchange"]
                        )
                        collected.append(history)
                        summary["success"] += 1
                        summary["rows"] += len(history)
                        summary["inserted_rows"] += len(history)
                        existing.update(history["trade_date"].astype(str))
            except Exception:
                summary["failed"] += 1
                logger.exception("failed SZSE historical margin fallback")
            try:
                eastmoney = _retry_fetch(
                    fetch_margin_szse_eastmoney_history, retries, start_date, end_date
                )
                if not eastmoney.empty:
                    official_dates = _existing_dates("SZSE", start_date, end_date)
                    official_frame = read_sql(
                        """
                        SELECT trade_date
                        FROM margin_market_daily
                        WHERE exchange='SZSE'
                          AND data_source=:data_source
                          AND trade_date BETWEEN :start_date AND :end_date
                        """,
                        {
                            "data_source": SZSE_SOURCE,
                            "start_date": start_date,
                            "end_date": end_date,
                        },
                    )
                    protected_dates = set(
                        pd.to_datetime(official_frame["trade_date"]).dt.strftime("%Y-%m-%d")
                    ) if not official_frame.empty else set()
                    eastmoney = eastmoney[
                        ~eastmoney["trade_date"].astype(str).isin(protected_dates)
                    ].copy()
                    before = len(
                        official_dates & set(eastmoney["trade_date"].astype(str))
                    )
                    if not eastmoney.empty:
                        upsert_dataframe(
                            eastmoney, "margin_market_daily", ["trade_date", "exchange"]
                        )
                        collected.append(eastmoney)
                        summary["success"] += 1
                        summary["rows"] += len(eastmoney)
                        summary["updated_rows"] += before
                        summary["inserted_rows"] += len(eastmoney) - before
                        existing.update(eastmoney["trade_date"].astype(str))
            except Exception:
                summary["failed"] += 1
                logger.exception("failed Eastmoney SZSE margin fallback")
            requested_dates = (
                [item for item in reference_dates if item not in existing]
                if fill_szse_official_gaps
                else []
            )
            for trade_date in requested_dates:
                try:
                    frame = _retry_fetch(fetch_margin_szse, retries, trade_date)
                    if frame.empty:
                        summary["empty"] += 1
                        continue
                    upsert_dataframe(frame, "margin_market_daily", ["trade_date", "exchange"])
                    collected.append(frame)
                    summary["success"] += 1
                    summary["rows"] += 1
                    if trade_date in existing:
                        summary["updated_rows"] += 1
                    else:
                        summary["inserted_rows"] += 1
                except Exception:
                    summary["failed"] += 1
                    logger.exception("failed margin exchange=SZSE date=%s", trade_date)
                if sleep_seconds:
                    time_module.sleep(sleep_seconds)

        run.fetched_rows = summary["rows"]
        run.inserted_rows = summary["inserted_rows"]
        run.updated_rows = summary["updated_rows"]
        run.error_rows = summary["failed"]
        run.finish("success" if summary["failed"] == 0 else "partial")
        if collected:
            batch = pd.concat(collected, ignore_index=True)
            metrics = evaluate_dataframe(batch, DATASETS["margin_market_daily"])
            metrics.extend(evaluate_margin_relationships(batch))
            persist_metrics("margin_market_daily:batch", metrics, date.today())
        output = project_path("data", "reports", "margin_market_v2_last_run.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    except Exception as exc:
        run.error_rows += 1
        run.finish("failed", str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集模型2.0沪深融资融券市场汇总。")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--exchange", action="append", choices=["SSE", "SZSE"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-szse-official-gaps", action="store_true")
    args = parser.parse_args()
    result = collect_margin_market(
        args.start_date,
        args.end_date,
        args.exchange,
        args.resume,
        args.sleep,
        args.retries,
        not args.skip_szse_official_gaps,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
