from __future__ import annotations

import argparse
import hashlib
import json
import time as sleep_time
from datetime import date, datetime, time
from typing import Any, Callable, TypeVar

import pandas as pd
import requests

from src.common.config import get_config, load_yaml, project_path
from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.network import disable_env_proxies
from src.quality.check_collection_data import DatasetSpec, evaluate_dataframe, persist_metrics


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_date}&coed={end_date}"
AKSHARE_GLOBAL_INDEX_URL = "https://akshare.akfamily.xyz/data/index/index.html"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
RETRY_SLEEP_SECONDS = 2.0
REQUEST_TIMEOUT_SECONDS = 30

T = TypeVar("T")


def _raw_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _retry_call(label: str, func: Callable[[], T], attempts: int = 3, sleep_seconds: float = RETRY_SLEEP_SECONDS) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt < attempts:
                sleep_time.sleep(sleep_seconds)
    raise RuntimeError(f"{label} failed after {attempts} attempts: {last_error}")


def _available_time(trade_dates: pd.Series, timezone: str) -> pd.Series:
    hour = 6 if timezone == "America/New_York" else 18
    return pd.to_datetime(trade_dates).map(lambda value: datetime.combine(value.date(), time(hour, 0)))


def fetch_fred_series(item: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    disable_env_proxies()
    url = FRED_CSV_URL.format(series_id=item["series_id"], start_date=start_date, end_date=end_date)
    from io import StringIO

    def _request() -> pd.DataFrame:
        with requests.Session() as session:
            response = session.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 data-collectors/2.0", "Connection": "close"},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))

    raw = _retry_call(f"FRED {item['series_id']}", _request)
    if "observation_date" not in raw.columns or item["series_id"] not in raw.columns:
        raise ValueError(f"FRED {item['series_id']} unexpected columns: {list(raw.columns)}")
    frame = raw.rename(columns={"observation_date": "trade_date", item["series_id"]: "close"}).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "close"])
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)]
    frame["symbol"] = item["symbol"]
    frame["symbol_name"] = item["name"]
    frame["asset_class"] = item["asset_class"]
    frame["open"] = None
    frame["high"] = None
    frame["low"] = None
    frame["settle"] = frame["close"]
    frame["volume"] = None
    frame["currency"] = item.get("currency")
    frame["timezone"] = "America/New_York"
    frame["data_source"] = f"fred:{item['series_id']}"
    frame["source_url"] = url
    frame["available_time"] = _available_time(frame["trade_date"], "America/New_York")
    frame["crawl_time"] = datetime.now()
    frame["raw_hash"] = frame.apply(lambda row: _raw_hash(row.to_dict()), axis=1)
    columns = [
        "trade_date", "symbol", "symbol_name", "asset_class", "open", "high", "low", "close",
        "settle", "volume", "currency", "timezone", "data_source", "source_url",
        "available_time", "crawl_time", "raw_hash",
    ]
    return frame[columns]


def normalize_akshare_global_index(raw: pd.DataFrame, item: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    mapping = {
        "日期": "trade_date",
        "date": "trade_date",
        "开盘": "open",
        "open": "open",
        "最高": "high",
        "high": "high",
        "最低": "low",
        "low": "low",
        "收盘": "close",
        "close": "close",
        "成交量": "volume",
        "volume": "volume",
        "最新值": "close",
    }
    frame = raw.rename(columns=mapping).copy()
    required = ["trade_date", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{item['symbol']} missing columns {missing}; got={list(raw.columns)}")
    for column in ["open", "high", "low", "volume"]:
        if column not in frame.columns:
            frame[column] = None
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "close"])
    ohlc_cols = ["open", "high", "low", "close"]
    ohlc = frame[ohlc_cols].apply(pd.to_numeric, errors="coerce")
    frame["high"] = ohlc.max(axis=1)
    frame["low"] = ohlc.min(axis=1)
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)]
    frame["symbol"] = item["symbol"]
    frame["symbol_name"] = item["name"]
    frame["asset_class"] = item["asset_class"]
    frame["settle"] = frame["close"]
    if "volume" not in frame.columns:
        frame["volume"] = None
    frame["currency"] = item.get("currency")
    frame["timezone"] = "America/New_York"
    frame["data_source"] = item.get("data_source", f"akshare:index_us_stock_sina:{item.get('ak_symbol')}")
    frame["source_url"] = AKSHARE_GLOBAL_INDEX_URL
    frame["available_time"] = _available_time(frame["trade_date"], "America/New_York")
    frame["crawl_time"] = datetime.now()
    frame["raw_hash"] = frame.apply(lambda row: _raw_hash(row.to_dict()), axis=1)
    columns = [
        "trade_date", "symbol", "symbol_name", "asset_class", "open", "high", "low", "close",
        "settle", "volume", "currency", "timezone", "data_source", "source_url",
        "available_time", "crawl_time", "raw_hash",
    ]
    return frame[columns]


def normalize_akshare_global_futures(raw: pd.DataFrame, item: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    mapping = {
        "日期": "trade_date",
        "date": "trade_date",
        "开盘": "open",
        "open": "open",
        "最高": "high",
        "high": "high",
        "最低": "low",
        "low": "low",
        "最新价": "close",
        "close": "close",
        "总量": "volume",
        "volume": "volume",
        "settlement": "settle",
    }
    frame = raw.rename(columns=mapping).copy()
    required = ["trade_date", "open", "high", "low", "close"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{item['symbol']} missing columns {missing}; got={list(raw.columns)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    for column in ["open", "high", "low", "close", "volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "close"])
    ohlc = frame[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    frame["high"] = ohlc.max(axis=1)
    frame["low"] = ohlc.min(axis=1)
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)]
    frame["symbol"] = item["symbol"]
    frame["symbol_name"] = item["name"]
    frame["asset_class"] = item["asset_class"]
    if "settle" not in frame.columns:
        frame["settle"] = frame["close"]
    frame["settle"] = pd.to_numeric(frame["settle"], errors="coerce").combine_first(frame["close"])
    frame["currency"] = item.get("currency")
    frame["timezone"] = "America/New_York"
    frame["data_source"] = f"akshare:{item.get('function', 'futures_foreign_hist')}:{item['ak_symbol']}"
    frame["source_url"] = "https://akshare.akfamily.xyz/data/futures/futures.html"
    frame["available_time"] = _available_time(frame["trade_date"], "America/New_York")
    frame["crawl_time"] = datetime.now()
    frame["raw_hash"] = frame.apply(lambda row: _raw_hash(row.to_dict()), axis=1)
    columns = [
        "trade_date", "symbol", "symbol_name", "asset_class", "open", "high", "low", "close",
        "settle", "volume", "currency", "timezone", "data_source", "source_url",
        "available_time", "crawl_time", "raw_hash",
    ]
    return frame[columns]


def normalize_akshare_us_bond_yield(raw: pd.DataFrame, item: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    if "日期" not in raw.columns or item["source_column"] not in raw.columns:
        raise ValueError(f"{item['symbol']} missing columns 日期/{item['source_column']}; got={list(raw.columns)}")
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(raw["日期"], errors="coerce").dt.date,
            "close": pd.to_numeric(raw[item["source_column"]], errors="coerce"),
        }
    ).dropna(subset=["trade_date", "close"])
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)]
    frame["symbol"] = item["symbol"]
    frame["symbol_name"] = item["name"]
    frame["asset_class"] = item["asset_class"]
    frame["open"] = None
    frame["high"] = None
    frame["low"] = None
    frame["settle"] = frame["close"]
    frame["volume"] = None
    frame["currency"] = item.get("currency")
    frame["timezone"] = "America/New_York"
    frame["data_source"] = "akshare:bond_zh_us_rate"
    frame["source_url"] = "https://akshare.akfamily.xyz/data/bond/bond.html"
    frame["available_time"] = _available_time(frame["trade_date"], "America/New_York")
    frame["crawl_time"] = datetime.now()
    frame["raw_hash"] = frame.apply(lambda row: _raw_hash(row.to_dict()), axis=1)
    columns = [
        "trade_date", "symbol", "symbol_name", "asset_class", "open", "high", "low", "close",
        "settle", "volume", "currency", "timezone", "data_source", "source_url",
        "available_time", "crawl_time", "raw_hash",
    ]
    return frame[columns]


def fetch_yahoo_chart(item: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    start_ts = int(pd.Timestamp(start_date, tz="UTC").timestamp())
    end_ts = int((pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(days=1)).timestamp())
    url = YAHOO_CHART_URL.format(symbol=item["yahoo_symbol"])
    response = requests.get(
        url,
        params={"period1": start_ts, "period2": end_ts, "interval": "1d"},
        headers={"User-Agent": "Mozilla/5.0 data-collectors/2.0", "Connection": "close"},
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result") or []
    if not result:
        raise RuntimeError(f"Yahoo chart returned no data for {item['symbol']}")
    block = result[0]
    timestamps = block.get("timestamp") or []
    quote = (block.get("indicators", {}).get("quote") or [{}])[0]
    frame = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(timestamps, unit="s", utc=True).date,
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    item = {**item, "data_source": f"yahoo_chart:{item['yahoo_symbol']}"}
    return normalize_akshare_global_index(frame, item, start_date, end_date)


def _load_items() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    cfg = load_yaml(project_path("config", "data_sources_v2.yaml"))["global_market"]
    return (
        cfg.get("fred_reference", []),
        cfg.get("akshare_bond_yield", []),
        cfg.get("akshare_us_index", []),
        cfg.get("akshare_sox_index", []),
        cfg.get("akshare_global_futures", []),
        cfg.get("akshare_us_proxy", []),
        cfg.get("yahoo_reference", []),
    )


def _existing_count(start_date: str, end_date: str) -> int:
    result = read_sql(
        """
        SELECT COUNT(*) AS row_count
        FROM global_market_daily
        WHERE trade_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    return int(result.iloc[0]["row_count"])


def _quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    spec = DatasetSpec(
        table="global_market_daily",
        date_column="trade_date",
        unique_columns=("trade_date", "symbol"),
        required_columns=("trade_date", "symbol", "asset_class", "close", "data_source", "available_time"),
        non_negative_columns=("volume",),
        ohlc=True,
    )
    metrics = evaluate_dataframe(frame, spec)
    persist_metrics("global_market_daily", metrics, date.today())
    return {"metrics": metrics}


def collect_global_market(
    start_date: str | None = None,
    end_date: str | None = None,
    include_akshare: bool = True,
    include_fred: bool = False,
    include_yahoo: bool = False,
) -> dict[str, Any]:
    config = get_config()
    start_date = start_date or config["project"]["start_date"]
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    ingestion = IngestionRun(
        dataset_name="global_market_daily",
        data_source="akshare" if not include_fred else "akshare+fred_reference",
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    try:
        fred_items, bond_items, ak_items, sox_items, futures_items, proxy_items, yahoo_items = _load_items()
        failed: list[str] = []
        frames = []
        if include_fred:
            for item in fred_items:
                try:
                    frames.append(fetch_fred_series(item, start_date, end_date))
                    sleep_time.sleep(RETRY_SLEEP_SECONDS)
                except Exception as exc:
                    failed.append(f"{item['symbol']}:{type(exc).__name__}:{exc}")
        if include_akshare:
            disable_env_proxies()
            import akshare as ak

            if bond_items:
                try:
                    bond_raw = _retry_call(
                        "akshare:bond_zh_us_rate",
                        lambda: ak.bond_zh_us_rate(start_date=start_date.replace("-", "")),
                    )
                    for item in bond_items:
                        frames.append(normalize_akshare_us_bond_yield(bond_raw, item, start_date, end_date))
                    sleep_time.sleep(RETRY_SLEEP_SECONDS)
                except Exception as exc:
                    for item in bond_items:
                        failed.append(f"{item['symbol']}:{type(exc).__name__}:{exc}")
            for item in ak_items:
                try:
                    raw = _retry_call(
                        f"akshare:index_us_stock_sina:{item['ak_symbol']}",
                        lambda item=item: ak.index_us_stock_sina(symbol=item["ak_symbol"]),
                    )
                    item = {**item, "data_source": f"akshare:index_us_stock_sina:{item['ak_symbol']}"}
                    frames.append(normalize_akshare_global_index(raw, item, start_date, end_date))
                    sleep_time.sleep(RETRY_SLEEP_SECONDS)
                except Exception as exc:
                    failed.append(f"{item['symbol']}:{type(exc).__name__}:{exc}")
            for item in sox_items:
                try:
                    raw = _retry_call("akshare:macro_global_sox_index", ak.macro_global_sox_index)
                    item = {**item, "data_source": "akshare:macro_global_sox_index"}
                    frames.append(normalize_akshare_global_index(raw, item, start_date, end_date))
                    sleep_time.sleep(RETRY_SLEEP_SECONDS)
                except Exception as exc:
                    failed.append(f"{item['symbol']}:{type(exc).__name__}:{exc}")
            for item in futures_items:
                try:
                    function_name = item.get("function", "futures_foreign_hist")
                    fetcher = getattr(ak, function_name)
                    raw = _retry_call(
                        f"akshare:{function_name}:{item['ak_symbol']}",
                        lambda item=item, fetcher=fetcher: fetcher(symbol=item["ak_symbol"]),
                    )
                    frames.append(normalize_akshare_global_futures(raw, item, start_date, end_date))
                    sleep_time.sleep(RETRY_SLEEP_SECONDS)
                except Exception as exc:
                    failed.append(f"{item['symbol']}:{type(exc).__name__}:{exc}")
            for item in proxy_items:
                try:
                    raw = _retry_call(
                        f"akshare:stock_us_daily:{item['ak_symbol']}",
                        lambda item=item: ak.stock_us_daily(symbol=item["ak_symbol"]),
                    )
                    item = {
                        **item,
                        "data_source": f"akshare:stock_us_daily:{item['ak_symbol']};proxy_for:{item.get('proxy_for')}",
                    }
                    frames.append(normalize_akshare_global_index(raw, item, start_date, end_date))
                    sleep_time.sleep(RETRY_SLEEP_SECONDS)
                except Exception as exc:
                    failed.append(f"{item['symbol']}:{type(exc).__name__}:{exc}")
            if include_yahoo:
                for item in yahoo_items:
                    try:
                        frames.append(_retry_call(f"yahoo_chart:{item['symbol']}", lambda item=item: fetch_yahoo_chart(item, start_date, end_date)))
                        sleep_time.sleep(RETRY_SLEEP_SECONDS)
                    except Exception as exc:
                        failed.append(f"{item['symbol']}:{type(exc).__name__}:{exc}")
        combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
        if combined.empty:
            raise RuntimeError(f"global_market_daily collected no rows; failed={failed}")
        existing = _existing_count(start_date, end_date)
        upsert_dataframe(combined, "global_market_daily", ["trade_date", "symbol"])
        ingestion.fetched_rows = len(combined)
        ingestion.inserted_rows = max(0, len(combined) - existing)
        ingestion.updated_rows = min(len(combined), existing)
        ingestion.finish("success")
        return {
            "rows": len(combined),
            "symbols": sorted(combined["symbol"].unique().tolist()),
            "trade_date_min": str(combined["trade_date"].min()),
            "trade_date_max": str(combined["trade_date"].max()),
            "failed": failed,
            "quality": _quality_report(combined),
        }
    except Exception as exc:
        ingestion.error_rows = 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集第二批外部风险映射行情。")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--no-akshare", action="store_true")
    parser.add_argument("--include-fred", action="store_true", help="额外采集FRED校验源，默认不跑。")
    parser.add_argument("--include-yahoo", action="store_true", help="额外采集Yahoo校验源，默认不跑。")
    args = parser.parse_args()
    print(
        collect_global_market(
            args.start_date,
            args.end_date,
            include_akshare=not args.no_akshare,
            include_fred=args.include_fred,
            include_yahoo=args.include_yahoo,
        )
    )


if __name__ == "__main__":
    main()
