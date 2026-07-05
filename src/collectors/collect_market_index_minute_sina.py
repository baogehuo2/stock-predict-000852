from __future__ import annotations

import argparse
import hashlib
import json
import time as time_module
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.migrations import apply_migrations
from src.quality.check_collection_data import DatasetSpec, evaluate_dataframe, persist_metrics


TABLE_NAME = "market_index_minute_raw"
DATA_SOURCE = "sina:CN_MarketData.getKLineData"
SINA_ENDPOINT = "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketData.getKLineData"
DEFAULT_DATALEN = 1023
DEFAULT_REQUEST_SLEEP_SECONDS = 0.2

DEFAULT_INDICES: dict[str, str] = {
    "000852.SH": "中证1000",
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000016.SH": "上证50",
    "000688.SH": "科创50",
}

FREQ_TO_SCALE = {
    "1min": "1",
    "5min": "5",
    "15min": "15",
    "30min": "30",
    "60min": "60",
}


def _index_code(ts_code: str) -> str:
    return ts_code.split(".")[0]


def _sina_symbol(ts_code: str) -> str:
    code, exchange = ts_code.split(".", 1)
    exchange = exchange.upper()
    if exchange == "SH":
        return f"sh{code}"
    if exchange == "SZ":
        return f"sz{code}"
    raise ValueError(f"unsupported ts_code exchange: {ts_code}")


def _freq_minutes(freq: str) -> int:
    return int(FREQ_TO_SCALE[freq])


def _raw_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_end_time(value: str | None) -> datetime:
    if not value or value.lower() == "now":
        return datetime.now()
    return pd.Timestamp(value).to_pydatetime()


def _parse_symbols(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_INDICES)
    parsed: dict[str, str] = {}
    for value in values:
        if "=" in value:
            ts_code, name = value.split("=", 1)
            parsed[ts_code.strip()] = name.strip()
        else:
            ts_code = value.strip()
            parsed[ts_code] = DEFAULT_INDICES.get(ts_code, ts_code)
    return parsed


def fetch_sina_index_minutes(
    ts_code: str,
    freq: str,
    datalen: int = DEFAULT_DATALEN,
    timeout: int = 20,
) -> pd.DataFrame:
    params = {
        "symbol": _sina_symbol(ts_code),
        "scale": FREQ_TO_SCALE[freq],
        "ma": "no",
        "datalen": str(datalen),
    }
    response = requests.get(
        SINA_ENDPOINT,
        params=params,
        headers={"User-Agent": "Mozilla/5.0 data-collectors/2.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if not data:
        return pd.DataFrame()
    if not isinstance(data, list):
        raise ValueError(f"unexpected Sina response type: {type(data).__name__}")
    return pd.DataFrame(data)


def normalize_sina_minutes(
    raw: pd.DataFrame,
    ts_code: str,
    index_name: str,
    freq: str,
    start_time: datetime | None,
    end_time: datetime,
    datalen: int = DEFAULT_DATALEN,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.rename(
        columns={
            "day": "trade_time",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
            "amount": "amount",
        }
    ).copy()
    required = ["trade_time", "open", "high", "low", "close", "volume", "amount"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Sina minute data missing columns {missing}; got={list(raw.columns)}")

    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame = frame.dropna(subset=["trade_time"])
    frame = frame[frame["trade_time"] <= pd.Timestamp(end_time)]
    if start_time is not None:
        frame = frame[frame["trade_time"] >= pd.Timestamp(start_time)]
    if frame.empty:
        return pd.DataFrame()

    for column in ["open", "high", "low", "close", "volume", "amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    if frame.empty:
        return pd.DataFrame()

    ohlc = frame[["open", "high", "low", "close"]]
    frame["high"] = ohlc.max(axis=1)
    frame["low"] = ohlc.min(axis=1)
    frame["trade_date"] = frame["trade_time"].dt.date
    frame["index_code"] = _index_code(ts_code)
    frame["ts_code"] = ts_code
    frame["index_name"] = index_name
    frame["freq"] = freq
    frame["currency"] = "CNY"
    frame["volume_unit"] = "share"
    frame["amount_unit"] = "CNY_yuan"
    frame["data_source"] = DATA_SOURCE
    frame["source_url"] = (
        f"{SINA_ENDPOINT}?symbol={_sina_symbol(ts_code)}"
        f"&scale={FREQ_TO_SCALE[freq]}&ma=no&datalen={datalen}"
    )
    frame["available_time"] = frame["trade_time"].map(
        lambda value: value.to_pydatetime() + timedelta(minutes=_freq_minutes(freq))
    )
    frame["crawl_time"] = datetime.now()
    columns = [
        "trade_time",
        "trade_date",
        "index_code",
        "ts_code",
        "index_name",
        "freq",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "currency",
        "volume_unit",
        "amount_unit",
        "data_source",
        "source_url",
        "available_time",
        "crawl_time",
    ]
    out = frame[columns].copy()
    out["raw_hash"] = out.apply(lambda row: _raw_hash(row.to_dict()), axis=1)
    return (
        out.sort_values(["ts_code", "trade_time"])
        .drop_duplicates(["index_code", "trade_time", "freq", "data_source"], keep="last")
        .reset_index(drop=True)
    )


def _existing_count(ts_code: str, freq: str, start_time: datetime | None, end_time: datetime) -> int:
    conditions = [
        "ts_code = :ts_code",
        "freq = :freq",
        "data_source = :data_source",
        "trade_time <= :end_time",
    ]
    params: dict[str, Any] = {
        "ts_code": ts_code,
        "freq": freq,
        "data_source": DATA_SOURCE,
        "end_time": end_time,
    }
    if start_time is not None:
        conditions.append("trade_time >= :start_time")
        params["start_time"] = start_time
    result = read_sql(
        f"""
        SELECT COUNT(*) AS row_count
        FROM {TABLE_NAME}
        WHERE {' AND '.join(conditions)}
        """,
        params,
    )
    return int(result.iloc[0]["row_count"])


def _quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    spec = DatasetSpec(
        table=TABLE_NAME,
        date_column="trade_date",
        unique_columns=("index_code", "trade_time", "freq", "data_source"),
        required_columns=(
            "trade_time",
            "trade_date",
            "index_code",
            "ts_code",
            "freq",
            "open",
            "high",
            "low",
            "close",
            "data_source",
            "available_time",
        ),
        non_negative_columns=("volume", "amount"),
        ohlc=True,
    )
    metrics = evaluate_dataframe(frame, spec)
    persist_metrics(TABLE_NAME, metrics, date.today())
    return {"metrics": metrics}


def collect_sina_index_minutes(
    *,
    symbols: dict[str, str],
    freq: str,
    start_time: datetime | None,
    end_time: datetime,
    datalen: int = DEFAULT_DATALEN,
    request_sleep_seconds: float = DEFAULT_REQUEST_SLEEP_SECONDS,
    max_jobs: int | None = None,
    skip_existing: bool = False,
    skip_existing_min_rows: int = 1,
    dry_run: bool = False,
    apply_db_migrations: bool = False,
) -> dict[str, Any]:
    if apply_db_migrations and not dry_run:
        apply_migrations()
    jobs = list(symbols.items())
    if max_jobs is not None:
        jobs = jobs[:max_jobs]

    summary: dict[str, Any] = {
        "table": TABLE_NAME,
        "data_source": DATA_SOURCE,
        "freq": freq,
        "symbol_count": len(symbols),
        "jobs": len(jobs),
        "datalen": datalen,
        "start_time": str(start_time) if start_time else None,
        "end_time": str(end_time),
        "fetched_rows": 0,
        "written_rows": 0,
        "inserted_rows": 0,
        "updated_rows": 0,
        "skipped_jobs": 0,
        "failed_jobs": [],
        "trade_time_min": None,
        "trade_time_max": None,
        "quality": None,
    }
    if dry_run:
        summary["planned_jobs"] = [{"ts_code": ts_code, "index_name": name} for ts_code, name in jobs]
        return summary

    run = IngestionRun(
        dataset_name=TABLE_NAME,
        data_source=DATA_SOURCE,
        requested_start_date=start_time.date() if start_time else None,
        requested_end_date=end_time.date(),
    )
    frames: list[pd.DataFrame] = []
    try:
        for index, (ts_code, index_name) in enumerate(jobs):
            if skip_existing and _existing_count(ts_code, freq, start_time, end_time) >= skip_existing_min_rows:
                summary["skipped_jobs"] += 1
                continue
            try:
                raw = fetch_sina_index_minutes(ts_code, freq, datalen=datalen)
                frame = normalize_sina_minutes(raw, ts_code, index_name, freq, start_time, end_time, datalen)
            except Exception as exc:
                summary["failed_jobs"].append(
                    {"ts_code": ts_code, "freq": freq, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue

            if frame.empty:
                summary["failed_jobs"].append({"ts_code": ts_code, "freq": freq, "error": "empty"})
            else:
                existing = _existing_count(ts_code, freq, start_time, end_time)
                written = upsert_dataframe(
                    frame,
                    TABLE_NAME,
                    ["index_code", "trade_time", "freq", "data_source"],
                )
                summary["fetched_rows"] += len(frame)
                summary["written_rows"] += written
                summary["inserted_rows"] += max(0, len(frame) - existing)
                summary["updated_rows"] += min(len(frame), existing)
                frames.append(frame)

            if index < len(jobs) - 1 and request_sleep_seconds > 0:
                time_module.sleep(request_sleep_seconds)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not combined.empty:
            summary["trade_time_min"] = str(combined["trade_time"].min())
            summary["trade_time_max"] = str(combined["trade_time"].max())
            summary["quality"] = _quality_report(combined)

        run.fetched_rows = int(summary["fetched_rows"])
        run.inserted_rows = int(summary["inserted_rows"])
        run.updated_rows = int(summary["updated_rows"])
        run.skipped_rows = int(summary["skipped_jobs"])
        run.error_rows = len(summary["failed_jobs"])
        run.finish("success" if not summary["failed_jobs"] else "partial")
        return summary
    except Exception as exc:
        run.fetched_rows = int(summary["fetched_rows"])
        run.inserted_rows = int(summary["inserted_rows"])
        run.updated_rows = int(summary["updated_rows"])
        run.skipped_rows = int(summary["skipped_jobs"])
        run.error_rows = len(summary["failed_jobs"]) + 1
        run.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="使用新浪直接接口采集 A 股指数分钟线，适合作为生产增量备份源。")
    parser.add_argument("--symbol", action="append", help="指数 ts_code，可重复；可写 000852.SH=中证1000。")
    parser.add_argument("--freq", default="1min", choices=sorted(FREQ_TO_SCALE))
    parser.add_argument("--all-freq", action="store_true", help="一次运行全部 1/5/15/30/60 分钟频率。")
    parser.add_argument("--start-time", help="过滤开始时间，例如 2026-06-30 00:00:00。为空则不过滤下界。")
    parser.add_argument("--end-time", default="now", help="过滤截止时间；生产 8 点运行用默认 now 即可。")
    parser.add_argument("--lookback-days", type=int, default=None, help="未指定 start-time 时，按 end-time 回看 N 天。")
    parser.add_argument("--datalen", type=int, default=DEFAULT_DATALEN, help="新浪单次返回条数，建议 1023。")
    parser.add_argument("--request-sleep-seconds", type=float, default=DEFAULT_REQUEST_SLEEP_SECONDS)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-existing-min-rows", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply-migrations", action="store_true")
    args = parser.parse_args()

    end_time = _parse_end_time(args.end_time)
    start_time = pd.Timestamp(args.start_time).to_pydatetime() if args.start_time else None
    if start_time is None and args.lookback_days is not None:
        start_time = end_time - timedelta(days=args.lookback_days)

    freqs = ["1min", "5min", "15min", "30min", "60min"] if args.all_freq else [args.freq]
    results = []
    for freq in freqs:
        results.append(
            collect_sina_index_minutes(
                symbols=_parse_symbols(args.symbol),
                freq=freq,
                start_time=start_time,
                end_time=end_time,
                datalen=args.datalen,
                request_sleep_seconds=args.request_sleep_seconds,
                max_jobs=args.max_jobs,
                skip_existing=args.skip_existing,
                skip_existing_min_rows=args.skip_existing_min_rows,
                dry_run=args.dry_run,
                apply_db_migrations=args.apply_migrations,
            )
        )
    print(json.dumps({"results": results}, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
