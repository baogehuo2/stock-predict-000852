from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.migrations import apply_migrations
from src.common.secrets import get_secret
from src.quality.check_collection_data import DatasetSpec, evaluate_dataframe, persist_metrics


TABLE_NAME = "market_index_minute_raw"
DATA_SOURCE = "tushare:stk_mins"
SOURCE_URL = "https://tushare.pro/document/2?doc_id=292"
DEFAULT_INDEX_CODE = "000852"
DEFAULT_TS_CODE = "000852.SH"
DEFAULT_INDEX_NAME = "中证1000"
DEFAULT_FREQ = "1min"
DEFAULT_REQUEST_INTERVAL_SECONDS = 3700.0
MAX_EXPECTED_ROWS_PER_REQUEST = 8000
RATE_LIMIT_KEYWORDS = ("频率超限", "每分钟最多访问", "每小时最多访问", "每天最多访问")


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    @property
    def start_text(self) -> str:
        return self.start.strftime("%Y-%m-%d %H:%M:%S")

    @property
    def end_text(self) -> str:
        return self.end.strftime("%Y-%m-%d %H:%M:%S")


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def _compact_date(value: date) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _raw_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return any(keyword in message for keyword in RATE_LIMIT_KEYWORDS)


def _month_windows(start_date: str, end_date: str, window_months: int = 1) -> list[Window]:
    if window_months < 1:
        raise ValueError("window_months must be >= 1")
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if start > end:
        raise ValueError(f"start_date must be <= end_date: {start_date} > {end_date}")

    windows: list[Window] = []
    current = pd.Timestamp(start).replace(day=1)
    final = pd.Timestamp(end)
    while current <= final:
        month_start = max(current.date(), start)
        month_end = min((current + pd.offsets.MonthEnd(window_months)).date(), end)
        windows.append(
            Window(
                start=datetime.combine(month_start, datetime.strptime("09:30:00", "%H:%M:%S").time()),
                end=datetime.combine(month_end, datetime.strptime("15:00:00", "%H:%M:%S").time()),
            )
        )
        current = current + pd.offsets.MonthBegin(window_months)
    return windows


def _load_tushare_client():
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("missing dependency: tushare. Please run `pip install tushare`.") from exc

    token = (
        get_secret("tushare.token")
        or get_secret("TUSHARE_TOKEN")
        or get_secret("tushare_token")
        or get_secret("tushare.pro_token")
    )
    if not token:
        raise RuntimeError("missing tushare token in encrypted secrets")
    ts.set_token(token)
    return ts.pro_api(token)


def fetch_tushare_index_minutes(
    pro: Any,
    ts_code: str,
    freq: str,
    window: Window,
) -> pd.DataFrame:
    return pro.stk_mins(
        ts_code=ts_code,
        freq=freq,
        start_date=window.start_text,
        end_date=window.end_text,
    )


def normalize_minutes(
    raw: pd.DataFrame,
    index_code: str,
    ts_code: str,
    index_name: str,
    freq: str,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    frame = raw.copy()
    required = ["ts_code", "trade_time", "open", "high", "low", "close", "vol", "amount"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"stk_mins missing columns {missing}; got={list(frame.columns)}")

    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame = frame.dropna(subset=["trade_time"]).copy()
    frame["trade_date"] = frame["trade_time"].dt.date
    for column in ["open", "high", "low", "close", "vol", "amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    if frame.empty:
        return frame

    ohlc = frame[["open", "high", "low", "close"]]
    frame["high"] = ohlc.max(axis=1)
    frame["low"] = ohlc.min(axis=1)
    frame["index_code"] = index_code
    frame["ts_code"] = ts_code
    frame["index_name"] = index_name
    frame["freq"] = freq
    frame["volume"] = frame["vol"]
    frame["currency"] = "CNY"
    frame["volume_unit"] = "share"
    frame["amount_unit"] = "CNY_yuan"
    frame["data_source"] = DATA_SOURCE
    frame["source_url"] = SOURCE_URL
    frame["available_time"] = frame["trade_time"].map(lambda value: value.to_pydatetime() + timedelta(minutes=1))
    frame["crawl_time"] = datetime.now()
    frame["raw_hash"] = frame.apply(lambda row: _raw_hash(row.to_dict()), axis=1)

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
        "raw_hash",
    ]
    return (
        frame[columns]
        .sort_values("trade_time")
        .drop_duplicates(["index_code", "trade_time", "freq", "data_source"], keep="last")
    )


def _existing_count(index_code: str, freq: str, window: Window) -> int:
    result = read_sql(
        f"""
        SELECT COUNT(*) AS row_count
        FROM {TABLE_NAME}
        WHERE index_code = :index_code
          AND freq = :freq
          AND data_source = :data_source
          AND trade_time BETWEEN :start_time AND :end_time
        """,
        {
            "index_code": index_code,
            "freq": freq,
            "data_source": DATA_SOURCE,
            "start_time": window.start,
            "end_time": window.end,
        },
    )
    return int(result.iloc[0]["row_count"])


def _window_is_complete(index_code: str, freq: str, window: Window, min_rows: int) -> bool:
    return _existing_count(index_code, freq, window) >= min_rows


def _quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    spec = DatasetSpec(
        table=TABLE_NAME,
        date_column="trade_time",
        unique_columns=("index_code", "trade_time", "freq", "data_source"),
        required_columns=("trade_time", "trade_date", "index_code", "ts_code", "freq", "open", "high", "low", "close", "data_source", "available_time"),
        non_negative_columns=("volume", "amount"),
        ohlc=True,
    )
    metrics = evaluate_dataframe(frame, spec)
    persist_metrics(TABLE_NAME, metrics, date.today())
    return {"metrics": metrics}


def collect_market_index_minutes(
    start_date: str,
    end_date: str,
    index_code: str = DEFAULT_INDEX_CODE,
    ts_code: str = DEFAULT_TS_CODE,
    index_name: str = DEFAULT_INDEX_NAME,
    freq: str = DEFAULT_FREQ,
    request_interval_seconds: float = DEFAULT_REQUEST_INTERVAL_SECONDS,
    window_months: int = 1,
    max_windows: int | None = None,
    skip_existing: bool = False,
    skip_existing_min_rows: int = 1,
    dry_run: bool = False,
    apply_db_migrations: bool = False,
) -> dict[str, Any]:
    if apply_db_migrations and not dry_run:
        apply_migrations()

    windows = _month_windows(start_date, end_date, window_months=window_months)
    if max_windows is not None:
        windows = windows[:max_windows]

    summary: dict[str, Any] = {
        "table": TABLE_NAME,
        "index_code": index_code,
        "ts_code": ts_code,
        "freq": freq,
        "windows": len(windows),
        "fetched_rows": 0,
        "written_rows": 0,
        "inserted_rows": 0,
        "updated_rows": 0,
        "skipped_windows": 0,
        "failed_windows": [],
        "rate_limited": False,
        "rate_limit_message": None,
        "next_start_date": None,
        "trade_time_min": None,
        "trade_time_max": None,
        "quality": None,
    }

    if dry_run:
        summary["planned_windows"] = [{"start": window.start_text, "end": window.end_text} for window in windows]
        return summary

    run = IngestionRun(
        dataset_name=TABLE_NAME,
        data_source=DATA_SOURCE,
        requested_start_date=_parse_date(start_date),
        requested_end_date=_parse_date(end_date),
    )

    pro = _load_tushare_client()
    frames: list[pd.DataFrame] = []
    try:
        for index, window in enumerate(windows):
            if skip_existing and _window_is_complete(index_code, freq, window, skip_existing_min_rows):
                summary["skipped_windows"] += 1
                continue

            try:
                raw = fetch_tushare_index_minutes(pro, ts_code, freq, window)
            except Exception as exc:
                failed = {
                    "start": window.start_text,
                    "end": window.end_text,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                summary["failed_windows"].append(failed)
                summary["next_start_date"] = window.start.strftime("%Y-%m-%d")
                if _is_rate_limit_error(exc):
                    summary["rate_limited"] = True
                    summary["rate_limit_message"] = str(exc)
                    break
                raise

            frame = normalize_minutes(raw, index_code, ts_code, index_name, freq)
            if len(frame) > MAX_EXPECTED_ROWS_PER_REQUEST:
                raise RuntimeError(
                    f"single request returned {len(frame)} rows, above expected {MAX_EXPECTED_ROWS_PER_REQUEST}; "
                    "please reduce window size"
                )
            if frame.empty:
                summary["failed_windows"].append({"start": window.start_text, "end": window.end_text, "error": "empty"})
            else:
                existing = _existing_count(index_code, freq, window)
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

            if index < len(windows) - 1 and request_interval_seconds > 0:
                time.sleep(request_interval_seconds)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not combined.empty:
            summary["trade_time_min"] = str(combined["trade_time"].min())
            summary["trade_time_max"] = str(combined["trade_time"].max())
            summary["quality"] = _quality_report(combined)

        run.fetched_rows = int(summary["fetched_rows"])
        run.inserted_rows = int(summary["inserted_rows"])
        run.updated_rows = int(summary["updated_rows"])
        run.skipped_rows = int(summary["skipped_windows"])
        run.error_rows = len(summary["failed_windows"])
        run.finish("success" if not summary["failed_windows"] else "partial", summary["rate_limit_message"])
        return summary
    except Exception as exc:
        run.fetched_rows = int(summary["fetched_rows"])
        run.inserted_rows = int(summary["inserted_rows"])
        run.updated_rows = int(summary["updated_rows"])
        run.skipped_rows = int(summary["skipped_windows"])
        run.error_rows = len(summary["failed_windows"]) + 1
        run.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集中证1000等指数 Tushare 历史分钟线。")
    parser.add_argument("--start-date", required=True, help="开始日期，例如 2015-01-01。")
    parser.add_argument("--end-date", required=True, help="结束日期，例如 2026-06-17。")
    parser.add_argument("--index-code", default=DEFAULT_INDEX_CODE, help="指数代码，默认 000852。")
    parser.add_argument("--ts-code", default=DEFAULT_TS_CODE, help="Tushare 代码，默认 000852.SH。")
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME, help="指数名称，默认 中证1000。")
    parser.add_argument("--freq", default=DEFAULT_FREQ, choices=["1min", "5min", "15min", "30min", "60min"])
    parser.add_argument("--request-interval-seconds", type=float, default=DEFAULT_REQUEST_INTERVAL_SECONDS)
    parser.add_argument("--window-months", type=int, default=1, help="每次请求覆盖几个自然月，1分钟线建议默认 1。")
    parser.add_argument("--max-windows", type=int, help="最多执行几个自然月窗口，测试时建议 1。")
    parser.add_argument("--skip-existing", action="store_true", help="窗口内已有数据时跳过，适合断点续跑。")
    parser.add_argument("--skip-existing-min-rows", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="只打印计划窗口，不调用 Tushare、不入库。")
    parser.add_argument("--apply-migrations", action="store_true", help="采集前执行 SQL migration。")
    args = parser.parse_args()

    result = collect_market_index_minutes(
        start_date=args.start_date,
        end_date=args.end_date,
        index_code=args.index_code,
        ts_code=args.ts_code,
        index_name=args.index_name,
        freq=args.freq,
        request_interval_seconds=args.request_interval_seconds,
        window_months=args.window_months,
        max_windows=args.max_windows,
        skip_existing=args.skip_existing,
        skip_existing_min_rows=args.skip_existing_min_rows,
        dry_run=args.dry_run,
        apply_db_migrations=args.apply_migrations,
    )
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
