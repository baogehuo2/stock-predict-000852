from __future__ import annotations

import argparse
import hashlib
import json
import os
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
DATA_SOURCE = "minishare:idx_mins"
SOURCE_URL = "https://minidoc.pages.dev/#idx-mins"
DEFAULT_FREQ = "1min"
DEFAULT_REQUEST_SLEEP_SECONDS = 0.5
MAX_WINDOW_DAYS = 31

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


@dataclass(frozen=True)
class Window:
    start: datetime
    end: datetime

    @property
    def start_text(self) -> str:
        return self.start.strftime("%Y%m%d %H:%M:%S")

    @property
    def end_text(self) -> str:
        return self.end.strftime("%Y%m%d %H:%M:%S")


def _parse_date(value: str) -> date:
    return pd.Timestamp(value).date()


def _raw_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _index_code(ts_code: str) -> str:
    return str(ts_code).split(".")[0]


def _load_token() -> str:
    token = (
        os.environ.get("MINISHARE_TOKEN")
        or get_secret("minishare.token")
        or get_secret("MINISHARE_TOKEN")
        or get_secret("minishare_token")
    )
    if not token:
        raise RuntimeError(
            "missing minishare token. Set env MINISHARE_TOKEN or encrypted secret minishare.token."
        )
    return str(token)


def _load_minishare_client(token: str | None = None):
    try:
        import minishare as ms
    except ImportError as exc:
        raise RuntimeError(
            "missing dependency: minishare. "
            "Install with `pip install minishare --extra-index-url https://minidoc.pages.dev/simple/ -U`."
        ) from exc
    return ms.pro_api(token or _load_token())


def _windows(start_date: str, end_date: str, window_days: int = MAX_WINDOW_DAYS) -> list[Window]:
    if window_days < 1 or window_days > MAX_WINDOW_DAYS:
        raise ValueError(f"window_days must be between 1 and {MAX_WINDOW_DAYS}")
    start = pd.Timestamp(_parse_date(start_date))
    end = pd.Timestamp(_parse_date(end_date))
    if start > end:
        raise ValueError(f"start_date must be <= end_date: {start_date} > {end_date}")

    result: list[Window] = []
    current = start
    while current <= end:
        window_end = min(current + pd.Timedelta(days=window_days - 1), end)
        result.append(
            Window(
                start=datetime.combine(current.date(), datetime.strptime("09:00:00", "%H:%M:%S").time()),
                end=datetime.combine(window_end.date(), datetime.strptime("19:00:00", "%H:%M:%S").time()),
            )
        )
        current = window_end + pd.Timedelta(days=1)
    return result


def normalize_idx_mins(
    raw: pd.DataFrame,
    freq: str,
    index_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    frame = raw.copy()
    required = ["ts_code", "trade_time", "open", "close", "high", "low", "vol", "amount"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"idx_mins missing columns {missing}; got={list(frame.columns)}")

    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame = frame.dropna(subset=["trade_time", "ts_code"]).copy()
    frame["trade_date"] = frame["trade_time"].dt.date
    for column in ["open", "close", "high", "low", "vol", "amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    if frame.empty:
        return frame

    ohlc = frame[["open", "high", "low", "close"]]
    frame["high"] = ohlc.max(axis=1)
    frame["low"] = ohlc.min(axis=1)
    frame["index_code"] = frame["ts_code"].map(_index_code)
    index_names = index_names or {}
    frame["index_name"] = frame["ts_code"].map(index_names)
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
        .sort_values(["ts_code", "trade_time"])
        .drop_duplicates(["index_code", "trade_time", "freq", "data_source"], keep="last")
    )


def fetch_range(
    pro: Any,
    ts_code: str,
    freq: str,
    window: Window,
) -> pd.DataFrame:
    return pro.idx_mins(
        ts_code=ts_code,
        freq=freq,
        start_date=window.start_text,
        end_date=window.end_text,
    )


def fetch_trade_time(pro: Any, freq: str, trade_time: str) -> pd.DataFrame:
    return pro.idx_mins(freq=freq, trade_time=trade_time)


def _existing_count(ts_code: str, freq: str, window: Window) -> int:
    result = read_sql(
        f"""
        SELECT COUNT(*) AS row_count
        FROM {TABLE_NAME}
        WHERE ts_code = :ts_code
          AND freq = :freq
          AND data_source = :data_source
          AND trade_time BETWEEN :start_time AND :end_time
        """,
        {
            "ts_code": ts_code,
            "freq": freq,
            "data_source": DATA_SOURCE,
            "start_time": window.start,
            "end_time": window.end,
        },
    )
    return int(result.iloc[0]["row_count"])


def _quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    spec = DatasetSpec(
        table=TABLE_NAME,
        date_column="trade_time",
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


def collect_minishare_index_minutes(
    start_date: str,
    end_date: str,
    symbols: dict[str, str],
    freq: str = DEFAULT_FREQ,
    window_days: int = MAX_WINDOW_DAYS,
    request_sleep_seconds: float = DEFAULT_REQUEST_SLEEP_SECONDS,
    max_windows: int | None = None,
    skip_existing: bool = False,
    skip_existing_min_rows: int = 1,
    dry_run: bool = False,
    apply_db_migrations: bool = False,
) -> dict[str, Any]:
    if apply_db_migrations and not dry_run:
        apply_migrations()

    windows = _windows(start_date, end_date, window_days=window_days)
    jobs = [(ts_code, window) for ts_code in symbols for window in windows]
    if max_windows is not None:
        jobs = jobs[:max_windows]

    summary: dict[str, Any] = {
        "table": TABLE_NAME,
        "data_source": DATA_SOURCE,
        "freq": freq,
        "symbol_count": len(symbols),
        "windows_per_symbol": len(windows),
        "jobs": len(jobs),
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
        summary["planned_jobs"] = [
            {"ts_code": ts_code, "start": window.start_text, "end": window.end_text}
            for ts_code, window in jobs
        ]
        return summary

    run = IngestionRun(
        dataset_name=TABLE_NAME,
        data_source=DATA_SOURCE,
        requested_start_date=_parse_date(start_date),
        requested_end_date=_parse_date(end_date),
    )
    pro = _load_minishare_client()
    frames: list[pd.DataFrame] = []
    try:
        for index, (ts_code, window) in enumerate(jobs):
            if skip_existing and _existing_count(ts_code, freq, window) >= skip_existing_min_rows:
                summary["skipped_jobs"] += 1
                continue

            try:
                raw = fetch_range(pro, ts_code, freq, window)
                frame = normalize_idx_mins(raw, freq=freq, index_names=symbols)
            except Exception as exc:
                summary["failed_jobs"].append(
                    {
                        "ts_code": ts_code,
                        "start": window.start_text,
                        "end": window.end_text,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            if frame.empty:
                summary["failed_jobs"].append(
                    {"ts_code": ts_code, "start": window.start_text, "end": window.end_text, "error": "empty"}
                )
            else:
                existing = _existing_count(ts_code, freq, window)
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
                time.sleep(request_sleep_seconds)

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


def discover_indices(
    trade_time: str,
    freq: str = DEFAULT_FREQ,
    write_db: bool = False,
    apply_db_migrations: bool = False,
) -> dict[str, Any]:
    if apply_db_migrations and write_db:
        apply_migrations()
    pro = _load_minishare_client()
    raw = fetch_trade_time(pro, freq=freq, trade_time=trade_time)
    frame = normalize_idx_mins(raw, freq=freq, index_names=DEFAULT_INDICES)
    if write_db and not frame.empty:
        upsert_dataframe(frame, TABLE_NAME, ["index_code", "trade_time", "freq", "data_source"])
        _quality_report(frame)
    codes = sorted(frame["ts_code"].dropna().astype(str).unique().tolist()) if not frame.empty else []
    return {
        "trade_time": trade_time,
        "freq": freq,
        "rows": len(frame),
        "ts_code_count": len(codes),
        "ts_codes": codes,
        "written_rows": len(frame) if write_db else 0,
    }


def _parse_symbols(values: list[str] | None) -> dict[str, str]:
    if not values:
        return dict(DEFAULT_INDICES)
    parsed: dict[str, str] = {}
    for value in values:
        if "=" in value:
            ts_code, name = value.split("=", 1)
            parsed[ts_code.strip()] = name.strip()
        else:
            parsed[value.strip()] = DEFAULT_INDICES.get(value.strip(), value.strip())
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 minishare idx_mins 采集指数历史分钟线。")
    parser.add_argument("--start-date", help="开始日期，例如 2024-07-01。")
    parser.add_argument("--end-date", help="结束日期，例如 2026-06-25。")
    parser.add_argument("--symbol", action="append", help="指数 ts_code，可重复；可写 000852.SH=中证1000。默认采集内置几大指数。")
    parser.add_argument("--freq", default=DEFAULT_FREQ, choices=["1min", "5min", "15min", "30min", "60min"])
    parser.add_argument("--window-days", type=int, default=MAX_WINDOW_DAYS, help="单次请求天数，最大31。")
    parser.add_argument("--request-sleep-seconds", type=float, default=DEFAULT_REQUEST_SLEEP_SECONDS)
    parser.add_argument("--max-windows", type=int, help="最多执行多少个 symbol-window 任务，测试时建议 1。")
    parser.add_argument("--skip-existing", action="store_true", help="窗口内已有数据时跳过，适合断点续跑。")
    parser.add_argument("--skip-existing-min-rows", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply-migrations", action="store_true")
    parser.add_argument("--discover-time", help="按某个交易时间点探测全量可用指数，例如 20260528 14:52:00。")
    parser.add_argument("--write-discovery", action="store_true", help="discover-time 返回的数据也写入分钟表。")
    args = parser.parse_args()

    if args.discover_time:
        try:
            result = discover_indices(
                trade_time=args.discover_time,
                freq=args.freq,
                write_db=args.write_discovery,
                apply_db_migrations=args.apply_migrations,
            )
        except RuntimeError as exc:
            result = {
                "status": "failed",
                "error": str(exc),
                "hint": "Set $env:MINISHARE_TOKEN for one run, or add minishare.token to encrypted secrets.",
            }
    else:
        if not args.start_date or not args.end_date:
            raise SystemExit("--start-date and --end-date are required unless --discover-time is used")
        try:
            result = collect_minishare_index_minutes(
                start_date=args.start_date,
                end_date=args.end_date,
                symbols=_parse_symbols(args.symbol),
                freq=args.freq,
                window_days=args.window_days,
                request_sleep_seconds=args.request_sleep_seconds,
                max_windows=args.max_windows,
                skip_existing=args.skip_existing,
                skip_existing_min_rows=args.skip_existing_min_rows,
                dry_run=args.dry_run,
                apply_db_migrations=args.apply_migrations,
            )
        except RuntimeError as exc:
            result = {
                "status": "failed",
                "error": str(exc),
                "hint": "Set $env:MINISHARE_TOKEN for one run, or add minishare.token to encrypted secrets.",
            }
    print(json.dumps(result, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
