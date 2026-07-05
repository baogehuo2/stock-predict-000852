from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.migrations import apply_migrations


TABLE_NAME = "market_index_minute_raw"
DATA_SOURCE = "local:index_minute_zip"
DEFAULT_ROOT = Path(r"D:\BaiduNetdiskDownload\指数数据")
DEFAULT_CHUNKSIZE = 100_000

FREQ_DIRS = {
    "1min": "1分钟_按年汇总",
    "5min": "5分钟_按年汇总",
    "15min": "15分钟_按年汇总",
    "30min": "30分钟_按年汇总",
    "60min": "60分钟_按年汇总",
}

FREQ_MINUTES = {
    "1min": 1,
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "60min": 60,
}

DEFAULT_INDICES = {
    "000852": {"ts_code": "000852.SH", "name": "中证1000", "start_year": 2014},
    "000300": {"ts_code": "000300.SH", "name": "沪深300", "start_year": 2005},
    "000905": {"ts_code": "000905.SH", "name": "中证500", "start_year": 2007},
    "000001": {"ts_code": "000001.SH", "name": "上证指数", "start_year": 2006},
    "399001": {"ts_code": "399001.SZ", "name": "深证成指", "start_year": 2006},
    "399006": {"ts_code": "399006.SZ", "name": "创业板指", "start_year": 2010},
    "000016": {"ts_code": "000016.SH", "name": "上证50", "start_year": 2004},
    "000688": {"ts_code": "000688.SH", "name": "科创50", "start_year": 2020},
}

CSV_COLUMN_MAP = {
    "时间": "trade_time",
    "代码": "index_code",
    "名称": "index_name",
    "开盘价": "open",
    "收盘价": "close",
    "最高价": "high",
    "最低价": "low",
    "成交额": "amount",
}

OUTPUT_COLUMNS = [
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


@dataclass(frozen=True)
class ImportJob:
    freq: str
    year: int
    index_code: str
    ts_code: str
    index_name: str
    zip_path: Path
    entry_name: str

    @property
    def source_url(self) -> str:
        return f"{self.zip_path}::{self.entry_name}"


def _raw_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_csv_years(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    years: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start, end = part.split("-", 1)
                years.update(range(int(start), int(end) + 1))
            else:
                years.add(int(part))
    return sorted(years)


def _parse_indices(values: list[str] | None) -> dict[str, dict[str, Any]]:
    if not values:
        return dict(DEFAULT_INDICES)
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        code = value.strip().split("=", 1)[0].split(".", 1)[0]
        if code not in DEFAULT_INDICES:
            raise ValueError(f"unsupported index code for this import plan: {value}")
        result[code] = dict(DEFAULT_INDICES[code])
    return result


def _freqs(values: list[str] | None) -> list[str]:
    if not values:
        return ["60min", "30min", "15min", "5min", "1min"]
    result: list[str] = []
    for value in values:
        for part in value.split(","):
            freq = part.strip()
            if freq:
                if freq not in FREQ_DIRS:
                    raise ValueError(f"unsupported freq: {freq}")
                result.append(freq)
    return result


def discover_jobs(
    root: Path,
    freqs: Iterable[str],
    indices: dict[str, dict[str, Any]],
    years: list[int] | None = None,
) -> tuple[list[ImportJob], list[dict[str, Any]]]:
    jobs: list[ImportJob] = []
    missing: list[dict[str, Any]] = []
    for freq in freqs:
        freq_dir = root / FREQ_DIRS[freq]
        for index_code, meta in indices.items():
            available_years = years or list(range(int(meta["start_year"]), 2026))
            for year in available_years:
                zip_path = freq_dir / f"{year}_{freq}.zip"
                entry_name = f"{index_code}_{year}.csv"
                if not zip_path.exists():
                    missing.append({"freq": freq, "year": year, "index_code": index_code, "reason": "zip_missing"})
                    continue
                try:
                    with zipfile.ZipFile(zip_path) as archive:
                        if entry_name not in archive.namelist():
                            missing.append(
                                {"freq": freq, "year": year, "index_code": index_code, "reason": "entry_missing"}
                            )
                            continue
                except Exception as exc:
                    missing.append(
                        {
                            "freq": freq,
                            "year": year,
                            "index_code": index_code,
                            "reason": f"zip_error:{type(exc).__name__}:{exc}",
                        }
                    )
                    continue
                jobs.append(
                    ImportJob(
                        freq=freq,
                        year=year,
                        index_code=index_code,
                        ts_code=str(meta["ts_code"]),
                        index_name=str(meta["name"]),
                        zip_path=zip_path,
                        entry_name=entry_name,
                    )
                )
    return jobs, missing


def _existing_count(job: ImportJob) -> int:
    result = read_sql(
        f"""
        SELECT COUNT(*) AS row_count
        FROM {TABLE_NAME}
        WHERE index_code = :index_code
          AND freq = :freq
          AND data_source = :data_source
          AND trade_time >= :start_time
          AND trade_time < :end_time
        """,
        {
            "index_code": job.index_code,
            "freq": job.freq,
            "data_source": DATA_SOURCE,
            "start_time": datetime(job.year, 1, 1),
            "end_time": datetime(job.year + 1, 1, 1),
        },
    )
    return int(result.iloc[0]["row_count"])


def _normalize_chunk(chunk: pd.DataFrame, job: ImportJob) -> pd.DataFrame:
    frame = chunk.rename(columns=CSV_COLUMN_MAP).copy()
    required = ["trade_time", "index_code", "index_name", "open", "close", "high", "low", "amount"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{job.entry_name} missing columns {missing}; got={list(chunk.columns)}")

    frame = frame[required].copy()
    frame["trade_time"] = pd.to_datetime(frame["trade_time"], errors="coerce")
    frame = frame.dropna(subset=["trade_time"]).copy()
    frame["trade_date"] = frame["trade_time"].dt.date
    frame["index_code"] = frame["index_code"].astype(str).str.zfill(6)
    frame["index_name"] = frame["index_name"].fillna(job.index_name).astype(str)
    for column in ["open", "close", "high", "low", "amount"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    ohlc = frame[["open", "high", "low", "close"]]
    frame["high"] = ohlc.max(axis=1)
    frame["low"] = ohlc.min(axis=1)
    frame["ts_code"] = job.ts_code
    frame["freq"] = job.freq
    frame["volume"] = None
    frame["currency"] = "CNY"
    frame["volume_unit"] = "share"
    frame["amount_unit"] = "CNY_yuan"
    frame["data_source"] = DATA_SOURCE
    frame["source_url"] = job.source_url
    frame["available_time"] = frame["trade_time"].map(
        lambda value: value.to_pydatetime() + timedelta(minutes=FREQ_MINUTES[job.freq])
    )
    frame["crawl_time"] = datetime.now()
    hash_columns = ["trade_time", "index_code", "freq", "open", "high", "low", "close", "amount", "data_source"]
    frame["raw_hash"] = frame[hash_columns].apply(lambda row: _raw_hash(row.to_dict()), axis=1)
    return (
        frame[OUTPUT_COLUMNS]
        .sort_values("trade_time")
        .drop_duplicates(["index_code", "trade_time", "freq", "data_source"], keep="last")
    )


def import_job(job: ImportJob, chunksize: int = DEFAULT_CHUNKSIZE, dry_run: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "freq": job.freq,
        "year": job.year,
        "index_code": job.index_code,
        "ts_code": job.ts_code,
        "entry": job.entry_name,
        "rows": 0,
        "written_rows": 0,
        "trade_time_min": None,
        "trade_time_max": None,
    }
    if dry_run:
        return result

    with zipfile.ZipFile(job.zip_path) as archive:
        with archive.open(job.entry_name) as source:
            reader = pd.read_csv(source, chunksize=chunksize, encoding="utf-8-sig")
            for chunk in reader:
                frame = _normalize_chunk(chunk, job)
                if frame.empty:
                    continue
                upsert_dataframe(frame, TABLE_NAME, ["index_code", "trade_time", "freq", "data_source"])
                result["rows"] += len(frame)
                result["written_rows"] += len(frame)
                min_time = frame["trade_time"].min()
                max_time = frame["trade_time"].max()
                result["trade_time_min"] = str(min_time) if result["trade_time_min"] is None else min(result["trade_time_min"], str(min_time))
                result["trade_time_max"] = str(max_time) if result["trade_time_max"] is None else max(result["trade_time_max"], str(max_time))
    return result


def summarize_imported() -> pd.DataFrame:
    return read_sql(
        f"""
        SELECT ts_code, index_code, index_name, freq,
               COUNT(*) AS rows,
               MIN(trade_time) AS min_time,
               MAX(trade_time) AS max_time,
               COUNT(DISTINCT trade_date) AS trade_days,
               SUM(volume IS NULL) AS null_volume_rows,
               SUM(amount < 0) AS negative_amount_rows,
               SUM(high < GREATEST(open, close, low) OR low > LEAST(open, close, high)) AS ohlc_error_rows,
               SUM(
                   NOT (
                       TIME(trade_time) BETWEEN '09:30:00' AND '11:30:00'
                       OR TIME(trade_time) BETWEEN '13:00:00' AND '15:00:00'
                   )
               ) AS non_session_rows
        FROM {TABLE_NAME}
        WHERE data_source = :data_source
        GROUP BY ts_code, index_code, index_name, freq
        ORDER BY ts_code, FIELD(freq, '1min', '5min', '15min', '30min', '60min')
        """,
        {"data_source": DATA_SOURCE},
    )


def collect_market_index_minute_zip(
    root: Path = DEFAULT_ROOT,
    freqs: list[str] | None = None,
    indices: dict[str, dict[str, Any]] | None = None,
    years: list[int] | None = None,
    chunksize: int = DEFAULT_CHUNKSIZE,
    max_jobs: int | None = None,
    skip_existing: bool = False,
    skip_existing_min_rows: int = 1,
    dry_run: bool = False,
    apply_db_migrations: bool = False,
) -> dict[str, Any]:
    if apply_db_migrations and not dry_run:
        apply_migrations()
    selected_freqs = freqs or ["60min", "30min", "15min", "5min", "1min"]
    selected_indices = indices or dict(DEFAULT_INDICES)
    jobs, missing = discover_jobs(root, selected_freqs, selected_indices, years)
    if max_jobs is not None:
        jobs = jobs[:max_jobs]

    summary: dict[str, Any] = {
        "table": TABLE_NAME,
        "data_source": DATA_SOURCE,
        "root": str(root),
        "jobs": len(jobs),
        "missing_jobs": missing,
        "skipped_jobs": 0,
        "failed_jobs": [],
        "rows": 0,
        "written_rows": 0,
        "job_results": [],
    }
    if dry_run:
        summary["planned_jobs"] = [
            {
                "freq": job.freq,
                "year": job.year,
                "index_code": job.index_code,
                "ts_code": job.ts_code,
                "zip": str(job.zip_path),
                "entry": job.entry_name,
            }
            for job in jobs
        ]
        return summary

    run = IngestionRun(dataset_name=TABLE_NAME, data_source=DATA_SOURCE)
    try:
        for job in jobs:
            if skip_existing and _existing_count(job) >= skip_existing_min_rows:
                summary["skipped_jobs"] += 1
                continue
            try:
                job_result = import_job(job, chunksize=chunksize)
                summary["rows"] += int(job_result["rows"])
                summary["written_rows"] += int(job_result["written_rows"])
                summary["job_results"].append(job_result)
            except Exception as exc:
                summary["failed_jobs"].append(
                    {
                        "freq": job.freq,
                        "year": job.year,
                        "index_code": job.index_code,
                        "entry": job.entry_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        run.fetched_rows = int(summary["rows"])
        run.inserted_rows = int(summary["written_rows"])
        run.skipped_rows = int(summary["skipped_jobs"])
        run.error_rows = len(summary["failed_jobs"])
        run.finish("success" if not summary["failed_jobs"] else "partial")
        return summary
    except Exception as exc:
        run.fetched_rows = int(summary["rows"])
        run.inserted_rows = int(summary["written_rows"])
        run.skipped_rows = int(summary["skipped_jobs"])
        run.error_rows = len(summary["failed_jobs"]) + 1
        run.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="导入本地压缩包中的 A 股指数分钟历史数据。")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--freq", action="append", help="可重复或逗号分隔：1min,5min,15min,30min,60min。默认全频率。")
    parser.add_argument("--index-code", action="append", help="可重复，默认 8 个目标指数。")
    parser.add_argument("--year", action="append", help="可重复、逗号分隔或区间，例如 2024,2025 或 2015-2025。")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-existing-min-rows", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply-migrations", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.summary_only:
        print(summarize_imported().to_string(index=False))
        return

    result = collect_market_index_minute_zip(
        root=args.root,
        freqs=_freqs(args.freq),
        indices=_parse_indices(args.index_code),
        years=_parse_csv_years(args.year),
        chunksize=args.chunksize,
        max_jobs=args.max_jobs,
        skip_existing=args.skip_existing,
        skip_existing_min_rows=args.skip_existing_min_rows,
        dry_run=args.dry_run,
        apply_db_migrations=args.apply_migrations,
    )
    compact = dict(result)
    if len(compact.get("job_results", [])) > 20:
        compact["job_results"] = compact["job_results"][:20] + [
            {"truncated": len(result["job_results"]) - 20}
        ]
    if len(compact.get("missing_jobs", [])) > 20:
        compact["missing_jobs"] = compact["missing_jobs"][:20] + [
            {"truncated": len(result["missing_jobs"]) - 20}
        ]
    print(json.dumps(compact, ensure_ascii=False, default=str, indent=2))


if __name__ == "__main__":
    main()
