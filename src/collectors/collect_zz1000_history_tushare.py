from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.collectors.collect_zz1000_snapshot_v2 import (
    DATA_SOURCE,
    INDEX_CODE,
    INDEX_CODE_TUSHARE,
    _fetch_tushare_index_weight,
    _latest_trade_date,
)


def _normalize_history(raw: pd.DataFrame, fetched_at: datetime) -> pd.DataFrame:
    frame = raw.copy()
    frame = frame[frame["index_code"].astype(str).eq(INDEX_CODE_TUSHARE)].copy()
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "index_code",
                "stock_code",
                "effective_date",
                "end_date",
                "weight",
                "adjustment_type",
                "announcement_time",
                "available_time",
                "data_source",
                "source_url",
                "crawl_time",
            ]
        )

    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce") / 100.0
    frame["stock_code"] = frame["con_code"].astype(str).str.zfill(6)
    frame["adjustment_type"] = "monthly_snapshot"
    frame["announcement_time"] = None
    frame["available_time"] = fetched_at
    frame["data_source"] = DATA_SOURCE
    frame["source_url"] = "https://tushare.pro/document/2?doc_id=96"
    frame["crawl_time"] = fetched_at
    frame["effective_date"] = frame["trade_date"]
    frame = frame.sort_values(["stock_code", "trade_date"]).reset_index(drop=True)
    next_date = frame.groupby("stock_code")["trade_date"].shift(-1)
    frame["end_date"] = (pd.to_datetime(next_date) - pd.Timedelta(days=1)).dt.date
    frame["end_date"] = frame["end_date"].where(pd.notna(frame["end_date"]), None)
    frame["index_code"] = INDEX_CODE
    return frame[
        [
            "index_code",
            "stock_code",
            "effective_date",
            "end_date",
            "weight",
            "adjustment_type",
            "announcement_time",
            "available_time",
            "data_source",
            "source_url",
            "crawl_time",
        ]
    ]


def _month_windows(start_date: date, end_date: date) -> list[tuple[str, str]]:
    if start_date > end_date:
        return []
    windows: list[tuple[str, str]] = []
    cursor = start_date.replace(day=1)
    while cursor <= end_date:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        window_end = min(end_date, next_month - timedelta(days=1))
        windows.append((cursor.strftime("%Y%m%d"), window_end.strftime("%Y%m%d")))
        cursor = next_month
    return windows


def _month_end_dates(start_date: date, end_date: date) -> list[date]:
    dates: list[date] = []
    if start_date > end_date:
        return dates
    cursor = start_date.replace(day=1)
    while cursor <= end_date:
        next_month = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        month_end = min(end_date, next_month - timedelta(days=1))
        dates.append(month_end)
        cursor = next_month
    return dates


def _last_effective_date() -> date | None:
    result = read_sql(
        """
        SELECT MAX(effective_date) AS effective_date
        FROM index_constituent_history
        WHERE index_code = :index_code
        """,
        {"index_code": INDEX_CODE},
    )
    if result.empty or pd.isna(result.iloc[0]["effective_date"]):
        return None
    return pd.to_datetime(result.iloc[0]["effective_date"]).date()


def collect_history(start_date: str, end_date: str, run_quality: bool = True) -> dict[str, Any]:
    ingestion = IngestionRun(
        dataset_name="index_constituent_history",
        data_source=DATA_SOURCE,
        requested_start_date=pd.to_datetime(start_date).date(),
        requested_end_date=pd.to_datetime(end_date).date(),
    )
    fetched_at = datetime.now()
    try:
        raw = _fetch_tushare_index_weight(
            start_date=pd.to_datetime(start_date).strftime("%Y%m%d"),
            end_date=pd.to_datetime(end_date).strftime("%Y%m%d"),
        )
        ingestion.fetched_rows = len(raw)
        frame = _normalize_history(raw, fetched_at)
        if frame.empty:
            raise RuntimeError("tushare index_weight returned no CSI 1000 history rows")
        upsert_dataframe(frame, "index_constituent_history", ["index_code", "stock_code", "effective_date"])
        ingestion.inserted_rows = len(frame)
        ingestion.finish("success")
        return {
            "rows": len(frame),
            "effective_date_min": str(pd.to_datetime(frame["effective_date"]).min().date()),
            "effective_date_max": str(pd.to_datetime(frame["effective_date"]).max().date()),
            "weight_sum_latest": float(frame[frame["effective_date"] == pd.to_datetime(frame["effective_date"]).max().date()]["weight"].sum()),
        }
    except Exception as exc:
        ingestion.error_rows += 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def collect_history_backfill(
    start_date: str | None = None,
    end_date: str | None = None,
    step_months: int = 1,
) -> list[dict[str, Any]]:
    last_effective = _last_effective_date()
    if start_date:
        cursor = pd.to_datetime(start_date).date()
    elif last_effective:
        cursor = last_effective + timedelta(days=1)
    else:
        cursor = date(2020, 1, 1)

    final_end = pd.to_datetime(end_date).date() if end_date else date.today()
    if cursor > final_end:
        return []

    results: list[dict[str, Any]] = []
    month_ends = _month_end_dates(cursor, final_end)
    if step_months > 1:
        month_ends = month_ends[::step_months]
    for month_end in month_ends:
        month_start = month_end.replace(day=1)
        try:
            results.append(collect_history(month_start.isoformat(), month_end.isoformat()))
        except RuntimeError as exc:
            if "returned no CSI 1000 history rows" in str(exc):
                continue
            raise
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 Tushare 采集中证1000历史成分（月度快照）")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--backfill", action="store_true", help="从数据库已有末尾继续按月回填")
    parser.add_argument("--step-months", type=int, default=1, help="回填时每次合并多少个月窗口")
    args = parser.parse_args()

    if args.backfill:
        print(collect_history_backfill(args.start_date, args.end_date, step_months=args.step_months))
        return
    if not args.start_date or not args.end_date:
        raise SystemExit("--start-date and --end-date are required unless --backfill is used")
    print(collect_history(args.start_date, args.end_date))


if __name__ == "__main__":
    main()
