from __future__ import annotations

import argparse
import hashlib
import json
import time as time_module
from datetime import date, datetime, time
from typing import Any

import pandas as pd

from src.collectors.stock_universe_sources import fetch_collection_stock_universe
from src.common.config import project_path
from src.common.db import execute_sql, read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.quality.check_collection_data import (
    DATASETS,
    evaluate_date_coverage,
    evaluate_dataframe,
    evaluate_stock_daily_boundaries,
    persist_metrics,
)


logger = get_logger(__name__)
SINA_SOURCE = "akshare:stock_zh_a_daily"
SINA_SOURCE_URL = "https://finance.sina.com.cn/realstock/"
EASTMONEY_SOURCE = "akshare:stock_zh_a_hist"
EASTMONEY_SOURCE_URL = "https://quote.eastmoney.com/"
COLLECTION_SOURCE = "akshare:stock_daily_sina_eastmoney"
BSE_MARKET_START = pd.Timestamp("2021-11-15")


def _source_symbol(stock_code: str, exchange: str) -> str:
    prefixes = {"SSE": "sh", "SZSE": "sz", "BSE": "bj"}
    if exchange not in prefixes:
        raise ValueError(f"未知交易所: {exchange}")
    return f"{prefixes[exchange]}{stock_code}"


def _row_hash(row: pd.Series) -> str:
    values = [
        row.get("trade_date"), row.get("stock_code"), row.get("open"), row.get("high"),
        row.get("low"), row.get("close"), row.get("volume"), row.get("amount"),
        row.get("turnover_rate"), row.get("data_source"),
    ]
    return hashlib.sha256("|".join("" if value is None else str(value) for value in values).encode()).hexdigest()


def normalize_sina_history(
    raw: pd.DataFrame,
    stock_code: str,
    exchange: str,
    stock_name: str | None,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frame = raw.rename(columns={"date": "trade_date", "turnover": "turnover_rate"}).copy()
    required = ["trade_date", "open", "high", "low", "close", "volume", "amount", "turnover_rate"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"新浪日线缺少字段: {missing}; got={list(raw.columns)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "open", "high", "low", "close"])
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date")
    suspension_placeholder = (
        frame[["open", "high", "low", "volume", "amount"]].fillna(0).eq(0).all(axis=1)
        & frame["close"].gt(0)
    )
    frame = frame.loc[~suspension_placeholder].copy()
    frame["pre_close"] = frame["close"].shift(1)
    frame["pct_chg"] = frame["close"] / frame["pre_close"] - 1
    frame["amplitude"] = (frame["high"] - frame["low"]) / frame["pre_close"]
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()
    if frame.empty:
        return frame
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["stock_code"] = stock_code
    frame["exchange"] = exchange
    frame["stock_name"] = stock_name
    frame["currency"] = "CNY"
    frame["data_source"] = SINA_SOURCE
    frame["source_url"] = f"{SINA_SOURCE_URL}{_source_symbol(stock_code, exchange)}/"
    frame["available_time"] = frame["trade_date"].map(
        lambda value: datetime.combine(value, time(15, 30))
    )
    frame["crawl_time"] = datetime.now()
    frame["raw_hash"] = frame.apply(_row_hash, axis=1)
    columns = [
        "trade_date", "stock_code", "exchange", "stock_name", "open", "high", "low",
        "close", "pre_close", "pct_chg", "volume", "amount", "turnover_rate", "amplitude",
        "currency", "data_source", "source_url", "available_time", "crawl_time", "raw_hash",
    ]
    return frame[columns].reset_index(drop=True)


def normalize_eastmoney_history(
    raw: pd.DataFrame,
    stock_code: str,
    exchange: str,
    stock_name: str | None,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    frame = raw.rename(
        columns={
            "\u65e5\u671f": "trade_date",
            "\u5f00\u76d8": "open",
            "\u6700\u9ad8": "high",
            "\u6700\u4f4e": "low",
            "\u6536\u76d8": "close",
            "\u6210\u4ea4\u91cf": "volume",
            "\u6210\u4ea4\u989d": "amount",
            "\u6362\u624b\u7387": "turnover_rate",
            "\u632f\u5e45": "amplitude",
            "\u6da8\u8dcc\u5e45": "pct_chg",
        }
    ).copy()
    required = [
        "trade_date", "open", "high", "low", "close", "volume", "amount",
        "turnover_rate", "amplitude", "pct_chg",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Eastmoney daily history missing columns: {missing}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "open", "high", "low", "close"])
    frame = frame.sort_values("trade_date").drop_duplicates("trade_date")
    frame["volume"] = frame["volume"] * 100.0
    frame["turnover_rate"] = frame["turnover_rate"] / 100.0
    frame["amplitude"] = frame["amplitude"] / 100.0
    frame["pct_chg"] = frame["pct_chg"] / 100.0
    frame["pre_close"] = frame["close"].shift(1)
    fallback = frame["pre_close"].isna() & frame["pct_chg"].notna() & frame["pct_chg"].ne(-1)
    frame.loc[fallback, "pre_close"] = (
        frame.loc[fallback, "close"] / (1 + frame.loc[fallback, "pct_chg"])
    )
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    frame = frame[(frame["trade_date"] >= start) & (frame["trade_date"] <= end)].copy()
    if frame.empty:
        return frame
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["stock_code"] = stock_code
    frame["exchange"] = exchange
    frame["stock_name"] = stock_name
    frame["currency"] = "CNY"
    frame["data_source"] = EASTMONEY_SOURCE
    market_prefix = {"SSE": "SH", "SZSE": "SZ", "BSE": "BJ"}[exchange]
    frame["source_url"] = f"{EASTMONEY_SOURCE_URL}{market_prefix}{stock_code}.html"
    frame["available_time"] = frame["trade_date"].map(
        lambda value: datetime.combine(value, time(15, 30))
    )
    frame["crawl_time"] = datetime.now()
    frame["raw_hash"] = frame.apply(_row_hash, axis=1)
    columns = [
        "trade_date", "stock_code", "exchange", "stock_name", "open", "high", "low",
        "close", "pre_close", "pct_chg", "volume", "amount", "turnover_rate", "amplitude",
        "currency", "data_source", "source_url", "available_time", "crawl_time", "raw_hash",
    ]
    return frame[columns].reset_index(drop=True)


def fetch_stock_history(
    stock_code: str,
    exchange: str,
    stock_name: str | None,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    disable_env_proxies()
    import akshare as ak

    symbol = _source_symbol(stock_code, exchange)
    requested_start = pd.Timestamp(start_date)
    try:
        raw = pd.DataFrame()
        normalized = pd.DataFrame()
        for lookback_days in (15, 90, 365, 1825):
            buffered_start = (requested_start - pd.Timedelta(days=lookback_days)).strftime("%Y%m%d")
            raw = ak.stock_zh_a_daily(
                symbol=symbol,
                start_date=buffered_start,
                end_date=end_date.replace("-", ""),
                adjust="",
            )
            if not isinstance(raw, pd.DataFrame) or raw.empty:
                break
            normalized = normalize_sina_history(
                raw, stock_code, exchange, stock_name, start_date, end_date
            )
            if normalized.empty or pd.notna(normalized.iloc[0]["pre_close"]):
                return normalized
            raw_first_date = pd.to_datetime(raw["date"], errors="coerce").min()
            if pd.notna(raw_first_date) and raw_first_date >= requested_start:
                continue
            return normalized
    except Exception as exc:
        logger.warning("Sina history unavailable code=%s error=%s", stock_code, exc)

    buffered_start = (requested_start - pd.Timedelta(days=15)).strftime("%Y%m%d")
    raw = ak.stock_zh_a_hist(
        symbol=stock_code,
        period="daily",
        start_date=buffered_start,
        end_date=end_date.replace("-", ""),
        adjust="",
    )
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    return normalize_eastmoney_history(
        raw, stock_code, exchange, stock_name, start_date, end_date
    )


def save_universe_snapshot(universe: pd.DataFrame, snapshot_date: date | None = None) -> int:
    snapshot_date = snapshot_date or date.today()
    snapshot_id = hashlib.sha256(
        "|".join(
            f"{row.exchange}:{row.stock_code}:{row.stock_name}"
            for row in universe.itertuples(index=False)
        ).encode()
    ).hexdigest()[:32]
    frame = universe.copy()
    frame["snapshot_id"] = snapshot_id
    frame["snapshot_date"] = snapshot_date
    frame["available_time"] = datetime.now()
    frame["crawl_time"] = datetime.now()
    frame["raw_hash"] = frame.apply(
        lambda row: hashlib.sha256(
            f"{row['exchange']}|{row['stock_code']}|{row['stock_name']}|{row['listing_date']}|"
            f"{row['security_status']}|{row['delisting_date']}".encode()
        ).hexdigest(),
        axis=1,
    )
    columns = [
        "snapshot_id", "snapshot_date", "stock_code", "exchange", "stock_name", "listing_date",
        "security_status", "delisting_date", "data_source", "source_url", "available_time",
        "crawl_time", "raw_hash",
    ]
    return upsert_dataframe(
        frame[columns], "stock_universe_snapshot_raw", ["snapshot_id", "exchange", "stock_code"]
    )


def _update_checkpoint(
    item: pd.Series,
    start_date: str,
    end_date: str,
    status: str,
    fetched_rows: int,
    last_success_date: date | None,
    error_message: str | None = None,
) -> None:
    execute_sql(
        """
        INSERT INTO stock_daily_collection_checkpoint (
            stock_code, exchange, data_source, requested_start_date, requested_end_date,
            last_success_date, fetched_rows, attempt_count, status, error_message
        ) VALUES (
            :stock_code, :exchange, :data_source, :requested_start_date, :requested_end_date,
            :last_success_date, :fetched_rows, 1, :status, :error_message
        ) ON DUPLICATE KEY UPDATE
            requested_start_date = VALUES(requested_start_date),
            requested_end_date = VALUES(requested_end_date),
            last_success_date = COALESCE(VALUES(last_success_date), last_success_date),
            fetched_rows = fetched_rows + VALUES(fetched_rows),
            attempt_count = attempt_count + 1,
            status = VALUES(status),
            error_message = VALUES(error_message)
        """,
        {
            "stock_code": item["stock_code"], "exchange": item["exchange"],
            "data_source": COLLECTION_SOURCE, "requested_start_date": start_date,
            "requested_end_date": end_date, "last_success_date": last_success_date,
            "fetched_rows": fetched_rows, "status": status, "error_message": error_message,
        },
    )


def _resume_start(item: pd.Series, requested_start: str, requested_end: str) -> str:
    checkpoint = read_sql(
        """
        SELECT requested_end_date, last_success_date, status
        FROM stock_daily_collection_checkpoint
        WHERE stock_code=:stock_code AND exchange=:exchange AND data_source=:data_source
        """,
        {
            "stock_code": item["stock_code"], "exchange": item["exchange"],
            "data_source": COLLECTION_SOURCE,
        },
    )
    if checkpoint.empty or pd.isna(checkpoint.iloc[0]["last_success_date"]):
        return requested_start
    saved_end = checkpoint.iloc[0]["requested_end_date"]
    if (
        checkpoint.iloc[0]["status"] == "success"
        and pd.notna(saved_end)
        and pd.Timestamp(saved_end) >= pd.Timestamp(requested_end)
    ):
        return (pd.Timestamp(requested_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    next_date = pd.Timestamp(checkpoint.iloc[0]["last_success_date"]) + pd.Timedelta(days=1)
    return max(pd.Timestamp(requested_start), next_date).strftime("%Y-%m-%d")


def _existing_rows(stock_code: str, start_date: str, end_date: str) -> int:
    frame = read_sql(
        """
        SELECT COUNT(*) AS row_count FROM market_stock_daily_raw
        WHERE stock_code=:stock_code AND trade_date BETWEEN :start_date AND :end_date
        """,
        {"stock_code": stock_code, "start_date": start_date, "end_date": end_date},
    )
    return int(frame.iloc[0]["row_count"])


def _effective_date_range(item: dict[str, Any], start_date: str, end_date: str) -> tuple[str, str]:
    effective_start = pd.Timestamp(start_date)
    effective_end = pd.Timestamp(end_date)
    if pd.notna(item.get("listing_date")):
        effective_start = max(effective_start, pd.Timestamp(item["listing_date"]))
    if item.get("exchange") == "BSE":
        effective_start = max(effective_start, BSE_MARKET_START)
    if pd.notna(item.get("delisting_date")):
        effective_end = min(effective_end, pd.Timestamp(item["delisting_date"]))
    return effective_start.strftime("%Y-%m-%d"), effective_end.strftime("%Y-%m-%d")


def _delete_outside_lifecycle(item: dict[str, Any]) -> None:
    conditions: list[str] = []
    params: dict[str, Any] = {"stock_code": item["stock_code"], "exchange": item["exchange"]}
    if pd.notna(item.get("listing_date")):
        conditions.append("trade_date < :listing_date")
        params["listing_date"] = item["listing_date"]
    if item.get("exchange") == "BSE":
        conditions.append("trade_date < :bse_market_start")
        params["bse_market_start"] = BSE_MARKET_START.date()
    if pd.notna(item.get("delisting_date")):
        conditions.append("trade_date > :delisting_date")
        params["delisting_date"] = item["delisting_date"]
    if not conditions:
        lifecycle_sql = ""
    else:
        lifecycle_sql = " OR " + " OR ".join(conditions)
    execute_sql(
        "DELETE FROM market_stock_daily_raw "
        "WHERE stock_code=:stock_code AND exchange=:exchange AND ("
        "(open=0 AND high=0 AND low=0 AND volume=0 AND amount=0 AND close>0)"
        + lifecycle_sql
        + ")",
        params,
    )


def _select_universe(
    universe: pd.DataFrame,
    codes: list[str] | None,
    exchanges: list[str] | None,
    offset: int,
    limit: int | None,
) -> pd.DataFrame:
    selected = universe.copy()
    if codes:
        normalized = {str(code).zfill(6) for code in codes}
        selected = selected[selected["stock_code"].isin(normalized)]
    if exchanges:
        selected = selected[selected["exchange"].isin(exchanges)]
    selected = selected.iloc[offset:]
    if limit is not None:
        selected = selected.head(limit)
    if selected.empty:
        raise ValueError("筛选后的股票列表为空")
    return selected.reset_index(drop=True)


def _quality_report(
    frame: pd.DataFrame,
    universe: pd.DataFrame,
    dataset_name: str,
    reference_dates: pd.Series,
) -> None:
    metrics = evaluate_dataframe(frame, DATASETS["market_stock_daily_raw"])
    metrics.extend(evaluate_date_coverage(frame, DATASETS["market_stock_daily_raw"], reference_dates))
    metrics.extend(evaluate_stock_daily_boundaries(frame, universe))
    persist_metrics(dataset_name, metrics, date.today())


def collect_stock_daily(
    start_date: str,
    end_date: str,
    codes: list[str] | None = None,
    exchanges: list[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
    resume: bool = False,
    sleep_seconds: float = 0.2,
    retries: int = 3,
    run_batch_quality: bool = True,
) -> dict[str, int]:
    if pd.Timestamp(start_date) > pd.Timestamp(end_date):
        raise ValueError("start_date must not be later than end_date")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if retries <= 0:
        raise ValueError("retries must be positive")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds must be non-negative")
    universe = fetch_collection_stock_universe(start_date)
    save_universe_snapshot(universe)
    selected = _select_universe(universe, codes, exchanges, offset, limit)
    reference = read_sql(
        """
        SELECT DISTINCT trade_date FROM market_index_daily
        WHERE index_code='000852' AND trade_date BETWEEN :start_date AND :end_date
        ORDER BY trade_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )["trade_date"]
    run = IngestionRun(
        dataset_name="market_stock_daily_raw",
        data_source=COLLECTION_SOURCE,
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    combined: list[pd.DataFrame] = []
    summary = {
        "stocks": len(selected), "success": 0, "empty": 0, "failed": 0,
        "rows": 0, "inserted_rows": 0, "updated_rows": 0,
    }
    try:
        for item in selected.to_dict(orient="records"):
            series = pd.Series(item)
            _delete_outside_lifecycle(item)
            lifecycle_start, actual_end = _effective_date_range(item, start_date, end_date)
            if pd.Timestamp(lifecycle_start) > pd.Timestamp(actual_end):
                summary["empty"] += 1
                _update_checkpoint(series, lifecycle_start, actual_end, "success", 0, None)
                continue
            actual_start = (
                _resume_start(series, lifecycle_start, actual_end) if resume else lifecycle_start
            )
            if pd.Timestamp(actual_start) > pd.Timestamp(actual_end):
                summary["success"] += 1
                continue
            try:
                frame = pd.DataFrame()
                errors: list[str] = []
                for attempt in range(retries):
                    try:
                        frame = fetch_stock_history(
                            item["stock_code"], item["exchange"], item.get("stock_name"),
                            actual_start, actual_end,
                        )
                        break
                    except Exception as exc:
                        errors.append(f"attempt={attempt + 1}: {type(exc).__name__}: {exc}")
                        if attempt + 1 < retries:
                            time_module.sleep(1 + attempt)
                else:
                    raise RuntimeError("; ".join(errors))
                if frame.empty:
                    summary["empty"] += 1
                    _update_checkpoint(series, actual_start, actual_end, "success", 0, None)
                    continue
                existing = _existing_rows(item["stock_code"], actual_start, actual_end)
                upsert_dataframe(frame, "market_stock_daily_raw", ["trade_date", "stock_code"])
                last_date = max(frame["trade_date"])
                _update_checkpoint(series, actual_start, actual_end, "success", len(frame), last_date)
                if run_batch_quality:
                    combined.append(frame)
                summary["success"] += 1
                summary["rows"] += len(frame)
                summary["inserted_rows"] += max(0, len(frame) - existing)
                summary["updated_rows"] += min(len(frame), existing)
                logger.info("collected stock code=%s exchange=%s rows=%s", item["stock_code"], item["exchange"], len(frame))
            except Exception as exc:
                summary["failed"] += 1
                _update_checkpoint(series, actual_start, actual_end, "failed", 0, None, str(exc))
                logger.exception("failed stock code=%s exchange=%s", item["stock_code"], item["exchange"])
            if sleep_seconds > 0:
                time_module.sleep(sleep_seconds)
        run.fetched_rows = summary["rows"]
        run.inserted_rows = summary["inserted_rows"]
        run.updated_rows = summary["updated_rows"]
        run.error_rows = summary["failed"]
        run.finish("success" if summary["failed"] == 0 else "partial")
        if combined:
            batch = pd.concat(combined, ignore_index=True)
            _quality_report(batch, selected, "market_stock_daily_raw:batch", reference)
        output = project_path("data", "reports", "stock_daily_v2_last_run.json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary
    except Exception as exc:
        run.error_rows += 1
        run.finish("failed", str(exc))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集模型2.0全市场个股不复权日行情。")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--code", action="append", dest="codes")
    parser.add_argument("--exchange", action="append", choices=["SSE", "SZSE", "BSE"])
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-batch-quality", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect_stock_daily(
        args.start_date, args.end_date, args.codes, args.exchange, args.offset, args.limit,
        args.resume, args.sleep, args.retries, not args.skip_batch_quality,
    ), ensure_ascii=False))


if __name__ == "__main__":
    main()
