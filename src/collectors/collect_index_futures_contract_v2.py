from __future__ import annotations

import argparse
import calendar
import re
import time as time_module
from datetime import date, datetime, time
from typing import Any

import pandas as pd

from src.common.db import execute_sql, read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.quality.check_collection_data import (
    DATASETS,
    evaluate_dataframe,
    evaluate_futures_contracts,
    persist_metrics,
)


logger = get_logger(__name__)
PRODUCTS = {"IF": 300.0, "IC": 200.0, "IM": 200.0}
DATA_SOURCE = "akshare:get_futures_daily:CFFEX"
SOURCE_URL = "http://www.cffex.com.cn/rtj/"


def _planned_expiry(contract_code: str) -> date:
    match = re.fullmatch(r"(?:IF|IC|IM)(\d{2})(\d{2})", contract_code)
    if not match:
        raise ValueError(f"invalid index futures contract code: {contract_code}")
    year = 2000 + int(match.group(1))
    month = int(match.group(2))
    fridays = [
        day for day in calendar.Calendar().itermonthdates(year, month)
        if day.month == month and day.weekday() == calendar.FRIDAY
    ]
    return fridays[2]


def _available_time(trade_date: date) -> datetime:
    return datetime.combine(
        (pd.Timestamp(trade_date) + pd.Timedelta(days=1)).date(), time(0, 0)
    )


def normalize_cffex_contracts(raw: pd.DataFrame) -> pd.DataFrame:
    columns = {
        "symbol": "contract_code",
        "date": "trade_date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "settle": "settle",
        "pre_settle": "pre_settle",
        "volume": "volume",
        "open_interest": "open_interest",
        "turnover": "raw_turnover_10k_cny",
        "variety": "product_code",
    }
    missing = sorted(set(columns) - set(raw.columns))
    if missing:
        raise ValueError(f"CFFEX daily data missing columns: {missing}")
    frame = raw.rename(columns=columns)[list(columns.values())].copy()
    frame["contract_code"] = frame["contract_code"].astype(str).str.strip().str.upper()
    frame["product_code"] = frame["product_code"].astype(str).str.strip().str.upper()
    frame = frame[
        frame["product_code"].isin(PRODUCTS)
        & frame["contract_code"].str.fullmatch(r"(?:IF|IC|IM)\d{4}")
    ].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    numeric = [
        "open", "high", "low", "close", "settle", "pre_settle",
        "volume", "open_interest", "raw_turnover_10k_cny",
    ]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["trade_date", "contract_code", "product_code"])
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["contract_multiplier"] = frame["product_code"].map(PRODUCTS)
    frame["amount"] = frame["raw_turnover_10k_cny"] * 10_000.0
    frame["expiry_date"] = frame["contract_code"].map(_planned_expiry)
    frame["expiry_date_source"] = "contract_rule_third_friday"
    frame["amount_unit"] = "CNY"
    frame["data_source"] = DATA_SOURCE
    frame["source_url"] = SOURCE_URL
    frame["available_time"] = frame["trade_date"].map(_available_time)
    frame["crawl_time"] = datetime.now()
    columns_out = [
        "trade_date", "product_code", "contract_code", "expiry_date",
        "contract_multiplier", "open", "high", "low", "close", "settle",
        "pre_settle", "volume", "open_interest", "amount",
        "raw_turnover_10k_cny", "expiry_date_source", "amount_unit",
        "data_source", "source_url", "available_time", "crawl_time",
    ]
    return frame[columns_out].drop_duplicates(
        ["trade_date", "contract_code"], keep="last"
    ).sort_values(["trade_date", "contract_code"]).reset_index(drop=True)


def fetch_cffex_contracts(start_date: str, end_date: str) -> pd.DataFrame:
    disable_env_proxies()
    import akshare as ak

    raw = ak.get_futures_daily(
        start_date=start_date.replace("-", ""),
        end_date=end_date.replace("-", ""),
        market="CFFEX",
    )
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        return pd.DataFrame()
    return normalize_cffex_contracts(raw)


def _reference_trade_dates(start_date: str, end_date: str) -> list[str]:
    frame = read_sql(
        """
        SELECT DISTINCT trade_date FROM market_index_daily
        WHERE index_code='000852' AND trade_date BETWEEN :start_date AND :end_date
        ORDER BY trade_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        raise RuntimeError("market_index_daily 中缺少 000852 基准交易日")
    return pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d").tolist()


def _retry(retries: int, start_date: str, end_date: str) -> pd.DataFrame:
    errors: list[str] = []
    for attempt in range(retries):
        try:
            return fetch_cffex_contracts(start_date, end_date)
        except Exception as exc:
            errors.append(f"attempt={attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt + 1 < retries:
                time_module.sleep(attempt + 1)
    raise RuntimeError("; ".join(errors))


def _correct_expired_contract_dates(as_of_date: str) -> None:
    execute_sql(
        """
        UPDATE index_futures_contract_daily target
        JOIN (
            SELECT contract_code, MAX(trade_date) AS actual_expiry
            FROM index_futures_contract_daily
            GROUP BY contract_code
        ) observed ON observed.contract_code = target.contract_code
        SET target.expiry_date = observed.actual_expiry,
            target.expiry_date_source = 'observed_last_trade_date'
        WHERE target.expiry_date < :as_of_date
        """,
        {"as_of_date": as_of_date},
    )


def collect_index_futures_contracts(
    start_date: str,
    end_date: str,
    retries: int = 3,
    run_quality: bool = True,
) -> dict[str, Any]:
    ingestion = IngestionRun(
        dataset_name="index_futures_contract_daily",
        data_source=DATA_SOURCE,
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    try:
        frames: list[pd.DataFrame] = []
        trade_dates = _reference_trade_dates(start_date, end_date)
        for index, trade_date in enumerate(trade_dates, 1):
            frame = _retry(retries, trade_date, trade_date)
            if not frame.empty:
                frames.append(frame)
            if index % 100 == 0:
                logger.info("CFFEX contracts: %s/%s trade dates completed", index, len(trade_dates))
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        ingestion.fetched_rows = len(combined)
        if combined.empty:
            raise RuntimeError("CFFEX source returned no IM/IC/IF contract rows")
        existing = read_sql(
            """
            SELECT trade_date, contract_code FROM index_futures_contract_daily
            WHERE trade_date BETWEEN :start_date AND :end_date
            """,
            {"start_date": start_date, "end_date": end_date},
        )
        old_keys = set(zip(existing["trade_date"].astype(str), existing["contract_code"])) \
            if not existing.empty else set()
        new_keys = set(zip(combined["trade_date"].astype(str), combined["contract_code"]))
        ingestion.inserted_rows = len(new_keys - old_keys)
        ingestion.updated_rows = len(new_keys & old_keys)
        upsert_dataframe(
            combined, "index_futures_contract_daily", ["trade_date", "contract_code"]
        )
        _correct_expired_contract_dates(end_date)

        metrics: list[dict[str, Any]] = []
        if run_quality:
            stored = read_sql(
                """
                SELECT * FROM index_futures_contract_daily
                WHERE trade_date BETWEEN :start_date AND :end_date
                """,
                {"start_date": start_date, "end_date": end_date},
            )
            metrics = evaluate_dataframe(stored, DATASETS["index_futures_contract_daily"])
            metrics.extend(evaluate_futures_contracts(stored))
            persist_metrics("index_futures_contract_daily", metrics, date.today())
        ingestion.finish("success")
        return {
            "rows": len(combined),
            "inserted_rows": ingestion.inserted_rows,
            "updated_rows": ingestion.updated_rows,
            "quality_metrics": metrics,
        }
    except Exception as exc:
        ingestion.error_rows += 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集中金所 IM/IC/IF 真实合约级日行情。")
    parser.add_argument("--start-date", default="2015-01-05")
    parser.add_argument("--end-date", default="2026-06-10")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    result = collect_index_futures_contracts(
        args.start_date,
        args.end_date,
        retries=args.retries,
        run_quality=not args.skip_quality,
    )
    print(result)


if __name__ == "__main__":
    main()
