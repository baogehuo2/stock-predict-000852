from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.migrations import apply_migrations
from src.quality.check_collection_data import DATASETS, evaluate_dataframe, persist_metrics


DATA_SOURCE = "local:cffex_option_csv"
SOURCE_URL_PREFIX = "file:///"
OPTION_MARKET = "cffex_index_option"
CONTRACT_RE = re.compile(
    r"^CFFEX\.(?P<product>HO|IO|MO)(?P<month>\d{4})-(?P<side>C|P)-(?P<strike>\d+(?:\.\d+)?)\.csv$",
    re.IGNORECASE,
)

PRODUCT_META = {
    "HO": {"underlying_code": "000016", "underlying_name": "上证50"},
    "IO": {"underlying_code": "000300", "underlying_name": "沪深300"},
    "MO": {"underlying_code": "000852", "underlying_name": "中证1000"},
}


@dataclass(frozen=True)
class ParsedContractFile:
    path: Path
    product: str
    contract_month: str
    option_side: str
    strike_price: float
    contract_code: str
    underlying_code: str
    underlying_name: str


def _default_source_dir() -> Path:
    return Path("D:/BaiduNetdiskDownload") / "\u671f\u6743" / "\u671f\u6743"


def _available_time(trade_date: date) -> datetime:
    return datetime.combine(
        (pd.Timestamp(trade_date) + pd.Timedelta(days=1)).date(), time(0, 0)
    )


def _hash_row(row: pd.Series, columns: list[str]) -> str:
    payload = "|".join("" if pd.isna(row[column]) else str(row[column]) for column in columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_contract_file(path: Path) -> ParsedContractFile | None:
    match = CONTRACT_RE.match(path.name)
    if not match:
        return None
    product = match.group("product").upper()
    meta = PRODUCT_META[product]
    side_raw = match.group("side").upper()
    side = "call" if side_raw == "C" else "put"
    strike_text = match.group("strike")
    contract_month = match.group("month")
    return ParsedContractFile(
        path=path,
        product=product,
        contract_month=contract_month,
        option_side=side,
        strike_price=float(strike_text),
        contract_code=f"{product}{contract_month}{side_raw}{strike_text}",
        underlying_code=meta["underlying_code"],
        underlying_name=meta["underlying_name"],
    )


def normalize_csv_frame(
    raw: pd.DataFrame,
    contract: ParsedContractFile,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    required = ["datetime", "open", "high", "low", "close", "volume", "open_oi", "close_oi"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"{contract.path.name} missing columns: {missing}")
    frame = raw[required].copy()
    frame["trade_date"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"])
    if start_date:
        frame = frame[frame["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["trade_date"] <= pd.Timestamp(end_date)]
    if frame.empty:
        return pd.DataFrame()
    for column in ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["underlying_code"] = contract.underlying_code
    frame["underlying_name"] = contract.underlying_name
    frame["option_market"] = OPTION_MARKET
    frame["contract_code"] = contract.contract_code
    frame["contract_month"] = contract.contract_month
    frame["option_side"] = contract.option_side
    frame["strike_price"] = contract.strike_price
    frame["amount"] = None
    frame["open_interest_open"] = frame["open_oi"]
    frame["open_interest"] = frame["close_oi"]
    frame["data_source"] = DATA_SOURCE
    frame["source_url"] = SOURCE_URL_PREFIX + contract.path.as_posix()
    frame["available_time"] = frame["trade_date"].map(_available_time)
    frame["crawl_time"] = datetime.now()
    out = frame[
        [
            "trade_date",
            "underlying_code",
            "underlying_name",
            "option_market",
            "contract_code",
            "contract_month",
            "option_side",
            "strike_price",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest_open",
            "amount",
            "open_interest",
            "data_source",
            "source_url",
            "available_time",
            "crawl_time",
        ]
    ].copy()
    hash_columns = [
        "trade_date",
        "contract_code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest_open",
        "open_interest",
        "data_source",
    ]
    out["raw_hash"] = out.apply(lambda row: _hash_row(row, hash_columns), axis=1)
    return out.drop_duplicates(["trade_date", "contract_code", "data_source"], keep="last")


def _existing_keys(start_date: str | None, end_date: str | None) -> set[tuple[str, str, str]]:
    conditions = ["data_source = :data_source"]
    params: dict[str, Any] = {"data_source": DATA_SOURCE}
    if start_date:
        conditions.append("trade_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("trade_date <= :end_date")
        params["end_date"] = end_date
    frame = read_sql(
        f"""
        SELECT trade_date, contract_code, data_source
        FROM option_contract_daily_raw
        WHERE {' AND '.join(conditions)}
        """,
        params,
    )
    if frame.empty:
        return set()
    return set(zip(frame["trade_date"].astype(str), frame["contract_code"], frame["data_source"]))


def _stored_summary(start_date: str | None, end_date: str | None) -> pd.DataFrame:
    conditions = ["data_source = :data_source"]
    params: dict[str, Any] = {"data_source": DATA_SOURCE}
    if start_date:
        conditions.append("trade_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("trade_date <= :end_date")
        params["end_date"] = end_date
    return read_sql(
        f"""
        SELECT underlying_code, underlying_name, COUNT(*) AS rows,
               COUNT(DISTINCT contract_code) AS contracts,
               MIN(trade_date) AS min_date, MAX(trade_date) AS max_date
        FROM option_contract_daily_raw
        WHERE {' AND '.join(conditions)}
        GROUP BY underlying_code, underlying_name
        ORDER BY underlying_code
        """,
        params,
    )


def collect_option_contract_csv(
    source_dir: str | Path | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    batch_files: int = 500,
    run_quality: bool = True,
) -> dict[str, Any]:
    root = Path(source_dir) if source_dir else _default_source_dir()
    if not root.exists():
        raise FileNotFoundError(f"source directory not found: {root}")

    ingestion = IngestionRun(
        dataset_name="option_contract_daily_raw",
        data_source=DATA_SOURCE,
        requested_start_date=pd.Timestamp(start_date).date() if start_date else None,
        requested_end_date=pd.Timestamp(end_date).date() if end_date else None,
    )
    try:
        files = sorted(root.glob("*.csv"))
        parsed = [parse_contract_file(path) for path in files]
        contracts = [item for item in parsed if item is not None]
        bad_files = [path.name for path, item in zip(files, parsed) if item is None]
        old_keys = _existing_keys(start_date, end_date)
        new_keys: set[tuple[str, str, str]] = set()
        product_contracts: dict[str, set[str]] = {product: set() for product in PRODUCT_META}
        product_rows: dict[str, int] = {product: 0 for product in PRODUCT_META}
        errors: list[dict[str, str]] = []
        buffer: list[pd.DataFrame] = []

        def flush() -> None:
            if not buffer:
                return
            combined = pd.concat(buffer, ignore_index=True)
            buffer.clear()
            upsert_dataframe(
                combined,
                "option_contract_daily_raw",
                ["trade_date", "contract_code", "data_source"],
            )

        for index, contract in enumerate(contracts, 1):
            try:
                raw = pd.read_csv(contract.path, encoding="utf-8-sig")
                frame = normalize_csv_frame(raw, contract, start_date, end_date)
            except Exception as exc:
                errors.append({"file": contract.path.name, "error": f"{type(exc).__name__}: {exc}"})
                continue
            if frame.empty:
                continue
            ingestion.fetched_rows += len(frame)
            product_rows[contract.product] += len(frame)
            product_contracts[contract.product].add(contract.contract_code)
            new_keys.update(zip(frame["trade_date"].astype(str), frame["contract_code"], frame["data_source"]))
            buffer.append(frame)
            if len(buffer) >= batch_files:
                flush()
        flush()

        ingestion.inserted_rows = len(new_keys - old_keys)
        ingestion.updated_rows = len(new_keys & old_keys)
        ingestion.error_rows = len(errors)

        metrics: list[dict[str, Any]] = []
        if run_quality:
            conditions = ["data_source = :data_source"]
            params: dict[str, Any] = {"data_source": DATA_SOURCE}
            if start_date:
                conditions.append("trade_date >= :start_date")
                params["start_date"] = start_date
            if end_date:
                conditions.append("trade_date <= :end_date")
                params["end_date"] = end_date
            stored = read_sql(
                f"SELECT * FROM option_contract_daily_raw WHERE {' AND '.join(conditions)}",
                params,
            )
            metrics = evaluate_dataframe(stored, DATASETS["option_contract_daily_raw"])
            persist_metrics("option_contract_daily_raw", metrics, date.today())

        ingestion.finish("success")
        return {
            "table": "option_contract_daily_raw",
            "data_source": DATA_SOURCE,
            "source_dir": str(root),
            "files": len(files),
            "parsed_files": len(contracts),
            "bad_files": bad_files[:20],
            "rows": ingestion.fetched_rows,
            "inserted_rows": ingestion.inserted_rows,
            "updated_rows": ingestion.updated_rows,
            "product_rows": product_rows,
            "product_contracts": {key: len(value) for key, value in product_contracts.items()},
            "errors": errors[:20],
            "error_count": len(errors),
            "summary": _stored_summary(start_date, end_date).to_dict("records"),
            "quality": metrics,
        }
    except Exception as exc:
        ingestion.error_rows += 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="导入本地 CFFEX 股指期权历史日线 CSV。")
    parser.add_argument("--source-dir", default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--batch-files", type=int, default=500)
    parser.add_argument("--apply-migrations", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    if args.apply_migrations:
        apply_migrations()
    print(
        collect_option_contract_csv(
            source_dir=args.source_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            batch_files=args.batch_files,
            run_quality=not args.skip_quality,
        )
    )


if __name__ == "__main__":
    main()
