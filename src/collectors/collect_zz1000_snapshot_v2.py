from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd
import requests

from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.network import disable_env_proxies
from src.common.secrets import get_secret
from src.quality.check_collection_data import (
    DATASETS,
    evaluate_constituent_snapshot,
    evaluate_dataframe,
    persist_metrics,
)


INDEX_CODE = "000852"
INDEX_CODE_TUSHARE = "000852.SH"
DATA_SOURCE = "tushare:index_weight"


def _content_hash(frame: pd.DataFrame) -> str:
    records = frame.sort_values("stock_code")[
        ["stock_code", "stock_name", "exchange", "weight"]
    ].to_dict(orient="records")
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _latest_trade_date(frame: pd.DataFrame) -> pd.Timestamp:
    trade_dates = pd.to_datetime(frame["trade_date"], errors="coerce")
    trade_dates = trade_dates.dropna()
    if trade_dates.empty:
        raise ValueError("tushare index_weight returned no valid trade_date values")
    return trade_dates.max()


def normalize_snapshot(
    raw: pd.DataFrame,
    snapshot_date: date,
    available_time: datetime,
) -> pd.DataFrame:
    required = {"index_code", "con_code", "trade_date", "weight"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"unexpected tushare columns, missing: {sorted(missing)}")

    frame = raw.copy()
    frame = frame[frame["index_code"].astype(str).eq(INDEX_CODE_TUSHARE)].copy()
    if frame.empty:
        raise ValueError("tushare index_weight returned no CSI 1000 rows")

    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.date
    frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce") / 100.0
    frame["stock_code"] = frame["con_code"].astype(str).str.zfill(6)
    if "con_name" in frame.columns:
        frame["stock_name"] = frame["con_name"].astype(str).str.strip()
    elif "name" in frame.columns:
        frame["stock_name"] = frame["name"].astype(str).str.strip()
    else:
        frame["stock_name"] = ""
    frame["exchange"] = frame["stock_code"].map(
        lambda code: "SZSE" if code.startswith(("0", "2", "3")) else "SSE"
    )
    frame["constituent_file_date"] = frame["trade_date"]
    frame["weight_file_date"] = frame["trade_date"]
    frame["source_effective_date"] = frame["trade_date"]
    frame["snapshot_date"] = snapshot_date
    frame["weight_unit"] = "decimal"
    frame["data_source"] = DATA_SOURCE
    frame["source_url"] = "https://tushare.pro/document/2?doc_id=96"
    frame["available_time"] = available_time
    frame["crawl_time"] = available_time
    content_hash = _content_hash(frame)
    frame["snapshot_id"] = f"{INDEX_CODE}-{snapshot_date:%Y%m%d}-{content_hash[:16]}"
    frame["raw_hash"] = frame.apply(
        lambda row: hashlib.sha256(
            json.dumps(
                {
                    "stock_code": row["stock_code"],
                    "stock_name": row["stock_name"],
                    "exchange": row["exchange"],
                    "weight": row["weight"],
                    "constituent_file_date": row["constituent_file_date"],
                    "weight_file_date": row["weight_file_date"],
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            ).encode("utf-8")
        ).hexdigest(),
        axis=1,
    )
    columns = [
        "snapshot_id",
        "snapshot_date",
        "index_code",
        "stock_code",
        "stock_name",
        "exchange",
        "weight",
        "weight_unit",
        "source_effective_date",
        "constituent_file_date",
        "weight_file_date",
        "data_source",
        "source_url",
        "available_time",
        "crawl_time",
        "raw_hash",
    ]
    frame = frame[columns].sort_values("stock_code").reset_index(drop=True)
    if len(frame) != 1000:
        raise ValueError(f"tushare CSI 1000 snapshot must contain 1000 members, got {len(frame)}")
    return frame


def _fetch_tushare_index_weight(start_date: str | None = None, end_date: str | None = None, trade_date: str | None = None) -> pd.DataFrame:
    disable_env_proxies()
    token = get_secret("tushare.token")
    if not token:
        raise RuntimeError("missing tushare.token in encrypted secrets")
    session = requests.Session()
    session.trust_env = False
    response = session.post(
        "https://api.tushare.pro",
        json={
            "api_name": "index_weight",
            "token": token,
            "params": {
                "index_code": INDEX_CODE_TUSHARE,
                **({"start_date": start_date, "end_date": end_date} if start_date and end_date else {}),
                **({"trade_date": trade_date} if trade_date else {}),
            },
            "fields": "index_code,con_code,con_name,trade_date,weight",
        },
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if int(payload.get("code", 0)) not in (0, 200):
        raise RuntimeError(f"tushare index_weight request failed: {payload}")
    data = payload.get("data") or {}
    fields = data.get("fields") or []
    items = data.get("items") or []
    frame = pd.DataFrame(items, columns=fields)
    if frame.empty:
        raise RuntimeError("tushare index_weight returned no rows")
    return frame


def fetch_current_snapshot(snapshot_date: date | None = None) -> pd.DataFrame:
    fetched_at = datetime.now()
    end_date = fetched_at.date()
    start_date = end_date - timedelta(days=120)
    raw = _fetch_tushare_index_weight(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
    latest_trade_date = _latest_trade_date(raw)
    latest = raw[pd.to_datetime(raw["trade_date"], errors="coerce").dt.date.eq(latest_trade_date.date())].copy()
    if latest.empty:
        raise RuntimeError("tushare index_weight returned no latest-date rows")
    return normalize_snapshot(latest, snapshot_date or latest_trade_date.date(), fetched_at)


def collect_current_snapshot(run_quality: bool = True) -> dict[str, Any]:
    ingestion = IngestionRun(
        dataset_name="index_constituent_snapshot_raw",
        data_source=DATA_SOURCE,
        requested_start_date=date.today(),
        requested_end_date=date.today(),
    )
    try:
        frame = fetch_current_snapshot()
        ingestion.fetched_rows = len(frame)
        snapshot_id = str(frame["snapshot_id"].iloc[0])
        existing = read_sql(
            "SELECT stock_code FROM index_constituent_snapshot_raw WHERE snapshot_id=:snapshot_id",
            {"snapshot_id": snapshot_id},
        )
        old_codes = set(existing["stock_code"].astype(str)) if not existing.empty else set()
        new_codes = set(frame["stock_code"])
        ingestion.inserted_rows = len(new_codes - old_codes)
        ingestion.updated_rows = len(new_codes & old_codes)
        upsert_dataframe(
            frame,
            "index_constituent_snapshot_raw",
            ["snapshot_id", "index_code", "stock_code"],
        )
        metrics: list[dict[str, Any]] = []
        if run_quality:
            stored = read_sql(
                "SELECT * FROM index_constituent_snapshot_raw WHERE snapshot_id=:snapshot_id",
                {"snapshot_id": snapshot_id},
            )
            metrics = evaluate_dataframe(stored, DATASETS["index_constituent_snapshot_raw"])
            metrics.extend(evaluate_constituent_snapshot(stored))
            persist_metrics("index_constituent_snapshot_raw", metrics, date.today())
        ingestion.finish("success")
        return {
            "snapshot_id": snapshot_id,
            "rows": len(frame),
            "inserted_rows": ingestion.inserted_rows,
            "updated_rows": ingestion.updated_rows,
            "constituent_file_date": str(frame["constituent_file_date"].iloc[0]),
            "weight_file_date": str(frame["weight_file_date"].iloc[0]),
            "weight_sum": float(frame["weight"].sum()),
            "quality_metrics": metrics,
        }
    except Exception as exc:
        ingestion.error_rows += 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="保存中证1000当前成分及权重快照（Tushare）")
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    print(collect_current_snapshot(run_quality=not args.skip_quality))


if __name__ == "__main__":
    main()
