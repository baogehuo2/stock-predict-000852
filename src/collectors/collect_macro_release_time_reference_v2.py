from __future__ import annotations

import argparse
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.collectors.collect_macro_forecast_calendar_v2 import (
    WALLSTREETCN_CALENDAR_URL,
    _indicator_frequency,
    _indicator_name,
    _infer_period_date,
    _match_mapping,
    _raw_hash,
    fetch_macro_calendar_day,
)
from src.common.config import load_yaml, project_path
from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.quality.check_collection_data import DatasetSpec, evaluate_dataframe, persist_metrics


def _calendar_days(start_date: str, end_date: str) -> list[date]:
    return [value.date() for value in pd.date_range(start_date, end_date, freq="D")]


def normalize_release_time_reference(
    raw: pd.DataFrame,
    mappings: list[dict[str, Any]],
    source_cfg: dict[str, Any],
    crawl_time: datetime,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    required = ["时间", "地区", "事件", "链接"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"macro_info_ws missing columns: {missing}; got={list(raw.columns)}")

    rows: list[dict[str, Any]] = []
    for record in raw.to_dict(orient="records"):
        region = str(record.get("地区", "")).strip()
        event_name = str(record.get("事件", "")).strip()
        mapping = _match_mapping(region, event_name, mappings)
        if not mapping:
            continue
        release_time = pd.to_datetime(record.get("时间"), errors="coerce")
        if pd.isna(release_time):
            continue
        key = str(mapping["key"])
        frequency = _indicator_frequency(key)
        period_date = _infer_period_date(event_name, release_time.to_pydatetime(), frequency)
        rows.append(
            {
                "indicator_key": key,
                "indicator_name": _indicator_name(key),
                "country_region": mapping.get("country_region"),
                "period_date": period_date,
                "release_time": release_time.to_pydatetime(),
                "event_name": event_name,
                "source_type": source_cfg.get("source_type", "economic_calendar"),
                "source_name": source_cfg.get("source", "akshare:macro_info_ws"),
                "is_official": int(source_cfg.get("is_official", 0)),
                "confidence": source_cfg.get("confidence", "calendar_scheduled"),
                "available_time": crawl_time,
                "data_source": source_cfg.get("source", "akshare:macro_info_ws"),
                "source_url": record.get("链接") or source_cfg.get("source_url") or WALLSTREETCN_CALENDAR_URL,
                "crawl_time": crawl_time,
                "raw_hash": _raw_hash(record),
                "note": source_cfg.get(
                    "note",
                    "结构化经济日历发布时间参考；用于校验macro_release_raw.release_time，不作为actual权威源。",
                ),
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(["indicator_key", "period_date", "release_time", "crawl_time"]).drop_duplicates(
        ["indicator_key", "period_date", "release_time", "source_name"],
        keep="last",
    )


def _quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    spec = DatasetSpec(
        table="macro_release_time_reference_raw",
        date_column="period_date",
        unique_columns=("indicator_key", "period_date", "release_time", "source_name"),
        required_columns=(
            "indicator_key",
            "indicator_name",
            "country_region",
            "period_date",
            "release_time",
            "event_name",
            "source_type",
            "source_name",
            "confidence",
            "available_time",
            "data_source",
        ),
    )
    metrics = evaluate_dataframe(frame, spec)
    persist_metrics("macro_release_time_reference_raw", metrics, date.today())
    return {"metrics": metrics}


def compare_release_time_reference(start_date: str, end_date: str) -> dict[str, Any]:
    frame = read_sql(
        """
        SELECT
            m.indicator_key,
            m.period_date,
            m.release_time AS macro_release_time,
            r.release_time AS reference_release_time,
            r.source_name,
            ABS(TIMESTAMPDIFF(MINUTE, m.release_time, r.release_time)) AS diff_minutes
        FROM macro_release_raw m
        JOIN macro_release_time_reference_raw r
          ON r.indicator_key = m.indicator_key
         AND r.period_date = m.period_date
        WHERE m.period_date BETWEEN :start_date AND :end_date
          AND m.actual_value IS NOT NULL
          AND m.data_source <> 'akshare:macro_info_ws'
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        return {
            "matched_actual_rows": 0,
            "diff_over_60min_rows": 0,
            "max_diff_minutes": None,
            "sample": [],
        }
    diff = pd.to_numeric(frame["diff_minutes"], errors="coerce")
    bad = frame[diff > 60].copy()
    return {
        "matched_actual_rows": int(len(frame)),
        "diff_over_60min_rows": int(len(bad)),
        "max_diff_minutes": None if diff.dropna().empty else int(diff.max()),
        "sample": bad.sort_values("diff_minutes", ascending=False).head(20).astype(str).to_dict(orient="records"),
    }


def collect_macro_release_time_reference(
    start_date: str | None = None,
    end_date: str | None = None,
    future_days: int | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(project_path("config", "data_sources_v2.yaml"))["macro_release"]
    source_cfg = cfg.get("release_time_reference", {})
    today = pd.Timestamp.today().date()
    start_date = start_date or today.isoformat()
    if end_date is None:
        future_days = int(future_days if future_days is not None else source_cfg.get("future_days", 30))
        end_date = (pd.Timestamp(start_date).date() + pd.Timedelta(days=future_days)).isoformat()

    ingestion = IngestionRun(
        dataset_name="macro_release_time_reference_raw",
        data_source=source_cfg.get("source", "akshare:macro_info_ws"),
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    try:
        crawl_time = datetime.now()
        mappings = cfg.get("forecast_calendar", {}).get("event_mappings", [])
        frames: list[pd.DataFrame] = []
        failed_days: list[str] = []
        for day in _calendar_days(start_date, end_date):
            try:
                raw = fetch_macro_calendar_day(day)
                frames.append(normalize_release_time_reference(raw, mappings, source_cfg, crawl_time))
            except KeyError as exc:
                if str(exc).strip("'\"") in {"public_date", "items"}:
                    frames.append(pd.DataFrame())
                else:
                    failed_days.append(f"{day.isoformat()}:KeyError:{exc}")
            except Exception as exc:
                failed_days.append(f"{day.isoformat()}:{type(exc).__name__}:{exc}")
        non_empty_frames = [frame for frame in frames if not frame.empty]
        combined = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else pd.DataFrame()
        if combined.empty:
            raise RuntimeError("macro release time reference collected no mapped rows")
        combined = combined.sort_values(["indicator_key", "period_date", "release_time", "crawl_time"]).drop_duplicates(
            ["indicator_key", "period_date", "release_time", "source_name"],
            keep="last",
        )
        upsert_dataframe(
            combined,
            "macro_release_time_reference_raw",
            ["indicator_key", "period_date", "release_time", "source_name"],
        )
        ingestion.fetched_rows = len(combined)
        ingestion.inserted_rows = len(combined)
        ingestion.finish("success")
        return {
            "rows": len(combined),
            "indicators": sorted(combined["indicator_key"].unique().tolist()),
            "period_date_min": str(combined["period_date"].min()),
            "period_date_max": str(combined["period_date"].max()),
            "release_time_min": str(combined["release_time"].min()),
            "release_time_max": str(combined["release_time"].max()),
            "source_type": source_cfg.get("source_type", "economic_calendar"),
            "failed_days": failed_days,
            "comparison": compare_release_time_reference(start_date, end_date),
            "quality": _quality_report(combined),
        }
    except Exception as exc:
        ingestion.error_rows = 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集并校验宏观指标准确release_time参考。")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--future-days", type=int)
    args = parser.parse_args()
    print(collect_macro_release_time_reference(args.start_date, args.end_date, args.future_days))


if __name__ == "__main__":
    main()
