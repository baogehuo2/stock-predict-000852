from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date, datetime
from typing import Any

import pandas as pd

from src.common.config import load_yaml, project_path
from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.network import disable_env_proxies
from src.quality.check_collection_data import DatasetSpec, evaluate_dataframe, persist_metrics


WALLSTREETCN_CALENDAR_URL = "https://wallstreetcn.com/calendar"


def _raw_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_float(value: Any) -> float | None:
    number = pd.to_numeric(value, errors="coerce")
    return None if pd.isna(number) else float(number)


def _infer_period_date(event_name: str, release_time: datetime, frequency: str) -> date:
    match = re.search(r"(\d{1,2})月", event_name)
    if match:
        month = int(match.group(1))
        year = release_time.year
        if month > release_time.month:
            year -= 1
        return pd.Period(f"{year}-{month:02d}", freq="M").end_time.date()
    if frequency == "quarterly":
        return (pd.Period(release_time.date(), freq="Q") - 1).end_time.date()
    return (pd.Period(release_time.date(), freq="M") - 1).end_time.date()


def _match_mapping(region: str, event_name: str, mappings: list[dict[str, Any]]) -> dict[str, Any] | None:
    match_text = f"{region}{event_name}"
    for item in mappings:
        include = [str(value) for value in item.get("include", [])]
        exclude = [str(value) for value in item.get("exclude", [])]
        if include and not all(token in match_text for token in include):
            continue
        if any(token in match_text for token in exclude):
            continue
        return item
    return None


def _indicator_name(key: str) -> str:
    cfg = load_yaml(project_path("config", "data_sources_v2.yaml"))["macro_release"]
    for group in ("cn_indicators", "cn_akshare_fallback_indicators", "us_indicators"):
        for item in cfg.get(group, []):
            if item.get("key") == key:
                return str(item.get("name", key))
    return key


def _indicator_frequency(key: str) -> str:
    cfg = load_yaml(project_path("config", "data_sources_v2.yaml"))["macro_release"]
    for group in ("cn_indicators", "cn_akshare_fallback_indicators", "us_indicators"):
        for item in cfg.get(group, []):
            if item.get("key") == key:
                return str(item.get("frequency", "monthly"))
    return "monthly"


def _indicator_unit(key: str) -> str | None:
    cfg = load_yaml(project_path("config", "data_sources_v2.yaml"))["macro_release"]
    for group in ("cn_indicators", "cn_akshare_fallback_indicators", "us_indicators"):
        for item in cfg.get(group, []):
            if item.get("key") == key:
                return item.get("unit")
    return None


def fetch_macro_calendar_day(day: date) -> pd.DataFrame:
    disable_env_proxies()
    import akshare as ak

    return ak.macro_info_ws(date=day.strftime("%Y%m%d"))


def normalize_macro_forecast(raw: pd.DataFrame, mappings: list[dict[str, Any]], crawl_time: datetime) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    required = ["时间", "地区", "事件", "预期", "前值", "链接"]
    missing = [column for column in required if column not in raw.columns]
    if missing:
        raise ValueError(f"macro_info_ws missing columns: {missing}; got={list(raw.columns)}")
    for record in raw.to_dict(orient="records"):
        region = str(record.get("地区", "")).strip()
        event_name = str(record.get("事件", "")).strip()
        mapping = _match_mapping(region, event_name, mappings)
        if not mapping:
            continue
        release_time = pd.to_datetime(record.get("时间"), errors="coerce")
        if pd.isna(release_time):
            continue
        key = mapping["key"]
        frequency = _indicator_frequency(key)
        period_date = _infer_period_date(event_name, release_time.to_pydatetime(), frequency)
        forecast = _to_float(record.get("预期"))
        previous = _to_float(record.get("前值"))
        if forecast is None and previous is None:
            continue
        rows.append(
            {
                "indicator_key": key,
                "indicator_name": _indicator_name(key),
                "country_region": mapping.get("country_region"),
                "frequency": frequency,
                "period_date": period_date,
                "release_time": release_time.to_pydatetime(),
                "actual_value": None,
                "forecast_value": forecast,
                "previous_value": previous,
                "revised_previous_value": None,
                "unit": _indicator_unit(key),
                "seasonal_adjustment": None,
                "revision_no": 0,
                "is_preliminary": 1,
                "available_time": crawl_time,
                "data_source": "akshare:macro_info_ws",
                "source_url": record.get("链接") or WALLSTREETCN_CALENDAR_URL,
                "crawl_time": crawl_time,
                "raw_hash": _raw_hash(record),
                "note": "第三方经济日历预期/前值；仅用于forecast/previous补充，不作为actual权威源；period_date由事件文本保守推断。",
            }
        )
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(["indicator_key", "release_time", "crawl_time"]).drop_duplicates(
        ["indicator_key", "period_date", "release_time", "revision_no"],
        keep="last",
    )


def _calendar_days(start_date: str, end_date: str) -> list[date]:
    return [value.date() for value in pd.date_range(start_date, end_date, freq="D")]


def _existing_count(start_date: str, end_date: str) -> int:
    result = read_sql(
        """
        SELECT COUNT(*) AS row_count
        FROM macro_release_raw
        WHERE release_time BETWEEN :start_date AND :end_date
          AND data_source = 'akshare:macro_info_ws'
        """,
        {"start_date": f"{start_date} 00:00:00", "end_date": f"{end_date} 23:59:59"},
    )
    return int(result.iloc[0]["row_count"])


def _quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    spec = DatasetSpec(
        table="macro_release_raw",
        date_column="period_date",
        unique_columns=("indicator_key", "period_date", "release_time", "revision_no"),
        required_columns=("indicator_key", "indicator_name", "country_region", "period_date", "release_time", "available_time", "data_source"),
    )
    metrics = evaluate_dataframe(frame, spec)
    persist_metrics("macro_release_forecast_calendar", metrics, date.today())
    return {"metrics": metrics}


def collect_macro_forecast_calendar(
    start_date: str | None = None,
    end_date: str | None = None,
    future_days: int | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(project_path("config", "data_sources_v2.yaml"))["macro_release"]["forecast_calendar"]
    today = pd.Timestamp.today().date()
    start_date = start_date or today.isoformat()
    if end_date is None:
        future_days = int(future_days if future_days is not None else cfg.get("future_days", 30))
        end_date = (pd.Timestamp(start_date).date() + pd.Timedelta(days=future_days)).isoformat()
    ingestion = IngestionRun(
        dataset_name="macro_release_forecast_calendar",
        data_source="akshare:macro_info_ws",
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    try:
        crawl_time = datetime.now()
        mappings = cfg.get("event_mappings", [])
        frames = []
        failed_days: list[str] = []
        for day in _calendar_days(start_date, end_date):
            try:
                raw = fetch_macro_calendar_day(day)
                frames.append(normalize_macro_forecast(raw, mappings, crawl_time))
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
            raise RuntimeError("macro forecast calendar collected no mapped rows")
        combined = combined.sort_values(["indicator_key", "release_time", "crawl_time"]).drop_duplicates(
            ["indicator_key", "period_date", "release_time", "revision_no"],
            keep="last",
        )
        existing = _existing_count(start_date, end_date)
        upsert_dataframe(combined, "macro_release_raw", ["indicator_key", "period_date", "release_time", "revision_no"])
        ingestion.fetched_rows = len(combined)
        ingestion.inserted_rows = max(0, len(combined) - existing)
        ingestion.updated_rows = min(len(combined), existing)
        ingestion.finish("success")
        return {
            "rows": len(combined),
            "indicators": sorted(combined["indicator_key"].unique().tolist()),
            "release_time_min": str(combined["release_time"].min()),
            "release_time_max": str(combined["release_time"].max()),
            "forecast_non_null": int(combined["forecast_value"].notna().sum()),
            "previous_non_null": int(combined["previous_value"].notna().sum()),
            "failed_days": failed_days,
            "quality": _quality_report(combined),
        }
    except Exception as exc:
        ingestion.error_rows = 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集第二批宏观经济日历forecast/previous。")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--future-days", type=int)
    args = parser.parse_args()
    print(collect_macro_forecast_calendar(args.start_date, args.end_date, args.future_days))


if __name__ == "__main__":
    main()
