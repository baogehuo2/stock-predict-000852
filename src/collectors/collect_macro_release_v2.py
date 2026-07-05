from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, time
from typing import Any

import pandas as pd
import requests

from src.common.config import get_config, load_yaml, project_path
from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.network import disable_env_proxies
from src.common.secrets import get_secret
from src.quality.check_collection_data import DatasetSpec, evaluate_dataframe, persist_metrics


TUSHARE_URL = "https://api.tushare.pro"


def _raw_hash(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _period_end(value: Any, frequency: str) -> date | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    cn_month = pd.Series([text]).str.extract(r"(?P<year>\d{4})年(?P<month>\d{1,2})月").iloc[0]
    if pd.notna(cn_month.get("year")) and pd.notna(cn_month.get("month")):
        return pd.Period(f"{int(cn_month['year']):04d}-{int(cn_month['month']):02d}", freq="M").end_time.date()
    normalized = text.replace("-", "").replace(".", "").replace("/", "")
    if len(normalized) == 8:
        parsed = pd.Timestamp(normalized)
        if frequency == "monthly":
            return pd.Period(parsed, freq="M").end_time.date()
        if frequency == "quarterly":
            return pd.Period(parsed, freq="Q").end_time.date()
        return parsed.date()
    if frequency == "quarterly":
        if "Q" in text.upper():
            period = pd.Period(text.upper().replace(" ", ""), freq="Q")
            return period.end_time.date()
        if len(normalized) >= 6:
            year = int(normalized[:4])
            quarter = int(normalized[-1])
            return pd.Period(f"{year}Q{quarter}", freq="Q").end_time.date()
    if len(normalized) == 6:
        return pd.Period(normalized, freq="M").end_time.date()
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    if frequency == "monthly":
        return pd.Period(parsed, freq="M").end_time.date()
    if frequency == "quarterly":
        return pd.Period(parsed, freq="Q").end_time.date()
    return parsed.date()


def _estimated_release_time(period_date: date, frequency: str) -> datetime:
    lag_days = 60 if frequency == "quarterly" else 45
    return datetime.combine(period_date + pd.Timedelta(days=lag_days), time(9, 30))


def _period_from_release_date(release_date: date, frequency: str) -> date:
    if frequency == "quarterly":
        return (pd.Period(release_date, freq="Q") - 1).end_time.date()
    if frequency == "annual":
        return (pd.Period(release_date, freq="Y") - 1).end_time.date()
    return (pd.Period(release_date, freq="M") - 1).end_time.date()


def _fetch_tushare(api_name: str) -> pd.DataFrame:
    disable_env_proxies()
    token = get_secret("tushare.token")
    if not token:
        raise RuntimeError("missing tushare.token in encrypted secrets")
    response = requests.post(
        TUSHARE_URL,
        json={"api_name": api_name, "token": token, "params": {}, "fields": ""},
        timeout=40,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"tushare {api_name} request failed: {payload}")
    data = payload.get("data", {})
    return pd.DataFrame(data.get("items", []), columns=data.get("fields", []))


def normalize_tushare_indicator(raw: pd.DataFrame, item: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    period_field = str(item["period_field"]).lower()
    value_field = str(item["value_field"]).lower()
    if period_field not in frame.columns or value_field not in frame.columns:
        raise ValueError(f"{item['key']} missing fields: {period_field}, {value_field}; got={list(frame.columns)}")
    rows: list[dict[str, Any]] = []
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    for record in frame.to_dict(orient="records"):
        period_date = _period_end(record.get(period_field), item.get("frequency", "monthly"))
        if period_date is None or period_date < start or period_date > end:
            continue
        actual = pd.to_numeric(record.get(value_field), errors="coerce")
        if pd.isna(actual):
            continue
        release_time = _estimated_release_time(period_date, item.get("frequency", "monthly"))
        rows.append(
            {
                "indicator_key": item["key"],
                "indicator_name": item["name"],
                "country_region": "CN",
                "frequency": item.get("frequency"),
                "period_date": period_date,
                "release_time": release_time,
                "actual_value": float(actual),
                "forecast_value": None,
                "previous_value": None,
                "revised_previous_value": None,
                "unit": item.get("unit"),
                "seasonal_adjustment": None,
                "revision_no": 0,
                "is_preliminary": 1,
                "available_time": release_time,
                "data_source": f"tushare:{item['api_name']}",
                "source_url": item.get("source_url", "https://tushare.pro/"),
                "crawl_time": datetime.now(),
                "raw_hash": _raw_hash(record),
                "note": "Tushare宏观实际值接口未返回官方发布时间；release_time/available_time按保守滞后估算，待发布日历校验补充。",
            }
        )
    return pd.DataFrame(rows)


def _pick_date_column(columns: list[str]) -> str | None:
    candidates = ["日期", "时间", "date", "统计时间", "公布时间", "月份"]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return columns[0] if columns else None


def _pick_value_column(columns: list[str]) -> str | None:
    preferred = ["今值", "现值", "公布值", "实际值", "value", "数值", "指标值"]
    for candidate in preferred:
        if candidate in columns:
            return candidate
    for column in reversed(columns):
        if column not in {"日期", "时间", "月份", "统计时间", "公布时间"}:
            return column
    return None


def normalize_akshare_us_indicator(raw: pd.DataFrame, item: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    date_column = _pick_date_column(list(frame.columns))
    value_column = _pick_value_column(list(frame.columns))
    if not date_column or not value_column:
        raise ValueError(f"{item['key']} cannot infer date/value columns: {list(frame.columns)}")
    rows: list[dict[str, Any]] = []
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    for record in frame.to_dict(orient="records"):
        if item.get("date_role") == "release":
            release_date = _period_end(record.get(date_column), "daily")
            if release_date is None:
                continue
            release_time = datetime.combine(release_date, time(21, 30))
            period_date = _period_from_release_date(release_date, item.get("frequency", "monthly"))
            filter_date = release_date
        else:
            period_date = _period_end(record.get(date_column), item.get("frequency", "monthly"))
            if period_date is None:
                continue
            release_time = _estimated_release_time(period_date, item.get("frequency", "monthly"))
            filter_date = period_date
        if filter_date < start or filter_date > end:
            continue
        actual = pd.to_numeric(record.get(value_column), errors="coerce")
        if pd.isna(actual):
            continue
        rows.append(
            {
                "indicator_key": item["key"],
                "indicator_name": item["name"],
                "country_region": "US",
                "frequency": item.get("frequency"),
                "period_date": period_date,
                "release_time": release_time,
                "actual_value": float(actual),
                "forecast_value": None,
                "previous_value": None,
                "revised_previous_value": None,
                "unit": item.get("unit"),
                "seasonal_adjustment": None,
                "revision_no": 0,
                "is_preliminary": 1,
                "available_time": release_time,
                "data_source": f"akshare:{item['function']}",
                "source_url": "https://akshare.akfamily.xyz/data/macro/macro.html",
                "crawl_time": datetime.now(),
                "raw_hash": _raw_hash(record),
                "note": "AKShare美国宏观接口按配置区分统计期日期和发布日期；缺少准确发布时间时使用保守估算。",
            }
        )
    return pd.DataFrame(rows)


def normalize_akshare_cn_indicator(raw: pd.DataFrame, item: dict[str, Any], start_date: str, end_date: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.copy()
    period_field = item["period_field"]
    value_field = item["value_field"]
    if period_field not in frame.columns or value_field not in frame.columns:
        raise ValueError(f"{item['key']} missing fields: {period_field}, {value_field}; got={list(frame.columns)}")
    rows: list[dict[str, Any]] = []
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    for record in frame.to_dict(orient="records"):
        if item.get("date_role") == "release":
            release_date = _period_end(record.get(period_field), "daily")
            if release_date is None:
                continue
            release_time = datetime.combine(release_date, time(10, 0))
            period_date = _period_from_release_date(release_date, item.get("frequency", "monthly"))
            filter_date = release_date
        else:
            period_date = _period_end(record.get(period_field), item.get("frequency", "monthly"))
            if period_date is None:
                continue
            release_time = _estimated_release_time(period_date, item.get("frequency", "monthly"))
            filter_date = period_date
        if filter_date < start or filter_date > end:
            continue
        actual = pd.to_numeric(record.get(value_field), errors="coerce")
        if pd.isna(actual):
            continue
        forecast = pd.to_numeric(record.get(item.get("forecast_field")), errors="coerce") if item.get("forecast_field") else None
        previous = pd.to_numeric(record.get(item.get("previous_field")), errors="coerce") if item.get("previous_field") else None
        rows.append(
            {
                "indicator_key": item["key"],
                "indicator_name": item["name"],
                "country_region": "CN",
                "frequency": item.get("frequency"),
                "period_date": period_date,
                "release_time": release_time,
                "actual_value": float(actual),
                "forecast_value": None if forecast is None or pd.isna(forecast) else float(forecast),
                "previous_value": None if previous is None or pd.isna(previous) else float(previous),
                "revised_previous_value": None,
                "unit": item.get("unit"),
                "seasonal_adjustment": None,
                "revision_no": 0,
                "is_preliminary": 1,
                "available_time": release_time,
                "data_source": f"akshare:{item['function']}",
                "source_url": "https://akshare.akfamily.xyz/data/macro/macro.html",
                "crawl_time": datetime.now(),
                "raw_hash": _raw_hash(record),
                "note": "Tushare未确认稳定接口，按AKShare结构化接口兜底；日期含义按配置区分统计期和发布时间。",
            }
        )
    return pd.DataFrame(rows)


def _load_items() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cfg = load_yaml(project_path("config", "data_sources_v2.yaml"))["macro_release"]
    return cfg.get("cn_indicators", []) + cfg.get("cn_akshare_fallback_indicators", []), cfg.get("us_indicators", [])


def _existing_count(start_date: str, end_date: str) -> int:
    result = read_sql(
        """
        SELECT COUNT(*) AS row_count
        FROM macro_release_raw
        WHERE period_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
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
    persist_metrics("macro_release_raw", metrics, date.today())
    return {"metrics": metrics}


def collect_macro_release(start_date: str | None = None, end_date: str | None = None, include_us: bool = True) -> dict[str, Any]:
    config = get_config()
    start_date = start_date or config["project"]["start_date"]
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    ingestion = IngestionRun(
        dataset_name="macro_release_raw",
        data_source="tushare+akshare",
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    try:
        cn_items, us_items = _load_items()
        frames: list[pd.DataFrame] = []
        ak_module = None
        for item in cn_items:
            if "api_name" in item:
                raw = _fetch_tushare(item["api_name"])
                frames.append(normalize_tushare_indicator(raw, item, start_date, end_date))
            else:
                if ak_module is None:
                    disable_env_proxies()
                    import akshare as ak

                    ak_module = ak
                raw = getattr(ak_module, item["function"])()
                frames.append(normalize_akshare_cn_indicator(raw, item, start_date, end_date))
        if include_us:
            disable_env_proxies()
            import akshare as ak

            for item in us_items:
                raw = getattr(ak, item["function"])()
                frames.append(normalize_akshare_us_indicator(raw, item, start_date, end_date))
        combined = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
        if combined.empty:
            raise RuntimeError("macro_release_raw collected no rows")
        combined = combined.sort_values(["indicator_key", "period_date", "release_time", "crawl_time"]).drop_duplicates(
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
            "period_date_min": str(combined["period_date"].min()),
            "period_date_max": str(combined["period_date"].max()),
            "quality": _quality_report(combined),
        }
    except Exception as exc:
        ingestion.error_rows = 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集第二批中美宏观实际值/vintage。")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--no-us", action="store_true")
    args = parser.parse_args()
    print(collect_macro_release(args.start_date, args.end_date, include_us=not args.no_us))


if __name__ == "__main__":
    main()
