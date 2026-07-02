from __future__ import annotations

import argparse
import re
from datetime import date, datetime, time
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.common.config import load_yaml, project_path
from src.common.db import execute_sql, read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.network import disable_env_proxies
from src.quality.check_collection_data import DatasetSpec, evaluate_dataframe, persist_metrics


HEADERS = {"User-Agent": "Mozilla/5.0 data-collectors/2.0"}
MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _fetch_text(url: str) -> str:
    disable_env_proxies()
    response = requests.get(url, headers=HEADERS, timeout=40)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n")


def _month_number(month: str) -> int:
    key = month.lower().strip(".")
    if key not in MONTHS:
        raise ValueError(f"unknown month: {month}")
    return MONTHS[key]


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_english_dates(text: str, default_year: int | None = None) -> list[date]:
    cleaned = re.sub(r"\s+", " ", text)
    dates: set[date] = set()
    month_names = r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?"
    for match in re.finditer(rf"\b({month_names})\.?\s+(\d{{1,2}})(?:\s*[-–]\s*(\d{{1,2}}))?(?:,\s*|\s+)(20\d{{2}})\b", cleaned, re.I):
        year = int(match.group(4))
        month = _month_number(match.group(1))
        day = int(match.group(3) or match.group(2))
        value = _safe_date(year, month, day)
        if value:
            dates.add(value)
    if default_year is not None:
        for match in re.finditer(rf"\b({month_names})\.?\s+(\d{{1,2}})(?:\s*[-–]\s*(\d{{1,2}}))?\b", cleaned, re.I):
            month = _month_number(match.group(1))
            day = int(match.group(3) or match.group(2))
            value = _safe_date(default_year, month, day)
            if value:
                dates.add(value)
    for match in re.finditer(rf"\b(\d{{1,2}})\s+({month_names})\.?\s+(20\d{{2}})\b", cleaned, re.I):
        day = int(match.group(1))
        month = _month_number(match.group(2))
        year = int(match.group(3))
        value = _safe_date(year, month, day)
        if value:
            dates.add(value)
    return sorted(dates)


def extract_dates_from_year_sections(text: str) -> list[date]:
    dates: set[date] = set()
    current_year: int | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        year_match = re.search(r"\b(20\d{2})\b", line)
        if year_match and len(line) <= 80:
            current_year = int(year_match.group(1))
        for value in extract_english_dates(line, default_year=current_year):
            dates.add(value)
    return sorted(dates)


def extract_boj_meeting_dates(url: str, start_date: str, end_date: str) -> list[date]:
    start_year = pd.Timestamp(start_date).year
    tables = pd.read_html(url)
    dates: set[date] = set()
    if not tables:
        return []
    table = tables[0]
    first_column = table.columns[0]
    for value in table[first_column].dropna().astype(str):
        for parsed in extract_english_dates(value, default_year=start_year):
            dates.add(parsed)
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    return sorted(value for value in dates if start <= value <= end)


def extract_fomc_meeting_dates(text: str) -> list[date]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dates: set[date] = set()
    current_year: int | None = None
    current_month: int | None = None
    for line in lines:
        year_match = re.fullmatch(r"(20\d{2}) FOMC Meetings", line)
        if year_match:
            current_year = int(year_match.group(1))
            current_month = None
            continue
        if current_year is None:
            continue
        month_key = line.lower().strip(".")
        if month_key in MONTHS:
            current_month = MONTHS[month_key]
            continue
        date_match = re.fullmatch(r"(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?\*?", line)
        if date_match and current_month is not None:
            day = int(date_match.group(2) or date_match.group(1))
            value = _safe_date(current_year, current_month, day)
            if value:
                dates.add(value)
    return sorted(dates)


def extract_ecb_monetary_policy_dates(text: str) -> list[date]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    dates: set[date] = set()
    pending_date: date | None = None
    for line in lines:
        date_match = re.fullmatch(r"(\d{2})/(\d{2})/(20\d{2})", line)
        if date_match:
            pending_date = date(int(date_match.group(3)), int(date_match.group(2)), int(date_match.group(1)))
            continue
        lower = line.lower()
        if pending_date and "monetary policy meeting" in lower and "non-monetary" not in lower:
            if "day 2" in lower or "press conference" in lower:
                dates.add(pending_date)
            pending_date = None
    return sorted(dates)


def _event_row(source: dict[str, Any], event_date: date, suffix: str = "MEETING") -> dict[str, Any]:
    scheduled = datetime.combine(event_date, time(14, 0))
    key = f"{source['key']}:{suffix}:{event_date.isoformat()}"
    return {
        "event_key": key,
        "event_layer": "macro",
        "event_type": "central_bank_meeting",
        "event_name": source["key"],
        "country_region": source.get("country_region"),
        "scheduled_time": scheduled,
        "actual_start_time": None,
        "actual_end_time": None,
        "is_confirmed": 1,
        "importance": 5,
        "announcement_time": None,
        "available_time": datetime.now(),
        "data_source": source.get("source"),
        "source_url": source.get("source_url"),
        "crawl_time": datetime.now(),
        "note": source.get("scope"),
    }


def fetch_central_bank_events(start_date: str, end_date: str) -> pd.DataFrame:
    cfg = load_yaml(project_path("config", "data_sources_v2.yaml"))["event_calendar"]["central_bank_meetings"]
    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    rows: list[dict[str, Any]] = []
    for source in cfg:
        if source["key"] == "BOJ_MPM":
            dates = extract_boj_meeting_dates(source["source_url"], start_date, end_date)
        elif source["key"] == "FOMC":
            text = _fetch_text(source["source_url"])
            dates = extract_fomc_meeting_dates(text)
        elif source["key"] == "ECB_MPC":
            text = _fetch_text(source["source_url"])
            dates = extract_ecb_monetary_policy_dates(text)
        else:
            text = _fetch_text(source["source_url"])
            dates = extract_dates_from_year_sections(text)
        for event_date in dates:
            if start <= event_date <= end:
                rows.append(_event_row(source, event_date))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["country_region", "scheduled_time"]).drop_duplicates(["event_key", "scheduled_time"])


def _existing_count(start_date: str, end_date: str) -> int:
    result = read_sql(
        """
        SELECT COUNT(*) AS row_count
        FROM event_calendar_raw
        WHERE event_type = 'central_bank_meeting'
          AND scheduled_time BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": f"{end_date} 23:59:59"},
    )
    return int(result.iloc[0]["row_count"])


def _delete_existing_range(start_date: str, end_date: str) -> None:
    execute_sql(
        """
        DELETE FROM event_calendar_raw
        WHERE event_type = 'central_bank_meeting'
          AND scheduled_time BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": f"{end_date} 23:59:59"},
    )


def _quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    spec = DatasetSpec(
        table="event_calendar_raw",
        date_column="scheduled_time",
        unique_columns=("event_key", "scheduled_time"),
        required_columns=("event_key", "event_layer", "event_type", "event_name", "scheduled_time", "available_time", "data_source"),
    )
    metrics = evaluate_dataframe(frame, spec)
    persist_metrics("event_calendar_raw:central_bank_meeting", metrics, date.today())
    return {"metrics": metrics}


def collect_event_calendar(start_date: str | None = None, end_date: str | None = None, future_days: int = 370) -> dict[str, Any]:
    start_date = start_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    end_date = end_date or (pd.Timestamp.today() + pd.Timedelta(days=future_days)).strftime("%Y-%m-%d")
    ingestion = IngestionRun(
        dataset_name="event_calendar_raw:central_bank_meeting",
        data_source="official_central_bank_pages",
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    try:
        frame = fetch_central_bank_events(start_date, end_date)
        if frame.empty:
            raise RuntimeError("event_calendar_raw collected no central bank rows")
        existing = _existing_count(start_date, end_date)
        _delete_existing_range(start_date, end_date)
        upsert_dataframe(frame, "event_calendar_raw", ["event_key", "scheduled_time"])
        ingestion.fetched_rows = len(frame)
        ingestion.inserted_rows = max(0, len(frame) - existing)
        ingestion.updated_rows = min(len(frame), existing)
        ingestion.finish("success")
        return {
            "rows": len(frame),
            "countries": sorted(frame["country_region"].dropna().unique().tolist()),
            "scheduled_time_min": str(frame["scheduled_time"].min()),
            "scheduled_time_max": str(frame["scheduled_time"].max()),
            "quality": _quality_report(frame),
        }
    except Exception as exc:
        ingestion.error_rows = 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集第二批央行固定议程日历。")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--future-days", type=int, default=370)
    args = parser.parse_args()
    print(collect_event_calendar(args.start_date, args.end_date, args.future_days))


if __name__ == "__main__":
    main()
