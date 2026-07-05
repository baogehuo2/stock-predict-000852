from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import get_config
from src.common.db import execute_sql, read_sql, upsert_dataframe
from src.common.logger import get_logger
from src.common.network import disable_env_proxies


logger = get_logger(__name__)


def stable_news_id(source: str, publish_date: date, title: str, url: str = "") -> str:
    key = f"{source}|{publish_date.isoformat()}|{title}|{url}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def iter_dates(start_date: str, end_date: str) -> Iterable[date]:
    current = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    while current <= end:
        yield current
        current += timedelta(days=1)


def _keyword_groups_from_config(cfg: dict) -> dict[str, list[str]]:
    news_cfg = cfg.get("news", {})
    groups = news_cfg.get("keyword_groups") or {}
    if groups:
        return {str(group): [str(keyword) for keyword in keywords] for group, keywords in groups.items()}
    keywords = news_cfg.get("keywords", [])
    return {"legacy": [str(keyword) for keyword in keywords]}


def _parse_groups(value: str | None, fallback: list[str]) -> list[str]:
    if not value:
        return fallback
    return [item.strip() for item in value.split(",") if item.strip()]


def _match_keyword_groups(text: str, keyword_groups: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    matched_keywords: list[str] = []
    matched_groups: list[str] = []
    for group, keywords in keyword_groups.items():
        group_hits = [keyword for keyword in keywords if keyword and keyword in text]
        if group_hits:
            matched_groups.append(group)
            matched_keywords.extend(group_hits)
    return sorted(set(matched_keywords)), matched_groups


def _match_keywords(text: str, keyword_groups: dict[str, list[str]], keyword_mode: str) -> tuple[bool, list[str], list[str]]:
    matched_keywords, matched_groups = _match_keyword_groups(text, keyword_groups)
    if keyword_mode == "all":
        return True, matched_keywords, matched_groups
    return bool(matched_keywords), matched_keywords, matched_groups


def _json_list(values: list[str]) -> str:
    return json.dumps(values, ensure_ascii=False)


def ensure_news_raw_columns() -> None:
    existing = read_sql(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'news_raw'"
    )
    columns = set(existing["COLUMN_NAME"].tolist())
    if "matched_keywords" not in columns:
        execute_sql("ALTER TABLE news_raw ADD COLUMN matched_keywords TEXT NULL AFTER url")
    if "matched_groups" not in columns:
        execute_sql("ALTER TABLE news_raw ADD COLUMN matched_groups TEXT NULL AFTER matched_keywords")


def _first_existing(row: pd.Series, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col]).strip()
    return None


def fetch_cctv_news(day: date, keyword_groups: dict[str, list[str]], keyword_mode: str = "match") -> list[dict]:
    disable_env_proxies()
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is not installed. Run: pip install -r requirements.txt") from exc

    raw = ak.news_cctv(date=day.strftime("%Y%m%d"))
    if raw is None or raw.empty:
        return []

    rows: list[dict] = []
    for _, item in raw.iterrows():
        title = _first_existing(item, ["title", "标题", "新闻标题"]) or ""
        content = _first_existing(item, ["content", "内容", "新闻内容", "detail", "详情"]) or title
        url = _first_existing(item, ["url", "链接", "新闻链接"]) or ""
        text = f"{title}\n{content}"
        keep, matched_keywords, matched_groups = _match_keywords(text, keyword_groups, keyword_mode)
        if not title or not keep:
            continue
        publish_time = datetime.combine(day, datetime.min.time())
        rows.append(
            {
                "news_id": stable_news_id("cctv_news", day, title, url),
                "source": "CCTV新闻联播",
                "publish_time": publish_time,
                "trade_date": day,
                "title": title[:500],
                "content": content,
                "url": url,
                "matched_keywords": _json_list(matched_keywords),
                "matched_groups": _json_list(matched_groups),
            }
        )
    return rows


def fetch_baidu_economic_calendar(day: date, keyword_groups: dict[str, list[str]], keyword_mode: str = "match") -> list[dict]:
    disable_env_proxies()
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is not installed. Run: pip install -r requirements.txt") from exc

    raw = ak.news_economic_baidu(date=day.strftime("%Y%m%d"))
    if raw is None or raw.empty:
        return []

    rows: list[dict] = []
    for _, item in raw.iterrows():
        title = _first_existing(item, ["事件", "title", "指标名称", "名称"]) or ""
        content_parts = [str(v) for v in item.to_dict().values() if pd.notna(v)]
        content = " | ".join(content_parts)
        keep, matched_keywords, matched_groups = _match_keywords(f"{title}\n{content}", keyword_groups, keyword_mode)
        if not title or not keep:
            continue
        publish_time = datetime.combine(day, datetime.min.time())
        rows.append(
            {
                "news_id": stable_news_id("baidu_economic_calendar", day, title, content),
                "source": "百度股市通经济日历",
                "publish_time": publish_time,
                "trade_date": day,
                "title": title[:500],
                "content": content,
                "url": "https://finance.baidu.com/calendar",
                "matched_keywords": _json_list(matched_keywords),
                "matched_groups": _json_list(matched_groups),
            }
        )
    return rows


SOURCE_FETCHERS = {
    "cctv": fetch_cctv_news,
    "baidu_economic": fetch_baidu_economic_calendar,
}


def collect_news_history(
    start_date: str,
    end_date: str,
    sources: list[str] | None = None,
    keyword_mode: str = "match",
    sleep_seconds: float = 0.5,
    batch_days: int = 10,
    extract_events: bool = False,
    event_priority_groups: list[str] | None = None,
    event_limit_per_day: int | None = 0,
    event_batch_size: int = 20,
    llm_retries: int = 3,
    retry_wait: float = 10.0,
) -> int:
    cfg = get_config()
    keyword_groups = _keyword_groups_from_config(cfg)
    sources = sources or ["cctv", "baidu_economic"]
    unknown = [source for source in sources if source not in SOURCE_FETCHERS]
    if unknown:
        raise ValueError(f"Unsupported news sources: {unknown}. Supported: {sorted(SOURCE_FETCHERS)}")

    total = 0
    buffer: list[dict] = []
    ensure_news_raw_columns()
    for idx, day in enumerate(iter_dates(start_date, end_date), start=1):
        for source in sources:
            try:
                rows = SOURCE_FETCHERS[source](day, keyword_groups, keyword_mode)
                buffer.extend(rows)
                logger.info("news history source=%s date=%s rows=%s", source, day, len(rows))
            except Exception as exc:
                logger.exception("failed news history source=%s date=%s: %s", source, day, exc)

        if buffer and idx % batch_days == 0:
            total += upsert_dataframe(pd.DataFrame(buffer), "news_raw", ["news_id"])
            buffer = []

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if buffer:
        total += upsert_dataframe(pd.DataFrame(buffer), "news_raw", ["news_id"])
    logger.info("finish news history total=%s", total)
    if extract_events:
        from src.llm.extract_event import audit_missing_priority_news, extract_events_for_history

        priority_groups = event_priority_groups or list(keyword_groups.keys())
        logger.info(
            "start event extraction from collector start=%s end=%s limit_per_day=%s priority_groups=%s",
            start_date,
            end_date,
            event_limit_per_day,
            priority_groups,
        )
        extracted = extract_events_for_history(
            start_date=start_date,
            end_date=end_date,
            limit_per_day=event_limit_per_day,
            batch_size=event_batch_size,
            include_unmatched=keyword_mode == "all",
            fail_fast=False,
            llm_retries=llm_retries,
            retry_wait=retry_wait,
            priority_groups=priority_groups,
        )
        missing = audit_missing_priority_news(start_date, end_date, priority_groups)
        logger.info("finish event extraction extracted=%s missing_priority=%s", extracted, missing)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill historical news with real publish dates.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--source", action="append", choices=sorted(SOURCE_FETCHERS), help="Repeatable. Defaults to cctv + baidu_economic.")
    parser.add_argument("--keyword-mode", choices=["match", "all"], default="match", help="match: keep only configured keywords; all: keep all rows.")
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--batch-days", type=int, default=10)
    parser.add_argument("--extract-events", action="store_true", help="Run event extraction after collection using the same date range.")
    parser.add_argument(
        "--event-priority-groups",
        help="Comma-separated priority groups for extraction. Defaults to all configured news.keyword_groups.",
    )
    parser.add_argument(
        "--event-limit-per-day",
        type=int,
        default=0,
        help="Daily extraction cap. Use 0 for no daily cap so matched priority news is not dropped.",
    )
    parser.add_argument("--event-batch-size", type=int, default=20)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=10.0)
    args = parser.parse_args()
    cfg = get_config()
    keyword_groups = _keyword_groups_from_config(cfg)
    print(
        collect_news_history(
            start_date=args.start_date,
            end_date=args.end_date,
            sources=args.source,
            keyword_mode=args.keyword_mode,
            sleep_seconds=args.sleep,
            batch_days=args.batch_days,
            extract_events=args.extract_events,
            event_priority_groups=_parse_groups(args.event_priority_groups, list(keyword_groups.keys())),
            event_limit_per_day=args.event_limit_per_day,
            event_batch_size=args.event_batch_size,
            llm_retries=args.llm_retries,
            retry_wait=args.retry_wait,
        )
    )


if __name__ == "__main__":
    main()
