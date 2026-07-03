from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, time, timedelta
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.common.config import get_config, load_yaml, project_path
from src.common.db import upsert_dataframe
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.collectors.collect_news_history import (
    _json_list,
    _keyword_groups_from_config,
    _match_keywords,
    ensure_news_raw_columns,
)


logger = get_logger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0"}
DEFAULT_AKSHARE_SYMBOLS = [
    "000300",
    "000905",
    "399006",
    "000688",
    "000016",
    "159845",
    "510300",
    "510500",
    "159915",
    "588000",
]
DEFAULT_LOOKBACK_DAYS = 3


def _news_id(source: str, url: str, title: str) -> str:
    return hashlib.sha1(f"{source}|{url}|{title}".encode("utf-8")).hexdigest()


def _parse_publish_time(value: object) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    return pd.Timestamp(value).to_pydatetime()


def _collection_window(
    *,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[datetime, datetime]:
    window_end = end_time or datetime.now()
    window_start = start_time or datetime.combine(window_end.date(), time.min) - timedelta(days=lookback_days)
    return window_start, window_end


def _akshare_symbols(cfg: dict) -> list[str]:
    news_cfg = cfg.get("news", {})
    configured = news_cfg.get("akshare_symbols")
    if configured:
        return [str(item) for item in configured]
    return DEFAULT_AKSHARE_SYMBOLS


def normalize_akshare_stock_news(
    raw: pd.DataFrame,
    symbol: str,
    keyword_groups: dict[str, list[str]],
    window_start: datetime,
    window_end: datetime,
) -> list[dict]:
    if raw is None or raw.empty:
        return []
    rows: list[dict] = []
    for _, item in raw.iterrows():
        title = str(item.get("新闻标题") or item.get("title") or "").strip()
        content = str(item.get("新闻内容") or item.get("content") or title).strip()
        url = str(item.get("新闻链接") or item.get("url") or "").strip()
        publisher = str(item.get("文章来源") or "").strip()
        publish_time = _parse_publish_time(item.get("发布时间"))
        if not title or publish_time is None:
            continue
        if publish_time < window_start or publish_time > window_end:
            continue
        keep, matched_keywords, matched_groups = _match_keywords(f"{title}\n{content}", keyword_groups, "match")
        if not keep:
            continue
        source = f"AKShare东方财富个股新闻:{symbol}"
        if publisher:
            source = f"{source}:{publisher}"
        rows.append(
            {
                "news_id": _news_id(source, url, title),
                "source": source[:50],
                "publish_time": publish_time,
                "trade_date": publish_time.date(),
                "title": title[:500],
                "content": content,
                "url": url,
                "matched_keywords": _json_list(matched_keywords),
                "matched_groups": _json_list(matched_groups),
            }
        )
    return rows


def collect_akshare_news(
    keyword_groups: dict[str, list[str]],
    symbols: list[str],
    window_start: datetime,
    window_end: datetime,
) -> list[dict]:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is not installed. Run: pip install -r requirements.txt") from exc

    rows: list[dict] = []
    for symbol in symbols:
        raw = ak.stock_news_em(symbol=symbol)
        got = normalize_akshare_stock_news(raw, symbol, keyword_groups, window_start, window_end)
        rows.extend(got)
        logger.info("collected akshare news symbol=%s rows=%s", symbol, len(got))
    return rows


def parse_news_home(source: str, url: str, keyword_groups: dict[str, list[str]], max_items: int) -> list[dict]:
    disable_env_proxies()
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")
    rows: list[dict] = []
    for a in soup.select("a[href]"):
        title = a.get_text(" ", strip=True)
        href = a.get("href")
        if not title or not href or len(title) < 6:
            continue
        keep, matched_keywords, matched_groups = _match_keywords(title, keyword_groups, "match")
        if not keep:
            continue
        full_url = urljoin(url, href)
        rows.append(
            {
                "news_id": _news_id(source, full_url, title),
                "source": source,
                "publish_time": datetime.now(),
                "trade_date": datetime.now().date(),
                "title": title[:500],
                "content": title,
                "url": full_url,
                "matched_keywords": _json_list(matched_keywords),
                "matched_groups": _json_list(matched_groups),
            }
        )
        if len(rows) >= max_items:
            break
    return rows


def collect_news(
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> int:
    cfg = get_config()
    news_cfg = cfg.get("news", {})
    keyword_groups = _keyword_groups_from_config(cfg)
    max_items = int(news_cfg.get("max_items_per_source", 80))
    akshare_symbols = _akshare_symbols(cfg)
    window_start, window_end = _collection_window(
        start_time=start_time,
        end_time=end_time,
        lookback_days=int(news_cfg.get("lookback_days", lookback_days)),
    )
    sources = load_yaml(project_path("config", "symbols.yaml"))["news_sources"]
    rows: list[dict] = []
    ensure_news_raw_columns()
    try:
        rows.extend(collect_akshare_news(keyword_groups, akshare_symbols, window_start, window_end))
    except Exception as exc:
        logger.exception("failed to collect akshare news, fallback to homepage sources: %s", exc)

    if rows:
        return upsert_dataframe(pd.DataFrame(rows), "news_raw", ["news_id"])

    logger.warning("akshare news returned no rows in window %s - %s, fallback to homepage sources", window_start, window_end)
    for source in sources:
        try:
            got = parse_news_home(source["name"], source["url"], keyword_groups, max_items)
            rows.extend(got)
            logger.info("collected news source=%s rows=%s", source["name"], len(got))
        except Exception as exc:
            logger.exception("failed to collect news source=%s: %s", source["name"], exc)
    if not rows:
        return 0
    return upsert_dataframe(pd.DataFrame(rows), "news_raw", ["news_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect news, preferring AKShare timestamped interfaces over homepage fallback.")
    parser.add_argument("--start-time", help="Window start time, e.g. 2026-07-03 08:00:00.")
    parser.add_argument("--end-time", help="Window end time. Defaults to now.")
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS, help="Look back N days from end-time if start-time is empty.")
    args = parser.parse_args()
    print(
        collect_news(
            start_time=_parse_time(args.start_time),
            end_time=_parse_time(args.end_time),
            lookback_days=args.lookback_days,
        )
    )


if __name__ == "__main__":
    main()
