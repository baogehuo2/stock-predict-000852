from __future__ import annotations

import argparse
import hashlib
import re
import time
from dataclasses import dataclass
from datetime import date, datetime
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.collectors.collect_news_history import _json_list
from src.collectors.collect_event_raw_v4 import ensure_news_raw_v4_columns, extract_content, extract_publish_time
from src.common.db import upsert_dataframe
from src.common.logger import get_logger
from src.common.network import disable_env_proxies


logger = get_logger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 data-collectors/2.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


@dataclass(frozen=True)
class HistorySourceMeta:
    key: str
    source: str
    event_layer_hint: str
    publisher_country: str
    source_type: str
    document_type: str
    source_priority: int
    language: str


SOURCE_META = {
    "fed_fomc": HistorySourceMeta(
        key="fed_fomc",
        source="Federal Reserve",
        event_layer_hint="macro",
        publisher_country="US",
        source_type="official_central_bank",
        document_type="statement",
        source_priority=1,
        language="en",
    ),
    "mfa_press": HistorySourceMeta(
        key="mfa_press",
        source="外交部",
        event_layer_hint="geo",
        publisher_country="CN",
        source_type="official_government",
        document_type="statement",
        source_priority=1,
        language="zh",
    ),
    "pbc_news": HistorySourceMeta(
        key="pbc_news",
        source="中国人民银行",
        event_layer_hint="cn",
        publisher_country="CN",
        source_type="official_government",
        document_type="policy",
        source_priority=1,
        language="zh",
    ),
    "ecb_monetary": HistorySourceMeta(
        key="ecb_monetary",
        source="European Central Bank",
        event_layer_hint="macro",
        publisher_country="EU",
        source_type="official_central_bank",
        document_type="statement",
        source_priority=1,
        language="en",
    ),
    "boj_monetary": HistorySourceMeta(
        key="boj_monetary",
        source="Bank of Japan",
        event_layer_hint="macro",
        publisher_country="JP",
        source_type="official_central_bank",
        document_type="statement",
        source_priority=1,
        language="en",
    ),
}


def _fetch(url: str) -> str:
    disable_env_proxies()
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_date_from_url(url: str) -> date | None:
    for pattern in (
        r"(?:monetary|bcreg|ecb\.pr|mpr_|state_)(20\d{2})(\d{2})(\d{2})",
        r"/(20\d{2})(\d{2})/t(20\d{2})(\d{2})(\d{2})_",
        r"/(20\d{2})[/-](\d{2})[/-](\d{2})/",
    ):
        match = re.search(pattern, url)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 5:
            return date(int(groups[2]), int(groups[3]), int(groups[4]))
        return date(int(groups[0]), int(groups[1]), int(groups[2]))
    return None


def _in_range(item_date: date | None, start_date: date, end_date: date) -> bool:
    if item_date is None:
        return True
    return start_date <= item_date <= end_date


def _years(start_date: date, end_date: date) -> range:
    return range(start_date.year, end_date.year + 1)


def _row(meta: HistorySourceMeta, title: str, url: str, content: str, publish_time: datetime, matched_keywords: list[str]) -> dict:
    crawl_time = datetime.now()
    canonical_url = url
    title = title[:500]
    content_hash = _sha256(f"{title}|{content}")
    raw_hash = _sha256("|".join([meta.key, canonical_url, title, content, publish_time.isoformat()]))
    return {
        "news_id": _sha1(f"{meta.key}|{canonical_url}|{title}"),
        "source": meta.source,
        "publish_time": publish_time,
        "trade_date": publish_time.date(),
        "title": title,
        "content": content,
        "url": canonical_url,
        "matched_keywords": _json_list(matched_keywords),
        "matched_groups": _json_list([meta.event_layer_hint]),
        "language": meta.language,
        "country_region": meta.publisher_country,
        "source_type": meta.source_type,
        "first_seen_time": crawl_time,
        "last_seen_time": crawl_time,
        "canonical_url": canonical_url,
        "content_hash": content_hash,
        "is_reprint": 0,
        "original_news_id": None,
        "event_layer_hint": meta.event_layer_hint,
        "publisher_country": meta.publisher_country,
        "document_type": meta.document_type,
        "official_source": 1,
        "source_priority": meta.source_priority,
        "crawl_time": crawl_time,
        "available_time": publish_time,
        "raw_hash": raw_hash,
    }


def _fetch_detail(meta: HistorySourceMeta, title: str, url: str, fallback_date: date | None, matched_keywords: list[str]) -> dict | None:
    try:
        html = _fetch(url)
    except Exception as exc:
        logger.warning("detail fetch failed source=%s url=%s error=%s", meta.key, url, exc)
        return None
    soup = BeautifulSoup(html, "html.parser")
    detail_title = _clean_text((soup.select_one("h1") or soup.select_one("title") or soup).get_text(" ", strip=True))
    if detail_title.lower().startswith("board of governors of the federal reserve system"):
        detail_title = ""
    final_title = detail_title or title
    content = extract_content(soup) or final_title
    publish_time = extract_publish_time(soup, url)
    if publish_time is None and fallback_date is not None:
        publish_time = datetime.combine(fallback_date, datetime.min.time())
    if publish_time is None:
        return None
    return _row(meta, final_title, url, content, publish_time, matched_keywords)


def _append_if_in_range(rows: list[dict], row: dict | None, start_date: date, end_date: date) -> bool:
    if row is None:
        return False
    publish_date = row["publish_time"].date()
    if not _in_range(publish_date, start_date, end_date):
        return False
    rows.append(row)
    return True


def collect_fed_fomc(start_date: date, end_date: date, max_items: int, sleep_seconds: float) -> list[dict]:
    meta = SOURCE_META["fed_fomc"]
    rows: list[dict] = []
    seen: set[str] = set()
    for year in _years(start_date, end_date):
        archive_url = f"https://www.federalreserve.gov/newsevents/pressreleases/{year}-press-fomc.htm"
        try:
            html = _fetch(archive_url)
        except Exception as exc:
            logger.warning("fed archive failed year=%s error=%s", year, exc)
            continue
        soup = BeautifulSoup(html, "html.parser")
        for link in soup.select("a[href]"):
            href = str(link.get("href") or "")
            if not re.search(r"/newsevents/pressreleases/monetary20\d{6}a\.htm", href):
                continue
            url = urljoin(archive_url, href)
            item_date = _parse_date_from_url(url)
            if url in seen or not _in_range(item_date, start_date, end_date):
                continue
            title = _clean_text(link.get_text(" ", strip=True)) or "Federal Reserve FOMC statement"
            row = _fetch_detail(meta, title, url, item_date, ["FOMC", "monetary policy"])
            if row:
                _append_if_in_range(rows, row, start_date, end_date)
                seen.add(url)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
            if len(rows) >= max_items:
                return rows
    return rows


def collect_mfa_press(start_date: date, end_date: date, max_items: int, sleep_seconds: float) -> list[dict]:
    meta = SOURCE_META["mfa_press"]
    list_url = "https://www.mfa.gov.cn/web/wjdt_674879/fyrbt_674889/"
    html = _fetch(list_url)
    rows: list[dict] = []
    seen: set[str] = set()
    for match in re.finditer(r"\./(20\d{4})/(t20\d{6}_\d+\.shtml)", html):
        url = urljoin(list_url, f"{match.group(1)}/{match.group(2)}")
        item_date = _parse_date_from_url(url)
        if url in seen or not _in_range(item_date, start_date, end_date):
            continue
        title = f"{item_date.isoformat()} 外交部例行记者会" if item_date else "外交部例行记者会"
        row = _fetch_detail(meta, title, url, item_date, ["外交部发言人", "例行记者会"])
        if row:
            _append_if_in_range(rows, row, start_date, end_date)
            seen.add(url)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if len(rows) >= max_items:
            break
    return rows


def collect_pbc_news(start_date: date, end_date: date, max_items: int, sleep_seconds: float) -> list[dict]:
    meta = SOURCE_META["pbc_news"]
    list_url = "http://www.pbc.gov.cn/goutongjiaoliu/113456/113469/index.html"
    html = _fetch(list_url)
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    seen: set[str] = set()
    keywords = ["货币政策", "公开市场", "降准", "利率", "金融市场", "流动性", "贷款", "融资"]
    for link in soup.select("a[href]"):
        title = _clean_text(link.get_text(" ", strip=True))
        href = str(link.get("href") or "")
        if not title or not any(keyword in title for keyword in keywords):
            continue
        url = urljoin(list_url, href)
        item_date = _parse_date_from_url(url)
        if url in seen or not _in_range(item_date, start_date, end_date):
            continue
        row = _fetch_detail(meta, title, url, item_date, [keyword for keyword in keywords if keyword in title])
        if row:
            _append_if_in_range(rows, row, start_date, end_date)
            seen.add(url)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if len(rows) >= max_items:
            break
    return rows


def collect_ecb_monetary(start_date: date, end_date: date, max_items: int, sleep_seconds: float) -> list[dict]:
    meta = SOURCE_META["ecb_monetary"]
    list_url = "https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html"
    html = _fetch(list_url)
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    seen: set[str] = set()
    for link in soup.select("a[href]"):
        href = str(link.get("href") or "")
        title = _clean_text(link.get_text(" ", strip=True))
        if "ecb.governingcouncil" not in href and "ecb.pr" not in href:
            continue
        url = urljoin(list_url, href)
        item_date = _parse_date_from_url(url)
        if url in seen or not _in_range(item_date, start_date, end_date):
            continue
        row = _fetch_detail(meta, title or "ECB monetary policy decision", url, item_date, ["monetary policy", "ECB"])
        if row:
            _append_if_in_range(rows, row, start_date, end_date)
            seen.add(url)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if len(rows) >= max_items:
            break
    return rows


def collect_boj_monetary(start_date: date, end_date: date, max_items: int, sleep_seconds: float) -> list[dict]:
    meta = SOURCE_META["boj_monetary"]
    list_url = f"https://www.boj.or.jp/en/mopo/mpmdeci/state_{end_date.year}/index.htm"
    try:
        html = _fetch(list_url)
    except Exception:
        html = _fetch("https://www.boj.or.jp/en/mopo/mpmdeci/index.htm")
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict] = []
    seen: set[str] = set()
    for link in soup.select("a[href]"):
        href = str(link.get("href") or "")
        title = _clean_text(link.get_text(" ", strip=True))
        if not re.search(r"/en/mopo/mpmdeci/state_20\d+/", href):
            continue
        url = urljoin(list_url, href)
        item_date = _parse_date_from_url(url)
        if url in seen or not _in_range(item_date, start_date, end_date):
            continue
        row = _fetch_detail(meta, title or "BOJ statement on monetary policy", url, item_date, ["Monetary Policy"])
        if row:
            _append_if_in_range(rows, row, start_date, end_date)
            seen.add(url)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        if len(rows) >= max_items:
            break
    return rows


COLLECTORS = {
    "fed_fomc": collect_fed_fomc,
    "mfa_press": collect_mfa_press,
    "pbc_news": collect_pbc_news,
    "ecb_monetary": collect_ecb_monetary,
    "boj_monetary": collect_boj_monetary,
}


def collect_event_history_v4(
    start_date: str,
    end_date: str,
    sources: list[str] | None = None,
    max_items_per_source: int = 50,
    sleep_seconds: float = 0.2,
) -> dict:
    ensure_news_raw_v4_columns()
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date).date()
    selected = sources or list(COLLECTORS)
    unknown = sorted(set(selected) - set(COLLECTORS))
    if unknown:
        raise ValueError(f"未知第四批历史来源: {unknown}")

    all_rows: list[dict] = []
    stats: list[dict] = []
    for source in selected:
        try:
            rows = COLLECTORS[source](start, end, max_items_per_source, sleep_seconds)
            all_rows.extend(rows)
            stats.append({
                "source": source,
                "rows": len(rows),
                "min_publish_time": min((row["publish_time"] for row in rows), default=None),
                "max_publish_time": max((row["publish_time"] for row in rows), default=None),
            })
            logger.info("event history source=%s rows=%s", source, len(rows))
        except Exception as exc:
            logger.exception("event history source=%s failed: %s", source, exc)
            stats.append({"source": source, "rows": 0, "error": str(exc)})

    written = upsert_dataframe(pd.DataFrame(all_rows), "news_raw", ["news_id"]) if all_rows else 0
    return {"written": written, "sources": stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="回补模型2.0第四批官方重大事件原始文本。")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--source", action="append", choices=sorted(COLLECTORS))
    parser.add_argument("--max-items-per-source", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    print(
        collect_event_history_v4(
            start_date=args.start_date,
            end_date=args.end_date,
            sources=args.source,
            max_items_per_source=args.max_items_per_source,
            sleep_seconds=args.sleep,
        )
    )


if __name__ == "__main__":
    main()
