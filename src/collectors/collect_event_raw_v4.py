from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.collectors.collect_news_history import _json_list
from src.common.db import execute_sql, read_sql, upsert_dataframe
from src.common.logger import get_logger
from src.common.network import disable_env_proxies


logger = get_logger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 data-collectors/2.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


@dataclass(frozen=True)
class EventSource:
    key: str
    name: str
    list_url: str
    event_layer_hint: str
    publisher_country: str
    source_type: str
    document_type: str
    official_source: int
    source_priority: int
    include_keywords: tuple[str, ...]
    exclude_keywords: tuple[str, ...] = ()
    include_url_patterns: tuple[str, ...] = ()


SOURCES: tuple[EventSource, ...] = (
    EventSource(
        key="gov_cn_policy",
        name="中国政府网政策",
        list_url="https://www.gov.cn/zhengce/zuixin/",
        event_layer_hint="cn",
        publisher_country="CN",
        source_type="official_government",
        document_type="policy",
        official_source=1,
        source_priority=1,
        include_keywords=("国务院", "政策", "金融", "资本市场", "房地产", "财政", "货币", "稳增长", "会议"),
        include_url_patterns=(r"/zhengce/", r"/yaowen/"),
    ),
    EventSource(
        key="pbc_news",
        name="中国人民银行",
        list_url="http://www.pbc.gov.cn/goutongjiaoliu/113456/113469/index.html",
        event_layer_hint="cn",
        publisher_country="CN",
        source_type="official_government",
        document_type="policy",
        official_source=1,
        source_priority=1,
        include_keywords=("货币政策", "公开市场", "降准", "利率", "金融市场", "流动性", "贷款", "融资"),
        include_url_patterns=(r"/goutongjiaoliu/113456/113469/",),
    ),
    EventSource(
        key="csrc_news",
        name="中国证监会",
        list_url="http://www.csrc.gov.cn/csrc/c100028/zfxxgk_zdgk.shtml",
        event_layer_hint="cn",
        publisher_country="CN",
        source_type="official_government",
        document_type="policy",
        official_source=1,
        source_priority=1,
        include_keywords=("资本市场", "上市公司", "IPO", "减持", "融券", "量化", "监管", "证券", "基金", "证监会"),
        exclude_keywords=("年度报表", "网站工作报表", "政府网站"),
        include_url_patterns=(r"/csrc/c100028/.*content\.shtml", r"/csrc/c100028/.*content\.html"),
    ),
    EventSource(
        key="mfa_press",
        name="外交部",
        list_url="https://www.mfa.gov.cn/web/wjdt_674879/fyrbt_674889/",
        event_layer_hint="geo",
        publisher_country="CN",
        source_type="official_government",
        document_type="statement",
        official_source=1,
        source_priority=1,
        include_keywords=("外交部发言人", "例行记者会", "记者会", "美国", "台湾", "亚太", "俄罗斯", "乌克兰", "中东", "红海", "制裁", "关税"),
        include_url_patterns=(r"/fyrbt_674889/.*t20\d{6}_\d+\.shtml",),
    ),
    EventSource(
        key="mofcom_news",
        name="商务部",
        list_url="https://www.mofcom.gov.cn/syxwfb/",
        event_layer_hint="geo",
        publisher_country="CN",
        source_type="official_government",
        document_type="statement",
        official_source=1,
        source_priority=1,
        include_keywords=("贸易", "关税", "制裁", "出口管制", "中美", "供应链", "商务部新闻发言人"),
        include_url_patterns=(r"/syxwfb/", r"/xwfbh/", r"/article/"),
    ),
    EventSource(
        key="fed_press",
        name="Federal Reserve",
        list_url="https://www.federalreserve.gov/newsevents/pressreleases.htm",
        event_layer_hint="macro",
        publisher_country="US",
        source_type="official_central_bank",
        document_type="statement",
        official_source=1,
        source_priority=1,
        include_keywords=("FOMC", "monetary policy", "interest rate", "Federal Reserve", "banking"),
        include_url_patterns=(r"/newsevents/pressreleases/monetary20\d{6}a\.htm", r"/newsevents/pressreleases/bcreg20\d{6}a\.htm"),
    ),
    EventSource(
        key="ecb_press",
        name="European Central Bank",
        list_url="https://www.ecb.europa.eu/press/pr/date/html/index.en.html",
        event_layer_hint="macro",
        publisher_country="EU",
        source_type="official_central_bank",
        document_type="statement",
        official_source=1,
        source_priority=1,
        include_keywords=("monetary policy", "interest rate", "ECB", "Governing Council", "inflation"),
        include_url_patterns=(r"/press/pr/date/20\d{2}/html/ecb\.pr\d+.*\.en\.html",),
    ),
    EventSource(
        key="boj_release",
        name="Bank of Japan",
        list_url="https://www.boj.or.jp/en/mopo/mpmdeci/index.htm",
        event_layer_hint="macro",
        publisher_country="JP",
        source_type="official_central_bank",
        document_type="statement",
        official_source=1,
        source_priority=1,
        include_keywords=("Monetary Policy", "Guideline", "yield curve", "interest rate", "Outlook"),
        include_url_patterns=(r"/en/mopo/mpmdeci/mpr_20\d+.*\.htm", r"/en/mopo/mpmdeci/state_20\d+.*\.htm"),
    ),
)


def ensure_news_raw_v4_columns() -> None:
    existing = read_sql(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'news_raw'"
    )
    columns = set(existing["COLUMN_NAME"].tolist())
    alters = {
        "language": "ALTER TABLE news_raw ADD COLUMN language VARCHAR(16) NULL AFTER matched_groups",
        "country_region": "ALTER TABLE news_raw ADD COLUMN country_region VARCHAR(30) NULL AFTER language",
        "source_type": "ALTER TABLE news_raw ADD COLUMN source_type VARCHAR(40) NULL AFTER country_region",
        "first_seen_time": "ALTER TABLE news_raw ADD COLUMN first_seen_time DATETIME NULL AFTER source_type",
        "last_seen_time": "ALTER TABLE news_raw ADD COLUMN last_seen_time DATETIME NULL AFTER first_seen_time",
        "canonical_url": "ALTER TABLE news_raw ADD COLUMN canonical_url TEXT NULL AFTER last_seen_time",
        "content_hash": "ALTER TABLE news_raw ADD COLUMN content_hash CHAR(64) NULL AFTER canonical_url",
        "is_reprint": "ALTER TABLE news_raw ADD COLUMN is_reprint TINYINT NULL AFTER content_hash",
        "original_news_id": "ALTER TABLE news_raw ADD COLUMN original_news_id VARCHAR(160) NULL AFTER is_reprint",
        "event_layer_hint": "ALTER TABLE news_raw ADD COLUMN event_layer_hint VARCHAR(16) NULL AFTER original_news_id",
        "publisher_country": "ALTER TABLE news_raw ADD COLUMN publisher_country VARCHAR(30) NULL AFTER event_layer_hint",
        "document_type": "ALTER TABLE news_raw ADD COLUMN document_type VARCHAR(30) NULL AFTER publisher_country",
        "official_source": "ALTER TABLE news_raw ADD COLUMN official_source TINYINT NULL AFTER document_type",
        "source_priority": "ALTER TABLE news_raw ADD COLUMN source_priority INT NULL AFTER official_source",
        "available_time": "ALTER TABLE news_raw ADD COLUMN available_time DATETIME NULL AFTER crawl_time",
        "raw_hash": "ALTER TABLE news_raw ADD COLUMN raw_hash CHAR(64) NULL AFTER available_time",
    }
    for column, statement in alters.items():
        if column not in columns:
            execute_sql(statement)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _fetch(url: str) -> str:
    disable_env_proxies()
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _same_host_or_relative(base_url: str, href: str) -> bool:
    parsed = urlparse(href)
    if not parsed.netloc:
        return True
    return parsed.netloc.endswith(urlparse(base_url).netloc)


def _matches_keywords(text: str, include: tuple[str, ...], exclude: tuple[str, ...]) -> tuple[bool, list[str]]:
    lower_text = text.lower()
    hits = [keyword for keyword in include if keyword.lower() in lower_text]
    blocked = any(keyword.lower() in lower_text for keyword in exclude)
    return bool(hits) and not blocked, hits


def _matches_url(url: str, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return True
    return any(re.search(pattern, url) for pattern in patterns)


def _discover_links_from_html(source: EventSource, base_url: str, html: str, max_items: int, seen: set[str]) -> list[tuple[str, str, list[str]]]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[tuple[str, str, list[str]]] = []
    for node in soup.select("a[href]"):
        title = _clean_text(node.get_text(" ", strip=True))
        href = str(node.get("href") or "").strip()
        if not title or len(title) < 4 or not href.startswith(("http", "/", ".")):
            continue
        if not _same_host_or_relative(base_url, href):
            continue
        full_url = urljoin(base_url, href)
        if not _matches_url(full_url, source.include_url_patterns):
            continue
        keep, matched = _matches_keywords(title, source.include_keywords, source.exclude_keywords)
        if not keep or full_url in seen:
            continue
        seen.add(full_url)
        links.append((title[:500], full_url, matched))
        if len(links) >= max_items:
            break
    return links


def discover_links(source: EventSource, max_items: int) -> list[tuple[str, str, list[str]]]:
    html = _fetch(source.list_url)
    seen: set[str] = set()
    links = _discover_links_from_html(source, source.list_url, html, max_items, seen)
    if len(links) >= max_items:
        return links

    soup = BeautifulSoup(html, "html.parser")
    archive_candidates: list[str] = []
    for node in soup.select("a[href]"):
        href = str(node.get("href") or "").strip()
        if not href:
            continue
        full_url = urljoin(source.list_url, href)
        if full_url in seen or not _same_host_or_relative(source.list_url, full_url):
            continue
        text = _clean_text(node.get_text(" ", strip=True)).lower()
        if (
            "press release" in text
            or "fomc" in text
            or re.search(r"20\d{2}", text)
            or re.search(r"20\d{2}", full_url)
        ):
            archive_candidates.append(full_url)
    for archive_url in archive_candidates[:8]:
        try:
            archive_html = _fetch(archive_url)
        except Exception:
            continue
        links.extend(_discover_links_from_html(source, archive_url, archive_html, max_items - len(links), seen))
        if len(links) >= max_items:
            break
    return links


def _parse_datetime_text(value: str) -> datetime | None:
    value = _clean_text(value)
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone().replace(tzinfo=None)
        return parsed
    except Exception:
        pass
    patterns = (
        r"(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})[日]?\s+(\d{1,2}):(\d{2})(?::(\d{2}))?",
        r"(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})[日]?",
    )
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            parts = [int(item) if item else 0 for item in match.groups()]
            return datetime(parts[0], parts[1], parts[2], parts[3] if len(parts) > 3 else 0, parts[4] if len(parts) > 4 else 0, parts[5] if len(parts) > 5 else 0)
    return None


def extract_publish_time(soup: BeautifulSoup, url: str) -> datetime | None:
    meta_names = (
        "article:published_time",
        "pubdate",
        "publishdate",
        "publish_date",
        "date",
        "dc.date",
        "weibo: article:create_at",
    )
    for meta in soup.select("meta"):
        key = (meta.get("property") or meta.get("name") or "").strip().lower()
        if key in meta_names:
            parsed = _parse_datetime_text(str(meta.get("content") or ""))
            if parsed:
                return parsed
    text = soup.get_text(" ", strip=True)
    parsed = _parse_datetime_text(text[:3000])
    if parsed:
        return parsed
    match = re.search(r"/(20\d{2})[/-]?(\d{2})[/-]?(\d{2})/", url)
    if match:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    match = re.search(r"(?:monetary|bcreg|ecb\.pr|mpr_|state_)(20\d{2})(\d{2})(\d{2})", url)
    if match:
        return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    return None


def extract_content(soup: BeautifulSoup) -> str:
    for selector in (
        "article",
        ".article",
        ".article-content",
        ".content",
        ".TRS_Editor",
        "#zoom",
        "#article",
        ".pages_content",
        ".newsContent",
        ".main-content",
    ):
        node = soup.select_one(selector)
        if node:
            text = _clean_text(node.get_text(" ", strip=True))
            if len(text) >= 40:
                return text
    paragraphs = [_clean_text(p.get_text(" ", strip=True)) for p in soup.select("p")]
    paragraphs = [item for item in paragraphs if len(item) >= 15]
    return _clean_text(" ".join(paragraphs))


def fetch_detail(source: EventSource, title: str, url: str, matched_keywords: list[str]) -> dict | None:
    try:
        html = _fetch(url)
    except Exception as exc:
        logger.warning("failed to fetch event detail source=%s url=%s error=%s", source.key, url, exc)
        return None
    soup = BeautifulSoup(html, "html.parser")
    detail_title = _clean_text((soup.select_one("h1") or soup.select_one("title") or soup).get_text(" ", strip=True))
    if detail_title.lower().startswith("board of governors of the federal reserve system"):
        detail_title = ""
    final_title = (detail_title or title)[:500]
    content = extract_content(soup)
    if not content:
        content = final_title
    keep, detail_matched = _matches_keywords(f"{final_title}\n{content[:1000]}", source.include_keywords, source.exclude_keywords)
    if not keep:
        return None
    if detail_matched:
        matched_keywords = sorted(set(matched_keywords + detail_matched))
    publish_time = extract_publish_time(soup, url)
    if publish_time is None:
        publish_time = datetime.now()
    crawl_time = datetime.now()
    canonical_url = url
    content_hash = _sha256(f"{final_title}|{content}")
    raw_hash = _sha256("|".join([source.key, canonical_url, final_title, content, publish_time.isoformat()]))
    news_id = _sha1(f"{source.key}|{canonical_url}|{final_title}")
    return {
        "news_id": news_id,
        "source": source.name,
        "publish_time": publish_time,
        "trade_date": publish_time.date(),
        "title": final_title,
        "content": content,
        "url": canonical_url,
        "matched_keywords": _json_list(matched_keywords),
        "matched_groups": _json_list([source.event_layer_hint]),
        "language": "en" if source.publisher_country in {"US", "EU", "JP"} else "zh",
        "country_region": source.publisher_country,
        "source_type": source.source_type,
        "first_seen_time": crawl_time,
        "last_seen_time": crawl_time,
        "canonical_url": canonical_url,
        "content_hash": content_hash,
        "is_reprint": 0,
        "original_news_id": None,
        "event_layer_hint": source.event_layer_hint,
        "publisher_country": source.publisher_country,
        "document_type": source.document_type,
        "official_source": source.official_source,
        "source_priority": source.source_priority,
        "crawl_time": crawl_time,
        "available_time": publish_time,
        "raw_hash": raw_hash,
    }


def collect_event_raw_v4(max_items_per_source: int = 10, source_keys: list[str] | None = None) -> dict:
    ensure_news_raw_v4_columns()
    wanted = set(source_keys or [])
    sources = [source for source in SOURCES if not wanted or source.key in wanted]
    unknown = wanted - {source.key for source in SOURCES}
    if unknown:
        raise ValueError(f"未知来源: {sorted(unknown)}")

    rows: list[dict] = []
    source_stats: list[dict] = []
    for source in sources:
        try:
            links = discover_links(source, max_items=max_items_per_source)
            got_rows = []
            for title, url, matched_keywords in links:
                row = fetch_detail(source, title, url, matched_keywords)
                if row:
                    got_rows.append(row)
            rows.extend(got_rows)
            source_stats.append({"source": source.key, "links": len(links), "rows": len(got_rows)})
            logger.info("collected event source=%s links=%s rows=%s", source.key, len(links), len(got_rows))
        except Exception as exc:
            logger.exception("failed event source=%s: %s", source.key, exc)
            source_stats.append({"source": source.key, "links": 0, "rows": 0, "error": str(exc)})

    written = upsert_dataframe(pd.DataFrame(rows), "news_raw", ["news_id"]) if rows else 0
    return {"written": written, "sources": source_stats}


def main() -> None:
    parser = argparse.ArgumentParser(description="采集模型2.0第四批重大事件原始文本。")
    parser.add_argument("--max-items-per-source", type=int, default=10)
    parser.add_argument("--source", action="append", help="按 source key 过滤，可重复。")
    args = parser.parse_args()
    print(collect_event_raw_v4(max_items_per_source=args.max_items_per_source, source_keys=args.source))


if __name__ == "__main__":
    main()
