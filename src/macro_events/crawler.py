from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime
from html import unescape
from typing import Iterable
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from src.common.network import disable_env_proxies
from src.macro_events.sources import EVENT_KEYWORDS, OFFICIAL_SOURCES, OfficialSource


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
    )
}

OFFICIAL_HOST_SUFFIXES = (
    "gov.cn",
    "news.cn",
    "xinhuanet.com",
    "people.com.cn",
    "npc.gov.cn",
    "cppcc.gov.cn",
    "12371.cn",
)

DISCOVERY_SITE_FILTER = " OR ".join(
    f"site:{host}"
    for host in (
        "gov.cn",
        "news.cn",
        "xinhuanet.com",
        "people.com.cn",
        "npc.gov.cn",
        "cppcc.gov.cn",
        "12371.cn",
    )
)

MACRO_TITLE_KEYWORDS = (
    "\u5168\u56fd\u4eba\u5927",
    "\u5168\u56fd\u653f\u534f",
    "\u4e24\u4f1a",
    "\u653f\u6cbb\u5c40\u4f1a\u8bae",
    "\u4e2d\u592e\u7ecf\u6d4e\u5de5\u4f5c\u4f1a\u8bae",
    "\u5168\u4f1a",
)

SEED_PAGES = [
    OfficialSource(
        org="\u5171\u4ea7\u515a\u5458\u7f51",
        home_url="https://www.12371.cn/special/lczyjjgzhy/",
        search_url_template="",
    ),
    OfficialSource(
        org="\u5171\u4ea7\u515a\u5458\u7f51",
        home_url="https://www.12371.cn/special/20jszqh/",
        search_url_template="",
    ),
    OfficialSource(
        org="\u65b0\u534e\u793e",
        home_url="https://www.news.cn/zt/ddesjszqh/index.html",
        search_url_template="",
    ),
    *[
        OfficialSource(
            org="\u4e2d\u56fd\u653f\u5e9c\u7f51",
            home_url=f"https://www.gov.cn/zhuanti/{year}qglh/",
            search_url_template="",
        )
        for year in range(2005, 2027)
    ],
]


class MacroEventCrawler:
    def __init__(self, timeout: int = 20, sleep_seconds: float = 0.5, logger: logging.Logger | None = None) -> None:
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.logger = logger or logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def crawl_history_candidates(
        self,
        start_date: date,
        end_date: date,
        max_pages_per_keyword: int = 3,
    ) -> list[dict]:
        articles: list[dict] = []
        seen_urls: set[str] = set()
        for source in SEED_PAGES:
            for title, url in self.seed_page_links(source):
                if url in seen_urls or not _is_official_url(url):
                    continue
                seen_urls.add(url)
                article = self.fetch_article(url, source, fallback_title=title)
                if not article:
                    continue
                article_date = _safe_date(article.get("publish_date", ""))
                if article_date and not (start_date <= article_date <= end_date):
                    continue
                article["event_type_hint"] = _guess_event_type_from_seed(source.home_url, title)
                articles.append(article)
                self._sleep()
        for event_type, keywords in EVENT_KEYWORDS.items():
            for keyword in keywords:
                for source in OFFICIAL_SOURCES:
                    if not source.enabled:
                        continue
                    links = self.search_links(source, keyword, max_pages_per_keyword)
                    for title, url in links:
                        if url in seen_urls or not _is_official_url(url):
                            continue
                        seen_urls.add(url)
                        article = self.fetch_article(url, source, fallback_title=title)
                        if not article:
                            continue
                        article_date = _safe_date(article.get("publish_date", ""))
                        if article_date and not (start_date <= article_date <= end_date):
                            continue
                        article["event_type_hint"] = event_type
                        articles.append(article)
                        self._sleep()
        return articles

    def crawl_daily_candidates(self, keywords: Iterable[str] | None = None) -> list[dict]:
        target_keywords = list(keywords or sorted({keyword for values in EVENT_KEYWORDS.values() for keyword in values}))
        articles: list[dict] = []
        seen_urls: set[str] = set()
        for keyword in target_keywords:
            for source in OFFICIAL_SOURCES:
                if not source.enabled:
                    continue
                for title, url in self.search_links(source, keyword, max_pages=1):
                    if url in seen_urls or not _is_official_url(url):
                        continue
                    seen_urls.add(url)
                    article = self.fetch_article(url, source, fallback_title=title)
                    if article:
                        articles.append(article)
                    self._sleep()
        return articles

    def search_links(self, source: OfficialSource, keyword: str, max_pages: int = 1) -> list[tuple[str, str]]:
        if max_pages <= 0:
            return []
        disable_env_proxies()
        links: list[tuple[str, str]] = []
        search_url = source.search_url(keyword)
        try:
            resp = self.session.get(search_url, timeout=self.timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
        except Exception as exc:
            self.logger.warning("search failed source=%s keyword=%s error=%s", source.org, keyword, exc)
        else:
            soup = BeautifulSoup(resp.text, "html.parser")
            for title, url in _extract_links_from_search_page(soup, search_url):
                if not _looks_like_article_url(url):
                    continue
                if not _matches_keyword_or_macro_title(title, keyword):
                    continue
                links.append((title[:300], url))
                if len(links) >= max_pages * 20:
                    break

        if not links:
            links.extend(self.search_links_via_duckduckgo(keyword, max_pages=max_pages))
        return _dedupe_links(links)

    def search_links_via_duckduckgo(self, keyword: str, max_pages: int = 1) -> list[tuple[str, str]]:
        if max_pages <= 0:
            return []
        disable_env_proxies()
        query = f"{keyword} ({DISCOVERY_SITE_FILTER})"
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
        except Exception as exc:
            self.logger.warning("fallback search failed keyword=%s error=%s", keyword, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        links: list[tuple[str, str]] = []
        for title, href in _extract_links_from_search_page(soup, url):
            resolved = _unwrap_search_redirect(href)
            if not _looks_like_article_url(resolved):
                continue
            if not _matches_keyword_or_macro_title(title, keyword):
                continue
            links.append((title[:300], resolved))
            if len(links) >= max_pages * 10:
                break
        return _dedupe_links(links)

    def seed_page_links(self, source: OfficialSource) -> list[tuple[str, str]]:
        disable_env_proxies()
        try:
            resp = self.session.get(source.home_url, timeout=self.timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
        except Exception as exc:
            self.logger.warning("seed page failed source=%s url=%s error=%s", source.org, source.home_url, exc)
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        links: list[tuple[str, str]] = []
        for title, url in _extract_links_from_search_page(soup, source.home_url):
            if not _looks_like_article_url(url):
                continue
            if not _matches_seed_title(title):
                continue
            links.append((title[:300], url))
        return _dedupe_links(links)

    def fetch_article(self, url: str, source: OfficialSource, fallback_title: str = "") -> dict | None:
        disable_env_proxies()
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
        except Exception as exc:
            self.logger.warning("fetch failed url=%s error=%s", url, exc)
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        title = _extract_title(soup) or fallback_title
        text = _extract_text(soup)
        publish_date = _extract_publish_date(soup.get_text("\n", strip=True), url)
        if not title or not text:
            return None
        return {
            "title": unescape(title),
            "text": unescape(text),
            "publish_date": publish_date,
            "source_org": source.org,
            "url": url,
        }

    def _sleep(self) -> None:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)


def _extract_links_from_search_page(soup: BeautifulSoup, base_url: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    for a in soup.select("a[href]"):
        title = a.get_text(" ", strip=True)
        href = a.get("href", "").strip()
        if not title or len(title) < 6:
            continue
        links.append((title, urljoin(base_url, href)))
    return links


def _unwrap_search_redirect(url: str) -> str:
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    for key in ("uddg", "url", "u"):
        if query.get(key):
            return unquote(query[key][0])
    return url


def _dedupe_links(links: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for title, url in links:
        clean_url = _normalize_url(url)
        if clean_url in seen:
            continue
        seen.add(clean_url)
        rows.append((title, clean_url))
    return rows


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return parsed._replace(fragment="").geturl()


def _extract_title(soup: BeautifulSoup) -> str:
    for selector in ("h1", ".title", "#title"):
        node = soup.select_one(selector)
        if node:
            text = node.get_text(" ", strip=True)
            if text:
                return text
    if soup.title and soup.title.string:
        return re.sub(r"[_\\-].*$", "", soup.title.string).strip()
    return ""


def _extract_text(soup: BeautifulSoup) -> str:
    candidates = []
    for selector in ("article", ".article", ".content", ".pages_content", "#detail", "#content", ".main"):
        node = soup.select_one(selector)
        if node:
            candidates.append(node.get_text("\n", strip=True))
    if not candidates:
        paragraphs = [p.get_text(" ", strip=True) for p in soup.select("p")]
        candidates.append("\n".join(p for p in paragraphs if p))
    text = max(candidates, key=len, default="")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_publish_date(text: str, url: str) -> str:
    match = re.search(r"(20\d{2})[\u5e74/-](\d{1,2})[\u6708/-](\d{1,2})\u65e5?", text)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    match = re.search(r"/(20\d{2})[-/]?(\d{2})[-/]?(\d{2})/", url)
    if match:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    return ""


def _safe_date(value: str) -> date | None:
    try:
        return datetime.fromisoformat(value[:10]).date()
    except ValueError:
        return None


def _looks_like_article_url(url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme.startswith("http"):
        return False
    lowered = url.lower()
    if any(skip in lowered for skip in ("javascript:", "#", "mailto:", ".jpg", ".png", ".pdf")):
        return False
    return _is_official_url(url)


def _is_official_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(host == suffix or host.endswith("." + suffix) for suffix in OFFICIAL_HOST_SUFFIXES)


def _matches_keyword_or_macro_title(title: str, keyword: str) -> bool:
    compact_title = title.replace(" ", "")
    compact_keyword = keyword.replace(" ", "")
    return compact_keyword in compact_title or any(item in title for item in MACRO_TITLE_KEYWORDS)


def _matches_seed_title(title: str) -> bool:
    return any(item in title for item in MACRO_TITLE_KEYWORDS) or any(
        item in title
        for item in (
            "\u516c\u62a5",
            "\u5f00\u5e55",
            "\u95ed\u5e55",
            "\u8bae\u7a0b",
            "\u4e2d\u592e\u7ecf\u6d4e\u5de5\u4f5c",
        )
    )


def _guess_event_type_from_seed(seed_url: str, title: str) -> str:
    if "lczyjjgzhy" in seed_url or "\u4e2d\u592e\u7ecf\u6d4e\u5de5\u4f5c" in title:
        return "CEWC"
    if "20jszqh" in seed_url or "ddesjszqh" in seed_url or "\u5168\u4f1a" in title or "\u516c\u62a5" in title:
        return "CPC_PLENUM"
    if "qglh" in seed_url or "\u5168\u56fd\u4e24\u4f1a" in title or "\u5168\u56fd\u4eba\u5927" in title or "\u5168\u56fd\u653f\u534f" in title:
        return "TWO_SESSIONS"
    return ""
