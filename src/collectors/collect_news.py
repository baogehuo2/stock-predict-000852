from __future__ import annotations

import argparse
import hashlib
from datetime import datetime
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


def _news_id(source: str, url: str, title: str) -> str:
    return hashlib.sha1(f"{source}|{url}|{title}".encode("utf-8")).hexdigest()


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


def collect_news() -> int:
    cfg = get_config()
    news_cfg = cfg.get("news", {})
    keyword_groups = _keyword_groups_from_config(cfg)
    max_items = int(news_cfg.get("max_items_per_source", 80))
    sources = load_yaml(project_path("config", "symbols.yaml"))["news_sources"]
    rows: list[dict] = []
    ensure_news_raw_columns()
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
    argparse.ArgumentParser().parse_args()
    print(collect_news())


if __name__ == "__main__":
    main()
