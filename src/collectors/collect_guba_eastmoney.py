from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.common.config import get_config, load_yaml, project_path
from src.common.db import upsert_dataframe
from src.common.logger import get_logger
from src.common.network import disable_env_proxies


logger = get_logger(__name__)
HEADERS = {"User-Agent": "Mozilla/5.0"}


def page_url(base_url: str, page: int) -> str:
    if page <= 1:
        return base_url
    if base_url.endswith(".html"):
        base_url = re.sub(r"_\d+\.html$", ".html", base_url)
        return base_url.replace(".html", f"_{page}.html")
    return base_url.rstrip("/") + f"_{page}.html"


def bar_code_from_url(base_url: str) -> str | None:
    match = re.search(r"list,([^._]+)", base_url)
    return match.group(1) if match else None


def _post_id(url: str, title: str = "") -> str:
    match = re.search(r"/news,([^,]+),(\d+)\.html", url)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    return hashlib.sha1(url.encode("utf-8")).hexdigest()


def _safe_int(value: str | int | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(str(value).replace(",", "").strip())
    except Exception:
        return None


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def enrich_row_metadata(row: dict) -> dict:
    publish_time = row.get("publish_time")
    crawl_time = row.get("crawl_time") or datetime.now()
    available_time = publish_time or crawl_time
    title = str(row.get("title") or "")
    content = str(row.get("content") or "")
    author = str(row.get("author") or "")
    row["crawl_time"] = crawl_time
    row["available_time"] = available_time
    row["raw_hash"] = _hash_text(
        "|".join(
            [
                str(row.get("post_id") or ""),
                str(row.get("source") or ""),
                str(row.get("bar_name") or ""),
                str(publish_time or ""),
                title,
                content,
                str(row.get("url") or ""),
            ]
        )
    )
    row["content_hash"] = _hash_text(f"{title}|{content}")
    row["author_id_hash"] = _hash_text(author) if author else None
    return row


def fetch_guba_detail(url: str) -> dict:
    if "guba.eastmoney.com/news," not in url:
        return {}
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    publish_time = None
    match = re.search(r'"post_publish_time"\s*:\s*"([^"]+)"', html)
    if match:
        publish_time = pd.to_datetime(match.group(1), errors="coerce")
    if publish_time is None or pd.isna(publish_time):
        time_node = soup.select_one(".time")
        if time_node:
            publish_time = pd.to_datetime(time_node.get_text(strip=True), errors="coerce")

    def _json_field(name: str) -> str | None:
        m = re.search(rf'"{name}"\s*:\s*"([^"]*)"', html)
        return m.group(1) if m else None

    title = _json_field("post_title")
    content = _json_field("post_content") or _json_field("post_abstract")
    if not content:
        content_node = soup.select_one(".newstext, .stockcodec, .article-body")
        content = content_node.get_text(" ", strip=True) if content_node else ""

    return {
        "title": title,
        "publish_time": None if publish_time is None or pd.isna(publish_time) else publish_time.to_pydatetime(),
        "trade_date": None if publish_time is None or pd.isna(publish_time) else publish_time.date(),
        "content": content,
        "read_count": _safe_int(re.search(r'"post_click_count"\s*:\s*(\d+)', html).group(1)) if re.search(r'"post_click_count"\s*:\s*(\d+)', html) else None,
        "comment_count": _safe_int(re.search(r'"post_comment_count"\s*:\s*(\d+)', html).group(1)) if re.search(r'"post_comment_count"\s*:\s*(\d+)', html) else None,
        "like_count": _safe_int(re.search(r'"post_like_count"\s*:\s*(\d+)', html).group(1)) if re.search(r'"post_like_count"\s*:\s*(\d+)', html) else None,
    }


def parse_guba_page(bar_name: str, base_url: str, page: int, fetch_detail: bool = False, detail_sleep: float = 0.0) -> list[dict]:
    disable_env_proxies()
    url = page_url(base_url, page)
    bar_code = bar_code_from_url(base_url)
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    resp.encoding = resp.apparent_encoding
    soup = BeautifulSoup(resp.text, "html.parser")
    rows: list[dict] = []
    for a in soup.select("a[href*='/news,'], a[href*='/news/']"):
        title = a.get_text(" ", strip=True)
        href = a.get("href")
        if not title or not href or len(title) < 4:
            continue
        full_url = urljoin("https://guba.eastmoney.com", href)
        if "guba.eastmoney.com/news," not in full_url:
            continue
        if bar_code and f"/news,{bar_code}," not in full_url:
            continue
        parent_text = a.find_parent().get_text(" ", strip=True) if a.find_parent() else ""
        row = {
            "post_id": _post_id(full_url, title),
            "source": "eastmoney_guba",
            "bar_name": bar_name,
            "publish_time": None,
            "trade_date": datetime.now().date(),
            "title": title[:500],
            "content": parent_text[:2000],
            "read_count": None,
            "comment_count": None,
            "like_count": None,
            "author": None,
            "url": full_url,
        }
        if fetch_detail:
            try:
                import time

                detail = fetch_guba_detail(full_url)
                row.update({k: v for k, v in detail.items() if v not in (None, "")})
                if detail_sleep > 0:
                    time.sleep(detail_sleep)
            except Exception as exc:
                logger.warning("failed to fetch guba detail url=%s error=%s", full_url, exc)
        row = enrich_row_metadata(row)
        rows.append(row)
    dedup = {row["post_id"]: row for row in rows}
    return list(dedup.values())


def collect_guba(pages: int | None = None, fetch_detail: bool = True, detail_sleep: float = 0.2) -> int:
    cfg = get_config()
    pages = pages or int(cfg.get("sentiment", {}).get("guba_pages", 5))
    bars = load_yaml(project_path("config", "symbols.yaml"))["guba_bars"]
    all_rows: list[dict] = []
    for bar in bars:
        for page in range(1, pages + 1):
            try:
                rows = parse_guba_page(
                    bar["bar_name"],
                    bar["url"],
                    page,
                    fetch_detail=fetch_detail,
                    detail_sleep=detail_sleep,
                )
                all_rows.extend(rows)
                logger.info("collected guba bar=%s page=%s rows=%s", bar["bar_name"], page, len(rows))
            except Exception as exc:
                logger.exception("failed to collect guba bar=%s page=%s: %s", bar["bar_name"], page, exc)
    if not all_rows:
        return 0
    df = pd.DataFrame(all_rows)
    return upsert_dataframe(df, "sentiment_guba_raw", ["post_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pages", type=int)
    parser.add_argument("--no-detail", action="store_true")
    parser.add_argument("--detail-sleep", type=float, default=0.2)
    args = parser.parse_args()
    print(collect_guba(args.pages, fetch_detail=not args.no_detail, detail_sleep=args.detail_sleep))


if __name__ == "__main__":
    main()
