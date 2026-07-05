from __future__ import annotations

import argparse
import time
from datetime import datetime

import pandas as pd
from sqlalchemy import text

from src.collectors.collect_guba_eastmoney import enrich_row_metadata, fetch_guba_detail
from src.common.db import get_engine, read_sql
from src.common.logger import get_logger


logger = get_logger(__name__)


def repair_guba_publish_time(limit: int | None = None, sleep_seconds: float = 0.2) -> dict:
    limit_clause = "" if limit is None else " LIMIT :limit"
    params = {} if limit is None else {"limit": int(limit)}
    rows = read_sql(
        "SELECT post_id, source, bar_name, title, content, url, crawl_time "
        "FROM sentiment_guba_raw "
        "WHERE publish_time IS NULL AND url IS NOT NULL AND url <> '' "
        "ORDER BY crawl_time DESC, id DESC"
        + limit_clause,
        params,
    )
    if rows.empty:
        return {"checked": 0, "updated": 0, "failed": 0}

    updated = 0
    failed = 0
    checked = 0
    engine = get_engine()
    with engine.begin() as conn:
        for _, item in rows.iterrows():
            checked += 1
            post_id = str(item["post_id"])
            url = str(item["url"])
            try:
                detail = fetch_guba_detail(url)
                publish_time = detail.get("publish_time")
                if publish_time is None or pd.isna(publish_time):
                    failed += 1
                    continue
                row = {
                    "post_id": post_id,
                    "source": item.get("source"),
                    "bar_name": item.get("bar_name"),
                    "publish_time": publish_time,
                    "trade_date": publish_time.date(),
                    "title": detail.get("title") or item.get("title"),
                    "content": detail.get("content") or item.get("content"),
                    "read_count": detail.get("read_count"),
                    "comment_count": detail.get("comment_count"),
                    "like_count": detail.get("like_count"),
                    "author": None,
                    "url": url,
                    "crawl_time": item.get("crawl_time") or datetime.now(),
                }
                row = enrich_row_metadata(row)
                conn.execute(
                    text(
                        """
                        UPDATE sentiment_guba_raw
                        SET publish_time = :publish_time,
                            trade_date = :trade_date,
                            title = :title,
                            content = :content,
                            read_count = COALESCE(:read_count, read_count),
                            comment_count = COALESCE(:comment_count, comment_count),
                            like_count = COALESCE(:like_count, like_count),
                            available_time = :available_time,
                            raw_hash = :raw_hash,
                            content_hash = :content_hash,
                            author_id_hash = :author_id_hash
                        WHERE post_id = :post_id
                        """
                    ),
                    row,
                )
                updated += 1
            except Exception as exc:
                failed += 1
                logger.warning("failed to repair guba post_id=%s url=%s error=%s", post_id, url, exc)
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    return {"checked": checked, "updated": updated, "failed": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description="补齐股吧历史帖子真实发布时间。")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()
    print(repair_guba_publish_time(limit=args.limit, sleep_seconds=args.sleep))


if __name__ == "__main__":
    main()
