from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.collectors.collect_guba_eastmoney import parse_guba_page
from src.common.config import load_yaml, project_path
from src.common.db import upsert_dataframe
from src.common.logger import get_logger


logger = get_logger(__name__)


def _selected_bars(bar_name: str | None = None) -> list[dict]:
    bars = load_yaml(project_path("config", "symbols.yaml"))["guba_bars"]
    if not bar_name:
        return bars
    return [bar for bar in bars if bar["bar_name"] == bar_name or bar.get("topic") == bar_name]


def collect_guba_history(
    start_page: int = 1,
    end_page: int = 200,
    bar_name: str | None = None,
    sleep_seconds: float = 0.8,
    detail_sleep_seconds: float = 0.2,
    batch_pages: int = 10,
    fetch_detail: bool = True,
) -> int:
    if start_page < 1 or end_page < start_page:
        raise ValueError("Invalid page range.")
    bars = _selected_bars(bar_name)
    if not bars:
        raise ValueError(f"No guba bar matched: {bar_name}")

    total = 0
    for bar in bars:
        buffer: list[dict] = []
        empty_pages = 0
        logger.info("start guba history bar=%s pages=%s-%s", bar["bar_name"], start_page, end_page)
        for page in range(start_page, end_page + 1):
            try:
                rows = parse_guba_page(
                    bar["bar_name"],
                    bar["url"],
                    page,
                    fetch_detail=fetch_detail,
                    detail_sleep=detail_sleep_seconds,
                )
                if not rows:
                    empty_pages += 1
                    logger.warning(
                        "empty guba history page bar=%s page=%s empty_pages=%s",
                        bar["bar_name"],
                        page,
                        empty_pages,
                    )
                else:
                    empty_pages = 0
                    buffer.extend(rows)
                    logger.info("guba history bar=%s page=%s rows=%s", bar["bar_name"], page, len(rows))

                if buffer and (page - start_page + 1) % batch_pages == 0:
                    total += upsert_dataframe(pd.DataFrame(buffer), "sentiment_guba_raw", ["post_id"])
                    buffer = []

                if empty_pages >= 5:
                    logger.warning("stop bar=%s after 5 continuous empty pages at page=%s", bar["bar_name"], page)
                    break
            except Exception as exc:
                logger.exception("failed guba history bar=%s page=%s: %s", bar["bar_name"], page, exc)
            finally:
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

        if buffer:
            total += upsert_dataframe(pd.DataFrame(buffer), "sentiment_guba_raw", ["post_id"])
        logger.info("finish guba history bar=%s total_so_far=%s", bar["bar_name"], total)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Eastmoney Guba history pages.")
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=200)
    parser.add_argument("--bar-name", help="Optional bar_name or topic from config/symbols.yaml")
    parser.add_argument("--sleep", type=float, default=0.8, help="Sleep seconds between pages.")
    parser.add_argument("--detail-sleep", type=float, default=0.2, help="Sleep seconds between detail pages.")
    parser.add_argument("--batch-pages", type=int, default=10)
    parser.add_argument("--no-detail", action="store_true", help="Do not fetch detail pages; faster but no real historical publish_time.")
    args = parser.parse_args()
    print(
        collect_guba_history(
            start_page=args.start_page,
            end_page=args.end_page,
            bar_name=args.bar_name,
            sleep_seconds=args.sleep,
            detail_sleep_seconds=args.detail_sleep,
            batch_pages=args.batch_pages,
            fetch_detail=not args.no_detail,
        )
    )


if __name__ == "__main__":
    main()
