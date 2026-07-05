from __future__ import annotations

import argparse
import logging
from datetime import date
from logging.handlers import RotatingFileHandler

from src.common.migrations import apply_migrations
from src.common.db import execute_sql
from src.macro_events.cleaner import clean_macro_events
from src.macro_events.config import HISTORY_FILE, HISTORY_START_DATE, LOG_DIR, REVIEW_FILE, ensure_macro_event_dirs
from src.macro_events.crawler import MacroEventCrawler
from src.macro_events.db_store import upsert_macro_events
from src.macro_events.json_store import merge_into, save_events
from src.macro_events.parser import parse_article_to_events
from src.macro_events.predictor import build_future_predictions
from src.macro_events.validator import split_valid_and_review


def build_history(
    start_date: date = HISTORY_START_DATE,
    end_date: date | None = None,
    max_pages_per_keyword: int = 3,
    sleep_seconds: float = 0.5,
    no_network: bool = False,
    rebuild: bool = False,
    write_db: bool = False,
    apply_db_migrations: bool = False,
) -> dict[str, int]:
    ensure_macro_event_dirs()
    logger = _get_logger("macro_events.history_build", LOG_DIR / "history_build.log")
    end_date = end_date or date.today()
    articles = []
    if not no_network:
        crawler = MacroEventCrawler(sleep_seconds=sleep_seconds, logger=logger)
        articles = crawler.crawl_history_candidates(start_date, end_date, max_pages_per_keyword=max_pages_per_keyword)
    events = []
    for article in articles:
        try:
            events.extend(parse_article_to_events(article))
        except Exception as exc:
            logger.exception("parse article failed url=%s error=%s", article.get("url"), exc)
    events = clean_macro_events(events)
    valid, review = split_valid_and_review(events)
    if rebuild:
        save_events(HISTORY_FILE, valid)
        save_events(REVIEW_FILE, review)
        history_rows = valid
        review_rows = review
    else:
        history_rows = merge_into(HISTORY_FILE, valid)
        review_rows = merge_into(REVIEW_FILE, review)
    predicted_rows = merge_into(
        ensure_macro_event_dirs().predicted_file,
        build_future_predictions(today=end_date),
    )
    db_rows = 0
    if write_db:
        if apply_db_migrations:
            apply_migrations()
        execute_sql("DELETE FROM cn_macro_political_event WHERE is_predicted = 1")
        db_rows = upsert_macro_events([*history_rows, *predicted_rows])
    logger.info(
        "history build done articles=%s valid=%s review=%s history_total=%s review_total=%s predicted_total=%s db_rows=%s",
        len(articles),
        len(valid),
        len(review),
        len(history_rows),
        len(review_rows),
        len(predicted_rows),
        db_rows,
    )
    return {
        "articles": len(articles),
        "new_valid_events": len(valid),
        "new_review_events": len(review),
        "history_total": len(history_rows),
        "review_total": len(review_rows),
        "predicted_total": len(predicted_rows),
        "db_upsert_rows": db_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build domestic macro political event JSON history.")
    parser.add_argument("--start-date", default=HISTORY_START_DATE.isoformat())
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--max-pages-per-keyword", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.5)
    parser.add_argument("--no-network", action="store_true", help="Only refresh future prediction JSON.")
    parser.add_argument("--rebuild", action="store_true", help="Overwrite history/review JSON with this run.")
    parser.add_argument("--write-db", action="store_true", help="Write confirmed history rows into MySQL.")
    parser.add_argument("--apply-migrations", action="store_true", help="Apply SQL migrations before writing DB.")
    args = parser.parse_args()
    result = build_history(
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        max_pages_per_keyword=args.max_pages_per_keyword,
        sleep_seconds=args.sleep,
        no_network=args.no_network,
        rebuild=args.rebuild,
        write_db=args.write_db,
        apply_db_migrations=args.apply_migrations,
    )
    print(result)


def _get_logger(name: str, log_path) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    error_handler = RotatingFileHandler(LOG_DIR / "error.log", maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)
    return logger


if __name__ == "__main__":
    main()
