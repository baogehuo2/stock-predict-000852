from __future__ import annotations

import argparse

from src.common.config import project_path
from src.common.db import execute_sql
from src.common.migrations import apply_migrations
from src.macro_events.config import HISTORY_FILE, REVIEW_FILE, ensure_macro_event_dirs
from src.macro_events.csv_store import load_events_csv
from src.macro_events.db_store import upsert_macro_events
from src.macro_events.json_store import save_events


DEFAULT_FULL_CSV = project_path("docs", "architecture", "macro_events_full_clean.csv")
DEFAULT_TODO_CSV = project_path("docs", "architecture", "macro_events_need_fill_clean.csv")


def import_existing_events(write_db: bool = False, apply_db_migrations: bool = False) -> dict[str, int]:
    ensure_macro_event_dirs()
    raw_history_events = load_events_csv(DEFAULT_FULL_CSV)
    history_events = [
        event for event in raw_history_events
        if not event.is_predicted and event.status.lower() != "predicted"
    ]
    review_events = load_events_csv(DEFAULT_TODO_CSV)
    save_events(HISTORY_FILE, history_events)
    save_events(REVIEW_FILE, review_events)
    db_rows = 0
    if write_db:
        if apply_db_migrations:
            apply_migrations()
        execute_sql("DELETE FROM cn_macro_political_event WHERE is_predicted = 0")
        db_rows = upsert_macro_events(history_events)
    return {
        "history_json_rows": len(history_events),
        "review_json_rows": len(review_events),
        "db_upsert_rows": db_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import confirmed domestic macro political events from curated CSV.")
    parser.add_argument("--write-db", action="store_true", help="Write confirmed history rows into MySQL.")
    parser.add_argument("--apply-migrations", action="store_true", help="Apply SQL migrations before writing DB.")
    args = parser.parse_args()
    print(import_existing_events(write_db=args.write_db, apply_db_migrations=args.apply_migrations))


if __name__ == "__main__":
    main()
