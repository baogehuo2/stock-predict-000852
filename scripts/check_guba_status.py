from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.db import read_sql


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar-name", default="中证1000吧")
    parser.add_argument("--bar-code", default="zssh000852")
    args = parser.parse_args()

    pattern = f"%/news,{args.bar_code},%"
    params = {"bar": args.bar_name, "pat": pattern}
    summary = read_sql(
        "SELECT COUNT(*) cnt, MIN(publish_time) min_time, MAX(publish_time) max_time, "
        "COUNT(DISTINCT trade_date) trade_days, "
        "SUM(CASE WHEN url NOT LIKE :pat THEN 1 ELSE 0 END) dirty "
        "FROM sentiment_guba_raw WHERE bar_name=:bar",
        params,
    )
    dup = read_sql(
        "SELECT COUNT(*) duplicate_url_count FROM ("
        "SELECT url FROM sentiment_guba_raw WHERE bar_name=:bar GROUP BY url HAVING COUNT(*)>1"
        ") t",
        {"bar": args.bar_name},
    )
    earliest = read_sql(
        "SELECT trade_date, COUNT(*) cnt FROM sentiment_guba_raw WHERE bar_name=:bar "
        "GROUP BY trade_date ORDER BY trade_date ASC LIMIT 15",
        {"bar": args.bar_name},
    )
    latest = read_sql(
        "SELECT trade_date, COUNT(*) cnt FROM sentiment_guba_raw WHERE bar_name=:bar "
        "GROUP BY trade_date ORDER BY trade_date DESC LIMIT 15",
        {"bar": args.bar_name},
    )
    comments = read_sql(
        "SELECT COUNT(*) comment_cnt, COUNT(DISTINCT post_id) commented_posts, "
        "MIN(publish_time) min_time, MAX(publish_time) max_time, "
        "COUNT(DISTINCT trade_date) trade_days "
        "FROM sentiment_guba_comment_raw WHERE bar_name=:bar",
        {"bar": args.bar_name},
    )
    comment_latest = read_sql(
        "SELECT trade_date, COUNT(*) cnt FROM sentiment_guba_comment_raw WHERE bar_name=:bar "
        "GROUP BY trade_date ORDER BY trade_date DESC LIMIT 15",
        {"bar": args.bar_name},
    )

    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nDUP")
    print(dup.to_string(index=False))
    print("\nEARLIEST_DAILY")
    print(earliest.to_string(index=False))
    print("\nLATEST_DAILY")
    print(latest.to_string(index=False))
    print("\nCOMMENT_SUMMARY")
    print(comments.to_string(index=False))
    print("\nCOMMENT_LATEST_DAILY")
    print(comment_latest.to_string(index=False))


if __name__ == "__main__":
    main()
