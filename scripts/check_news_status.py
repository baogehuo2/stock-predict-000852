from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.db import read_sql


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    args = parser.parse_args()

    where = "WHERE source=:source" if args.source else ""
    params = {"source": args.source} if args.source else {}
    summary = read_sql(
        f"SELECT COUNT(*) cnt, MIN(publish_time) min_time, MAX(publish_time) max_time, "
        f"COUNT(DISTINCT trade_date) trade_days FROM news_raw {where}",
        params,
    )
    by_source = read_sql(
        f"SELECT source, COUNT(*) cnt, MIN(publish_time) min_time, MAX(publish_time) max_time "
        f"FROM news_raw {where} GROUP BY source ORDER BY cnt DESC",
        params,
    )
    latest = read_sql(
        f"SELECT source, publish_time, trade_date, title, url FROM news_raw {where} "
        f"ORDER BY publish_time DESC LIMIT 15",
        params,
    )
    print("SUMMARY")
    print(summary.to_string(index=False))
    print("\nBY_SOURCE")
    print(by_source.to_string(index=False))
    print("\nLATEST")
    print(latest.to_string(index=False))


if __name__ == "__main__":
    main()

