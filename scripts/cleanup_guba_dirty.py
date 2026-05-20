from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.common.db import execute_sql, read_sql


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bar-name", required=True)
    parser.add_argument("--bar-code", required=True)
    args = parser.parse_args()

    pattern = f"%/news,{args.bar_code},%"
    execute_sql(
        "DELETE FROM sentiment_guba_raw WHERE bar_name=:bar_name AND url NOT LIKE :pattern",
        {"bar_name": args.bar_name, "pattern": pattern},
    )
    df = read_sql(
        "SELECT COUNT(*) cnt, MIN(publish_time) min_time, MAX(publish_time) max_time, "
        "SUM(CASE WHEN url NOT LIKE :pattern THEN 1 ELSE 0 END) dirty "
        "FROM sentiment_guba_raw WHERE bar_name=:bar_name",
        {"bar_name": args.bar_name, "pattern": pattern},
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()

