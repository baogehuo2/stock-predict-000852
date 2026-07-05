from __future__ import annotations

import argparse

import pandas as pd
from sqlalchemy import text

from src.common.db import get_engine
from src.common.logger import get_logger


logger = get_logger(__name__)

NEWS_TABLES = {
    "news_raw": "news_id",
    "event_daily": "source_ids",
}


def _columns(database: str, table: str) -> list[str]:
    sql = (
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=:database AND TABLE_NAME=:table ORDER BY ORDINAL_POSITION"
    )
    with get_engine(database=database).connect() as conn:
        rows = conn.execute(text(sql), {"database": database, "table": table}).fetchall()
    return [str(row[0]) for row in rows]


def _table_summary(database: str, table: str) -> dict:
    with get_engine(database=database).connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM `{database}`.`{table}`")).scalar()
        dates = conn.execute(
            text(f"SELECT MIN(trade_date), MAX(trade_date) FROM `{database}`.`{table}`")
        ).fetchone()
    return {
        "database": database,
        "table": table,
        "rows": int(count or 0),
        "min_trade_date": dates[0] if dates else None,
        "max_trade_date": dates[1] if dates else None,
    }


def sync_table(
    source_database: str,
    target_database: str,
    table: str,
    unique_column: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    source_cols = _columns(source_database, table)
    target_cols = _columns(target_database, table)
    common_cols = [
        column
        for column in source_cols
        if column in target_cols and column != "id"
    ]
    if unique_column not in common_cols:
        raise RuntimeError(f"{table} common columns do not include unique column {unique_column}.")

    before = _table_summary(target_database, table)
    col_sql = ", ".join(f"`{column}`" for column in common_cols)
    select_sql = ", ".join(f"s.`{column}`" for column in common_cols)
    update_cols = [column for column in common_cols if column != unique_column]
    update_sql = ", ".join(f"`{column}`=VALUES(`{column}`)" for column in update_cols)
    where_parts = [f"s.`{unique_column}` IS NOT NULL", f"s.`{unique_column}` <> ''"]
    params: dict[str, str] = {}
    if start_date:
        where_parts.append("s.`trade_date` >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where_parts.append("s.`trade_date` <= :end_date")
        params["end_date"] = end_date
    where_sql = " AND ".join(where_parts)
    sql = (
        f"INSERT INTO `{target_database}`.`{table}` ({col_sql}) "
        f"SELECT {select_sql} FROM `{source_database}`.`{table}` s "
        f"WHERE {where_sql} "
        f"ON DUPLICATE KEY UPDATE {update_sql}"
    )
    with get_engine(database=target_database).begin() as conn:
        result = conn.execute(text(sql), params)
    after = _table_summary(target_database, table)
    report = {
        "table": table,
        "affected_rows": int(result.rowcount or 0),
        "target_rows_before": before["rows"],
        "target_rows_after": after["rows"],
        "target_min_trade_date": after["min_trade_date"],
        "target_max_trade_date": after["max_trade_date"],
    }
    logger.info("synced table=%s report=%s", table, report)
    return report


def sync_news_databases(
    source_database: str = "zz1000_botumn",
    target_database: str = "zz1000_predict",
    tables: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    if source_database == target_database:
        raise ValueError("source_database and target_database must be different.")
    table_list = tables or list(NEWS_TABLES)
    reports = []
    for table in table_list:
        if table not in NEWS_TABLES:
            raise ValueError(f"Unsupported table: {table}. Supported: {sorted(NEWS_TABLES)}")
        reports.append(
            sync_table(
                source_database=source_database,
                target_database=target_database,
                table=table,
                unique_column=NEWS_TABLES[table],
                start_date=start_date,
                end_date=end_date,
            )
        )
    return pd.DataFrame(reports)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync news/event tables between MySQL databases.")
    parser.add_argument("--source-database", default="zz1000_botumn")
    parser.add_argument("--target-database", default="zz1000_predict")
    parser.add_argument("--table", action="append", choices=sorted(NEWS_TABLES), help="Repeatable. Defaults to news_raw + event_daily.")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    report = sync_news_databases(
        source_database=args.source_database,
        target_database=args.target_database,
        tables=args.table,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
