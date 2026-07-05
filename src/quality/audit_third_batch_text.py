from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import inspect, text

from src.common.config import project_path
from src.common.db import get_engine


TABLES = {
    "sentiment_guba_raw": {
        "date_column": "trade_date",
        "time_column": "publish_time",
        "id_column": "post_id",
        "source_column": "source",
        "required": [
            "post_id",
            "source",
            "bar_name",
            "publish_time",
            "trade_date",
            "title",
            "content",
            "url",
            "available_time",
            "raw_hash",
        ],
    },
    "sentiment_guba_comment_raw": {
        "date_column": "trade_date",
        "time_column": "publish_time",
        "id_column": "comment_id",
        "source_column": "source",
        "required": [
            "comment_id",
            "post_id",
            "source",
            "bar_name",
            "publish_time",
            "trade_date",
            "content",
            "url",
            "available_time",
            "raw_hash",
        ],
    },
    "news_raw": {
        "date_column": "trade_date",
        "time_column": "publish_time",
        "id_column": "news_id",
        "source_column": "source",
        "required": [
            "news_id",
            "source",
            "publish_time",
            "trade_date",
            "title",
            "content",
            "available_time",
            "raw_hash",
            "content_hash",
        ],
    },
    "sentiment_social_raw": {
        "date_column": "trade_date",
        "time_column": "publish_time",
        "id_column": "content_id",
        "source_column": "source",
        "required": [
            "content_id",
            "source",
            "publish_time",
            "trade_date",
            "available_time",
            "raw_hash",
        ],
    },
}


def _read_sql(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def _table_columns(table: str) -> list[str]:
    rows = _read_sql(
        """
        SELECT COLUMN_NAME
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table
        ORDER BY ORDINAL_POSITION
        """,
        {"table": table},
    )
    return [str(item) for item in rows["COLUMN_NAME"].tolist()]


def _json_ready(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _records_json_ready(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.astype(object).where(pd.notna(frame), None).to_dict(orient="records")
    return [
        {key: _json_ready(value) for key, value in row.items()}
        for row in records
    ]


def audit_table(table: str, spec: dict[str, Any]) -> dict[str, Any]:
    engine = get_engine()
    if not inspect(engine).has_table(table):
        return {"table": table, "exists": False}

    columns = _table_columns(table)
    date_col = spec["date_column"]
    time_col = spec["time_column"]
    id_col = spec["id_column"]
    source_col = spec["source_column"]
    required = [column for column in spec["required"] if column in columns]
    missing_required_columns = [column for column in spec["required"] if column not in columns]

    null_selects = [
        f"SUM(CASE WHEN `{column}` IS NULL OR CAST(`{column}` AS CHAR) = '' THEN 1 ELSE 0 END) AS `null__{column}`"
        for column in required
    ]
    summary = _read_sql(
        f"""
        SELECT
            COUNT(*) AS rows_count,
            MIN(`{time_col}`) AS min_publish_time,
            MAX(`{time_col}`) AS max_publish_time,
            MIN(`{date_col}`) AS min_trade_date,
            MAX(`{date_col}`) AS max_trade_date,
            COUNT(DISTINCT `{date_col}`) AS distinct_trade_days,
            COUNT(DISTINCT `{id_col}`) AS distinct_ids,
            COUNT(*) - COUNT(DISTINCT `{id_col}`) AS duplicate_id_rows
            {',' if null_selects else ''}
            {', '.join(null_selects)}
        FROM `{table}`
        """
    ).iloc[0].to_dict()

    rows_count = int(summary.get("rows_count") or 0)
    null_rates = {}
    for column in required:
        null_count = int(summary.get(f"null__{column}") or 0)
        null_rates[column] = null_count / rows_count if rows_count else None

    source_rows = _read_sql(
        f"""
        SELECT `{source_col}` AS source, COUNT(*) AS rows_count,
               MIN(`{date_col}`) AS min_trade_date,
               MAX(`{date_col}`) AS max_trade_date
        FROM `{table}`
        GROUP BY `{source_col}`
        ORDER BY rows_count DESC
        LIMIT 20
        """
    )
    monthly_rows = _read_sql(
        f"""
        SELECT DATE_FORMAT(`{date_col}`, '%Y-%m') AS month,
               COUNT(*) AS rows_count,
               COUNT(DISTINCT `{date_col}`) AS active_days
        FROM `{table}`
        WHERE `{date_col}` IS NOT NULL
        GROUP BY month
        ORDER BY month DESC
        LIMIT 24
        """
    )

    summary_clean = {
        key: _json_ready(value)
        for key, value in summary.items()
        if not key.startswith("null__")
    }
    return {
        "table": table,
        "exists": True,
        "columns": columns,
        "missing_required_columns": missing_required_columns,
        "summary": summary_clean,
        "null_rates": null_rates,
        "sources": _records_json_ready(source_rows),
        "recent_monthly_coverage": _records_json_ready(monthly_rows),
    }


def run_audit(output: str | None = None) -> dict[str, Any]:
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch": "third_batch_text_raw",
        "scope": "sentiment/news raw data audit only; no feature/model/LLM extraction",
        "tables": [audit_table(table, spec) for table, spec in TABLES.items()],
    }
    output_path = Path(output) if output else project_path("data", "reports", "third_batch_text_audit.json")
    if not output_path.is_absolute():
        output_path = project_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["output"] = str(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="审计模型2.0第三批情绪/新闻原始数据。")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_audit(args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
