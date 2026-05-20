from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.parse import quote_plus

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .config import get_config, load_yaml, project_path
from .secrets import get_secret


def _mysql_url(database: str | None = None) -> str:
    db_cfg = load_yaml(project_path("config", "db.yaml"))["mysql"]
    password = quote_plus(str(get_secret(db_cfg.get("password_secret_path", "database.password"), "")))
    host = db_cfg["host"]
    port = int(db_cfg.get("port", 3306))
    user = db_cfg["user"]
    charset = db_cfg.get("charset", "utf8mb4")
    db_name = database if database is not None else db_cfg["database"]
    db_part = f"/{db_name}" if db_name else ""
    return f"mysql+pymysql://{user}:{password}@{host}:{port}{db_part}?charset={charset}"


def get_engine(database: str | None = None) -> Engine:
    return create_engine(_mysql_url(database), pool_pre_ping=True, future=True)


def get_database_name() -> str:
    return load_yaml(project_path("config", "db.yaml"))["mysql"]["database"]


def init_database(sql_path: str | Path | None = None) -> None:
    database = get_database_name()
    admin_engine = get_engine(database="")
    with admin_engine.begin() as conn:
        conn.execute(text(f"CREATE DATABASE IF NOT EXISTS `{database}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"))

    sql_file = Path(sql_path) if sql_path else project_path("sql", "create_tables.sql")
    sql_text = sql_file.read_text(encoding="utf-8")
    engine = get_engine()
    statements = [stmt.strip() for stmt in sql_text.split(";") if stmt.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def read_sql(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        result = conn.execute(text(sql), params or {})
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def execute_sql(sql: str, params: dict | None = None) -> None:
    with get_engine().begin() as conn:
        conn.execute(text(sql), params or {})


def upsert_dataframe(df: pd.DataFrame, table: str, unique_cols: Iterable[str]) -> int:
    if df.empty:
        return 0
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype(object).where(pd.notna(df), None)
    cols = list(df.columns)
    unique_cols = list(unique_cols)
    placeholders = ", ".join([f":{c}" for c in cols])
    col_names = ", ".join([f"`{c}`" for c in cols])
    update_cols = [c for c in cols if c not in unique_cols and c != "id"]
    updates = ", ".join([f"`{c}` = VALUES(`{c}`)" for c in update_cols])
    sql = f"INSERT INTO `{table}` ({col_names}) VALUES ({placeholders})"
    if updates:
        sql += f" ON DUPLICATE KEY UPDATE {updates}"
    records = df.to_dict(orient="records")
    with get_engine().begin() as conn:
        conn.execute(text(sql), records)
    return len(records)
