from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import text

from src.common.config import project_path
from src.common.db import get_engine


MIGRATION_PATTERN = re.compile(r"^\d{3}_[a-z0-9_]+\.sql$")


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    sql: str
    checksum: str


def split_sql_statements(sql: str) -> list[str]:
    lines = [line for line in sql.splitlines() if not line.lstrip().startswith("--")]
    cleaned = "\n".join(lines)
    return [statement.strip() for statement in cleaned.split(";") if statement.strip()]


def discover_migrations(directory: str | Path | None = None) -> list[Migration]:
    migration_dir = Path(directory) if directory else project_path("sql", "migrations")
    migrations: list[Migration] = []
    for path in sorted(migration_dir.glob("*.sql")):
        if not MIGRATION_PATTERN.match(path.name):
            raise ValueError(f"迁移文件名不符合规则: {path.name}")
        sql = path.read_text(encoding="utf-8")
        migrations.append(
            Migration(
                version=path.stem,
                path=path,
                sql=sql,
                checksum=hashlib.sha256(sql.encode("utf-8")).hexdigest(),
            )
        )
    return migrations


def apply_migrations(directory: str | Path | None = None, dry_run: bool = False) -> list[str]:
    migrations = discover_migrations(directory)
    if dry_run:
        return [migration.version for migration in migrations]

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS schema_migration (
                    version VARCHAR(80) PRIMARY KEY,
                    checksum CHAR(64) NOT NULL,
                    applied_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )
        )
        rows = conn.execute(text("SELECT version, checksum FROM schema_migration")).mappings()
        applied = {str(row["version"]): str(row["checksum"]) for row in rows}

    completed: list[str] = []
    for migration in migrations:
        old_checksum = applied.get(migration.version)
        if old_checksum:
            if old_checksum != migration.checksum:
                raise RuntimeError(f"已执行迁移被修改: {migration.version}")
            continue
        statements = split_sql_statements(migration.sql)
        with engine.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))
            conn.execute(
                text("INSERT INTO schema_migration (version, checksum) VALUES (:version, :checksum)"),
                {"version": migration.version, "checksum": migration.checksum},
            )
        completed.append(migration.version)
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description="执行数据采集 SQL migration。")
    parser.add_argument("--dry-run", action="store_true", help="仅列出待识别的迁移文件，不连接数据库。")
    args = parser.parse_args()
    versions = apply_migrations(dry_run=args.dry_run)
    for version in versions:
        print(version)


if __name__ == "__main__":
    main()

