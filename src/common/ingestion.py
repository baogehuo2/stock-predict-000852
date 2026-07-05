from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime

from src.common.db import execute_sql


def _git_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class IngestionRun:
    dataset_name: str
    data_source: str | None = None
    requested_start_date: date | None = None
    requested_end_date: date | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    start_time: datetime = field(default_factory=datetime.now)
    fetched_rows: int = 0
    inserted_rows: int = 0
    updated_rows: int = 0
    skipped_rows: int = 0
    error_rows: int = 0

    def finish(self, status: str, error_message: str | None = None) -> None:
        execute_sql(
            """
            INSERT INTO data_ingestion_log (
                run_id, dataset_name, data_source, start_time, end_time,
                requested_start_date, requested_end_date, fetched_rows,
                inserted_rows, updated_rows, skipped_rows, error_rows,
                status, error_message, code_version
            ) VALUES (
                :run_id, :dataset_name, :data_source, :start_time, :end_time,
                :requested_start_date, :requested_end_date, :fetched_rows,
                :inserted_rows, :updated_rows, :skipped_rows, :error_rows,
                :status, :error_message, :code_version
            ) ON DUPLICATE KEY UPDATE
                end_time = VALUES(end_time), fetched_rows = VALUES(fetched_rows),
                inserted_rows = VALUES(inserted_rows), updated_rows = VALUES(updated_rows),
                skipped_rows = VALUES(skipped_rows), error_rows = VALUES(error_rows),
                status = VALUES(status), error_message = VALUES(error_message),
                code_version = VALUES(code_version)
            """,
            {
                "run_id": self.run_id,
                "dataset_name": self.dataset_name,
                "data_source": self.data_source,
                "start_time": self.start_time,
                "end_time": datetime.now(),
                "requested_start_date": self.requested_start_date,
                "requested_end_date": self.requested_end_date,
                "fetched_rows": self.fetched_rows,
                "inserted_rows": self.inserted_rows,
                "updated_rows": self.updated_rows,
                "skipped_rows": self.skipped_rows,
                "error_rows": self.error_rows,
                "status": status,
                "error_message": error_message,
                "code_version": _git_version(),
            },
        )

