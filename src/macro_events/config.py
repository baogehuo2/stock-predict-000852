from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from src.common.config import project_path


HISTORY_START_DATE = date(2005, 1, 1)
FUTURE_MONTHS = 18

EVENT_TYPES = {
    "TWO_SESSIONS",
    "POLITBURO_MEETING",
    "CEWC",
    "CPC_PLENUM",
}

EVENT_STATUSES = {
    "confirmed",
    "predicted",
    "review",
    "cancelled",
}

DATA_DIR = project_path("data", "macro_events")
LOG_DIR = project_path("logs", "macro_events")

HISTORY_FILE = DATA_DIR / "macro_events_history.json"
PREDICTED_FILE = DATA_DIR / "macro_events_predicted.json"
REVIEW_FILE = DATA_DIR / "macro_events_review_queue.json"


@dataclass(frozen=True)
class RuntimePaths:
    data_dir: Path = DATA_DIR
    log_dir: Path = LOG_DIR
    history_file: Path = HISTORY_FILE
    predicted_file: Path = PREDICTED_FILE
    review_file: Path = REVIEW_FILE


def ensure_macro_event_dirs() -> RuntimePaths:
    paths = RuntimePaths()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    return paths

