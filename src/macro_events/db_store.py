from __future__ import annotations

import json
from datetime import datetime
from typing import Iterable

import pandas as pd

from src.common.db import upsert_dataframe
from src.macro_events.schemas import MacroEvent


TABLE_NAME = "cn_macro_political_event"


def events_to_dataframe(events: Iterable[MacroEvent | dict]) -> pd.DataFrame:
    rows = []
    now = datetime.now()
    for item in events:
        event = item if isinstance(item, MacroEvent) else MacroEvent.from_dict(item)
        row = event.to_dict()
        row["tags"] = json.dumps(row.get("tags", []), ensure_ascii=False)
        row["is_predicted"] = 1 if row.get("is_predicted") else 0
        row["created_at"] = now
        row["updated_at"] = now
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    date_cols = ["start_date", "end_date", "publish_date"]
    for col in date_cols:
        frame[col] = pd.to_datetime(frame[col], errors="coerce").dt.date
    return frame


def upsert_macro_events(events: Iterable[MacroEvent | dict]) -> int:
    frame = events_to_dataframe(events)
    if frame.empty:
        return 0
    return upsert_dataframe(frame, TABLE_NAME, ["event_id"])
