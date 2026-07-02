from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from src.macro_events.config import ensure_macro_event_dirs
from src.macro_events.schemas import MacroEvent


def load_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError(f"JSON root must be list: {path}")
    return data


def save_events(path: Path, events: Iterable[MacroEvent | dict]) -> None:
    ensure_macro_event_dirs()
    rows = [event.to_dict() if isinstance(event, MacroEvent) else MacroEvent.from_dict(event).to_dict() for event in events]
    rows = dedupe_rows(rows)
    rows.sort(key=lambda row: (row.get("start_date", ""), row.get("event_type", ""), row.get("event_name", "")))
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def dedupe_rows(rows: Iterable[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for row in rows:
        normalized = MacroEvent.from_dict(row).to_dict()
        key = normalized["event_id"]
        old = merged.get(key)
        if old is None:
            merged[key] = normalized
            continue
        merged[key] = choose_better(old, normalized)
    return list(merged.values())


def choose_better(left: dict, right: dict) -> dict:
    left_score = _quality_score(left)
    right_score = _quality_score(right)
    if right_score > left_score:
        return right
    if right_score < left_score:
        return left
    merged = dict(left)
    for key, value in right.items():
        if not merged.get(key) and value:
            merged[key] = value
    merged["tags"] = sorted(set(left.get("tags", [])) | set(right.get("tags", [])))
    return merged


def _quality_score(row: dict) -> int:
    score = 0
    for field in ("start_date", "end_date", "publish_date", "source_org", "source_title", "source_url"):
        if row.get(field):
            score += 1
    if row.get("status") == "confirmed":
        score += 3
    if row.get("source_text"):
        score += 1
    return score


def merge_into(path: Path, new_events: Iterable[MacroEvent | dict]) -> list[dict]:
    old_rows = load_events(path)
    new_rows = [event.to_dict() if isinstance(event, MacroEvent) else event for event in new_events]
    rows = dedupe_rows([*old_rows, *new_rows])
    save_events(path, rows)
    return rows

