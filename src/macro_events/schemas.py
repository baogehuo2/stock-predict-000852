from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any

from src.macro_events.config import EVENT_STATUSES, EVENT_TYPES


def parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    text = text.replace("/", "-").replace(".", "-")
    for fmt in ("%Y-%m-%d", "%Y年%m月%d日", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def stable_event_id(event_type: str, start_date: str, event_name: str) -> str:
    key = f"{event_type.strip().upper()}|{start_date}|{event_name.strip()}"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return f"{event_type.strip().lower()}_{digest}"


@dataclass
class MacroEvent:
    event_id: str
    event_type: str
    event_name: str
    event_level: str = "national"
    start_date: str = ""
    end_date: str = ""
    publish_date: str = ""
    source_org: str = ""
    source_title: str = ""
    source_url: str = ""
    source_text: str = ""
    is_predicted: bool = False
    prediction_confidence: float = 1.0
    status: str = "confirmed"
    tags: list[str] = field(default_factory=list)
    note: str = ""

    def normalized(self) -> "MacroEvent":
        event_type = self.event_type.strip().upper()
        status = self.status.strip().lower()
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unsupported event_type: {self.event_type}")
        if status not in EVENT_STATUSES:
            raise ValueError(f"Unsupported status: {self.status}")
        start = parse_date(self.start_date)
        end = parse_date(self.end_date) or start
        publish = parse_date(self.publish_date)
        start_text = start.isoformat() if start else ""
        end_text = end.isoformat() if end else ""
        publish_text = publish.isoformat() if publish else ""
        event_name = self.event_name.strip()
        event_id = self.event_id or stable_event_id(event_type, start_text, event_name)
        return MacroEvent(
            event_id=event_id,
            event_type=event_type,
            event_name=event_name,
            event_level=self.event_level or "national",
            start_date=start_text,
            end_date=end_text,
            publish_date=publish_text,
            source_org=self.source_org.strip(),
            source_title=self.source_title.strip(),
            source_url=self.source_url.strip(),
            source_text=self.source_text.strip(),
            is_predicted=bool(self.is_predicted),
            prediction_confidence=float(self.prediction_confidence),
            status=status,
            tags=sorted({str(tag).strip() for tag in self.tags if str(tag).strip()}),
            note=self.note.strip(),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MacroEvent":
        tags = row.get("tags", []) or []
        if isinstance(tags, str):
            tags = [item.strip() for item in tags.replace("；", ";").split(";") if item.strip()]
        return cls(
            event_id=str(row.get("event_id", "")),
            event_type=str(row.get("event_type", "")),
            event_name=str(row.get("event_name", "")),
            event_level=str(row.get("event_level", "national")),
            start_date=str(row.get("start_date", "")),
            end_date=str(row.get("end_date", "")),
            publish_date=str(row.get("publish_date", "")),
            source_org=str(row.get("source_org", "")),
            source_title=str(row.get("source_title", "")),
            source_url=str(row.get("source_url", "")),
            source_text=str(row.get("source_text", "")),
            is_predicted=_to_bool(row.get("is_predicted", False)),
            prediction_confidence=float(row.get("prediction_confidence", 1.0) or 1.0),
            status=str(row.get("status", "confirmed")),
            tags=list(tags),
            note=str(row.get("note", "")),
        )


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}
