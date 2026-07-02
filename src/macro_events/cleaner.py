from __future__ import annotations

import re
from collections.abc import Iterable

from src.macro_events.schemas import MacroEvent, stable_event_id


MOJIBAKE_RE = re.compile(r"[�]{1,}|\?{3,}|[ȫЭϯίλ쵼]+")
INTERNAL_MEETING_RE = re.compile(r"(第?[一二三四五六七八九十]+次(?:全体|主席团)会议|代表团会议|分组会议)")


def clean_macro_events(events: Iterable[MacroEvent | dict]) -> list[MacroEvent]:
    rows: list[MacroEvent] = []
    for item in events:
        event = item if isinstance(item, MacroEvent) else MacroEvent.from_dict(item)
        try:
            normalized = event.normalized()
        except ValueError:
            continue
        if _has_mojibake(normalized):
            continue
        normalized = _normalize_two_sessions(normalized)
        normalized = _normalize_cpc_plenum(normalized)
        normalized.event_id = stable_event_id(normalized.event_type, normalized.start_date, normalized.event_name)
        rows.append(normalized)
    return _dedupe(rows)


def _normalize_two_sessions(event: MacroEvent) -> MacroEvent:
    if event.event_type != "TWO_SESSIONS":
        return event
    year = event.start_date[:4]
    kind = "全国政协" if "政协" in " ".join([event.event_name, event.source_title, event.source_text, ";".join(event.tags)]) else "全国人大"
    event.event_name = f"{year}年{kind}会议" if year else f"{kind}会议"
    event.source_title = f"{event.event_name}官方报道"
    event.source_text = (
        f"{event.event_name}于{event.start_date}开幕，{event.end_date}闭幕。"
        "本库仅保留年度大会的时间窗口、官方来源和宏观事件摘要，不采集大会期间内部会议。"
    )
    event.tags = sorted({kind, "两会", year, *[tag for tag in event.tags if "届" not in tag and "次" not in tag]})
    event.note = INTERNAL_MEETING_RE.sub("", event.note).strip()
    return event


def _normalize_cpc_plenum(event: MacroEvent) -> MacroEvent:
    if event.event_type != "CPC_PLENUM":
        return event
    event.event_name = event.event_name.replace("公报", "").strip()
    event.tags = sorted({"中央全会", event.event_name, *event.tags})
    return event


def _has_mojibake(event: MacroEvent) -> bool:
    text = " ".join(
        [
            event.event_name,
            event.source_org,
            event.source_title,
            event.source_text,
            event.note,
            ";".join(event.tags),
        ]
    )
    return bool(MOJIBAKE_RE.search(text))


def _dedupe(events: list[MacroEvent]) -> list[MacroEvent]:
    best: dict[str, MacroEvent] = {}
    for event in events:
        old = best.get(event.event_id)
        if old is None or _quality_score(event) > _quality_score(old):
            best[event.event_id] = event
    return sorted(best.values(), key=lambda row: (row.start_date, row.event_type, row.event_name))


def _quality_score(event: MacroEvent) -> int:
    score = 0
    for field in ("start_date", "end_date", "publish_date", "source_org", "source_title", "source_url"):
        if getattr(event, field):
            score += 2
    if event.status == "confirmed":
        score += 5
    if event.source_text:
        score += min(len(event.source_text) // 100, 10)
    return score
