from __future__ import annotations

from src.macro_events.schemas import MacroEvent, parse_date


def split_valid_and_review(events: list[MacroEvent]) -> tuple[list[MacroEvent], list[MacroEvent]]:
    valid: list[MacroEvent] = []
    review: list[MacroEvent] = []
    for event in events:
        normalized = event.normalized()
        reasons = validate_event(normalized)
        if reasons:
            normalized.status = "review"
            normalized.note = "; ".join([normalized.note, *reasons]).strip("; ")
            review.append(normalized)
        else:
            valid.append(normalized)
    return valid, review


def validate_event(event: MacroEvent) -> list[str]:
    reasons: list[str] = []
    if not event.event_name:
        reasons.append("缺少事件名称")
    if not parse_date(event.start_date):
        reasons.append("缺少或无法解析开始日期")
    if not parse_date(event.publish_date) and not event.is_predicted:
        reasons.append("缺少或无法解析发布日期")
    if event.status == "confirmed":
        if not event.source_org:
            reasons.append("缺少来源机构")
        if not event.source_url:
            reasons.append("缺少原文链接")
        if not event.source_title:
            reasons.append("缺少来源标题")
    start = parse_date(event.start_date)
    end = parse_date(event.end_date)
    if start and end and end < start:
        reasons.append("结束日期早于开始日期")
    return reasons
