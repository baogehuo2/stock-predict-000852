from __future__ import annotations

import re
from datetime import date, timedelta

from src.macro_events.config import EVENT_TYPES
from src.macro_events.schemas import MacroEvent, stable_event_id
from src.macro_events.sources import ECONOMIC_POLITBURO_KEYWORDS


DATE_RE = re.compile(r"(?P<year>20\d{2})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日")
RANGE_RE = re.compile(
    r"(?P<year>20\d{2})年(?P<start_month>\d{1,2})月(?P<start_day>\d{1,2})日"
    r"(?:至|到|-|—|--)"
    r"(?:(?P<end_month>\d{1,2})月)?(?P<end_day>\d{1,2})日"
)
PARTIAL_RANGE_RE = re.compile(
    r"(?P<start_month>\d{1,2})月(?P<start_day>\d{1,2})日"
    r"(?:至|到|-|—|--)"
    r"(?:(?P<end_month>\d{1,2})月)?(?P<end_day>\d{1,2})日"
)
PLENUM_RE = re.compile(r"(?P<session>[十二一三四五六七八九〇零]+届)(?P<number>[一二三四五六七八九十]+中)全会")


def parse_article_to_events(article: dict) -> list[MacroEvent]:
    title = str(article.get("title", "")).strip()
    text = str(article.get("text", "") or title).strip()
    publish_date = str(article.get("publish_date", "")).strip()
    source_org = str(article.get("source_org", "")).strip()
    source_url = str(article.get("url", "")).strip()
    body = f"{title}\n{text}"
    event_type_hint = str(article.get("event_type_hint", "")).strip().upper()
    event_types = [event_type_hint] if event_type_hint in EVENT_TYPES else detect_event_types(body)
    events: list[MacroEvent] = []
    for event_type in event_types:
        start, end = extract_date_range(body, publish_date)
        name = normalize_event_name(event_type, title, body, start)
        tags = extract_tags(event_type, body, start)
        status = "confirmed" if start and source_url else "review"
        events.append(
            MacroEvent(
                event_id=stable_event_id(event_type, start or "", name),
                event_type=event_type,
                event_name=name,
                start_date=start or "",
                end_date=end or start or "",
                publish_date=publish_date,
                source_org=source_org,
                source_title=title,
                source_url=source_url,
                source_text=_clean_summary(event_type, name, text, start, end),
                is_predicted=False,
                prediction_confidence=1.0,
                status=status,
                tags=tags,
                note="" if status == "confirmed" else "自动抽取信息不完整，需人工确认",
            )
        )
    return events


def detect_event_types(text: str) -> list[str]:
    event_types: list[str] = []
    if any(keyword in text for keyword in ("全国人民代表大会", "全国人大", "全国政协", "中国人民政治协商会议")):
        if any(keyword in text for keyword in ("会议", "开幕", "闭幕", "议程")):
            event_types.append("TWO_SESSIONS")
    if "中共中央政治局" in text and "会议" in text:
        event_types.append("POLITBURO_MEETING")
    if "中央经济工作会议" in text:
        event_types.append("CEWC")
    if ("中央委员会" in text and "全体会议" in text) or PLENUM_RE.search(text):
        event_types.append("CPC_PLENUM")
    return sorted(set(event_types))


def extract_date_range(text: str, fallback_publish_date: str = "") -> tuple[str, str]:
    range_match = RANGE_RE.search(text)
    if range_match:
        year = int(range_match.group("year"))
        start_month = int(range_match.group("start_month"))
        start_day = int(range_match.group("start_day"))
        end_month = int(range_match.group("end_month") or start_month)
        end_day = int(range_match.group("end_day"))
        return date(year, start_month, start_day).isoformat(), date(year, end_month, end_day).isoformat()

    fallback_year = int(fallback_publish_date[:4]) if re.match(r"20\d{2}-\d{2}-\d{2}", fallback_publish_date) else None
    partial_match = PARTIAL_RANGE_RE.search(text)
    if partial_match and fallback_year:
        start_month = int(partial_match.group("start_month"))
        start_day = int(partial_match.group("start_day"))
        end_month = int(partial_match.group("end_month") or start_month)
        end_day = int(partial_match.group("end_day"))
        return date(fallback_year, start_month, start_day).isoformat(), date(fallback_year, end_month, end_day).isoformat()

    dates = [date(int(m.group("year")), int(m.group("month")), int(m.group("day"))) for m in DATE_RE.finditer(text)]
    if dates:
        start = min(dates)
        end = max(dates)
        if (end - start) > timedelta(days=30):
            end = start
        return start.isoformat(), end.isoformat()
    return fallback_publish_date, fallback_publish_date


def normalize_event_name(event_type: str, title: str, text: str, start_date: str = "") -> str:
    year = int(start_date[:4]) if re.match(r"20\d{2}-\d{2}-\d{2}", start_date) else _first_year(title) or _first_year(text)
    if event_type == "CEWC":
        return f"{year}年中央经济工作会议" if year else "中央经济工作会议"
    if event_type == "POLITBURO_MEETING":
        return title if "政治局" in title else "中共中央政治局会议"
    if event_type == "CPC_PLENUM":
        match = PLENUM_RE.search(text)
        if match:
            return f"{match.group('session')}{match.group('number')}全会"
        return title.replace("公报", "").strip()
    if event_type == "TWO_SESSIONS":
        if "政协" in text and "人大" not in text:
            return f"{year}年全国政协会议" if year else "全国政协会议"
        if "人大" in text and "政协" not in text:
            return f"{year}年全国人大会议" if year else "全国人大会议"
        return f"{year}年全国两会" if year else "全国两会"
    return title


def extract_tags(event_type: str, text: str, start_date: str = "") -> list[str]:
    tags: list[str] = []
    if event_type == "POLITBURO_MEETING":
        tags.extend(keyword for keyword in ECONOMIC_POLITBURO_KEYWORDS if keyword in text)
        month = _first_month(text) or (int(start_date[5:7]) if re.match(r"20\d{2}-\d{2}-\d{2}", start_date) else None)
        if month in {4, 7, 12}:
            tags.append(f"{month}月重点政治局会议")
    if event_type == "CPC_PLENUM":
        match = PLENUM_RE.search(text)
        if match:
            tags.extend(["中央全会", f"{match.group('session')}{match.group('number')}全会"])
    if event_type == "CEWC":
        for keyword in ("稳中求进", "高质量发展", "扩大内需", "房地产", "资本市场", "财政政策", "货币政策"):
            if keyword in text:
                tags.append(keyword)
    if event_type == "TWO_SESSIONS":
        if "政协" in text:
            tags.append("全国政协")
        if "人大" in text:
            tags.append("全国人大")
        tags.append("两会")
    return sorted(set(tags))


def _clean_summary(event_type: str, name: str, text: str, start: str, end: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if event_type == "TWO_SESSIONS":
        if start and end:
            return f"{name}于{start}开幕，{end}闭幕。本库仅保留年度大会的时间窗口、官方来源和宏观事件摘要，不采集大会期间内部会议。"
        return f"{name}。本库仅保留年度大会的时间窗口、官方来源和宏观事件摘要。"
    return text[:3000]


def _first_year(text: str) -> int | None:
    match = re.search(r"(20\d{2})年", text)
    return int(match.group(1)) if match else None


def _first_month(text: str) -> int | None:
    match = re.search(r"20\d{2}年(\d{1,2})月", text)
    return int(match.group(1)) if match else None
