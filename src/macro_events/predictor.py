from __future__ import annotations

import calendar
import re
from datetime import date

from src.macro_events.config import FUTURE_MONTHS
from src.macro_events.schemas import MacroEvent, stable_event_id


def build_future_predictions(today: date | None = None, months: int = FUTURE_MONTHS) -> list[MacroEvent]:
    today = today or date.today()
    end = _add_months(today, months)
    events: list[MacroEvent] = []
    for year in range(today.year, end.year + 1):
        events.extend(_two_sessions_predictions(year))
        events.append(_cewc_prediction(year))
        for month in (4, 7, 12):
            events.append(_politburo_prediction(year, month))
    return [event for event in events if _in_range(event.start_date, today, end)]


def build_plenum_predictions_from_articles(articles: list[dict]) -> list[MacroEvent]:
    events: list[MacroEvent] = []
    pattern = re.compile(r"决定于?(?P<date>20\d{2}年\d{1,2}月(?:\d{1,2}日)?)召开(?P<name>[^，。；\s]+全会)")
    for article in articles:
        text = f"{article.get('title', '')}\n{article.get('text', '')}"
        for match in pattern.finditer(text):
            name = match.group("name")
            start_date = _date_text_to_iso(match.group("date"))
            if not start_date:
                continue
            events.append(
                MacroEvent(
                    event_id=stable_event_id("CPC_PLENUM", start_date, name),
                    event_type="CPC_PLENUM",
                    event_name=name,
                    start_date=start_date,
                    end_date=start_date,
                    publish_date=str(article.get("publish_date", "")),
                    source_org=str(article.get("source_org", "")),
                    source_title=str(article.get("title", "")),
                    source_url=str(article.get("url", "")),
                    source_text=str(article.get("text", ""))[:3000],
                    is_predicted=True,
                    prediction_confidence=0.85,
                    status="predicted",
                    tags=["政治局会议公告", "中央全会预测"],
                    note="根据政治局会议公告生成的中央全会预测记录",
                )
            )
    return events


def _two_sessions_predictions(year: int) -> list[MacroEvent]:
    return [
        _prediction(
            "TWO_SESSIONS",
            f"{year}年全国政协会议预测",
            date(year, 3, 4),
            date(year, 3, 10),
            0.75,
            ["全国政协", "预测窗口±3天"],
        ),
        _prediction(
            "TWO_SESSIONS",
            f"{year}年全国人大会议预测",
            date(year, 3, 5),
            date(year, 3, 12),
            0.75,
            ["全国人大", "预测窗口±3天"],
        ),
    ]


def _cewc_prediction(year: int) -> MacroEvent:
    return _prediction(
        "CEWC",
        f"{year}年中央经济工作会议预测",
        date(year, 12, 8),
        date(year, 12, 18),
        0.7,
        ["预测窗口", "12月8日至12月18日"],
    )


def _politburo_prediction(year: int, month: int) -> MacroEvent:
    last_day = calendar.monthrange(year, month)[1]
    return _prediction(
        "POLITBURO_MEETING",
        f"{year}年{month}月中共中央政治局经济相关会议预测",
        date(year, month, 20),
        date(year, month, last_day),
        0.65,
        [f"{month}月重点政治局会议", "预测窗口20日至月底"],
    )


def _prediction(
    event_type: str,
    name: str,
    start: date,
    end: date,
    confidence: float,
    tags: list[str],
) -> MacroEvent:
    return MacroEvent(
        event_id=stable_event_id(event_type, start.isoformat(), name),
        event_type=event_type,
        event_name=name,
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        publish_date="",
        source_org="",
        source_title="",
        source_url="",
        source_text="",
        is_predicted=True,
        prediction_confidence=confidence,
        status="predicted",
        tags=tags,
        note="规则预测，待官方公告确认后转入历史库",
    )


def _in_range(start_date: str, today: date, end: date) -> bool:
    start = date.fromisoformat(start_date)
    return today <= start <= end


def _date_text_to_iso(text: str) -> str:
    match = re.match(r"(20\d{2})年(\d{1,2})月(?:(\d{1,2})日)?", text)
    if not match:
        return ""
    year = int(match.group(1))
    month = int(match.group(2))
    day = int(match.group(3) or 1)
    return date(year, month, day).isoformat()


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)
