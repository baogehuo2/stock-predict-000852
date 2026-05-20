from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from requests import exceptions as request_exceptions

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import get_config
from src.common.db import read_sql, upsert_dataframe
from src.common.logger import get_logger
from src.common.utils import extract_json
from src.llm.llm_client import LLMError, OpenAICompatibleClient
from src.llm.prompts import render_prompt


logger = get_logger(__name__)


def _json_list(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    try:
        data = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [str(item) for item in data if str(item)]


def score_event(event: dict) -> float:
    direction = {"利多": 1.0, "利空": -1.0, "中性": 0.0}.get(event.get("impact_direction"), 0.0)
    expectation = {"预期不足": 1.5, "中性": 1.0, "预期充分": 0.7, "极度过热": 0.2}.get(event.get("expectation_level"), 1.0)
    stage = {"传闻": 0.6, "发酵": 0.9, "确认": 1.0, "落地": 0.8, "兑现": 0.4}.get(event.get("event_stage"), 1.0)
    surprise = {"超预期": 1.8, "符合预期": 1.0, "不及预期": -0.8, "尚未落地": 0.8}.get(event.get("surprise_level"), 1.0)
    try:
        strength = max(1, min(5, int(event.get("impact_strength") or 1)))
    except Exception:
        strength = 1
    return direction * strength * expectation * stage * surprise


def _neutral_event(item_data: dict, reason: str) -> dict:
    return {
        "event_name": str(item_data.get("title") or "未识别事件")[:200],
        "event_type": "其他",
        "event_stage": "确认",
        "expectation_level": "中性",
        "surprise_level": "符合预期",
        "affected_style": "全市场",
        "impact_direction": "中性",
        "impact_strength": 1,
        "reason": reason,
    }


def _select_news_candidates(
    *,
    limit: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit_per_day: int | None = None,
    include_unmatched: bool = True,
) -> pd.DataFrame:
    where = ["news_id NOT IN (SELECT source_ids FROM event_daily WHERE source_ids IS NOT NULL)"]
    params: dict[str, object] = {}
    if start_date:
        where.append("trade_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        where.append("trade_date <= :end_date")
        params["end_date"] = end_date
    if not include_unmatched:
        where.append("matched_groups IS NOT NULL AND matched_groups <> '[]'")

    sql = (
        "SELECT id, news_id, trade_date, publish_time, title, content, matched_keywords, matched_groups "
        "FROM news_raw WHERE "
        + " AND ".join(where)
        + " ORDER BY trade_date DESC, "
        + "CASE WHEN matched_groups IS NOT NULL AND matched_groups <> '[]' THEN 0 ELSE 1 END, "
        + "publish_time DESC, id DESC"
    )
    if limit and not limit_per_day:
        sql += " LIMIT :limit"
        params["limit"] = limit

    news = read_sql(sql, params)
    if news.empty or not limit_per_day:
        return news

    news = news.groupby("trade_date", group_keys=False, sort=False).head(limit_per_day)
    if limit:
        news = news.head(limit)
    return news.reset_index(drop=True)


def _flush_events(rows: list[dict]) -> int:
    if not rows:
        return 0
    return upsert_dataframe(pd.DataFrame(rows), "event_daily", ["trade_date", "event_name"])


def _is_transient_llm_error(exc: Exception) -> bool:
    if isinstance(exc, request_exceptions.RequestException):
        return True
    if isinstance(exc, LLMError):
        text = str(exc)
        return any(code in text for code in ["429", "500", "502", "503", "504"])
    return False


def _chat_with_retry(client: OpenAICompatibleClient, prompt: str, retries: int, retry_wait: float) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return client.chat(prompt)
        except Exception as exc:
            last_error = exc
            if not _is_transient_llm_error(exc) or attempt >= retries:
                raise
            logger.warning(
                "transient LLM request failed attempt=%s/%s wait=%.1fs error=%s",
                attempt,
                retries,
                retry_wait,
                exc,
            )
            time.sleep(retry_wait)
    raise RuntimeError("LLM request failed") from last_error


def _extract_events(
    news: pd.DataFrame,
    fail_fast: bool,
    batch_size: int = 20,
    llm_retries: int = 3,
    retry_wait: float = 10.0,
) -> int:
    if news.empty:
        logger.info("no news available for event extraction")
        return 0
    client = OpenAICompatibleClient()
    rows = []
    total = len(news)
    saved = 0
    for index, item in enumerate(news.itertuples(index=False), start=1):
        item_data = item._asdict()
        logger.info(
            "extract event progress %s/%s trade_date=%s news_id=%s title=%s",
            index,
            total,
            item_data.get("trade_date"),
            item_data.get("news_id"),
            str(item_data.get("title") or "")[:80],
        )
        matched_groups = _json_list(item_data.get("matched_groups"))
        matched_keywords = _json_list(item_data.get("matched_keywords"))
        news_text = (
            f"关键词分组：{', '.join(matched_groups) or '无'}\n"
            f"命中关键词：{', '.join(matched_keywords) or '无'}\n"
            f"标题：{item_data['title']}\n"
            f"内容：{item_data.get('content') or ''}"
        )
        prompt = render_prompt("news_event_extract", news_text=news_text[:4000])
        try:
            content = _chat_with_retry(client, prompt, llm_retries, retry_wait)
            if not content or not content.strip():
                raise ValueError("LLM returned empty content")
            event = extract_json(content)
            rows.append(
                {
                    "trade_date": pd.to_datetime(item_data["trade_date"]).date(),
                    "event_name": str(event.get("event_name", ""))[:200],
                    "event_type": event.get("event_type"),
                    "event_stage": event.get("event_stage"),
                    "expectation_level": event.get("expectation_level"),
                    "surprise_level": event.get("surprise_level"),
                    "affected_style": event.get("affected_style"),
                    "impact_direction": event.get("impact_direction"),
                    "impact_strength": event.get("impact_strength") or 1,
                    "event_score": score_event(event),
                    "llm_reason": event.get("reason") or json.dumps(event, ensure_ascii=False),
                    "source_ids": item_data["news_id"],
                }
            )
        except Exception as exc:
            logger.exception("failed to extract event for news_id=%s: %s", item_data.get("news_id"), exc)
            if _is_transient_llm_error(exc):
                saved += _flush_events(rows)
                logger.error(
                    "stop event extraction because LLM/network is unavailable; saved buffered rows=%s total_saved=%s",
                    len(rows),
                    saved,
                )
                raise RuntimeError("LLM/network unavailable, stop to avoid writing false neutral events") from exc
            if fail_fast:
                raise
            event = _neutral_event(item_data, f"LLM解析失败，已按中性事件占位：{exc}")
            rows.append(
                {
                    "trade_date": pd.to_datetime(item_data["trade_date"]).date(),
                    "event_name": event["event_name"],
                    "event_type": event["event_type"],
                    "event_stage": event["event_stage"],
                    "expectation_level": event["expectation_level"],
                    "surprise_level": event["surprise_level"],
                    "affected_style": event["affected_style"],
                    "impact_direction": event["impact_direction"],
                    "impact_strength": event["impact_strength"],
                    "event_score": 0.0,
                    "llm_reason": event["reason"],
                    "source_ids": item_data["news_id"],
                }
            )
        if batch_size > 0 and len(rows) >= batch_size:
            saved += _flush_events(rows)
            logger.info("saved event batch rows=%s total_saved=%s progress=%s/%s", len(rows), saved, index, total)
            rows.clear()
    saved += _flush_events(rows)
    logger.info("extracted event rows=%s", saved)
    return saved


def extract_events_for_recent_news(limit: int = 30) -> int:
    cfg = get_config()
    fail_fast = bool(cfg.get("run", {}).get("fail_fast_llm", True))
    news = _select_news_candidates(limit=limit, include_unmatched=True)
    return _extract_events(news, fail_fast)


def extract_events_for_history(
    start_date: str,
    end_date: str,
    limit_per_day: int | None = 20,
    limit: int | None = None,
    include_unmatched: bool = False,
    batch_size: int = 20,
    fail_fast: bool | None = None,
    llm_retries: int = 3,
    retry_wait: float = 10.0,
) -> int:
    cfg = get_config()
    if fail_fast is None:
        fail_fast = False
    news = _select_news_candidates(
        limit=limit,
        start_date=start_date,
        end_date=end_date,
        limit_per_day=limit_per_day,
        include_unmatched=include_unmatched,
    )
    logger.info(
        "selected historical news for event extraction rows=%s start=%s end=%s limit_per_day=%s include_unmatched=%s",
        len(news),
        start_date,
        end_date,
        limit_per_day,
        include_unmatched,
    )
    return _extract_events(
        news,
        fail_fast,
        batch_size=batch_size,
        llm_retries=llm_retries,
        retry_wait=retry_wait,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-date", help="Historical extraction start trade_date, YYYY-MM-DD.")
    parser.add_argument("--end-date", help="Historical extraction end trade_date, YYYY-MM-DD.")
    parser.add_argument("--limit-per-day", type=int, default=20, help="Historical extraction max news per trade_date.")
    parser.add_argument("--include-unmatched", action="store_true", help="Also extract news without matched keyword groups.")
    parser.add_argument("--batch-size", type=int, default=20, help="Write extracted events to database every N rows.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when one LLM extraction fails.")
    parser.add_argument("--llm-retries", type=int, default=3, help="Retry transient LLM/network errors before stopping.")
    parser.add_argument("--retry-wait", type=float, default=10.0, help="Seconds to wait between transient LLM retries.")
    args = parser.parse_args()
    if args.start_date or args.end_date:
        if not args.start_date or not args.end_date:
            parser.error("--start-date and --end-date must be used together")
        print(
            extract_events_for_history(
                start_date=args.start_date,
                end_date=args.end_date,
                limit_per_day=args.limit_per_day,
                limit=args.limit,
                include_unmatched=args.include_unmatched,
                batch_size=args.batch_size,
                fail_fast=args.fail_fast,
                llm_retries=args.llm_retries,
                retry_wait=args.retry_wait,
            )
        )
    else:
        print(extract_events_for_recent_news(args.limit or 30))


if __name__ == "__main__":
    main()
