from __future__ import annotations

from datetime import datetime


CSV_COLUMNS = [
    "region_id",
    "start_date",
    "end_date",
    "region_type",
    "cycle",
    "level",
    "label_freq",
    "confidence",
    "usable_for_signal",
    "entry_start",
    "entry_end",
    "exit_start",
    "exit_end",
    "reason",
    "notes",
]

REGION_TYPES = {"bottom", "top"}
CYCLES = {"intraday", "short", "medium", "long"}
LEVELS = {"minor", "swing", "major"}
LABEL_FREQS = {"intraday", "daily", "weekly", "monthly"}
CONFIDENCES = {"1", "2", "3"}
USABLE_VALUES = {"0", "1"}


def _clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _date(value: object, field: str) -> str:
    text = _clean_text(value)
    if not text:
        raise ValueError(f"{field} 不能为空")
    try:
        return datetime.strptime(text[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"{field} 必须是 YYYY-MM-DD") from exc


def _optional_date(value: object, field: str) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    return _date(text, field)


def normalize_label(raw: dict) -> dict:
    item = {col: _clean_text(raw.get(col, "")) for col in CSV_COLUMNS}
    item["start_date"] = _date(item["start_date"], "start_date")
    item["end_date"] = _date(item["end_date"], "end_date")
    item["entry_start"] = _optional_date(item["entry_start"], "entry_start")
    item["entry_end"] = _optional_date(item["entry_end"], "entry_end")
    item["exit_start"] = _optional_date(item["exit_start"], "exit_start")
    item["exit_end"] = _optional_date(item["exit_end"], "exit_end")
    item["confidence"] = str(item["confidence"])
    item["usable_for_signal"] = str(item["usable_for_signal"])
    return item


def validate_label(raw: dict, existing_ids: set[str] | None = None, original_id: str | None = None) -> dict:
    item = normalize_label(raw)
    existing_ids = existing_ids or set()
    region_id = item["region_id"]
    if not region_id:
        raise ValueError("region_id 不能为空")
    if region_id != original_id and region_id in existing_ids:
        raise ValueError(f"region_id 已存在: {region_id}")

    if item["start_date"] > item["end_date"]:
        raise ValueError("start_date 必须小于等于 end_date")
    if item["region_type"] not in REGION_TYPES:
        raise ValueError("region_type 只能是 bottom 或 top")
    if item["cycle"] not in CYCLES:
        raise ValueError("cycle 只能是 intraday/short/medium/long")
    if item["level"] not in LEVELS:
        raise ValueError("level 只能是 minor/swing/major")
    if item["label_freq"] not in LABEL_FREQS:
        raise ValueError("label_freq 只能是 intraday/daily/weekly/monthly")
    if item["confidence"] not in CONFIDENCES:
        raise ValueError("confidence 只能是 1/2/3")
    if item["usable_for_signal"] not in USABLE_VALUES:
        raise ValueError("usable_for_signal 只能是 0/1")
    if item["cycle"] == "intraday" and item["label_freq"] != "intraday":
        raise ValueError("cycle=intraday 时 label_freq 必须是 intraday")
    if item["label_freq"] == "intraday" and item["cycle"] != "intraday":
        raise ValueError("label_freq=intraday 时 cycle 必须是 intraday")

    if item["region_type"] == "bottom":
        if not item["entry_start"] or not item["entry_end"]:
            raise ValueError("bottom 标注必须填写 entry_start 和 entry_end")
        if item["exit_start"] or item["exit_end"]:
            raise ValueError("bottom 标注的 exit_start/exit_end 必须为空")
        if not (item["start_date"] <= item["entry_start"] <= item["entry_end"] <= item["end_date"]):
            raise ValueError("bottom 标注必须满足 start_date <= entry_start <= entry_end <= end_date")
    else:
        if not item["exit_start"] or not item["exit_end"]:
            raise ValueError("top 标注必须填写 exit_start 和 exit_end")
        if item["entry_start"] or item["entry_end"]:
            raise ValueError("top 标注的 entry_start/entry_end 必须为空")
        if not (item["start_date"] <= item["exit_start"] <= item["exit_end"] <= item["end_date"]):
            raise ValueError("top 标注必须满足 start_date <= exit_start <= exit_end <= end_date")
    return item

