from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

import pandas as pd


def normalize_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    return pd.to_datetime(value).date()


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(str(value).replace(",", "").replace("%", ""))
    except Exception:
        return None


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        text = match.group(0)
    return json.loads(text)

