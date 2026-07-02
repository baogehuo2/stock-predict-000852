from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from typing import Any

import pandas as pd
import requests

from src.common.config import get_config, load_yaml, project_path
from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.network import disable_env_proxies
from src.common.secrets import get_secret
from src.quality.check_collection_data import DatasetSpec, evaluate_dataframe, persist_metrics


TUSHARE_URL = "https://api.tushare.pro"
TUSHARE_TRADE_CAL_URL = "https://tushare.pro/document/2?doc_id=26"
HOLIDAY_CN_URL_TEMPLATES = [
    "https://raw.githubusercontent.com/NateScarlet/holiday-cn/master/{year}.json",
    "https://cdn.jsdelivr.net/gh/NateScarlet/holiday-cn@master/{year}.json",
    "https://ghfast.top/https://raw.githubusercontent.com/NateScarlet/holiday-cn/master/{year}.json",
]


def _compact(value: str) -> str:
    return value.replace("-", "")


def _date_range(start_date: str, end_date: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start_date), pd.Timestamp(end_date), freq="D")


def fetch_tushare_trade_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    disable_env_proxies()
    token = get_secret("tushare.token")
    if not token:
        raise RuntimeError("missing tushare.token in encrypted secrets")
    response = requests.post(
        TUSHARE_URL,
        json={
            "api_name": "trade_cal",
            "token": token,
            "params": {
                "exchange": "SSE",
                "start_date": _compact(start_date),
                "end_date": _compact(end_date),
            },
            "fields": "exchange,cal_date,is_open,pretrade_date",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("code") != 0:
        raise RuntimeError(f"tushare trade_cal request failed: {payload}")
    data = payload.get("data", {})
    rows = data.get("items", [])
    fields = data.get("fields", [])
    if not rows:
        raise RuntimeError("tushare trade_cal returned no rows")
    return pd.DataFrame(rows, columns=fields)


def _fetch_holiday_year(year: int) -> pd.DataFrame:
    disable_env_proxies()
    last_error: Exception | None = None
    payload: Any | None = None
    for template in HOLIDAY_CN_URL_TEMPLATES:
        url = template.format(year=year)
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 data-collectors/2.0"}, timeout=30)
            response.raise_for_status()
            payload = response.json()
            break
        except Exception as exc:
            last_error = exc
    if payload is None:
        raise RuntimeError(f"holiday-cn fetch failed for year={year}: {last_error}")
    days = payload.get("days", payload if isinstance(payload, list) else [])
    if not isinstance(days, list):
        raise RuntimeError(f"holiday-cn unexpected payload for year={year}")
    frame = pd.DataFrame(days)
    if frame.empty:
        return pd.DataFrame(columns=["date", "name", "isOffDay"])
    return frame


def fetch_holidays(start_date: str, end_date: str) -> pd.DataFrame:
    start = pd.Timestamp(start_date).year
    end = pd.Timestamp(end_date).year
    frames = [_fetch_holiday_year(year) for year in range(start, end + 1)]
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        return pd.DataFrame(columns=["calendar_date", "holiday_name", "is_holiday"])
    date_col = "date" if "date" in frame.columns else "calendar_date"
    name_col = "name" if "name" in frame.columns else "holiday_name"
    off_col = "isOffDay" if "isOffDay" in frame.columns else "is_holiday"
    out = pd.DataFrame(
        {
            "calendar_date": pd.to_datetime(frame[date_col], errors="coerce").dt.date,
            "holiday_name": frame[name_col].astype(str),
            "is_holiday": frame[off_col].astype(bool).astype(int),
        }
    )
    out = out.dropna(subset=["calendar_date"])
    out = out[(out["calendar_date"] >= pd.Timestamp(start_date).date()) & (out["calendar_date"] <= pd.Timestamp(end_date).date())]
    return out.sort_values("calendar_date").drop_duplicates("calendar_date", keep="last")


def normalize_calendar(trade_cal: pd.DataFrame, holidays: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    cal = pd.DataFrame({"calendar_date": _date_range(start_date, end_date).date})
    trade = trade_cal.copy()
    trade["calendar_date"] = pd.to_datetime(trade["cal_date"], errors="coerce").dt.date
    trade["is_trading_day"] = pd.to_numeric(trade["is_open"], errors="coerce").fillna(0).astype(int)
    trade["pre_trade_date"] = pd.to_datetime(trade.get("pretrade_date"), errors="coerce").dt.date
    trade = trade[["calendar_date", "is_trading_day", "pre_trade_date"]].dropna(subset=["calendar_date"])
    cal = cal.merge(trade, on="calendar_date", how="left")
    if cal["is_trading_day"].isna().any():
        missing = cal.loc[cal["is_trading_day"].isna(), "calendar_date"].astype(str).head(5).tolist()
        raise RuntimeError(f"trade calendar missing dates: {missing}")
    cal["is_trading_day"] = cal["is_trading_day"].astype(int)
    cal["is_exchange_closed"] = (1 - cal["is_trading_day"]).astype(int)
    holiday = holidays.copy()
    if holiday.empty:
        holiday = pd.DataFrame(columns=["calendar_date", "is_holiday", "holiday_name"])
    cal = cal.merge(holiday, on="calendar_date", how="left")
    cal["is_holiday"] = cal["is_holiday"].fillna(0).astype(int)
    cal["holiday_name"] = cal["holiday_name"].where(cal["is_holiday"].eq(1), None)
    trading_days = cal.loc[cal["is_trading_day"].eq(1), "calendar_date"].tolist()
    next_map: dict[date, date | None] = {}
    for idx, current in enumerate(trading_days):
        next_map[current] = trading_days[idx + 1] if idx + 1 < len(trading_days) else None
    cal["next_trade_date"] = cal["calendar_date"].map(next_map)
    cal["market"] = "CN_STOCK"
    cal["source_type"] = "tushare:trade_cal+holiday-cn"
    cal["source_url"] = f"{TUSHARE_TRADE_CAL_URL}|https://github.com/NateScarlet/holiday-cn"
    cal["available_time"] = datetime.now()
    cal["crawl_time"] = datetime.now()
    cal["note"] = "统一A股日历；不区分SSE/SZSE；节假日接受holiday-cn结构化主源。"
    columns = [
        "calendar_date", "market", "is_trading_day", "is_exchange_closed", "is_holiday",
        "holiday_name", "pre_trade_date", "next_trade_date", "source_type", "source_url",
        "available_time", "crawl_time", "note",
    ]
    return cal[columns]


def _existing_count(start_date: str, end_date: str) -> int:
    result = read_sql(
        """
        SELECT COUNT(*) AS row_count
        FROM calendar_daily_raw
        WHERE market = 'CN_STOCK' AND calendar_date BETWEEN :start_date AND :end_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    return int(result.iloc[0]["row_count"])


def _quality_report(frame: pd.DataFrame) -> dict[str, Any]:
    spec = DatasetSpec(
        table="calendar_daily_raw",
        date_column="calendar_date",
        unique_columns=("calendar_date", "market"),
        required_columns=("calendar_date", "market", "is_trading_day", "is_exchange_closed", "source_type", "available_time"),
    )
    metrics = evaluate_dataframe(frame, spec)
    persist_metrics("calendar_daily_raw:CN_STOCK", metrics, date.today())
    return {"metrics": metrics}


def collect_calendar_daily(start_date: str | None = None, end_date: str | None = None, future_days: int = 370) -> dict[str, Any]:
    config = get_config()
    start_date = start_date or config["project"]["start_date"]
    end_date = end_date or (pd.Timestamp.today() + pd.Timedelta(days=future_days)).strftime("%Y-%m-%d")
    run = IngestionRun(
        dataset_name="calendar_daily_raw:CN_STOCK",
        data_source="tushare:trade_cal+holiday-cn",
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    try:
        trade_cal = fetch_tushare_trade_calendar(start_date, end_date)
        holidays = fetch_holidays(start_date, end_date)
        frame = normalize_calendar(trade_cal, holidays, start_date, end_date)
        existing = _existing_count(start_date, end_date)
        upsert_dataframe(frame, "calendar_daily_raw", ["calendar_date", "market"])
        run.fetched_rows = len(frame)
        run.inserted_rows = max(0, len(frame) - existing)
        run.updated_rows = min(len(frame), existing)
        run.finish("success")
        return {
            "rows": len(frame),
            "start_date": start_date,
            "end_date": end_date,
            "holiday_rows": int(frame["is_holiday"].sum()),
            "trading_rows": int(frame["is_trading_day"].sum()),
            "quality": _quality_report(frame),
        }
    except Exception as exc:
        run.error_rows = 1
        run.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集第二批统一A股交易日历和中国节假日。")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--future-days", type=int, default=370)
    args = parser.parse_args()
    print(collect_calendar_daily(args.start_date, args.end_date, args.future_days))


if __name__ == "__main__":
    main()
