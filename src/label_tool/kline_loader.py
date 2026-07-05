from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import numpy as np
from sqlalchemy.exc import SQLAlchemyError

from src.common.config import get_config
from src.common.db import read_sql
from src.label_tool.indicators import add_indicators


@dataclass(frozen=True)
class KlineRequest:
    freq: str = "daily"
    start_date: str | None = None
    end_date: str | None = None


DAILY_FREQS = {"daily", "weekly", "monthly"}
ALL_FREQS = DAILY_FREQS | {"intraday"}
_TARGET_INDEX_OVERRIDE: str | None = None


def set_target_index(index_code: str | None) -> None:
    global _TARGET_INDEX_OVERRIDE
    _TARGET_INDEX_OVERRIDE = str(index_code).strip() if index_code else None


def _target_index() -> str:
    if _TARGET_INDEX_OVERRIDE:
        return _TARGET_INDEX_OVERRIDE
    return str(get_config().get("project", {}).get("target_index", "000852"))


def _date_filter_sql(column: str, req: KlineRequest) -> tuple[str, dict]:
    where = []
    params: dict[str, object] = {}
    if req.start_date:
        where.append(f"{column} >= :start_date")
        params["start_date"] = req.start_date
    if req.end_date:
        where.append(f"{column} <= :end_date")
        params["end_date"] = req.end_date
    return (" AND " + " AND ".join(where)) if where else "", params


def _load_daily(req: KlineRequest) -> pd.DataFrame:
    params = {"index_code": _target_index()}
    sql = (
        "SELECT trade_date, open, high, low, close, volume, amount "
        "FROM market_index_daily "
        "WHERE index_code=:index_code "
        "ORDER BY trade_date"
    )
    df = read_sql(sql, params)
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["trade_date", "open", "high", "low", "close"]).sort_values("trade_date")


def _load_periodic_table(freq: str) -> pd.DataFrame:
    if freq not in {"weekly", "monthly"}:
        raise ValueError(f"不支持的周期表频率: {freq}")
    table = "market_index_weekly" if freq == "weekly" else "market_index_monthly"
    sql = (
        "SELECT trade_date, open, high, low, close, volume, amount, "
        "period_start_date AS period_start, period_end_date AS period_end "
        f"FROM {table} "
        "WHERE index_code=:index_code "
        "ORDER BY trade_date"
    )
    try:
        df = read_sql(sql, {"index_code": _target_index()})
    except SQLAlchemyError:
        return pd.DataFrame()
    if df.empty:
        return df
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["period_start"] = pd.to_datetime(df["period_start"]).fillna(df["trade_date"])
    df["period_end"] = pd.to_datetime(df["period_end"]).fillna(df["trade_date"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["trade_date", "open", "high", "low", "close"]).sort_values("trade_date")


def _resample_daily(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if freq == "daily" or df.empty:
        out = df.copy()
        out["period_start"] = out["trade_date"]
        out["period_end"] = out["trade_date"]
        return out

    rule = "W-FRI" if freq == "weekly" else "ME"
    data = df.set_index("trade_date").sort_index()
    out = data.resample(rule).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
        period_start=("open", lambda s: s.index.min()),
        period_end=("open", lambda s: s.index.max()),
    )
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    out["trade_date"] = out["period_end"]
    return out


def _load_minute_1m() -> pd.DataFrame:
    sql = (
        "SELECT trade_time, trade_date, open, high, low, close, volume, amount "
        "FROM market_index_minute_raw "
        "WHERE index_code=:index_code AND freq='1min' "
        "ORDER BY trade_time"
    )
    try:
        df = read_sql(sql, {"index_code": _target_index()})
    except SQLAlchemyError as exc:
        raise RuntimeError("未找到可用 1 分钟K表 market_index_minute_raw。") from exc
    if df.empty:
        return df
    df["trade_time"] = pd.to_datetime(df["trade_time"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for col in ("open", "high", "low", "close", "volume", "amount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["trade_time", "open", "high", "low", "close"]).sort_values("trade_time")


def load_aggregated_minute(freq: str = "60min") -> pd.DataFrame:
    minute = _load_minute_1m()
    if minute.empty:
        return pd.DataFrame()
    if freq not in {"60min", "120min"}:
        raise ValueError(f"不支持的分钟聚合频率: {freq}")

    data = minute.copy()
    clock = data["trade_time"].dt.strftime("%H:%M")
    data["bucket_time"] = None
    if freq == "60min":
        data.loc[(clock >= "09:30") & (clock <= "10:30"), "bucket_time"] = "10:30"
        data.loc[(clock > "10:30") & (clock <= "11:30"), "bucket_time"] = "11:30"
        data.loc[(clock >= "13:00") & (clock <= "14:00"), "bucket_time"] = "14:00"
        data.loc[(clock > "14:00") & (clock <= "15:00"), "bucket_time"] = "15:00"
    else:
        data.loc[(clock >= "09:30") & (clock <= "11:30"), "bucket_time"] = "11:30"
        data.loc[(clock >= "13:00") & (clock <= "15:00"), "bucket_time"] = "15:00"
    data = data.dropna(subset=["bucket_time"])
    if data.empty:
        return pd.DataFrame()

    data["bucket_end"] = pd.to_datetime(data["trade_date"].dt.strftime("%Y-%m-%d") + " " + data["bucket_time"])
    out = (
        data.groupby("bucket_end", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
            period_start=("trade_time", "min"),
            period_end=("trade_time", "max"),
        )
        .sort_values("bucket_end")
    )
    out["trade_date"] = out["bucket_end"]
    return out.drop(columns=["bucket_end"])


def _load_intraday(req: KlineRequest) -> pd.DataFrame:
    return load_aggregated_minute("60min")


def _apply_request_filter(df: pd.DataFrame, req: KlineRequest) -> pd.DataFrame:
    if df.empty or (not req.start_date and not req.end_date):
        return df
    anchor = pd.to_datetime(df.get("trade_date"))
    mask = pd.Series(True, index=df.index)
    if req.start_date:
        start = pd.Timestamp(req.start_date)
        if (anchor.dt.time != pd.Timestamp("00:00").time()).any():
            mask &= anchor.dt.date >= start.date()
        else:
            mask &= anchor >= start
    if req.end_date:
        end = pd.Timestamp(req.end_date)
        if (anchor.dt.time != pd.Timestamp("00:00").time()).any():
            mask &= anchor.dt.date <= end.date()
        else:
            mask &= anchor <= end
    return df[mask].copy()


def load_kline(req: KlineRequest) -> pd.DataFrame:
    if req.freq not in ALL_FREQS:
        raise ValueError(f"不支持的K线频率: {req.freq}")
    if req.freq == "intraday":
        df = _load_intraday(req)
    elif req.freq in {"weekly", "monthly"}:
        df = _load_periodic_table(req.freq)
        if df.empty:
            df = _resample_daily(_load_daily(req), req.freq)
    elif req.freq == "daily":
        df = _resample_daily(_load_daily(req), req.freq)
    else:
        raise ValueError(f"不支持的K线频率: {req.freq}")
    if df.empty:
        return df
    df = add_indicators(df)
    df = _apply_request_filter(df, req)
    cols = [
        "trade_date",
        "period_start",
        "period_end",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "ma5",
        "ma10",
        "ma20",
        "ma60",
        "ma120",
        "ma250",
        "boll_upper",
        "boll_mid",
        "boll_lower",
        "boll_upper_2",
        "boll_lower_2",
        "boll_upper_3",
        "boll_lower_3",
        "macd_dif",
        "macd_dea",
        "macd_bar",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "rsi6",
        "rsi12",
        "rsi24",
        "cci14",
        "wr14",
        "atr14",
        "upper_shadow_ratio",
        "lower_shadow_ratio",
        "body_ratio",
        "vol_ma5",
        "vol_ma10",
    ]
    existing = [c for c in cols if c in df.columns]
    return df[existing]


def kline_to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    out = df.copy()
    for col in ("trade_date", "period_start", "period_end"):
        if col in out:
            dt = pd.to_datetime(out[col])
            has_time = (dt.dt.time != pd.Timestamp("00:00").time()).any()
            out[col] = dt.dt.strftime("%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d")
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.astype(object).where(pd.notna(out), None)
    return out.to_dict(orient="records")
