from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import get_config, load_yaml, project_path
from src.common.db import upsert_dataframe
from src.common.logger import get_logger
from src.common.network import disable_env_proxies


logger = get_logger(__name__)


def _rename_market_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "date": "trade_date",
        "日期": "trade_date",
        "成交金额": "amount",
        "开盘": "open",
        "开盘价": "open",
        "最高": "high",
        "最高价": "high",
        "最低": "low",
        "最低价": "low",
        "收盘": "close",
        "收盘价": "close",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
        "涨跌额": "change",
        "振幅": "amplitude",
    }
    return df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})


def _prefixed_index_symbol(symbol: str) -> str:
    if symbol.startswith("399"):
        return f"sz{symbol}"
    return f"sh{symbol}"


def _normalize_index_df(raw: pd.DataFrame, symbol: str, name: str, source: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    df = _rename_market_columns(raw).copy()
    required = ["trade_date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{source} index result missing columns {missing}; got {list(raw.columns)}")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    start = pd.to_datetime(start_date).date()
    end = pd.to_datetime(end_date or pd.Timestamp.today()).date()
    df = df[(df["trade_date"] >= start) & (df["trade_date"] <= end)].sort_values("trade_date")
    if df.empty:
        raise ValueError(f"{source} returned empty rows for {symbol}")
    df["index_code"] = symbol
    df["index_name"] = name
    df["pre_close"] = pd.to_numeric(df["close"], errors="coerce").shift(1)
    if "pct_chg" not in df:
        df["pct_chg"] = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100
    if "volume" not in df:
        df["volume"] = None
    if "amount" not in df:
        df["amount"] = None
    df["data_source"] = source
    return df[
        [
            "trade_date",
            "index_code",
            "index_name",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "pct_chg",
            "volume",
            "amount",
            "data_source",
        ]
    ]


def fetch_index_history(symbol: str, name: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    disable_env_proxies()
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is not installed. Run: pip install -r requirements.txt") from exc

    start = start_date.replace("-", "")
    end = (end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
    prefixed = _prefixed_index_symbol(symbol)
    attempts = [
        ("akshare:stock_zh_index_daily", lambda: ak.stock_zh_index_daily(symbol=prefixed)),
        ("akshare:stock_zh_index_daily_tx", lambda: ak.stock_zh_index_daily_tx(symbol=prefixed, start_date=start_date, end_date=end_date or "")),
        ("akshare:stock_zh_index_hist_csindex", lambda: ak.stock_zh_index_hist_csindex(symbol=symbol, start_date=start, end_date=end)),
        ("akshare:index_zh_a_hist", lambda: ak.index_zh_a_hist(symbol=symbol, period="daily", start_date=start, end_date=end)),
    ]
    errors: list[str] = []
    for source, getter in attempts:
        try:
            raw = getter()
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                return _normalize_index_df(raw, symbol, name, source, start_date, end_date)
            errors.append(f"{source}: empty")
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            logger.warning("index source failed symbol=%s source=%s error=%s", symbol, source, exc)
    raise RuntimeError("; ".join(errors))


def collect_indices(start_date: str | None = None, end_date: str | None = None) -> int:
    cfg = get_config()
    symbols = load_yaml(project_path("config", "symbols.yaml"))["indices"]
    start_date = start_date or cfg["project"]["start_date"]
    total = 0
    for item in symbols:
        try:
            df = fetch_index_history(item["code"], item["name"], start_date, end_date)
            total += upsert_dataframe(df, "market_index_daily", ["trade_date", "index_code"])
            logger.info("collected index %s rows=%s", item["code"], len(df))
        except Exception as exc:
            logger.exception("failed to collect index %s: %s", item.get("code"), exc)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    print(collect_indices(args.start_date, args.end_date))


if __name__ == "__main__":
    main()
