from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import get_config, load_yaml, project_path
from src.common.db import upsert_dataframe
from src.common.logger import get_logger
from src.common.network import disable_env_proxies


logger = get_logger(__name__)


def _normalize(raw: pd.DataFrame, code: str, name: str, category: str, source: str) -> pd.DataFrame:
    mapping = {
        "date": "trade_date",
        "日期": "trade_date",
        "open": "open",
        "开盘价": "open",
        "high": "high",
        "最高价": "high",
        "low": "low",
        "最低价": "low",
        "close": "close",
        "收盘价": "close",
        "settle": "settle",
        "结算价": "settle",
        "volume": "volume",
        "成交量": "volume",
        "hold": "open_interest",
        "持仓量": "open_interest",
    }
    df = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns}).copy()
    required = ["trade_date", "open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"futures result missing columns {missing}; got {list(raw.columns)}")
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["future_code"] = code
    df["future_name"] = name
    df["category"] = category
    if "settle" not in df:
        df["settle"] = None
    if "volume" not in df:
        df["volume"] = None
    if "open_interest" not in df:
        df["open_interest"] = None
    df["pct_chg"] = pd.to_numeric(df["close"], errors="coerce").pct_change() * 100
    df["amount"] = None
    df["data_source"] = source
    return df[
        [
            "trade_date",
            "future_code",
            "future_name",
            "category",
            "open",
            "high",
            "low",
            "close",
            "settle",
            "pct_chg",
            "volume",
            "open_interest",
            "amount",
            "data_source",
        ]
    ]


def fetch_future_history(code: str, name: str, category: str) -> pd.DataFrame:
    disable_env_proxies()
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is not installed. Run: pip install -r requirements.txt") from exc

    errors: list[str] = []
    for func_name in ("futures_main_sina", "futures_zh_daily_sina"):
        func = getattr(ak, func_name, None)
        if func is None:
            continue
        try:
            raw = func(symbol=code)
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                return _normalize(raw, code, name, category, f"akshare:{func_name}")
        except Exception as exc:
            errors.append(f"{func_name}: {exc}")
    raise RuntimeError("; ".join(errors) or f"No AKShare futures function available for {code}")


def collect_futures() -> int:
    symbols = load_yaml(project_path("config", "symbols.yaml"))
    items = symbols.get("index_futures", []) + symbols.get("commodity_futures", [])
    total = 0
    for item in items:
        try:
            df = fetch_future_history(item["code"], item["name"], item["category"])
            total += upsert_dataframe(df, "market_futures_daily", ["trade_date", "future_code"])
            logger.info("collected future %s rows=%s", item["code"], len(df))
        except Exception as exc:
            logger.exception("failed to collect future %s: %s", item.get("code"), exc)
    return total


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(collect_futures())


if __name__ == "__main__":
    main()
