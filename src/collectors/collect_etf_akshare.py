from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import get_config, load_yaml, project_path
from src.common.db import upsert_dataframe
from src.common.logger import get_logger
from src.common.network import disable_env_proxies


logger = get_logger(__name__)


def fetch_etf_history(symbol: str, name: str, start_date: str, end_date: str | None = None) -> pd.DataFrame:
    disable_env_proxies()
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is not installed. Run: pip install -r requirements.txt") from exc

    start = start_date.replace("-", "")
    end = (end_date or pd.Timestamp.today().strftime("%Y-%m-%d")).replace("-", "")
    market_prefix = "sh" if symbol.startswith(("5", "6")) else "sz"
    attempts = [
        ("akshare:fund_etf_hist_em", lambda: ak.fund_etf_hist_em(symbol=symbol, period="daily", start_date=start, end_date=end, adjust="")),
        ("akshare:fund_etf_hist_sina", lambda: ak.fund_etf_hist_sina(symbol=f"{market_prefix}{symbol}")),
        ("akshare:stock_zh_a_hist_tx", lambda: ak.stock_zh_a_hist_tx(symbol=f"{market_prefix}{symbol}", start_date=start_date, end_date=end_date or "", adjust="")),
    ]
    mapping = {
        "date": "trade_date",
        "日期": "trade_date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "涨跌幅": "pct_chg",
        "成交量": "volume",
        "成交额": "amount",
        "换手率": "turnover",
    }
    errors: list[str] = []
    df = pd.DataFrame()
    source = ""
    for source, getter in attempts:
        try:
            raw = getter()
            if raw is None or raw.empty:
                errors.append(f"{source}: empty")
                continue
            df = raw.rename(columns={k: v for k, v in mapping.items() if k in raw.columns})
            required = ["trade_date", "open", "high", "low", "close"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                errors.append(f"{source}: missing {missing}, columns={list(raw.columns)}")
                continue
            break
        except Exception as exc:
            errors.append(f"{source}: {exc}")
            logger.warning("etf source failed symbol=%s source=%s error=%s", symbol, source, exc)
    else:
        raise RuntimeError("; ".join(errors))
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    start_d = pd.to_datetime(start_date).date()
    end_d = pd.to_datetime(end_date or pd.Timestamp.today()).date()
    df = df[(df["trade_date"] >= start_d) & (df["trade_date"] <= end_d)].sort_values("trade_date")
    df["etf_code"] = symbol
    df["etf_name"] = name
    if "turnover" not in df:
        df["turnover"] = None
    if "volume" not in df:
        df["volume"] = None
    if "amount" not in df:
        df["amount"] = None
    if "pct_chg" not in df:
        df["pct_chg"] = df["close"].pct_change() * 100
    df["data_source"] = source
    return df[
        [
            "trade_date",
            "etf_code",
            "etf_name",
            "open",
            "high",
            "low",
            "close",
            "pct_chg",
            "volume",
            "amount",
            "turnover",
            "data_source",
        ]
    ]


def collect_etfs(start_date: str | None = None, end_date: str | None = None) -> int:
    cfg = get_config()
    symbols = load_yaml(project_path("config", "symbols.yaml"))["etfs"]
    start_date = start_date or cfg["project"]["start_date"]
    total = 0
    for item in symbols:
        try:
            df = fetch_etf_history(item["code"], item["name"], start_date, end_date)
            total += upsert_dataframe(df, "market_etf_daily", ["trade_date", "etf_code"])
            logger.info("collected etf %s rows=%s", item["code"], len(df))
        except Exception as exc:
            logger.exception("failed to collect etf %s: %s", item.get("code"), exc)
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    print(collect_etfs(args.start_date, args.end_date))


if __name__ == "__main__":
    main()
