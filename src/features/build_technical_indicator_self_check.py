from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import read_sql
from src.features.technical_indicators import atr, boll, cci, kdj, macd, rsi_cn, wr


SELF_CHECK_DATE = "2025-03-24"
EXPECTED_CLOSE = 6360.341
EXPECTED_RSI6 = 29.14


def build_technical_indicator_self_check(
    output_csv: str = "data/reports/technical_indicator_self_check.csv",
) -> pd.DataFrame:
    target = str(get_config()["project"]["target_index"])
    daily = read_sql(
        "SELECT trade_date,index_code,open,high,low,close,volume,amount "
        "FROM market_index_daily WHERE index_code=:target ORDER BY trade_date",
        {"target": target},
    )
    if daily.empty:
        raise RuntimeError(f"No market_index_daily rows found for {target}.")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")

    macd_frame = macd(daily["close"])
    kdj_frame = kdj(daily["high"], daily["low"], daily["close"])
    boll_frame = boll(daily["close"], ddof=0)
    check = daily[["trade_date", "open", "high", "low", "close"]].copy()
    check["freq"] = "daily"
    check["trade_time"] = check["trade_date"].dt.strftime("%Y-%m-%d")
    check["rsi6"] = rsi_cn(daily["close"], 6)
    check["rsi14"] = rsi_cn(daily["close"], 14)
    check["macd_dif"] = macd_frame["macd_dif"]
    check["macd_dea"] = macd_frame["macd_dea"]
    check["macd_bar"] = macd_frame["macd_bar"]
    check["kdj_k"] = kdj_frame["kdj_k"]
    check["kdj_d"] = kdj_frame["kdj_d"]
    check["kdj_j"] = kdj_frame["kdj_j"]
    check["boll_mid"] = boll_frame["boll_mid"]
    check["boll_upper"] = boll_frame["boll_upper"]
    check["boll_lower"] = boll_frame["boll_lower"]
    check["cci14"] = cci(daily["high"], daily["low"], daily["close"], 14)
    check["wr14"] = wr(daily["high"], daily["low"], daily["close"], 14, sign="negative")
    check["atr14"] = atr(daily["high"], daily["low"], daily["close"], 14, normalize=True)
    check["source"] = "market_index_daily/akshare_em_daily"
    check["check_status"] = ""
    check["note"] = ""

    sample_mask = check["trade_time"].eq(SELF_CHECK_DATE)
    if sample_mask.any():
        sample = check.loc[sample_mask].iloc[0]
        close_ok = abs(float(sample["close"]) - EXPECTED_CLOSE) < 0.001
        rsi_ok = abs(float(sample["rsi6"]) - EXPECTED_RSI6) < 0.05
        status = "PASS" if close_ok and rsi_ok else "FAIL"
        note = (
            f"expected close={EXPECTED_CLOSE}, rsi6~={EXPECTED_RSI6}; "
            f"actual close={float(sample['close']):.3f}, rsi6={float(sample['rsi6']):.6f}; "
            "boll_ddof=0; wr14 negative; atr14 normalized by close"
        )
        check.loc[sample_mask, "check_status"] = status
        check.loc[sample_mask, "note"] = note

    output = check[
        [
            "freq",
            "trade_time",
            "open",
            "high",
            "low",
            "close",
            "rsi6",
            "rsi14",
            "macd_dif",
            "macd_dea",
            "macd_bar",
            "kdj_k",
            "kdj_d",
            "kdj_j",
            "boll_mid",
            "boll_upper",
            "boll_lower",
            "cci14",
            "wr14",
            "atr14",
            "source",
            "check_status",
            "note",
        ]
    ]
    path = project_path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build technical indicator self-check report.")
    parser.add_argument("--output-csv", default="data/reports/technical_indicator_self_check.csv")
    args = parser.parse_args()
    report = build_technical_indicator_self_check(args.output_csv)
    sample = report[report["trade_time"].eq(SELF_CHECK_DATE)]
    print(sample.to_string(index=False) if not sample.empty else f"No sample row for {SELF_CHECK_DATE}")


if __name__ == "__main__":
    main()
