from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import project_path
from src.common.db import get_database_name, read_sql
from src.features.technical_indicators import atr, boll, cci, kdj, macd, rsi_cn, wr


OUTPUT_PATH = "data/reports/technical_indicator_self_check.csv"
TARGET_INDEX = "000852"
SAMPLE_DATE = pd.Timestamp("2025-03-24")
EXPECTED_CLOSE = 6360.341
EXPECTED_RSI6 = 29.14


def _load_daily_history(target_index: str) -> pd.DataFrame:
    data = read_sql(
        """
        SELECT trade_date, open, high, low, close, volume, amount
        FROM market_index_daily
        WHERE index_code = :target_index
        ORDER BY trade_date
        """,
        {"target_index": target_index},
    )
    if data.empty:
        raise RuntimeError(f"No market_index_daily rows found for {target_index}.")
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values("trade_date").reset_index(drop=True)


def _add_daily_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]

    result["rsi6"] = rsi_cn(close, 6)
    result["rsi14"] = rsi_cn(close, 14)

    macd_frame = macd(close)
    result["macd_dif"] = macd_frame["macd_dif"]
    result["macd_dea"] = macd_frame["macd_dea"]
    result["macd_bar"] = macd_frame["macd_bar"]

    kdj_frame = kdj(high, low, close)
    result["kdj_k"] = kdj_frame["kdj_k"]
    result["kdj_d"] = kdj_frame["kdj_d"]
    result["kdj_j"] = kdj_frame["kdj_j"]

    boll_frame = boll(close, window=20, multiplier=2.0, ddof=0)
    result["boll_mid"] = boll_frame["boll_mid"]
    result["boll_upper"] = boll_frame["boll_upper"]
    result["boll_lower"] = boll_frame["boll_lower"]

    result["cci14"] = cci(high, low, close, window=14)
    result["wr14"] = wr(high, low, close, window=14, sign="negative")
    result["atr14"] = atr(high, low, close, window=14, normalize=True)
    return result


def _daily_sample_row(target_index: str) -> dict:
    data = _add_daily_indicators(_load_daily_history(target_index))
    sample = data[data["trade_date"] == SAMPLE_DATE]
    if sample.empty:
        raise RuntimeError(f"No daily sample found for {target_index} {SAMPLE_DATE.date()}.")
    row = sample.iloc[0]
    close_ok = np.isclose(float(row["close"]), EXPECTED_CLOSE, atol=0.001)
    rsi_ok = np.isclose(float(row["rsi6"]), EXPECTED_RSI6, atol=0.02)
    status = "ok" if close_ok and rsi_ok else "mismatch"
    note = (
        "daily full-history warm-up; RSI=SMA_CN; MACD_BAR=2*(DIF-DEA); "
        "BOLL ddof=0; WR negative; atr14=ATR/CLOSE"
    )
    if not close_ok:
        note += f"; expected_close={EXPECTED_CLOSE}"
    if not rsi_ok:
        note += f"; expected_rsi6~={EXPECTED_RSI6}"

    return {
        "freq": "daily",
        "trade_time": row["trade_date"].strftime("%Y-%m-%d"),
        "open": row["open"],
        "high": row["high"],
        "low": row["low"],
        "close": row["close"],
        "rsi6": row["rsi6"],
        "rsi14": row["rsi14"],
        "macd_dif": row["macd_dif"],
        "macd_dea": row["macd_dea"],
        "macd_bar": row["macd_bar"],
        "kdj_k": row["kdj_k"],
        "kdj_d": row["kdj_d"],
        "kdj_j": row["kdj_j"],
        "boll_mid": row["boll_mid"],
        "boll_upper": row["boll_upper"],
        "boll_lower": row["boll_lower"],
        "cci14": row["cci14"],
        "wr14": row["wr14"],
        "atr14": row["atr14"],
        "source": "market_index_daily",
        "check_status": status,
        "note": note,
    }


def _table_exists(table_name: str) -> bool:
    database = get_database_name()
    result = read_sql(
        """
        SELECT COUNT(*) AS cnt
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = :database
          AND TABLE_NAME = :table_name
        """,
        {"database": database, "table_name": table_name},
    )
    return bool(int(result["cnt"].iloc[0]))


def _minute_check_rows(target_index: str) -> list[dict]:
    candidate_tables = [
        "market_index_minute",
        "market_index_minute_raw",
        "market_index_intraday",
        "market_index_1m",
    ]
    existing = [table for table in candidate_tables if _table_exists(table)]
    if not existing:
        return [
            _empty_check_row(
                "60min",
                "skipped",
                "no minute K table found; expected 2025-03-24 bars: 10:30/11:30/14:00/15:00",
            ),
            _empty_check_row(
                "120min",
                "skipped",
                "no minute K table found; expected 2025-03-24 bars: 11:30/15:00",
            ),
        ]

    return [
        _empty_check_row(
            "60min",
            "manual_check_required",
            f"minute table exists ({','.join(existing)}); verify 10:30/11:30/14:00/15:00 bars before using",
        ),
        _empty_check_row(
            "120min",
            "manual_check_required",
            f"minute table exists ({','.join(existing)}); verify 11:30/15:00 bars before using",
        ),
    ]


def _empty_check_row(freq: str, status: str, note: str) -> dict:
    return {
        "freq": freq,
        "trade_time": SAMPLE_DATE.strftime("%Y-%m-%d"),
        "open": np.nan,
        "high": np.nan,
        "low": np.nan,
        "close": np.nan,
        "rsi6": np.nan,
        "rsi14": np.nan,
        "macd_dif": np.nan,
        "macd_dea": np.nan,
        "macd_bar": np.nan,
        "kdj_k": np.nan,
        "kdj_d": np.nan,
        "kdj_j": np.nan,
        "boll_mid": np.nan,
        "boll_upper": np.nan,
        "boll_lower": np.nan,
        "cci14": np.nan,
        "wr14": np.nan,
        "atr14": np.nan,
        "source": "",
        "check_status": status,
        "note": note,
    }


def build_self_check(output_path: str = OUTPUT_PATH, target_index: str = TARGET_INDEX) -> pd.DataFrame:
    rows = [_daily_sample_row(target_index), *_minute_check_rows(target_index)]
    result = pd.DataFrame(rows)
    path = project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build technical indicator formula self-check report.")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--target-index", default=TARGET_INDEX)
    args = parser.parse_args()

    result = build_self_check(args.output, args.target_index)
    pd.set_option("display.max_columns", None)
    print(result.to_string(index=False))
    print(f"\nwritten: {Path(project_path(args.output)).as_posix()}")


if __name__ == "__main__":
    main()
