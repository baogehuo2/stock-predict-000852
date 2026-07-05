from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.common.config import project_path
from src.label_tool.indicators import add_indicators
from src.label_tool.kline_loader import KlineRequest, load_aggregated_minute, load_kline


SELF_CHECK_COLUMNS = [
    "freq",
    "trade_time",
    "open",
    "high",
    "low",
    "close",
    "rsi6",
    "rsi12",
    "rsi24",
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


def _format_time(value: object) -> str:
    if pd.isna(value):
        return ""
    ts = pd.to_datetime(value)
    return ts.strftime("%Y-%m-%d %H:%M" if ts.time() != pd.Timestamp("00:00").time() else "%Y-%m-%d")


def _row_from_series(row: pd.Series, freq: str, status: str, note: str) -> dict:
    result = {col: "" for col in SELF_CHECK_COLUMNS}
    result.update(
        {
            "freq": freq,
            "trade_time": _format_time(row.get("trade_date", "")),
            "source": "market_index_daily" if freq == "daily" else "market_index_minute_raw(freq=1min)",
            "check_status": status,
            "note": note,
        }
    )
    for col in SELF_CHECK_COLUMNS:
        if col in result and result[col] != "":
            continue
        if col in row:
            value = row[col]
            if pd.isna(value):
                result[col] = ""
            elif isinstance(value, float):
                result[col] = round(value, 6)
            else:
                result[col] = value
    return result


def build_self_check_frame() -> pd.DataFrame:
    rows: list[dict] = []

    daily = load_kline(KlineRequest(freq="daily"))
    sample = daily[pd.to_datetime(daily["trade_date"]).dt.date == pd.Timestamp("2025-03-24").date()]
    if sample.empty:
        rows.append({"freq": "daily", "trade_time": "2025-03-24", "check_status": "FAIL", "note": "日K样本缺失"})
    else:
        row = sample.iloc[0]
        rsi6 = float(row["rsi6"])
        close = float(row["close"])
        status = "PASS" if abs(rsi6 - 29.14) <= 0.05 and abs(close - 6360.341) <= 0.001 else "WARN"
        note = f"RSI6 expected ~=29.14, actual={rsi6:.4f}; close expected=6360.341, actual={close:.3f}; BOLL std ddof=0"
        rows.append(_row_from_series(row, "daily", status, note))

    for freq, expected_times in (
        ("60min", ["10:30", "11:30", "14:00", "15:00"]),
        ("120min", ["11:30", "15:00"]),
    ):
        try:
            data = load_aggregated_minute(freq)
        except Exception as exc:
            rows.append({"freq": freq, "trade_time": "2025-03-24", "check_status": "FAIL", "note": str(exc)})
            continue
        if data.empty:
            rows.append({"freq": freq, "trade_time": "2025-03-24", "check_status": "FAIL", "note": "分钟K样本缺失"})
            continue
        data = add_indicators(data)
        part = data[pd.to_datetime(data["trade_date"]).dt.date == pd.Timestamp("2025-03-24").date()].copy()
        actual_times = pd.to_datetime(part["trade_date"]).dt.strftime("%H:%M").tolist()
        status = "PASS" if actual_times == expected_times else "FAIL"
        note = f"expected_times={','.join(expected_times)}; actual_times={','.join(actual_times)}"
        if part.empty:
            rows.append({"freq": freq, "trade_time": "2025-03-24", "check_status": "FAIL", "note": note})
        else:
            for _, row in part.iterrows():
                rows.append(_row_from_series(row, freq, status, note))

    out = pd.DataFrame(rows)
    for col in SELF_CHECK_COLUMNS:
        if col not in out:
            out[col] = ""
    return out[SELF_CHECK_COLUMNS]


def write_self_check_csv(path: str | Path | None = None) -> Path:
    out_path = Path(path) if path else project_path("data", "reports", "technical_indicator_self_check.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame = build_self_check_frame()
    frame.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> None:
    path = write_self_check_csv()
    print(path)


if __name__ == "__main__":
    main()

