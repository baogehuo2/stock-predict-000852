from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
import platform
import time
from queue import Empty
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.common.config import load_yaml, project_path
from src.common.network import disable_env_proxies


DATE_CANDIDATES = ("日期", "date", "trade_date", "交易日期", "报告日", "净值日期")
OHLC_COLUMN_SETS = (("开盘", "最高", "最低", "收盘"), ("open", "high", "low", "close"))


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "row_count": int(len(df)),
        "columns": [str(column) for column in df.columns],
        "dtypes": {str(column): str(dtype) for column, dtype in df.dtypes.items()},
        "null_rate": {
            str(column): round(float(df[column].isna().mean()), 8) for column in df.columns
        },
        "sample_rows": [
            {str(key): _json_value(value) for key, value in row.items()}
            for row in df.head(3).to_dict(orient="records")
        ],
    }
    for column in DATE_CANDIDATES:
        if column not in df.columns:
            continue
        values = pd.to_datetime(df[column], errors="coerce").dropna()
        if not values.empty:
            summary["date_column"] = column
            summary["min_date"] = values.min().date().isoformat()
            summary["max_date"] = values.max().date().isoformat()
            break
    for columns in OHLC_COLUMN_SETS:
        if not all(column in df.columns for column in columns):
            continue
        numeric = df[list(columns)].apply(pd.to_numeric, errors="coerce")
        summary["incomplete_ohlc_rows"] = int(numeric.isna().any(axis=1).sum())
        summary["adjacent_duplicate_ohlc_rows"] = int(numeric.eq(numeric.shift()).all(axis=1).sum())
        break
    return summary


def probe_function(ak: Any, dataset_name: str, probe: dict[str, Any], dataset: dict[str, Any]) -> dict[str, Any]:
    function_name = str(probe["function"])
    parameters = dict(probe.get("parameters") or {})
    result: dict[str, Any] = {
        "dataset_name": dataset_name,
        "probe_name": probe["name"],
        "function": function_name,
        "parameters": parameters,
        "official_reference": str(probe.get("official_reference", dataset.get("official_reference", ""))),
        "underlying_source": str(probe.get("underlying_source", dataset.get("underlying_source", ""))),
        "declared_units": probe.get("units", dataset.get("units", {})),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    if function_name == "local:fetch_bse_stock_list":
        from src.collectors.stock_universe_sources import fetch_bse_stock_list

        function = fetch_bse_stock_list
    elif function_name == "local:fetch_margin_szse_eastmoney_history":
        from src.collectors.collect_margin_market_v2 import (
            fetch_margin_szse_eastmoney_history,
        )

        function = fetch_margin_szse_eastmoney_history
    elif function_name == "local:fetch_szse_etf_shares_http":
        from src.collectors.collect_etf_fund_v2 import fetch_szse_shares_http

        function = fetch_szse_shares_http
    elif function_name == "local:fetch_etf_nav_history":
        from src.collectors.collect_etf_fund_v2 import fetch_nav_history

        function = fetch_nav_history
    else:
        function = getattr(ak, function_name, None)
    if function is None:
        result.update(status="unavailable", error="AKShare中不存在该函数")
        return result
    result["signature"] = str(inspect.signature(function))
    started = time.perf_counter()
    try:
        raw = function(**parameters)
        if not isinstance(raw, pd.DataFrame):
            raise TypeError(f"返回类型不是DataFrame: {type(raw).__name__}")
        result.update(status="ok" if not raw.empty else "empty", **summarize_dataframe(raw))
        if raw.attrs.get("data_source"):
            result["actual_data_source"] = raw.attrs["data_source"]
        if raw.attrs.get("source_url"):
            result["actual_source_url"] = raw.attrs["source_url"]
    except Exception as exc:
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    return result


def _probe_worker(dataset_name: str, probe: dict[str, Any], dataset: dict[str, Any], queue: Any) -> None:
    try:
        import akshare as ak

        queue.put(probe_function(ak, dataset_name, probe, dataset))
    except Exception as exc:
        queue.put(
            {
                "dataset_name": dataset_name,
                "probe_name": probe.get("name"),
                "function": probe.get("function"),
                "parameters": probe.get("parameters", {}),
                "official_reference": str(probe.get("official_reference", dataset.get("official_reference", ""))),
                "underlying_source": str(probe.get("underlying_source", dataset.get("underlying_source", ""))),
                "declared_units": probe.get("units", dataset.get("units", {})),
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )


def probe_with_timeout(dataset_name: str, probe: dict[str, Any], dataset: dict[str, Any],
                       timeout_seconds: int) -> dict[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue(maxsize=1)
    process = context.Process(target=_probe_worker, args=(dataset_name, probe, dataset, queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(5)
        return {
            "dataset_name": dataset_name,
            "probe_name": probe.get("name"),
            "function": probe.get("function"),
            "parameters": probe.get("parameters", {}),
            "official_reference": str(probe.get("official_reference", dataset.get("official_reference", ""))),
            "underlying_source": str(probe.get("underlying_source", dataset.get("underlying_source", ""))),
            "declared_units": probe.get("units", dataset.get("units", {})),
            "status": "timeout",
            "error": f"单接口超过{timeout_seconds}秒未完成",
            "elapsed_seconds": timeout_seconds,
        }
    try:
        return queue.get_nowait()
    except Empty:
        pass
    return {
        "dataset_name": dataset_name,
        "probe_name": probe.get("name"),
        "function": probe.get("function"),
        "parameters": probe.get("parameters", {}),
        "official_reference": str(probe.get("official_reference", dataset.get("official_reference", ""))),
        "underlying_source": str(probe.get("underlying_source", dataset.get("underlying_source", ""))),
        "declared_units": probe.get("units", dataset.get("units", {})),
        "status": "error",
        "error": f"探测子进程退出且无结果，exitcode={process.exitcode}",
    }


def run_probes(dataset_names: list[str] | None = None, timeout_seconds: int = 45) -> dict[str, Any]:
    disable_env_proxies()
    import akshare as ak

    config = load_yaml(project_path("config", "data_sources_v2.yaml"))
    selected = dataset_names or list(config)
    unknown = sorted(set(selected) - set(config))
    if unknown:
        raise ValueError(f"未知数据集: {', '.join(unknown)}")
    results: list[dict[str, Any]] = []
    for dataset_name in selected:
        dataset = config[dataset_name]
        for probe in dataset.get("probes", []):
            results.append(probe_with_timeout(dataset_name, probe, dataset, timeout_seconds))
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "akshare_version": getattr(ak, "__version__", "unknown"),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="探测模型2.0第一批AKShare数据来源，不写业务表。")
    parser.add_argument("--dataset", action="append", help="可重复指定数据集；省略时探测全部。")
    parser.add_argument(
        "--output",
        default="data/reports/source_probe_v2.json",
        help="JSON报告路径。",
    )
    parser.add_argument("--timeout-seconds", type=int, default=45, help="单个AKShare接口硬超时。")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds 必须大于0")
    report = run_probes(args.dataset, timeout_seconds=args.timeout_seconds)
    output = Path(args.output)
    if not output.is_absolute():
        output = project_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in report["results"]:
        print(
            f"{item['dataset_name']} | {item['probe_name']} | {item['status']} | "
            f"rows={item.get('row_count', 0)} | {item.get('error', '')}"
        )
    print(output)


if __name__ == "__main__":
    main()
