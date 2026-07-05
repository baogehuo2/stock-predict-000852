from __future__ import annotations

import argparse
import hashlib
import re
import time as time_module
from dataclasses import dataclass
from datetime import date, datetime, time
from typing import Any, Callable

import pandas as pd

from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.migrations import apply_migrations
from src.common.network import disable_env_proxies
from src.quality.check_collection_data import DATASETS, evaluate_dataframe, persist_metrics


CONTRACT_DATA_SOURCE = "akshare:sina:option_contract_daily"
QVIX_DATA_SOURCE = "akshare:index_option_qvix"
SINA_CFFEX_URL = "https://stock.finance.sina.com.cn/futures/view/optionsCffexDP.php"
SINA_SSE_URL = "https://stock.finance.sina.com.cn/option/quotes.html"
QVIX_URL = "https://www.akshare.xyz/data/index/index.html"


@dataclass(frozen=True)
class CffexOptionSource:
    underlying_code: str
    underlying_name: str
    product_prefix: str
    list_func: str
    spot_func: str
    daily_func: str


@dataclass(frozen=True)
class SseOptionSource:
    underlying_code: str
    underlying_name: str
    list_symbol: str
    underlying_etf: str


@dataclass(frozen=True)
class QvixSource:
    underlying_code: str
    underlying_name: str
    option_market: str
    qvix_code: str
    func_name: str


CFFEX_SOURCES: tuple[CffexOptionSource, ...] = (
    CffexOptionSource("000852", "中证1000", "MO", "option_cffex_zz1000_list_sina", "option_cffex_zz1000_spot_sina", "option_cffex_zz1000_daily_sina"),
    CffexOptionSource("000300", "沪深300", "IO", "option_cffex_hs300_list_sina", "option_cffex_hs300_spot_sina", "option_cffex_hs300_daily_sina"),
    CffexOptionSource("000016", "上证50", "HO", "option_cffex_sz50_list_sina", "option_cffex_sz50_spot_sina", "option_cffex_sz50_daily_sina"),
)

SSE_SOURCES: tuple[SseOptionSource, ...] = (
    SseOptionSource("000016", "上证50ETF期权", "50ETF", "510050"),
    SseOptionSource("000300", "沪深300ETF期权", "300ETF", "510300"),
    SseOptionSource("000905", "中证500ETF期权", "500ETF", "510500"),
)

QVIX_SOURCES: tuple[QvixSource, ...] = (
    QvixSource("000016", "上证50ETF期权波动率指数", "sse_etf_option", "50etf_qvix", "index_option_50etf_qvix"),
    QvixSource("000300", "沪深300ETF期权波动率指数", "sse_etf_option", "300etf_qvix", "index_option_300etf_qvix"),
    QvixSource("000905", "中证500ETF期权波动率指数", "sse_etf_option", "500etf_qvix", "index_option_500etf_qvix"),
    QvixSource("399006", "创业板ETF期权波动率指数", "szse_etf_option", "cyb_qvix", "index_option_cyb_qvix"),
    QvixSource("000688", "科创50ETF期权波动率指数", "sse_etf_option", "kcb_qvix", "index_option_kcb_qvix"),
)


def _available_time(trade_date: date) -> datetime:
    return datetime.combine(
        (pd.Timestamp(trade_date) + pd.Timedelta(days=1)).date(), time(0, 0)
    )


def _hash_row(row: pd.Series, columns: list[str]) -> str:
    payload = "|".join("" if pd.isna(row[column]) else str(row[column]) for column in columns)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _call_with_retry(fn: Callable[..., Any], *args: Any, retries: int = 3, **kwargs: Any) -> Any:
    errors: list[str] = []
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt + 1 < retries:
                time_module.sleep(1 + attempt)
    raise RuntimeError("; ".join(errors))


def _is_source_no_data_error(exc: Exception) -> bool:
    message = str(exc)
    return "SyntaxError: invalid syntax" in message or "invalid syntax (<string>, line 0)" in message


def _contract_month(contract_code: str) -> str | None:
    match = re.search(r"(\d{4})", str(contract_code))
    return match.group(1) if match else None


def _option_side(contract_code: str) -> str | None:
    code = str(contract_code).upper()
    if "C" in code:
        return "call"
    if "P" in code:
        return "put"
    return None


def _strike_from_contract(contract_code: str) -> float | None:
    match = re.search(r"[CP](\d+(?:\.\d+)?)$", str(contract_code).upper())
    return float(match.group(1)) if match else None


def normalize_contract_daily(
    raw: pd.DataFrame,
    *,
    underlying_code: str,
    underlying_name: str,
    option_market: str,
    contract_code: str,
    data_source: str = CONTRACT_DATA_SOURCE,
    source_url: str,
    start_date: str | None = None,
    end_date: str | None = None,
    strike_price: float | None = None,
    option_side: str | None = None,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    aliases = {
        "date": "trade_date",
        "日期": "trade_date",
        "open": "open",
        "开盘": "open",
        "high": "high",
        "最高": "high",
        "low": "low",
        "最低": "low",
        "close": "close",
        "收盘": "close",
        "volume": "volume",
        "成交量": "volume",
        "amount": "amount",
        "成交额": "amount",
        "open_interest": "open_interest",
        "持仓量": "open_interest",
    }
    columns = [column for column in aliases if column in raw.columns]
    frame = raw[columns].rename(columns=aliases).copy()
    if "trade_date" not in frame.columns:
        raise ValueError(f"{contract_code} missing trade date column")
    for column in ("open", "high", "low", "close", "volume", "amount", "open_interest"):
        if column not in frame.columns:
            frame[column] = None
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"])
    if start_date:
        frame = frame[frame["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        frame = frame[frame["trade_date"] <= pd.Timestamp(end_date)]
    if frame.empty:
        return pd.DataFrame()
    for column in ("open", "high", "low", "close", "volume", "amount", "open_interest"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["underlying_code"] = underlying_code
    frame["underlying_name"] = underlying_name
    frame["option_market"] = option_market
    frame["contract_code"] = str(contract_code)
    frame["contract_month"] = _contract_month(contract_code)
    frame["option_side"] = option_side if option_side else _option_side(contract_code)
    frame["strike_price"] = strike_price if strike_price is not None else _strike_from_contract(contract_code)
    frame["data_source"] = data_source
    frame["source_url"] = source_url
    frame["available_time"] = frame["trade_date"].map(_available_time)
    frame["crawl_time"] = datetime.now()
    out = frame[
        [
            "trade_date", "underlying_code", "underlying_name", "option_market",
            "contract_code", "contract_month", "option_side", "strike_price",
            "open", "high", "low", "close", "volume", "amount", "open_interest",
            "data_source", "source_url", "available_time", "crawl_time",
        ]
    ].copy()
    hash_columns = [
        "trade_date", "contract_code", "open", "high", "low", "close",
        "volume", "amount", "open_interest", "data_source",
    ]
    out["raw_hash"] = out.apply(lambda row: _hash_row(row, hash_columns), axis=1)
    return out.drop_duplicates(["trade_date", "contract_code", "data_source"], keep="last")


def normalize_qvix_daily(
    raw: pd.DataFrame,
    source: QvixSource,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    frame = raw.rename(columns={"date": "trade_date"}).copy()
    required = {"trade_date", "open", "high", "low", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source.qvix_code} missing columns: {sorted(missing)}")
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame = frame.dropna(subset=["trade_date"])
    frame = frame[
        (frame["trade_date"] >= pd.Timestamp(start_date))
        & (frame["trade_date"] <= pd.Timestamp(end_date))
    ].copy()
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"], how="all")
    if frame.empty:
        return pd.DataFrame()
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["underlying_code"] = source.underlying_code
    frame["underlying_name"] = source.underlying_name
    frame["option_market"] = source.option_market
    frame["qvix_code"] = source.qvix_code
    frame["data_source"] = QVIX_DATA_SOURCE
    frame["source_url"] = QVIX_URL
    frame["available_time"] = frame["trade_date"].map(_available_time)
    frame["crawl_time"] = datetime.now()
    out = frame[
        [
            "trade_date", "underlying_code", "underlying_name", "option_market",
            "qvix_code", "open", "high", "low", "close", "data_source",
            "source_url", "available_time", "crawl_time",
        ]
    ].copy()
    out["raw_hash"] = out.apply(
        lambda row: _hash_row(row, ["trade_date", "qvix_code", "open", "high", "low", "close", "data_source"]),
        axis=1,
    )
    return out.drop_duplicates(["trade_date", "qvix_code", "data_source"], keep="last")


def _discover_cffex_contracts(ak: Any, source: CffexOptionSource, retries: int) -> list[tuple[str, float | None, str | None]]:
    listed = _call_with_retry(getattr(ak, source.list_func), retries=retries)
    months = next(iter(listed.values())) if isinstance(listed, dict) and listed else []
    contracts: dict[str, tuple[float | None, str | None]] = {}
    for month in months:
        spot = _call_with_retry(getattr(ak, source.spot_func), month, retries=retries)
        if not isinstance(spot, pd.DataFrame) or spot.empty:
            continue
        for side_column, side in (("看涨合约-标识", "call"), ("看跌合约-标识", "put")):
            if side_column not in spot.columns:
                continue
            for _, row in spot.dropna(subset=[side_column]).iterrows():
                code = str(row[side_column]).strip()
                if code:
                    strike = pd.to_numeric(row.get("行权价"), errors="coerce")
                    contracts[code] = (None if pd.isna(strike) else float(strike), side)
    return [(code, values[0], values[1]) for code, values in sorted(contracts.items())]


def _sse_strike_price(ak: Any, contract_code: str, retries: int) -> float | None:
    try:
        spot = _call_with_retry(getattr(ak, "option_sse_spot_price_sina"), contract_code, retries=retries)
    except Exception:
        return None
    if not isinstance(spot, pd.DataFrame) or not {"字段", "值"}.issubset(spot.columns):
        return None
    row = spot[spot["字段"].astype(str) == "行权价"]
    if row.empty:
        return None
    value = pd.to_numeric(row.iloc[0]["值"], errors="coerce")
    return None if pd.isna(value) else float(value)


def _discover_sse_contracts(ak: Any, source: SseOptionSource, retries: int) -> list[tuple[str, float | None, str | None]]:
    months = _call_with_retry(getattr(ak, "option_sse_list_sina"), source.list_symbol, retries=retries)
    contracts: dict[str, tuple[float | None, str | None]] = {}
    for month in months:
        for side_name, side in (("看涨期权", "call"), ("看跌期权", "put")):
            codes = _call_with_retry(
                getattr(ak, "option_sse_codes_sina"),
                side_name,
                month,
                source.underlying_etf,
                retries=retries,
            )
            if not isinstance(codes, pd.DataFrame) or "期权代码" not in codes.columns:
                continue
            for code in codes["期权代码"].dropna().astype(str):
                code = code.strip()
                contracts[code] = (None, side)
    return [
        (code, strike if strike is not None else _sse_strike_price(ak, code, retries), side)
        for code, (strike, side) in sorted(contracts.items())
    ]


def collect_option_contracts(
    start_date: str,
    end_date: str,
    retries: int = 3,
    max_contracts: int | None = None,
    run_quality: bool = True,
) -> dict[str, Any]:
    disable_env_proxies()
    import akshare as ak

    ingestion = IngestionRun(
        dataset_name="option_contract_daily_raw",
        data_source=CONTRACT_DATA_SOURCE,
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    try:
        frames: list[pd.DataFrame] = []
        discovered: dict[str, int] = {}
        failed: list[dict[str, str]] = []
        skipped: list[dict[str, str]] = []

        for source in CFFEX_SOURCES:
            contracts = _discover_cffex_contracts(ak, source, retries)
            discovered[f"cffex:{source.product_prefix}"] = len(contracts)
            for contract_code, strike_price, option_side in contracts[:max_contracts]:
                try:
                    raw = _call_with_retry(getattr(ak, source.daily_func), contract_code, retries=retries)
                    frame = normalize_contract_daily(
                        raw,
                        underlying_code=source.underlying_code,
                        underlying_name=source.underlying_name,
                        option_market="cffex_index_option",
                        contract_code=contract_code,
                        source_url=SINA_CFFEX_URL,
                        start_date=start_date,
                        end_date=end_date,
                        strike_price=strike_price,
                        option_side=option_side,
                    )
                    if not frame.empty:
                        frames.append(frame)
                except Exception as exc:
                    record = {"contract_code": contract_code, "error": f"{type(exc).__name__}: {exc}"}
                    if _is_source_no_data_error(exc):
                        skipped.append(record)
                    else:
                        failed.append(record)

        for source in SSE_SOURCES:
            contracts = _discover_sse_contracts(ak, source, retries)
            discovered[f"sse:{source.underlying_etf}"] = len(contracts)
            for contract_code, strike_price, option_side in contracts[:max_contracts]:
                try:
                    raw = _call_with_retry(getattr(ak, "option_sse_daily_sina"), contract_code, retries=retries)
                    frame = normalize_contract_daily(
                        raw,
                        underlying_code=source.underlying_code,
                        underlying_name=source.underlying_name,
                        option_market="sse_etf_option",
                        contract_code=contract_code,
                        source_url=SINA_SSE_URL,
                        start_date=start_date,
                        end_date=end_date,
                        strike_price=strike_price,
                        option_side=option_side,
                    )
                    if not frame.empty:
                        frames.append(frame)
                except Exception as exc:
                    record = {"contract_code": contract_code, "error": f"{type(exc).__name__}: {exc}"}
                    if _is_source_no_data_error(exc):
                        skipped.append(record)
                    else:
                        failed.append(record)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        ingestion.fetched_rows = len(combined)
        if not combined.empty:
            existing = read_sql(
                """
                SELECT trade_date, contract_code, data_source FROM option_contract_daily_raw
                WHERE trade_date BETWEEN :start_date AND :end_date
                """,
                {"start_date": start_date, "end_date": end_date},
            )
            old_keys = set(zip(existing["trade_date"].astype(str), existing["contract_code"], existing["data_source"])) if not existing.empty else set()
            new_keys = set(zip(combined["trade_date"].astype(str), combined["contract_code"], combined["data_source"]))
            ingestion.inserted_rows = len(new_keys - old_keys)
            ingestion.updated_rows = len(new_keys & old_keys)
            upsert_dataframe(
                combined,
                "option_contract_daily_raw",
                ["trade_date", "contract_code", "data_source"],
            )
        metrics: list[dict[str, Any]] = []
        if run_quality:
            stored = read_sql(
                """
                SELECT * FROM option_contract_daily_raw
                WHERE trade_date BETWEEN :start_date AND :end_date
                """,
                {"start_date": start_date, "end_date": end_date},
            )
            metrics = evaluate_dataframe(stored, DATASETS["option_contract_daily_raw"])
            persist_metrics("option_contract_daily_raw", metrics, date.today())
        ingestion.error_rows = len(failed)
        ingestion.finish("success")
        return {
            "table": "option_contract_daily_raw",
            "rows": len(combined),
            "inserted_rows": ingestion.inserted_rows,
            "updated_rows": ingestion.updated_rows,
            "discovered_contracts": discovered,
            "failed_contracts": failed[:20],
            "failed_count": len(failed),
            "skipped_contracts": skipped[:20],
            "skipped_count": len(skipped),
            "quality": metrics,
        }
    except Exception as exc:
        ingestion.error_rows += 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def collect_option_qvix(
    start_date: str,
    end_date: str,
    retries: int = 3,
    run_quality: bool = True,
) -> dict[str, Any]:
    disable_env_proxies()
    import akshare as ak

    ingestion = IngestionRun(
        dataset_name="option_qvix_daily_raw",
        data_source=QVIX_DATA_SOURCE,
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    try:
        frames: list[pd.DataFrame] = []
        failed: list[dict[str, str]] = []
        for source in QVIX_SOURCES:
            try:
                raw = _call_with_retry(getattr(ak, source.func_name), retries=retries)
                frame = normalize_qvix_daily(raw, source, start_date, end_date)
                if not frame.empty:
                    frames.append(frame)
            except Exception as exc:
                failed.append({"qvix_code": source.qvix_code, "error": f"{type(exc).__name__}: {exc}"})
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        ingestion.fetched_rows = len(combined)
        if not combined.empty:
            existing = read_sql(
                """
                SELECT trade_date, qvix_code, data_source FROM option_qvix_daily_raw
                WHERE trade_date BETWEEN :start_date AND :end_date
                """,
                {"start_date": start_date, "end_date": end_date},
            )
            old_keys = set(zip(existing["trade_date"].astype(str), existing["qvix_code"], existing["data_source"])) if not existing.empty else set()
            new_keys = set(zip(combined["trade_date"].astype(str), combined["qvix_code"], combined["data_source"]))
            ingestion.inserted_rows = len(new_keys - old_keys)
            ingestion.updated_rows = len(new_keys & old_keys)
            upsert_dataframe(combined, "option_qvix_daily_raw", ["trade_date", "qvix_code", "data_source"])
        metrics: list[dict[str, Any]] = []
        if run_quality:
            stored = read_sql(
                """
                SELECT * FROM option_qvix_daily_raw
                WHERE trade_date BETWEEN :start_date AND :end_date
                """,
                {"start_date": start_date, "end_date": end_date},
            )
            metrics = evaluate_dataframe(stored, DATASETS["option_qvix_daily_raw"])
            persist_metrics("option_qvix_daily_raw", metrics, date.today())
        ingestion.error_rows = len(failed)
        ingestion.finish("success")
        return {
            "table": "option_qvix_daily_raw",
            "rows": len(combined),
            "inserted_rows": ingestion.inserted_rows,
            "updated_rows": ingestion.updated_rows,
            "failed": failed,
            "quality": metrics,
        }
    except Exception as exc:
        ingestion.error_rows += 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def collect_option_history(
    start_date: str,
    end_date: str,
    dataset: str = "all",
    retries: int = 3,
    max_contracts: int | None = None,
    run_quality: bool = True,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if dataset in ("all", "contracts"):
        result["contracts"] = collect_option_contracts(
            start_date,
            end_date,
            retries=retries,
            max_contracts=max_contracts,
            run_quality=run_quality,
        )
    if dataset in ("all", "qvix"):
        result["qvix"] = collect_option_qvix(
            start_date,
            end_date,
            retries=retries,
            run_quality=run_quality,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="采集指数相关历史期权数据：中金所指数期权、上交所 ETF 期权、ETF 期权 QVIX。")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--dataset", choices=("all", "contracts", "qvix"), default="all")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-contracts", type=int, default=None, help="调试用：限制每个来源最多拉取多少个合约。")
    parser.add_argument("--apply-migrations", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    if args.apply_migrations:
        apply_migrations()
    result = collect_option_history(
        args.start_date,
        args.end_date,
        dataset=args.dataset,
        retries=args.retries,
        max_contracts=args.max_contracts,
        run_quality=not args.skip_quality,
    )
    print(result)


if __name__ == "__main__":
    main()
