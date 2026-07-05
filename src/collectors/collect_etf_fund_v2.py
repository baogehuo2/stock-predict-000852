from __future__ import annotations

import argparse
import io
import math
import random
import time as time_module
from datetime import date, datetime, time
from typing import Any

import pandas as pd
import requests

from src.common.config import load_yaml, project_path
from src.common.db import read_sql, upsert_dataframe
from src.common.ingestion import IngestionRun
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.quality.check_collection_data import (
    DATASETS,
    evaluate_dataframe,
    evaluate_etf_field_coverage,
    persist_metrics,
)


logger = get_logger(__name__)
SSE_SHARE_SOURCE = "akshare:fund_etf_scale_sse"
SSE_SHARE_FALLBACK_SOURCE = "sse:COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L"
SZSE_SHARE_SOURCE = "akshare:fund_scale_daily_szse"
SZSE_HTTP_SOURCE = "szse:http:scsj_fund_jjgm"
NAV_SOURCE = "eastmoney:fund_f10_lsjz"
SSE_SHARE_URL = "https://www.sse.com.cn/assortment/fund/etf/list/scale/"
SZSE_SHARE_URL = "https://www.szse.cn/market/fund/volume/etf/index.html"
SZSE_HTTP_API_URL = "http://www.szse.cn/api/report/ShowReport"
NAV_URL_TEMPLATE = "https://fundf10.eastmoney.com/jjjz_{code}.html"


def _exchange(code: str) -> str:
    return "SSE" if code.startswith(("5", "6")) else "SZSE"


def _conservative_available_time(trade_date: date) -> datetime:
    return datetime.combine(
        (pd.Timestamp(trade_date) + pd.Timedelta(days=1)).date(), time(0, 0)
    )


def _configured_etfs(codes: list[str] | None = None) -> pd.DataFrame:
    config = load_yaml(project_path("config", "symbols.yaml"))
    frame = pd.DataFrame(config.get("etfs") or [])
    if frame.empty:
        raise RuntimeError("config/symbols.yaml 中没有 etfs 配置")
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if codes:
        requested = {str(code).zfill(6) for code in codes}
        unknown = sorted(requested - set(frame["code"]))
        if unknown:
            raise ValueError(f"ETF 不在配置中: {', '.join(unknown)}")
        frame = frame[frame["code"].isin(requested)].copy()
    frame["exchange"] = frame["code"].map(_exchange)
    return frame.reset_index(drop=True)


def normalize_sse_shares(
    raw: pd.DataFrame, selected_codes: set[str], data_source: str = SSE_SHARE_SOURCE
) -> pd.DataFrame:
    required = {"基金代码", "基金简称", "统计日期", "基金份额"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"SSE ETF share data missing columns: {missing}")
    frame = raw.rename(
        columns={
            "基金代码": "etf_code",
            "基金简称": "fund_name",
            "统计日期": "trade_date",
            "基金份额": "fund_share",
        }
    )[["trade_date", "etf_code", "fund_name", "fund_share"]].copy()
    frame["etf_code"] = frame["etf_code"].astype(str).str.extract(r"(\d{6})", expand=False)
    frame = frame[frame["etf_code"].isin(selected_codes)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["fund_share"] = pd.to_numeric(frame["fund_share"], errors="coerce")
    frame = frame.dropna(subset=["trade_date", "etf_code", "fund_share"])
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["exchange"] = "SSE"
    frame["share_data_source"] = data_source
    frame["share_source_url"] = SSE_SHARE_URL
    frame["share_available_time"] = frame["trade_date"].map(_conservative_available_time)
    return frame.drop_duplicates(["trade_date", "etf_code"], keep="last")


def normalize_szse_shares(
    raw: pd.DataFrame, selected_codes: set[str], data_source: str = SZSE_SHARE_SOURCE
) -> pd.DataFrame:
    if len(raw.columns) < 4:
        raise ValueError(f"SZSE ETF share data expected at least 4 columns, got {len(raw.columns)}")
    frame = raw.iloc[:, :4].copy()
    frame.columns = ["trade_date", "etf_code", "fund_name", "fund_share"]
    frame["etf_code"] = frame["etf_code"].astype(str).str.extract(r"(\d{6})", expand=False)
    frame = frame[frame["etf_code"].isin(selected_codes)].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["fund_share"] = pd.to_numeric(
        frame["fund_share"].astype(str).str.replace(",", "", regex=False), errors="coerce"
    )
    frame = frame.dropna(subset=["trade_date", "etf_code", "fund_share"])
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["exchange"] = "SZSE"
    frame["share_data_source"] = data_source
    frame["share_source_url"] = SZSE_SHARE_URL
    frame["share_available_time"] = frame["trade_date"].map(_conservative_available_time)
    return frame.drop_duplicates(["trade_date", "etf_code"], keep="last")


def normalize_nav(raw: pd.DataFrame, code: str, fund_name: str) -> pd.DataFrame:
    columns = {
        "FSRQ": "trade_date",
        "DWJZ": "unit_nav",
        "LJJZ": "accumulated_nav",
        "SGZT": "subscription_status",
        "SHZT": "redemption_status",
    }
    missing = sorted(set(columns) - set(raw.columns))
    if missing:
        raise ValueError(f"ETF NAV data missing fields: {missing}")
    frame = raw.rename(columns=columns)[list(columns.values())].copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    for column in ("unit_nav", "accumulated_nav"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["trade_date"])
    frame["trade_date"] = frame["trade_date"].dt.date
    frame["etf_code"] = code
    frame["exchange"] = _exchange(code)
    frame["fund_name"] = fund_name
    frame["nav_data_source"] = NAV_SOURCE
    frame["nav_source_url"] = NAV_URL_TEMPLATE.format(code=code)
    frame["nav_available_time"] = frame["trade_date"].map(_conservative_available_time)
    return frame.drop_duplicates(["trade_date", "etf_code"], keep="last")


def fetch_sse_share_date(
    trade_date: str, selected_codes: set[str], prefer_akshare: bool = True
) -> pd.DataFrame:
    compact = trade_date.replace("-", "")
    if prefer_akshare:
        disable_env_proxies()
        import akshare as ak

        try:
            raw = ak.fund_etf_scale_sse(date=compact)
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                return normalize_sse_shares(raw, selected_codes)
        except Exception as exc:
            logger.warning("SSE AKShare share request failed for %s: %s", trade_date, exc)

    response = requests.get(
        "https://query.sse.com.cn/commonQuery.do",
        params={
            "isPagination": "true",
            "pageHelp.pageSize": "10000",
            "pageHelp.pageNo": "1",
            "pageHelp.beginPage": "1",
            "pageHelp.cacheSize": "1",
            "pageHelp.endPage": "1",
            "sqlId": "COMMON_SSE_ZQPZ_ETFZL_XXPL_ETFGM_SEARCH_L",
            "STAT_DATE": pd.Timestamp(trade_date).strftime("%Y-%m-%d"),
        },
        headers={"Referer": "https://www.sse.com.cn/", "User-Agent": "Mozilla/5.0"},
        timeout=30,
    )
    response.raise_for_status()
    raw = pd.DataFrame(response.json().get("result") or []).rename(
        columns={
            "SEC_CODE": "基金代码",
            "SEC_NAME": "基金简称",
            "STAT_DATE": "统计日期",
            "TOT_VOL": "基金份额",
        }
    )
    if not raw.empty:
        raw["基金份额"] = pd.to_numeric(raw["基金份额"], errors="coerce") * 10_000
    return normalize_sse_shares(raw, selected_codes, SSE_SHARE_FALLBACK_SOURCE)


def fetch_szse_shares_http(
    start_date: str = "2025-01-02",
    end_date: str = "2025-01-10",
    selected_codes: set[str] | None = None,
) -> pd.DataFrame:
    response = requests.get(
        SZSE_HTTP_API_URL,
        params={
            "SHOWTYPE": "xlsx",
            "CATALOGID": "scsj_fund_jjgm",
            "TABKEY": "tab1",
            "txtStart": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
            "txtEnd": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
            "jjlb": "ETF",
            "random": str(random.random()),
        },
        headers={"Referer": SZSE_SHARE_URL.replace("https://", "http://"), "User-Agent": "Mozilla/5.0"},
        timeout=90,
    )
    response.raise_for_status()
    raw = pd.read_excel(io.BytesIO(response.content), engine="openpyxl", dtype=str)
    result = normalize_szse_shares(
        raw, selected_codes or set(raw.iloc[:, 1].astype(str)), SZSE_HTTP_SOURCE
    )
    result.attrs["data_source"] = SZSE_HTTP_SOURCE
    result.attrs["source_url"] = SZSE_SHARE_URL
    return result


def fetch_szse_share_window(
    start_date: str,
    end_date: str,
    selected_codes: set[str],
    prefer_akshare: bool = True,
) -> tuple[pd.DataFrame, bool]:
    if prefer_akshare:
        disable_env_proxies()
        import akshare as ak

        try:
            raw = ak.fund_scale_daily_szse(
                start_date=start_date.replace("-", ""),
                end_date=end_date.replace("-", ""),
                symbol="ETF",
            )
            if isinstance(raw, pd.DataFrame) and not raw.empty:
                return normalize_szse_shares(raw, selected_codes), True
        except Exception as exc:
            logger.warning("SZSE AKShare HTTPS unavailable, switching to official HTTP: %s", exc)
    return fetch_szse_shares_http(start_date, end_date, selected_codes), False


def fetch_nav_history(
    code: str, fund_name: str, start_date: str, end_date: str
) -> pd.DataFrame:
    disable_env_proxies()
    session = requests.Session()
    session.trust_env = False
    url = "https://api.fund.eastmoney.com/f10/lsjz"
    # The API returns at most 20 rows per page even when a larger pageSize is requested.
    page_size = 20
    rows: list[dict[str, Any]] = []
    page = 1
    pages = 1
    while page <= pages:
        response = session.get(
            url,
            params={
                "fundCode": code,
                "pageIndex": page,
                "pageSize": page_size,
                "startDate": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                "endDate": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                "_": round(time_module.time() * 1000),
            },
            headers={"User-Agent": "Mozilla/5.0", "Referer": NAV_URL_TEMPLATE.format(code=code)},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        if int(payload.get("ErrCode") or 0) != 0:
            raise RuntimeError(f"Eastmoney NAV API error: {payload.get('ErrMsg')}")
        rows.extend(((payload.get("Data") or {}).get("LSJZList") or []))
        pages = max(1, math.ceil(int(payload.get("TotalCount") or 0) / page_size))
        page += 1
    if not rows:
        return pd.DataFrame()
    frame = normalize_nav(pd.DataFrame(rows), code, fund_name)
    mask = pd.to_datetime(frame["trade_date"]).between(start_date, end_date)
    return frame[mask].reset_index(drop=True)


def _six_month_ranges(start_date: str, end_date: str) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    ranges: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        # Although SZSE documents a six-month limit, large ETF universes can make
        # long XLSX responses silently lose rows near the window start.
        window_end = min(cursor + pd.Timedelta(days=89), end)
        ranges.append((cursor.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")))
        cursor = window_end + pd.Timedelta(days=1)
    return ranges


def _reference_trade_dates(start_date: str, end_date: str) -> list[str]:
    frame = read_sql(
        """
        SELECT DISTINCT trade_date FROM market_index_daily
        WHERE index_code='000852' AND trade_date BETWEEN :start_date AND :end_date
        ORDER BY trade_date
        """,
        {"start_date": start_date, "end_date": end_date},
    )
    if frame.empty:
        raise RuntimeError("market_index_daily 中缺少 000852 基准交易日")
    return pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d").tolist()


def _retry(function: Any, retries: int, *args: Any, **kwargs: Any) -> Any:
    errors: list[str] = []
    for attempt in range(retries):
        try:
            return function(*args, **kwargs)
        except Exception as exc:
            errors.append(f"attempt={attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt + 1 < retries:
                time_module.sleep(attempt + 1)
    raise RuntimeError("; ".join(errors))


def _merge_sources(shares: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    keys = ["trade_date", "etf_code"]
    if shares.empty:
        merged = nav.copy()
    elif nav.empty:
        merged = shares.copy()
    else:
        merged = shares.merge(nav, on=keys, how="outer", suffixes=("_share", "_nav"))
        for column in ("exchange", "fund_name"):
            merged[column] = merged.pop(f"{column}_share").combine_first(
                merged.pop(f"{column}_nav")
            )
    for column in (
        "fund_share", "unit_nav", "accumulated_nav", "subscription_status",
        "redemption_status", "share_data_source", "nav_data_source",
        "share_source_url", "nav_source_url", "share_available_time", "nav_available_time",
    ):
        if column not in merged:
            merged[column] = None
    merged["available_time"] = merged[["share_available_time", "nav_available_time"]].max(axis=1)
    merged["data_source"] = merged.apply(
        lambda row: " | ".join(
            value for value in (row["share_data_source"], row["nav_data_source"])
            if isinstance(value, str) and value
        ),
        axis=1,
    )
    merged["source_url"] = merged.apply(
        lambda row: " | ".join(
            value for value in (row["share_source_url"], row["nav_source_url"])
            if isinstance(value, str) and value
        ),
        axis=1,
    )
    merged["currency"] = "CNY"
    merged["crawl_time"] = datetime.now()
    columns = [
        "trade_date", "etf_code", "exchange", "fund_name", "fund_share",
        "unit_nav", "accumulated_nav", "subscription_status", "redemption_status",
        "currency", "share_data_source", "nav_data_source", "share_source_url",
        "nav_source_url", "share_available_time", "nav_available_time", "data_source",
        "source_url", "available_time", "crawl_time",
    ]
    return merged[columns].sort_values(keys).reset_index(drop=True)


def collect_etf_fund(
    start_date: str,
    end_date: str,
    codes: list[str] | None = None,
    retries: int = 3,
    run_quality: bool = True,
) -> dict[str, Any]:
    etfs = _configured_etfs(codes)
    selected = set(etfs["code"])
    ingestion = IngestionRun(
        dataset_name="etf_fund_daily",
        data_source="AKShare/SSE/SZSE/Eastmoney",
        requested_start_date=pd.Timestamp(start_date).date(),
        requested_end_date=pd.Timestamp(end_date).date(),
    )
    try:
        share_frames: list[pd.DataFrame] = []
        sse_codes = set(etfs.loc[etfs["exchange"].eq("SSE"), "code"])
        if sse_codes:
            for index, trade_date in enumerate(_reference_trade_dates(start_date, end_date), 1):
                frame = _retry(fetch_sse_share_date, retries, trade_date, sse_codes)
                if not frame.empty:
                    share_frames.append(frame)
                if index % 100 == 0:
                    logger.info("SSE ETF shares: %s dates completed", index)
        szse_codes = set(etfs.loc[etfs["exchange"].eq("SZSE"), "code"])
        if szse_codes:
            prefer_akshare = True
            for range_start, range_end in _six_month_ranges(start_date, end_date):
                frame, used_akshare = _retry(
                    fetch_szse_share_window,
                    retries,
                    range_start,
                    range_end,
                    szse_codes,
                    prefer_akshare,
                )
                prefer_akshare = prefer_akshare and used_akshare
                if not frame.empty:
                    share_frames.append(frame)
        shares = (
            pd.concat(share_frames, ignore_index=True).drop_duplicates(
                ["trade_date", "etf_code"], keep="last"
            )
            if share_frames else pd.DataFrame()
        )

        nav_frames: list[pd.DataFrame] = []
        for item in etfs.to_dict(orient="records"):
            frame = _retry(
                fetch_nav_history, retries, item["code"], item["name"], start_date, end_date
            )
            if not frame.empty:
                nav_frames.append(frame)
        nav = (
            pd.concat(nav_frames, ignore_index=True).drop_duplicates(
                ["trade_date", "etf_code"], keep="last"
            )
            if nav_frames else pd.DataFrame()
        )
        combined = _merge_sources(shares, nav)
        ingestion.fetched_rows = len(combined)
        if combined.empty:
            raise RuntimeError("ETF share and NAV sources returned no rows")
        existing = read_sql(
            """
            SELECT trade_date, etf_code FROM etf_fund_daily
            WHERE trade_date BETWEEN :start_date AND :end_date
            """,
            {"start_date": start_date, "end_date": end_date},
        )
        old_keys = set(
            zip(existing["trade_date"].astype(str), existing["etf_code"].astype(str))
        ) if not existing.empty else set()
        new_keys = set(zip(combined["trade_date"].astype(str), combined["etf_code"]))
        ingestion.inserted_rows = len(new_keys - old_keys)
        ingestion.updated_rows = len(new_keys & old_keys)
        upsert_dataframe(combined, "etf_fund_daily", ["trade_date", "etf_code"])

        quality_metrics: list[dict[str, Any]] = []
        if run_quality:
            stored = read_sql(
                "SELECT * FROM etf_fund_daily WHERE trade_date BETWEEN :start_date AND :end_date",
                {"start_date": start_date, "end_date": end_date},
            )
            quality_metrics = evaluate_dataframe(stored, DATASETS["etf_fund_daily"])
            quality_metrics.extend(
                evaluate_etf_field_coverage(
                    stored, pd.Series(_reference_trade_dates(start_date, end_date)), selected
                )
            )
            persist_metrics("etf_fund_daily", quality_metrics, date.today())
        ingestion.finish("success")
        return {
            "rows": len(combined),
            "inserted_rows": ingestion.inserted_rows,
            "updated_rows": ingestion.updated_rows,
            "share_rows": len(shares),
            "nav_rows": len(nav),
            "quality_metrics": quality_metrics,
        }
    except Exception as exc:
        ingestion.error_rows += 1
        ingestion.finish("failed", f"{type(exc).__name__}: {exc}")
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="采集模型 2.0 ETF 份额与历史净值。")
    parser.add_argument("--start-date", default="2015-01-05")
    parser.add_argument("--end-date", default="2026-06-10")
    parser.add_argument("--code", action="append", help="仅采集指定配置 ETF，可重复。")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--skip-quality", action="store_true")
    args = parser.parse_args()
    result = collect_etf_fund(
        args.start_date,
        args.end_date,
        codes=args.code,
        retries=args.retries,
        run_quality=not args.skip_quality,
    )
    print(result)


if __name__ == "__main__":
    main()
