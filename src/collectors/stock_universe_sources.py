from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import requests

from src.common.network import disable_env_proxies


BSE_INFO_API_URL = "https://www.bseinfo.net/nqxxController/nqxxCnzq.do"
BSE_INFO_PAGE_URL = "https://www.bseinfo.net/nq/listedcompany.html"


def _decode_bse_jsonp(text: str) -> list[dict[str, Any]]:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < start:
        raise ValueError("北交所接口未返回有效JSONP")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, list) or not payload:
        raise ValueError("北交所接口返回空JSONP")
    return payload


def _fetch_bse_official(timeout: int = 20, retries: int = 3) -> pd.DataFrame:
    request_data = {
        "page": "0",
        "typejb": "T",
        "xxfcbj[]": "2",
        "xxzqdm": "",
        "sortfield": "xxzqdm",
        "sorttype": "asc",
    }
    errors: list[str] = []
    for attempt in range(retries):
        try:
            session = requests.Session()
            session.trust_env = False
            headers = {"User-Agent": "Mozilla/5.0", "Referer": BSE_INFO_PAGE_URL}
            first_response = session.post(
                BSE_INFO_API_URL, data=request_data, headers=headers, timeout=timeout
            )
            first_response.raise_for_status()
            first_payload = _decode_bse_jsonp(first_response.text)[0]
            total_pages = int(first_payload["totalPages"])
            rows = list(first_payload.get("content") or [])
            for page in range(1, total_pages):
                request_data["page"] = str(page)
                response = session.post(
                    BSE_INFO_API_URL,
                    data=request_data,
                    headers=headers,
                    timeout=timeout,
                )
                response.raise_for_status()
                rows.extend((_decode_bse_jsonp(response.text)[0].get("content") or []))
            frame = pd.DataFrame(rows).rename(
                columns={"xxzqdm": "证券代码", "xxzqjc": "证券简称", "fxssrq": "上市日期"}
            )
            if frame.empty or not {"证券代码", "证券简称"}.issubset(frame.columns):
                raise ValueError("北交所官方接口返回空数据或字段缺失")
            frame["证券代码"] = frame["证券代码"].astype(str).str.zfill(6)
            frame["exchange"] = "BSE"
            frame = frame.drop_duplicates("证券代码").sort_values("证券代码").reset_index(drop=True)
            frame.attrs["data_source"] = "bseinfo:listed_company_official"
            frame.attrs["source_url"] = BSE_INFO_API_URL
            return frame[["证券代码", "证券简称", "上市日期", "exchange"]]
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        if attempt + 1 < retries:
            time.sleep(1 + attempt)
    raise RuntimeError("; ".join(errors))


def fetch_bse_stock_list() -> pd.DataFrame:
    disable_env_proxies()
    try:
        import akshare as ak

        frame = ak.stock_info_bj_name_code()
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            result = frame.rename(columns={"证券代码": "证券代码", "证券简称": "证券简称"}).copy()
            result["证券代码"] = result["证券代码"].astype(str).str.zfill(6)
            result["exchange"] = "BSE"
            result.attrs["data_source"] = "akshare:stock_info_bj_name_code"
            result.attrs["source_url"] = "https://www.bse.cn/nq/listedcompany.html"
            return result[["证券代码", "证券简称", "exchange"]]
    except Exception:
        pass
    return _fetch_bse_official()


def fetch_current_stock_universe() -> pd.DataFrame:
    disable_env_proxies()
    import akshare as ak

    frames: list[pd.DataFrame] = []

    sh_main = ak.stock_info_sh_name_code(symbol="主板A股").rename(
        columns={"证券代码": "stock_code", "证券简称": "stock_name", "上市日期": "listing_date"}
    )
    sh_main["exchange"] = "SSE"
    sh_main["data_source"] = "akshare:stock_info_sh_name_code"
    sh_main["source_url"] = "https://www.sse.com.cn/assortment/stock/list/share/"
    frames.append(sh_main)

    sh_star = ak.stock_info_sh_name_code(symbol="科创板").rename(
        columns={"证券代码": "stock_code", "证券简称": "stock_name", "上市日期": "listing_date"}
    )
    sh_star["exchange"] = "SSE"
    sh_star["data_source"] = "akshare:stock_info_sh_name_code"
    sh_star["source_url"] = "https://www.sse.com.cn/assortment/stock/list/share/"
    frames.append(sh_star)

    sz = ak.stock_info_sz_name_code(symbol="A股列表").rename(
        columns={"A股代码": "stock_code", "A股简称": "stock_name", "A股上市日期": "listing_date"}
    )
    sz["exchange"] = "SZSE"
    sz["data_source"] = "akshare:stock_info_sz_name_code"
    sz["source_url"] = "https://www.szse.cn/market/product/stock/list/"
    frames.append(sz)

    bj = fetch_bse_stock_list().rename(
        columns={"证券代码": "stock_code", "证券简称": "stock_name", "上市日期": "listing_date"}
    )
    bj["data_source"] = bj.attrs.get("data_source", "bseinfo:listed_company_official")
    bj["source_url"] = bj.attrs.get("source_url", BSE_INFO_PAGE_URL)
    frames.append(bj)

    universe = pd.concat(frames, ignore_index=True, sort=False)
    universe["stock_code"] = (
        universe["stock_code"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )
    universe["listing_date"] = pd.to_datetime(universe.get("listing_date"), errors="coerce").dt.date
    universe["security_status"] = "listed"
    universe["delisting_date"] = pd.NaT
    universe = universe[
        [
            "stock_code", "exchange", "stock_name", "listing_date", "security_status",
            "delisting_date", "data_source", "source_url",
        ]
    ]
    universe = universe.drop_duplicates(["exchange", "stock_code"]).sort_values(
        ["exchange", "stock_code"]
    )
    return universe.reset_index(drop=True)


def fetch_delisted_stock_universe(start_date: str = "2015-01-01") -> pd.DataFrame:
    disable_env_proxies()
    import akshare as ak

    sh = ak.stock_info_sh_delist(symbol="\u5168\u90e8").rename(
        columns={
            "\u516c\u53f8\u4ee3\u7801": "stock_code",
            "\u516c\u53f8\u7b80\u79f0": "stock_name",
            "\u4e0a\u5e02\u65e5\u671f": "listing_date",
            "\u6682\u505c\u4e0a\u5e02\u65e5\u671f": "delisting_date",
        }
    )
    sh = sh[sh["stock_code"].astype(str).str.match(r"^(600|601|603|605|688)\d{3}$")].copy()
    sh["exchange"] = "SSE"
    sh["data_source"] = "akshare:stock_info_sh_delist"
    sh["source_url"] = "https://www.sse.com.cn/assortment/stock/list/delisting/"

    sz = ak.stock_info_sz_delist(symbol="\u7ec8\u6b62\u4e0a\u5e02\u516c\u53f8").rename(
        columns={
            "\u8bc1\u5238\u4ee3\u7801": "stock_code",
            "\u8bc1\u5238\u7b80\u79f0": "stock_name",
            "\u4e0a\u5e02\u65e5\u671f": "listing_date",
            "\u7ec8\u6b62\u4e0a\u5e02\u65e5\u671f": "delisting_date",
        }
    )
    sz = sz[sz["stock_code"].astype(str).str.match(r"^(000|001|002|003|300|301)\d{3}$")].copy()
    sz["exchange"] = "SZSE"
    sz["data_source"] = "akshare:stock_info_sz_delist"
    sz["source_url"] = "https://www.szse.cn/market/stock/suspend/index.html"

    universe = pd.concat([sh, sz], ignore_index=True, sort=False)
    universe["stock_code"] = universe["stock_code"].astype(str).str.zfill(6)
    universe["listing_date"] = pd.to_datetime(universe["listing_date"], errors="coerce").dt.date
    universe["delisting_date"] = pd.to_datetime(
        universe["delisting_date"], errors="coerce"
    ).dt.date
    universe["security_status"] = "delisted"
    cutoff = pd.Timestamp(start_date).date()
    universe = universe[
        universe["delisting_date"].isna() | (universe["delisting_date"] >= cutoff)
    ]
    columns = [
        "stock_code", "exchange", "stock_name", "listing_date", "security_status",
        "delisting_date", "data_source", "source_url",
    ]
    return universe[columns].drop_duplicates(["exchange", "stock_code"]).reset_index(drop=True)


def fetch_collection_stock_universe(start_date: str = "2015-01-01") -> pd.DataFrame:
    current = fetch_current_stock_universe()
    delisted = fetch_delisted_stock_universe(start_date)
    universe = pd.concat([current, delisted], ignore_index=True, sort=False)
    universe["_priority"] = universe["security_status"].map({"listed": 0, "delisted": 1})
    universe = universe.sort_values(["exchange", "stock_code", "_priority"])
    universe = universe.drop_duplicates(["exchange", "stock_code"], keep="first")
    return universe.drop(columns="_priority").reset_index(drop=True)
