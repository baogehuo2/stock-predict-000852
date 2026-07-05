from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import inspect, text

from src.common.config import load_yaml, project_path
from src.common.db import get_engine, upsert_dataframe


@dataclass(frozen=True)
class DatasetSpec:
    table: str
    date_column: str
    unique_columns: tuple[str, ...]
    required_columns: tuple[str, ...]
    non_negative_columns: tuple[str, ...] = ()
    ohlc: bool = False
    count_group_column: str | None = None


DATASETS: dict[str, DatasetSpec] = {
    "market_index_daily": DatasetSpec(
        table="market_index_daily",
        date_column="trade_date",
        unique_columns=("trade_date", "index_code"),
        required_columns=("trade_date", "index_code", "data_source", "available_time"),
        non_negative_columns=("volume", "amount"),
        ohlc=True,
    ),
    "market_index_weekly": DatasetSpec(
        table="market_index_weekly",
        date_column="trade_date",
        unique_columns=("trade_date", "index_code"),
        required_columns=("trade_date", "index_code", "open", "high", "low", "close", "data_source", "available_time"),
        non_negative_columns=("volume", "amount"),
        ohlc=True,
    ),
    "market_index_monthly": DatasetSpec(
        table="market_index_monthly",
        date_column="trade_date",
        unique_columns=("trade_date", "index_code"),
        required_columns=("trade_date", "index_code", "open", "high", "low", "close", "data_source", "available_time"),
        non_negative_columns=("volume", "amount"),
        ohlc=True,
    ),
    "market_stock_daily_raw": DatasetSpec(
        table="market_stock_daily_raw",
        date_column="trade_date",
        unique_columns=("trade_date", "stock_code"),
        required_columns=("trade_date", "stock_code", "exchange", "data_source", "available_time"),
        non_negative_columns=("volume", "amount", "turnover_rate", "amplitude"),
        ohlc=True,
        count_group_column="stock_code",
    ),
    "margin_market_daily": DatasetSpec(
        table="margin_market_daily",
        date_column="trade_date",
        unique_columns=("trade_date", "exchange"),
        required_columns=("trade_date", "exchange", "data_source", "available_time"),
        non_negative_columns=(
            "financing_balance", "financing_buy_amount", "securities_lending_balance",
            "securities_lending_sell_volume", "securities_lending_remaining_volume",
            "margin_balance",
        ),
    ),
    "etf_fund_daily": DatasetSpec(
        table="etf_fund_daily",
        date_column="trade_date",
        unique_columns=("trade_date", "etf_code"),
        required_columns=("trade_date", "etf_code", "data_source", "available_time"),
        non_negative_columns=("fund_share", "unit_nav", "accumulated_nav"),
    ),
    "index_futures_contract_daily": DatasetSpec(
        table="index_futures_contract_daily",
        date_column="trade_date",
        unique_columns=("trade_date", "contract_code"),
        required_columns=(
            "trade_date", "product_code", "contract_code", "data_source", "available_time"
        ),
        non_negative_columns=("volume", "open_interest", "amount"),
        ohlc=True,
    ),
    "index_constituent_snapshot_raw": DatasetSpec(
        table="index_constituent_snapshot_raw",
        date_column="snapshot_date",
        unique_columns=("snapshot_id", "index_code", "stock_code"),
        required_columns=(
            "snapshot_id", "snapshot_date", "index_code", "stock_code",
            "data_source", "available_time",
        ),
        non_negative_columns=("weight",),
    ),
    "calendar_daily_raw": DatasetSpec(
        table="calendar_daily_raw",
        date_column="calendar_date",
        unique_columns=("calendar_date", "market"),
        required_columns=("calendar_date", "market", "is_trading_day", "is_exchange_closed", "source_type", "available_time"),
    ),
    "event_calendar_raw": DatasetSpec(
        table="event_calendar_raw",
        date_column="scheduled_time",
        unique_columns=("event_key", "scheduled_time"),
        required_columns=("event_key", "event_layer", "event_type", "event_name", "scheduled_time", "available_time", "data_source"),
    ),
    "macro_release_raw": DatasetSpec(
        table="macro_release_raw",
        date_column="period_date",
        unique_columns=("indicator_key", "period_date", "release_time", "revision_no"),
        required_columns=("indicator_key", "indicator_name", "country_region", "period_date", "release_time", "available_time", "data_source"),
    ),
    "macro_release_time_reference_raw": DatasetSpec(
        table="macro_release_time_reference_raw",
        date_column="period_date",
        unique_columns=("indicator_key", "period_date", "release_time", "source_name"),
        required_columns=(
            "indicator_key", "indicator_name", "country_region", "period_date",
            "release_time", "event_name", "source_type", "source_name",
            "confidence", "available_time", "data_source",
        ),
    ),
    "global_market_daily": DatasetSpec(
        table="global_market_daily",
        date_column="trade_date",
        unique_columns=("trade_date", "symbol"),
        required_columns=("trade_date", "symbol", "asset_class", "close", "data_source", "available_time"),
        non_negative_columns=("volume",),
        ohlc=True,
    ),
    "option_contract_daily_raw": DatasetSpec(
        table="option_contract_daily_raw",
        date_column="trade_date",
        unique_columns=("trade_date", "contract_code", "data_source"),
        required_columns=(
            "trade_date", "underlying_code", "underlying_name", "option_market",
            "contract_code", "data_source", "available_time",
        ),
        non_negative_columns=("volume", "amount", "open_interest"),
        ohlc=True,
    ),
    "option_qvix_daily_raw": DatasetSpec(
        table="option_qvix_daily_raw",
        date_column="trade_date",
        unique_columns=("trade_date", "qvix_code", "data_source"),
        required_columns=(
            "trade_date", "underlying_code", "underlying_name", "option_market",
            "qvix_code", "data_source", "available_time",
        ),
    ),
    "sentiment_guba_raw": DatasetSpec(
        table="sentiment_guba_raw",
        date_column="trade_date",
        unique_columns=("post_id",),
        required_columns=("post_id", "source", "bar_name", "publish_time", "trade_date", "title", "content", "url", "available_time", "raw_hash"),
        non_negative_columns=("read_count", "comment_count", "like_count"),
    ),
    "sentiment_guba_comment_raw": DatasetSpec(
        table="sentiment_guba_comment_raw",
        date_column="trade_date",
        unique_columns=("comment_id",),
        required_columns=("comment_id", "post_id", "source", "bar_name", "publish_time", "trade_date", "content", "url", "available_time", "raw_hash"),
        non_negative_columns=("like_count", "reply_count"),
    ),
    "news_raw": DatasetSpec(
        table="news_raw",
        date_column="trade_date",
        unique_columns=("news_id",),
        required_columns=("news_id", "source", "publish_time", "trade_date", "title", "content", "available_time", "raw_hash", "content_hash"),
    ),
    "sentiment_social_raw": DatasetSpec(
        table="sentiment_social_raw",
        date_column="trade_date",
        unique_columns=("source", "content_id"),
        required_columns=("content_id", "source", "publish_time", "trade_date", "available_time", "raw_hash"),
        non_negative_columns=("read_count", "comment_count", "like_count", "share_count"),
    ),
}


def _metric(name: str, value: float | int | None, status: str, details: dict[str, Any] | None = None,
            expected_min: float | None = None, expected_max: float | None = None) -> dict[str, Any]:
    return {
        "metric_name": name,
        "metric_value": value,
        "expected_min": expected_min,
        "expected_max": expected_max,
        "status": status,
        "details": details or {},
    }


def evaluate_dataframe(df: pd.DataFrame, spec: DatasetSpec) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    metrics.append(_metric("row_count", len(df), "pass" if len(df) else "warn", expected_min=1))
    missing_columns = [column for column in spec.required_columns if column not in df.columns]
    metrics.append(_metric("missing_required_columns", len(missing_columns), "fail" if missing_columns else "pass",
                           {"columns": missing_columns}, expected_max=0))
    if missing_columns or df.empty:
        return metrics

    dates = pd.to_datetime(df[spec.date_column], errors="coerce")
    invalid_dates = int(dates.isna().sum())
    metrics.append(_metric("invalid_date_rows", invalid_dates, "fail" if invalid_dates else "pass", expected_max=0))
    valid_dates = dates.dropna()
    metrics.append(_metric("min_date_ordinal", valid_dates.min().date().toordinal() if not valid_dates.empty else None,
                           "pass" if not valid_dates.empty else "fail",
                           {"date": valid_dates.min().date().isoformat() if not valid_dates.empty else None}))
    metrics.append(_metric("max_date_ordinal", valid_dates.max().date().toordinal() if not valid_dates.empty else None,
                           "pass" if not valid_dates.empty else "fail",
                           {"date": valid_dates.max().date().isoformat() if not valid_dates.empty else None}))

    duplicate_rows = int(df.duplicated(list(spec.unique_columns), keep=False).sum())
    metrics.append(_metric("duplicate_key_rows", duplicate_rows, "fail" if duplicate_rows else "pass", expected_max=0))
    for column in df.columns:
        if column == "id":
            continue
        null_rate = float(df[column].isna().mean())
        required = column in spec.required_columns
        status = "fail" if required and null_rate > 0 else "warn" if null_rate > 0 else "pass"
        metrics.append(_metric(f"null_rate__{column}", null_rate, status,
                               {"required": required}, expected_max=0 if required else None))
    for column in spec.non_negative_columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        count = int((values < 0).sum())
        metrics.append(_metric(f"negative_rows__{column}", count, "fail" if count else "pass", expected_max=0))
    if spec.ohlc and all(column in df.columns for column in ("open", "high", "low", "close")):
        numeric = df[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
        bad = (
            (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
            | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
        )
        count = int(bad.fillna(False).sum())
        metrics.append(_metric("ohlc_logic_error_rows", count, "fail" if count else "pass", expected_max=0))
    if spec.count_group_column:
        counts = df.assign(_date=dates).groupby("_date")[spec.count_group_column].nunique().sort_index()
        changes = counts.pct_change().abs()
        abrupt = changes[changes > 0.10]
        metrics.append(_metric("daily_count_change_over_10pct", len(abrupt), "warn" if len(abrupt) else "pass",
                               {"dates": {index.date().isoformat(): round(float(value), 6) for index, value in abrupt.items()}},
                               expected_max=0))
    return metrics


def evaluate_date_coverage(df: pd.DataFrame, spec: DatasetSpec,
                           reference_dates: pd.Series | None) -> list[dict[str, Any]]:
    if reference_dates is None or reference_dates.empty or df.empty:
        return [_metric("date_coverage_available", 0, "warn", {"reason": "缺少基准交易日"})]
    observed = set(pd.to_datetime(df[spec.date_column], errors="coerce").dropna().dt.date)
    reference = sorted(set(pd.to_datetime(reference_dates, errors="coerce").dropna().dt.date))
    if not reference:
        return [_metric("date_coverage_available", 0, "warn", {"reason": "基准交易日为空"})]
    metrics = [_metric("date_coverage_available", 1, "pass")]
    reference_by_year: dict[int, list[date]] = {}
    for item in reference:
        reference_by_year.setdefault(item.year, []).append(item)
    for year, dates in sorted(reference_by_year.items()):
        missing = [item for item in dates if item not in observed]
        rate = len(missing) / len(dates)
        metrics.append(_metric(
            f"missing_trade_date_rate__{year}", rate, "warn" if missing else "pass",
            {"missing_count": len(missing), "expected_count": len(dates),
             "missing_dates": [item.isoformat() for item in missing[:50]]}, expected_max=0,
        ))
    missing_flags = [item not in observed for item in reference]
    gaps: list[dict[str, Any]] = []
    start_index: int | None = None
    for index, missing in enumerate(missing_flags + [False]):
        if missing and start_index is None:
            start_index = index
        elif not missing and start_index is not None:
            gaps.append({
                "start": reference[start_index].isoformat(),
                "end": reference[index - 1].isoformat(),
                "trade_days": index - start_index,
            })
            start_index = None
    longest = max((gap["trade_days"] for gap in gaps), default=0)
    metrics.append(_metric("longest_continuous_gap_trade_days", longest,
                           "warn" if longest else "pass", {"gaps": gaps[:50]}, expected_max=0))
    return metrics


def evaluate_margin_relationships(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = {"financing_balance", "securities_lending_balance", "margin_balance", "exchange"}
    if df.empty or not columns.issubset(df.columns):
        return [_metric(
            "margin_relationship_check_available", 0, "warn",
            {"reason": "missing columns or empty data"},
        )]
    numeric = df[list(columns - {"exchange"})].apply(pd.to_numeric, errors="coerce")
    difference = (
        numeric["margin_balance"]
        - numeric["financing_balance"]
        - numeric["securities_lending_balance"]
    ).abs()
    tolerance = df["exchange"].map({"SSE": 1.0, "SZSE": 1_000_000.0}).fillna(1.0)
    bad = difference > tolerance
    return [
        _metric("margin_relationship_check_available", 1, "pass"),
        _metric(
            "margin_balance_equation_error_rows",
            int(bad.fillna(False).sum()),
            "fail" if bad.fillna(False).any() else "pass",
            {
                "max_abs_difference": float(difference.max()) if not difference.empty else None,
                "tolerance_sse_cny": 1.0,
                "tolerance_szse_cny": 1_000_000.0,
            },
            expected_max=0,
        ),
    ]


def evaluate_etf_field_coverage(
    df: pd.DataFrame, reference_dates: pd.Series | None, expected_codes: set[str]
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    if df.empty or reference_dates is None or reference_dates.empty:
        return [_metric("etf_field_coverage_available", 0, "warn")]
    reference = set(pd.to_datetime(reference_dates, errors="coerce").dropna().dt.date)
    metrics.append(_metric("etf_field_coverage_available", 1, "pass"))
    for code in sorted(expected_codes):
        fund = df[df["etf_code"].astype(str).eq(code)].copy()
        share_dates = set(
            pd.to_datetime(
                fund.loc[fund.get("fund_share", pd.Series(index=fund.index)).notna(), "trade_date"],
                errors="coerce",
            ).dropna().dt.date
        )
        lifecycle_start = min(share_dates) if share_dates else None
        for field in ("fund_share", "unit_nav", "accumulated_nav"):
            observed = fund.loc[fund[field].notna(), "trade_date"] if field in fund else pd.Series(dtype=object)
            observed_dates = set(pd.to_datetime(observed, errors="coerce").dropna().dt.date)
            if not observed_dates:
                metrics.append(_metric(
                    f"missing_trade_date_rate__{field}__{code}", 1.0, "fail",
                    {"observed_dates": 0}, expected_max=0,
                ))
                continue
            first_date = max(min(observed_dates), lifecycle_start) if lifecycle_start else min(observed_dates)
            expected = {item for item in reference if item >= first_date}
            missing = expected - observed_dates
            rate = len(missing) / len(expected) if expected else 0.0
            metrics.append(_metric(
                f"missing_trade_date_rate__{field}__{code}", rate,
                "warn" if rate > 0 else "pass",
                {
                    "first_observed_date": first_date.isoformat(),
                    "expected_dates": len(expected),
                    "observed_reference_dates": len(observed_dates & reference),
                    "missing_dates_sample": sorted(item.isoformat() for item in missing)[:50],
                },
                expected_max=0,
            ))
    if {"available_time", "share_available_time", "nav_available_time"}.issubset(df.columns):
        actual = pd.to_datetime(df["available_time"], errors="coerce")
        components = df[["share_available_time", "nav_available_time"]].apply(
            pd.to_datetime, errors="coerce"
        ).max(axis=1)
        bad = actual.ne(components) & components.notna()
        metrics.append(_metric(
            "available_time_not_latest_component_rows", int(bad.sum()),
            "fail" if bad.any() else "pass", expected_max=0,
        ))
    return metrics


def evaluate_futures_contracts(df: pd.DataFrame) -> list[dict[str, Any]]:
    needed = {
        "trade_date", "product_code", "contract_code", "expiry_date",
        "contract_multiplier", "settle", "volume", "amount",
        "raw_turnover_10k_cny",
    }
    if df.empty or not needed.issubset(df.columns):
        return [_metric("futures_contract_check_available", 0, "warn")]
    metrics = [_metric("futures_contract_check_available", 1, "pass")]
    valid_code = df["contract_code"].astype(str).str.fullmatch(r"(?:IF|IC|IM)\d{4}")
    metrics.append(_metric(
        "invalid_contract_code_rows", int((~valid_code).sum()),
        "fail" if (~valid_code).any() else "pass", expected_max=0,
    ))
    expected_multiplier = df["product_code"].map({"IF": 300.0, "IC": 200.0, "IM": 200.0})
    multiplier = pd.to_numeric(df["contract_multiplier"], errors="coerce")
    bad_multiplier = multiplier.ne(expected_multiplier)
    metrics.append(_metric(
        "contract_multiplier_error_rows", int(bad_multiplier.sum()),
        "fail" if bad_multiplier.any() else "pass", expected_max=0,
    ))
    amount = pd.to_numeric(df["amount"], errors="coerce")
    raw_turnover = pd.to_numeric(df["raw_turnover_10k_cny"], errors="coerce")
    unit_error = (amount - raw_turnover * 10_000.0).abs() > 1.0
    metrics.append(_metric(
        "amount_unit_conversion_error_rows", int(unit_error.fillna(False).sum()),
        "fail" if unit_error.fillna(False).any() else "pass", expected_max=0,
    ))
    trade_date = pd.to_datetime(df["trade_date"], errors="coerce")
    expiry_date = pd.to_datetime(df["expiry_date"], errors="coerce")
    after_expiry = trade_date > expiry_date
    metrics.append(_metric(
        "trade_after_expiry_rows", int(after_expiry.fillna(False).sum()),
        "fail" if after_expiry.fillna(False).any() else "pass", expected_max=0,
    ))
    daily_products = df.groupby(["trade_date", "product_code"])["contract_code"].nunique()
    too_few = daily_products[daily_products < 4]
    metrics.append(_metric(
        "product_days_with_fewer_than_four_contracts", len(too_few),
        "warn" if len(too_few) else "pass",
        {"rows": {f"{d}|{p}": int(v) for (d, p), v in too_few.head(50).items()}},
        expected_min=4,
    ))
    return metrics


def evaluate_futures_product_coverage(
    df: pd.DataFrame, reference_dates: pd.Series | None, product_code: str
) -> list[dict[str, Any]]:
    product = df[df["product_code"].astype(str).eq(product_code)].copy()
    if product.empty or reference_dates is None or reference_dates.empty:
        return [_metric(f"{product_code.lower()}__date_coverage_available", 0, "warn")]
    first_date = pd.to_datetime(product["trade_date"], errors="coerce").min()
    clipped_reference = pd.to_datetime(reference_dates, errors="coerce")
    clipped_reference = clipped_reference[clipped_reference >= first_date]
    metrics = evaluate_date_coverage(
        product, DATASETS["index_futures_contract_daily"], clipped_reference
    )
    output: list[dict[str, Any]] = []
    for metric in metrics:
        item = dict(metric)
        item["metric_name"] = f"{product_code.lower()}__{metric['metric_name']}"
        details = dict(item.get("details") or {})
        details["first_observed_date"] = first_date.date().isoformat()
        item["details"] = details
        output.append(item)
    return output


def evaluate_constituent_snapshot(df: pd.DataFrame) -> list[dict[str, Any]]:
    needed = {
        "snapshot_id", "stock_code", "weight", "weight_unit",
        "constituent_file_date", "weight_file_date", "raw_hash",
    }
    if df.empty or not needed.issubset(df.columns):
        return [_metric("constituent_snapshot_check_available", 0, "warn")]
    metrics = [_metric("constituent_snapshot_check_available", 1, "pass")]
    snapshots = df["snapshot_id"].nunique()
    metrics.append(_metric(
        "snapshot_id_count", snapshots, "fail" if snapshots != 1 else "pass",
        expected_min=1, expected_max=1,
    ))
    stock_count = df["stock_code"].nunique()
    metrics.append(_metric(
        "constituent_count", stock_count, "fail" if stock_count != 1000 else "pass",
        expected_min=1000, expected_max=1000,
    ))
    weight = pd.to_numeric(df["weight"], errors="coerce")
    weight_sum = float(weight.sum())
    metrics.append(_metric(
        "weight_sum", weight_sum,
        "pass" if 0.995 <= weight_sum <= 1.005 else "fail",
        expected_min=0.995, expected_max=1.005,
    ))
    invalid_weight = weight.isna() | weight.lt(0) | weight.gt(1)
    metrics.append(_metric(
        "invalid_weight_rows", int(invalid_weight.sum()),
        "fail" if invalid_weight.any() else "pass", expected_max=0,
    ))
    invalid_unit = ~df["weight_unit"].astype(str).eq("decimal")
    metrics.append(_metric(
        "invalid_weight_unit_rows", int(invalid_unit.sum()),
        "fail" if invalid_unit.any() else "pass", expected_max=0,
    ))
    return metrics


def evaluate_stock_daily_boundaries(
    df: pd.DataFrame, universe: pd.DataFrame
) -> list[dict[str, Any]]:
    needed = {"stock_code", "exchange", "trade_date", "pre_close", "pct_chg", "amplitude"}
    universe_needed = {"stock_code", "exchange", "listing_date"}
    if df.empty or not needed.issubset(df.columns) or not universe_needed.issubset(universe.columns):
        return [_metric(
            "stock_boundary_check_available", 0, "warn",
            {"reason": "missing columns or empty data"},
        )]
    listing = universe[list(universe_needed)].drop_duplicates(["stock_code", "exchange"])
    checked = df.merge(listing, on=["stock_code", "exchange"], how="left")
    trade_dates = pd.to_datetime(checked["trade_date"], errors="coerce")
    listing_dates = pd.to_datetime(checked["listing_date"], errors="coerce")
    boundary_null = checked[["pre_close", "pct_chg", "amplitude"]].isna().any(axis=1)
    listing_day = trade_dates.eq(listing_dates)
    unexpected = checked.loc[boundary_null & ~listing_day, ["stock_code", "exchange", "trade_date"]]
    expected = checked.loc[boundary_null & listing_day, ["stock_code", "exchange", "trade_date"]]
    return [
        _metric("stock_boundary_check_available", 1, "pass"),
        _metric(
            "listing_day_boundary_null_rows", len(expected), "pass",
            {"rows": expected.head(50).astype(str).to_dict(orient="records")},
        ),
        _metric(
            "unexpected_boundary_null_rows", len(unexpected),
            "fail" if len(unexpected) else "pass",
            {"rows": unexpected.head(50).astype(str).to_dict(orient="records")},
            expected_max=0,
        ),
    ]


def load_dataset(spec: DatasetSpec, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    engine = get_engine()
    if not inspect(engine).has_table(spec.table):
        raise RuntimeError(f"数据表不存在: {spec.table}，请先执行migration")
    conditions: list[str] = []
    params: dict[str, Any] = {}
    if start_date:
        conditions.append(f"`{spec.date_column}` >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append(f"`{spec.date_column}` <= :end_date")
        params["end_date"] = end_date
    where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT * FROM `{spec.table}`{where}"), params)
        return pd.DataFrame(result.fetchall(), columns=result.keys())


def load_reference_dates(start_date: str | None, end_date: str | None) -> pd.Series | None:
    engine = get_engine()
    if not inspect(engine).has_table("market_index_daily"):
        return None
    conditions = ["index_code = '000852'"]
    params: dict[str, Any] = {}
    if start_date:
        conditions.append("trade_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("trade_date <= :end_date")
        params["end_date"] = end_date
    with engine.connect() as conn:
        result = conn.execute(
            text(f"SELECT DISTINCT trade_date FROM market_index_daily WHERE {' AND '.join(conditions)}"),
            params,
        )
        frame = pd.DataFrame(result.fetchall(), columns=result.keys())
    return frame["trade_date"] if "trade_date" in frame else None


def persist_metrics(dataset_name: str, metrics: list[dict[str, Any]], check_date: date) -> None:
    rows = pd.DataFrame(
        [
            {
                "check_date": check_date,
                "dataset_name": dataset_name,
                "metric_name": metric["metric_name"],
                "metric_value": metric["metric_value"],
                "expected_min": metric["expected_min"],
                "expected_max": metric["expected_max"],
                "status": metric["status"],
                "details": json.dumps(metric["details"], ensure_ascii=False),
            }
            for metric in metrics
        ]
    )
    upsert_dataframe(rows, "data_quality_daily", ["check_date", "dataset_name", "metric_name"])


def run_quality_check(dataset_name: str, start_date: str | None = None, end_date: str | None = None,
                      persist: bool = True) -> dict[str, Any]:
    if dataset_name not in DATASETS:
        raise ValueError(f"未知数据集: {dataset_name}")
    spec = DATASETS[dataset_name]
    df = load_dataset(spec, start_date, end_date)
    metrics = evaluate_dataframe(df, spec)
    reference_dates = load_reference_dates(start_date, end_date)
    if dataset_name == "index_constituent_snapshot_raw":
        metrics.append(_metric(
            "date_coverage_not_applicable", 1, "pass",
            {"reason": "当前成分快照不是日频行情，按 snapshot_id 和成员数量验收"},
        ))
    elif dataset_name in {"sentiment_guba_raw", "sentiment_guba_comment_raw", "news_raw", "sentiment_social_raw"}:
        metrics.append(_metric(
            "date_coverage_not_applicable", 1, "pass",
            {"reason": "文本与社区原始数据不要求按每个A股交易日强制覆盖，覆盖连续性单独看月度和来源统计。"},
        ))
    else:
        metrics.extend(evaluate_date_coverage(df, spec, reference_dates))
    if dataset_name == "margin_market_daily":
        metrics.extend(evaluate_margin_relationships(df))
        if "exchange" in df.columns:
            for exchange in ("SSE", "SZSE"):
                exchange_metrics = evaluate_date_coverage(
                    df[df["exchange"] == exchange], spec, reference_dates
                )
                for metric in exchange_metrics:
                    metric = dict(metric)
                    metric["metric_name"] = f"{exchange.lower()}__{metric['metric_name']}"
                    metrics.append(metric)
    if dataset_name == "etf_fund_daily":
        symbols = load_yaml(project_path("config", "symbols.yaml"))
        expected_codes = {
            str(item["code"]).zfill(6) for item in symbols.get("etfs", [])
        }
        metrics.extend(evaluate_etf_field_coverage(df, reference_dates, expected_codes))
    if dataset_name == "index_futures_contract_daily":
        metrics.extend(evaluate_futures_contracts(df))
        if "product_code" in df.columns and reference_dates is not None:
            for product in ("IF", "IC", "IM"):
                metrics.extend(evaluate_futures_product_coverage(df, reference_dates, product))
    if dataset_name == "index_constituent_snapshot_raw":
        if "snapshot_id" in df.columns and not df.empty:
            latest_id = df.sort_values(["snapshot_date", "crawl_time"])["snapshot_id"].iloc[-1]
            metrics.extend(evaluate_constituent_snapshot(df[df["snapshot_id"] == latest_id]))
    if persist:
        persist_metrics(dataset_name, metrics, date.today())
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "start_date": start_date,
        "end_date": end_date,
        "metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="检查模型2.0采集表的数据质量。")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--no-persist", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_quality_check(
        args.dataset, args.start_date, args.end_date, persist=not args.no_persist
    )
    output = Path(args.output) if args.output else project_path(
        "data", "reports", f"quality_{args.dataset}.json"
    )
    if not output.is_absolute():
        output = project_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for metric in report["metrics"]:
        print(f"{metric['status']} | {metric['metric_name']} | {metric['metric_value']}")
    print(output)


if __name__ == "__main__":
    main()
