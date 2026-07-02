from __future__ import annotations

import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from src.collectors.collect_etf_fund_v2 import _merge_sources, _six_month_ranges, normalize_nav, normalize_sse_shares, normalize_szse_shares
from src.collectors.collect_index_futures_contract_v2 import _planned_expiry, normalize_cffex_contracts
from src.collectors.collect_margin_market_v2 import _year_ranges, normalize_margin_sse, normalize_margin_szse, normalize_margin_szse_eastmoney, normalize_margin_szse_history
from src.collectors.collect_stock_daily_v2 import _effective_date_range, _resume_start, _source_symbol, normalize_eastmoney_history, normalize_sina_history
from src.collectors.collect_style_indices_v2 import _normalize_csindex
from src.collectors.collect_zz1000_snapshot_v2 import normalize_snapshot
from src.collectors.probe_sources_v2 import summarize_dataframe
from src.collectors.stock_universe_sources import _decode_bse_jsonp, fetch_collection_stock_universe
from src.common.migrations import discover_migrations, split_sql_statements
from src.quality.check_collection_data import (
    DatasetSpec,
    evaluate_constituent_snapshot,
    evaluate_dataframe,
    evaluate_date_coverage,
    evaluate_etf_field_coverage,
    evaluate_futures_contracts,
    evaluate_futures_product_coverage,
    evaluate_margin_relationships,
    evaluate_stock_daily_boundaries,
)


class MigrationTests(unittest.TestCase):
    def test_split_sql_statements_ignores_line_comments(self) -> None:
        sql = "-- comment\nCREATE TABLE a (id INT);\nINSERT INTO a VALUES (1);"
        self.assertEqual(split_sql_statements(sql), ["CREATE TABLE a (id INT)", "INSERT INTO a VALUES (1)"])

    def test_discover_migrations_is_sorted_and_checksummed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "002_second.sql").write_text("SELECT 2;", encoding="utf-8")
            (root / "001_first.sql").write_text("SELECT 1;", encoding="utf-8")
            migrations = discover_migrations(root)
            self.assertEqual([item.version for item in migrations], ["001_first", "002_second"])
            self.assertTrue(all(len(item.checksum) == 64 for item in migrations))


class SourceProbeTests(unittest.TestCase):
    def test_decode_bse_jsonp(self) -> None:
        payload = _decode_bse_jsonp('null([{"totalPages":1,"content":[]}])')
        self.assertEqual(payload[0]["totalPages"], 1)

    def test_summarize_dataframe_reports_dates_and_nulls(self) -> None:
        df = pd.DataFrame({"trade_date": ["2025-01-02", "2025-01-03"], "close": [100.0, None]})
        summary = summarize_dataframe(df)
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["min_date"], "2025-01-02")
        self.assertEqual(summary["max_date"], "2025-01-03")

    def test_csindex_pre_close_falls_back_to_pct_change(self) -> None:
        raw = pd.DataFrame(
            {
                "trade_date": ["2025-01-02"],
                "open": [100.0],
                "high": [102.0],
                "low": [99.0],
                "close": [101.0],
                "pct_chg": [1.0],
                "change": [1.0],
                "volume": [1000.0],
                "amount": [2.0],
            }
        )
        normalized = _normalize_csindex(raw, "932000", "中证2000")
        self.assertAlmostEqual(normalized.iloc[0]["pre_close"], 100.0)

    def test_normalize_sina_history_uses_buffer_for_pre_close(self) -> None:
        raw = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-02"],
                "open": [9.5, 10.0],
                "high": [10.0, 11.0],
                "low": [9.0, 9.8],
                "close": [10.0, 10.5],
                "volume": [100.0, 120.0],
                "amount": [1000.0, 1260.0],
                "turnover": [0.01, 0.012],
            }
        )
        normalized = normalize_sina_history(raw, "000001", "SZSE", "平安银行", "2025-01-02", "2025-01-02")
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized.iloc[0]["pre_close"], 10.0)

    def test_normalize_eastmoney_history_converts_units(self) -> None:
        raw = pd.DataFrame(
            {
                "日期": ["2025-01-01", "2025-01-02"],
                "开盘": [9.5, 10.0],
                "最高": [10.0, 11.0],
                "最低": [9.0, 9.8],
                "收盘": [10.0, 10.5],
                "成交量": [100.0, 120.0],
                "成交额": [1000.0, 1260.0],
                "换手率": [1.0, 1.2],
                "振幅": [10.0, 12.0],
                "涨跌幅": [0.0, 5.0],
            }
        )
        normalized = normalize_eastmoney_history(raw, "000001", "SZSE", "平安银行", "2025-01-02", "2025-01-02")
        self.assertEqual(normalized.iloc[0]["volume"], 12000.0)
        self.assertAlmostEqual(normalized.iloc[0]["turnover_rate"], 0.012)

    def test_stock_source_symbol_supports_three_exchanges(self) -> None:
        self.assertEqual(_source_symbol("600000", "SSE"), "sh600000")
        self.assertEqual(_source_symbol("000001", "SZSE"), "sz000001")
        self.assertEqual(_source_symbol("920000", "BSE"), "bj920000")

    def test_normalize_sse_etf_share_keeps_share_unit(self) -> None:
        raw = pd.DataFrame([{"基金代码": "510300", "基金简称": "300ETF", "统计日期": "2025-01-10", "基金份额": 123456789.0}])
        result = normalize_sse_shares(raw, {"510300"})
        self.assertEqual(result.iloc[0]["fund_share"], 123456789.0)

    def test_normalize_szse_etf_share_by_position(self) -> None:
        raw = pd.DataFrame([["2025-01-10", "159845", "1000ETF", "11,682,425,606"]])
        result = normalize_szse_shares(raw, {"159845"})
        self.assertEqual(result.iloc[0]["fund_share"], 11682425606.0)

    def test_normalize_nav_uses_named_api_fields(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "FSRQ": "2025-01-10",
                    "DWJZ": "2.2591",
                    "LJJZ": "0.9365",
                    "SGZT": "场内买入",
                    "SHZT": "场内卖出",
                }
            ]
        )
        result = normalize_nav(raw, "159845", "中证1000ETF华夏")
        self.assertEqual(str(result.iloc[0]["trade_date"]), "2025-01-10")
        self.assertAlmostEqual(result.iloc[0]["unit_nav"], 2.2591)

    def test_six_month_ranges_do_not_exceed_limit(self) -> None:
        self.assertEqual(
            _six_month_ranges("2025-01-01", "2026-01-10"),
            [
                ("2025-01-01", "2025-03-31"),
                ("2025-04-01", "2025-06-29"),
                ("2025-06-30", "2025-09-27"),
                ("2025-09-28", "2025-12-26"),
                ("2025-12-27", "2026-01-10"),
            ],
        )

    def test_merge_etf_sources_uses_latest_available_time(self) -> None:
        shares = normalize_szse_shares(pd.DataFrame([["2025-01-10", "159845", "1000ETF", "100"]]), {"159845"})
        nav = normalize_nav(
            pd.DataFrame([{"FSRQ": "2025-01-10", "DWJZ": "2", "LJJZ": "2", "SGZT": "open", "SHZT": "open"}]),
            "159845",
            "1000ETF",
        )
        nav["nav_available_time"] = pd.Timestamp("2025-01-12 00:00:00")
        result = _merge_sources(shares, nav)
        self.assertEqual(str(result.iloc[0]["available_time"]), "2025-01-12 00:00:00")

    def test_planned_expiry_is_third_friday(self) -> None:
        self.assertEqual(str(_planned_expiry("IM2606")), "2026-06-19")

    def test_normalize_cffex_contract_converts_turnover_to_cny(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "symbol": "IM2606",
                    "date": "20260610",
                    "open": 8190,
                    "high": 8214.8,
                    "low": 8053,
                    "close": 8162.2,
                    "volume": 154308,
                    "open_interest": 177936,
                    "turnover": 25101086.816,
                    "settle": 8147,
                    "pre_settle": 8237.2,
                    "variety": "IM",
                }
            ]
        )
        result = normalize_cffex_contracts(raw)
        self.assertEqual(result.iloc[0]["contract_multiplier"], 200.0)
        self.assertAlmostEqual(result.iloc[0]["amount"], 251010868160.0)

    def test_normalize_constituent_snapshot_converts_percent_weight(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "index_code": "000852.SH",
                    "con_code": f"{item:06d}",
                    "con_name": f"S{item}",
                    "trade_date": "20260612",
                    "weight": 0.25,
                }
                for item in range(1000)
            ]
        )
        result = normalize_snapshot(raw, date(2026, 6, 13), datetime(2026, 6, 13, 12, 0))
        self.assertEqual(len(result), 1000)
        self.assertEqual(result.iloc[0]["stock_code"], "000000")
        self.assertAlmostEqual(result.iloc[0]["weight"], 0.0025)

    def test_normalize_constituent_snapshot_allows_missing_con_name(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "index_code": "000852.SH",
                    "con_code": f"{item:06d}",
                    "trade_date": "20260612",
                    "weight": 0.25,
                }
                for item in range(1000)
            ]
        )
        result = normalize_snapshot(raw, date(2026, 6, 13), datetime(2026, 6, 13, 12, 0))
        self.assertEqual(len(result), 1000)
        self.assertEqual(result.iloc[0]["stock_code"], "000000")
        self.assertEqual(result.iloc[0]["stock_name"], "")

    def test_collection_universe_prefers_current_security_record(self) -> None:
        from unittest.mock import patch

        current = pd.DataFrame([{"stock_code": "600000", "exchange": "SSE", "security_status": "listed"}])
        delisted = pd.DataFrame([{"stock_code": "600000", "exchange": "SSE", "security_status": "delisted"}])
        with patch("src.collectors.stock_universe_sources.fetch_current_stock_universe", return_value=current), patch(
            "src.collectors.stock_universe_sources.fetch_delisted_stock_universe",
            return_value=delisted,
        ):
            result = fetch_collection_stock_universe("2015-01-01")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["security_status"], "listed")


class QualityTests(unittest.TestCase):
    def test_valid_dataframe_passes_core_checks(self) -> None:
        spec = DatasetSpec(
            table="sample",
            date_column="trade_date",
            unique_columns=("trade_date", "code"),
            required_columns=("trade_date", "code", "data_source", "available_time"),
            non_negative_columns=("volume",),
            ohlc=True,
            count_group_column="code",
        )
        df = pd.DataFrame(
            {
                "trade_date": ["2025-01-02", "2025-01-03"],
                "code": ["A", "A"],
                "data_source": ["test", "test"],
                "available_time": ["2025-01-02 15:30:00", "2025-01-03 15:30:00"],
                "open": [10, 11],
                "high": [12, 12],
                "low": [9, 10],
                "close": [11, 11.5],
                "volume": [100, 200],
            }
        )
        metrics = {item["metric_name"]: item for item in evaluate_dataframe(df, spec)}
        self.assertEqual(metrics["duplicate_key_rows"]["status"], "pass")
        self.assertEqual(metrics["negative_rows__volume"]["status"], "pass")

    def test_date_coverage_reports_year_and_continuous_gap(self) -> None:
        spec = DatasetSpec(
            table="sample",
            date_column="trade_date",
            unique_columns=("trade_date", "code"),
            required_columns=("trade_date", "code", "data_source", "available_time"),
        )
        df = pd.DataFrame({"trade_date": ["2025-01-02", "2025-01-06"], "code": ["A", "A"]})
        reference = pd.Series(["2025-01-02", "2025-01-03", "2025-01-06"])
        metrics = {item["metric_name"]: item for item in evaluate_date_coverage(df, spec, reference)}
        self.assertAlmostEqual(metrics["missing_trade_date_rate__2025"]["metric_value"], 1 / 3)
        self.assertEqual(metrics["longest_continuous_gap_trade_days"]["metric_value"], 1)

    def test_stock_boundary_nulls_are_allowed_only_on_listing_day(self) -> None:
        df = pd.DataFrame(
            {
                "stock_code": ["A", "A", "B"],
                "exchange": ["SSE", "SSE", "SZSE"],
                "trade_date": ["2025-01-02", "2025-01-03", "2025-01-03"],
                "pre_close": [None, 10.0, None],
                "pct_chg": [None, 0.01, None],
                "amplitude": [None, 0.02, None],
            }
        )
        universe = pd.DataFrame(
            {"stock_code": ["A", "B"], "exchange": ["SSE", "SZSE"], "listing_date": ["2025-01-02", "2025-01-02"]}
        )
        metrics = {item["metric_name"]: item for item in evaluate_stock_daily_boundaries(df, universe)}
        self.assertEqual(metrics["unexpected_boundary_null_rows"]["status"], "fail")

    def test_margin_relationship_check_uses_exchange_tolerance(self) -> None:
        df = pd.DataFrame(
            {
                "exchange": ["SSE", "SZSE"],
                "financing_balance": [100.0, 10_000_000_000.0],
                "securities_lending_balance": [5.0, 500_000_000.0],
                "margin_balance": [105.0, 10_500_500_000.0],
            }
        )
        metrics = {item["metric_name"]: item for item in evaluate_margin_relationships(df)}
        self.assertEqual(metrics["margin_balance_equation_error_rows"]["status"], "pass")

    def test_etf_field_coverage_detects_missing_dates(self) -> None:
        df = pd.DataFrame(
            {
                "trade_date": ["2025-01-02", "2025-01-06"],
                "etf_code": ["159845", "159845"],
                "fund_share": [100, 101],
                "unit_nav": [1, 1.1],
                "accumulated_nav": [1, 1.1],
            }
        )
        metrics = {
            item["metric_name"]: item
            for item in evaluate_etf_field_coverage(df, pd.Series(["2025-01-02", "2025-01-03", "2025-01-06"]), {"159845"})
        }
        self.assertAlmostEqual(metrics["missing_trade_date_rate__fund_share__159845"]["metric_value"], 1 / 3)

    def test_futures_quality_checks_multiplier_and_amount_unit(self) -> None:
        frame = normalize_cffex_contracts(
            pd.DataFrame(
                [
                    {
                        "symbol": "IF2606",
                        "date": "20260610",
                        "open": 4700,
                        "high": 4750,
                        "low": 4650,
                        "close": 4720,
                        "volume": 10,
                        "open_interest": 20,
                        "turnover": 1410,
                        "settle": 4700,
                        "pre_settle": 4710,
                        "variety": "IF",
                    }
                ]
            )
        )
        metrics = {item["metric_name"]: item for item in evaluate_futures_contracts(frame)}
        self.assertEqual(metrics["contract_multiplier_error_rows"]["status"], "pass")
        self.assertEqual(metrics["amount_unit_conversion_error_rows"]["status"], "pass")

    def test_futures_product_coverage_starts_at_listing(self) -> None:
        frame = pd.DataFrame({"trade_date": ["2022-07-22", "2022-07-25"], "product_code": ["IM", "IM"], "contract_code": ["IM2208", "IM2208"]})
        reference = pd.Series(["2022-07-21", "2022-07-22", "2022-07-25"])
        metrics = {item["metric_name"]: item for item in evaluate_futures_product_coverage(frame, reference, "IM")}
        self.assertEqual(metrics["im__missing_trade_date_rate__2022"]["metric_value"], 0)


if __name__ == "__main__":
    unittest.main()
