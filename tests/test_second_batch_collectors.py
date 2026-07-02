from __future__ import annotations

import unittest

import pandas as pd

from src.collectors.collect_calendar_daily_v2 import normalize_calendar
from src.collectors.collect_event_calendar_v2 import extract_english_dates
from src.collectors.collect_global_market_v2 import normalize_akshare_global_index
from src.collectors.collect_macro_forecast_calendar_v2 import _infer_period_date, normalize_macro_forecast
from src.collectors.collect_macro_release_time_reference_v2 import normalize_release_time_reference
from src.collectors.collect_macro_release_v2 import normalize_akshare_us_indicator, normalize_tushare_indicator
from src.collectors.collect_market_index_periodic_v25 import normalize_index_periodic


class SecondBatchCollectorTests(unittest.TestCase):
    def test_normalize_calendar_merges_trade_and_holiday(self) -> None:
        trade = pd.DataFrame(
            [
                {"cal_date": "20250101", "is_open": 0, "pretrade_date": "20241231"},
                {"cal_date": "20250102", "is_open": 1, "pretrade_date": "20241231"},
            ]
        )
        holidays = pd.DataFrame([{"calendar_date": pd.Timestamp("2025-01-01").date(), "is_holiday": 1, "holiday_name": "元旦"}])
        result = normalize_calendar(trade, holidays, "2025-01-01", "2025-01-02")
        self.assertEqual(result.iloc[0]["market"], "CN_STOCK")
        self.assertEqual(result.iloc[0]["is_exchange_closed"], 1)
        self.assertEqual(result.iloc[0]["holiday_name"], "元旦")
        self.assertEqual(result.iloc[1]["is_trading_day"], 1)

    def test_extract_english_dates_supports_ranges(self) -> None:
        text = "FOMC January 28-29, 2025 and 11 September 2025 meetings."
        result = extract_english_dates(text)
        self.assertIn(pd.Timestamp("2025-01-29").date(), result)
        self.assertIn(pd.Timestamp("2025-09-11").date(), result)

    def test_normalize_tushare_indicator_keeps_vintage_key(self) -> None:
        raw = pd.DataFrame([{"month": "202501", "nt_yoy": 0.5, "nt_val": 100.0}])
        item = {
            "key": "CN_CPI_YOY",
            "name": "中国CPI同比",
            "api_name": "cn_cpi",
            "period_field": "month",
            "value_field": "nt_yoy",
            "frequency": "monthly",
            "unit": "percent",
            "source_url": "https://tushare.pro/",
        }
        result = normalize_tushare_indicator(raw, item, "2025-01-01", "2025-12-31")
        self.assertEqual(result.iloc[0]["indicator_key"], "CN_CPI_YOY")
        self.assertEqual(str(result.iloc[0]["period_date"]), "2025-01-31")
        self.assertEqual(result.iloc[0]["actual_value"], 0.5)

    def test_normalize_akshare_us_indicator_infers_columns(self) -> None:
        raw = pd.DataFrame([{"日期": "2025-01-01", "今值": 2.1, "预测值": 2.0}])
        item = {
            "key": "US_CPI_MOM",
            "name": "美国CPI环比",
            "function": "macro_usa_cpi_monthly",
            "frequency": "monthly",
            "unit": "percent",
        }
        result = normalize_akshare_us_indicator(raw, item, "2025-01-01", "2025-12-31")
        self.assertEqual(result.iloc[0]["country_region"], "US")
        self.assertEqual(result.iloc[0]["actual_value"], 2.1)

    def test_normalize_akshare_global_index_maps_ohlc(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "日期": "2025-01-02",
                    "开盘": 100.0,
                    "最高": 102.0,
                    "最低": 99.0,
                    "收盘": 101.0,
                    "成交量": 12345,
                }
            ]
        )
        item = {"symbol": "SPX", "name": "S&P 500", "asset_class": "equity_index", "ak_symbol": "标普500", "currency": "USD"}
        result = normalize_akshare_global_index(raw, item, "2025-01-01", "2025-01-31")
        self.assertEqual(result.iloc[0]["symbol"], "SPX")
        self.assertEqual(result.iloc[0]["close"], 101.0)
        self.assertEqual(result.iloc[0]["data_source"], "akshare:index_us_stock_sina:标普500")

    def test_macro_forecast_calendar_maps_event(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "时间": "2026-06-17 20:30:00",
                    "地区": "美国",
                    "事件": "5月CPI环比",
                    "重要性": 4,
                    "今值": None,
                    "预期": 0.2,
                    "前值": 0.3,
                    "链接": "https://wallstreetcn.com/calendar/US_CPI",
                }
            ]
        )
        mappings = [{"key": "US_CPI_MOM", "country_region": "US", "include": ["美国", "CPI", "环比"]}]
        result = normalize_macro_forecast(raw, mappings, pd.Timestamp("2026-06-16 10:00:00").to_pydatetime())
        self.assertEqual(result.iloc[0]["indicator_key"], "US_CPI_MOM")
        self.assertEqual(str(result.iloc[0]["period_date"]), "2026-05-31")
        self.assertIsNone(result.iloc[0]["actual_value"])
        self.assertEqual(result.iloc[0]["forecast_value"], 0.2)
        self.assertEqual(result.iloc[0]["previous_value"], 0.3)

    def test_infer_period_date_uses_month_in_event_name(self) -> None:
        release_time = pd.Timestamp("2026-01-15 10:00:00").to_pydatetime()
        self.assertEqual(str(_infer_period_date("美国12月CPI环比", release_time, "monthly")), "2025-12-31")

    def test_normalize_release_time_reference_uses_region_plus_event(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "时间": "2026-07-15 10:00:00",
                    "地区": "中国",
                    "事件": "6月规模以上工业增加值同比",
                    "重要性": 3,
                    "今值": None,
                    "预期": None,
                    "前值": 5.8,
                    "链接": "https://wallstreetcn.com/calendar/CN_INDUSTRIAL",
                }
            ]
        )
        mappings = [
            {
                "key": "CN_INDUSTRIAL_PRODUCTION_YOY",
                "country_region": "CN",
                "include": ["中国", "规模以上工业增加值", "同比"],
            }
        ]
        source_cfg = {
            "source": "akshare:macro_info_ws",
            "source_type": "economic_calendar",
            "is_official": 0,
            "confidence": "calendar_scheduled",
        }
        result = normalize_release_time_reference(
            raw,
            mappings,
            source_cfg,
            pd.Timestamp("2026-06-16 10:00:00").to_pydatetime(),
        )
        self.assertEqual(result.iloc[0]["indicator_key"], "CN_INDUSTRIAL_PRODUCTION_YOY")
        self.assertEqual(str(result.iloc[0]["period_date"]), "2026-06-30")
        self.assertEqual(str(result.iloc[0]["release_time"]), "2026-07-15 10:00:00")
        self.assertEqual(result.iloc[0]["source_type"], "economic_calendar")

    def test_normalize_index_periodic_keeps_direct_weekly_bar(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "日期": "2025-01-10",
                    "开盘": 100.0,
                    "最高": 99.0,
                    "最低": 101.0,
                    "收盘": 102.0,
                    "涨跌幅": 2.0,
                    "成交量": 10000.0,
                    "成交额": 20000.0,
                }
            ]
        )
        item = {"code": "000852", "name": "中证1000"}
        result = normalize_index_periodic(raw, item, "weekly", "2025-01-01", "2025-01-31")
        self.assertEqual(result.iloc[0]["index_code"], "000852")
        self.assertEqual(str(result.iloc[0]["period_start_date"]), "2025-01-06")
        self.assertEqual(str(result.iloc[0]["period_end_date"]), "2025-01-10")
        self.assertEqual(result.iloc[0]["high"], 102.0)
        self.assertEqual(result.iloc[0]["low"], 99.0)
        self.assertAlmostEqual(result.iloc[0]["pre_close"], 100.0)


if __name__ == "__main__":
    unittest.main()
