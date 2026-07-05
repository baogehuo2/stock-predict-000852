from __future__ import annotations

import unittest
from datetime import datetime

import pandas as pd

from unittest.mock import patch

from src.collectors.collect_zz1000_history_tushare import _month_windows, _normalize_history, collect_history_backfill


class TushareHistoryTests(unittest.TestCase):
    def test_normalize_history_builds_effective_ranges(self) -> None:
        raw = pd.DataFrame(
            [
                {"index_code": "000852.SH", "con_code": "1", "trade_date": "20250131", "weight": 0.25},
                {"index_code": "000852.SH", "con_code": "1", "trade_date": "20250228", "weight": 0.30},
                {"index_code": "000852.SH", "con_code": "2", "trade_date": "20250131", "weight": 0.10},
            ]
        )

        frame = _normalize_history(raw, datetime(2025, 3, 1, 12, 0))

        self.assertEqual(len(frame), 3)
        self.assertEqual(frame.iloc[0]["effective_date"].isoformat(), "2025-01-31")
        self.assertEqual(frame.iloc[0]["end_date"].isoformat(), "2025-02-27")
        self.assertEqual(frame.iloc[1]["effective_date"].isoformat(), "2025-02-28")
        self.assertIsNone(frame.iloc[1]["end_date"])
        self.assertEqual(frame.iloc[2]["stock_code"], "000002")

    def test_month_windows_splits_by_calendar_month(self) -> None:
        self.assertEqual(
            _month_windows(pd.Timestamp("2025-01-15").date(), pd.Timestamp("2025-03-10").date()),
            [("20250101", "20250131"), ("20250201", "20250228"), ("20250301", "20250310")],
        )

    def test_backfill_starts_after_last_effective_date(self) -> None:
        with patch("src.collectors.collect_zz1000_history_tushare._last_effective_date", return_value=pd.Timestamp("2025-01-31").date()), patch(
            "src.collectors.collect_zz1000_history_tushare.collect_history",
            return_value={"rows": 1},
        ) as mocked_collect:
            result = collect_history_backfill(end_date="2025-03-31")
        self.assertEqual(result, [{"rows": 1}, {"rows": 1}])
        self.assertEqual(mocked_collect.call_args_list[0].args[:2], ("2025-02-01", "2025-02-28"))
        self.assertEqual(mocked_collect.call_args_list[1].args[:2], ("2025-03-01", "2025-03-31"))


if __name__ == "__main__":
    unittest.main()
