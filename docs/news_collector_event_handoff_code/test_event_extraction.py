from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.llm.extract_event import _select_news_candidates


class EventExtractionCandidateTests(unittest.TestCase):
    def test_priority_group_news_is_not_pushed_out_by_macro_calendar(self) -> None:
        rows = []
        for idx in range(25):
            rows.append(
                {
                    "id": 1000 + idx,
                    "news_id": f"macro-{idx}",
                    "trade_date": "2024-09-24",
                    "publish_time": "2024-09-24 00:00:00",
                    "title": f"海外宏观日历 {idx}",
                    "content": "",
                    "matched_keywords": '["PMI"]',
                    "matched_groups": '["macro"]',
                }
            )
        rows.insert(
            0,
            {
                "id": 900,
                "news_id": "policy-news",
                "trade_date": "2024-09-24",
                "publish_time": "2024-09-24 00:00:00",
                "title": "多项政策发力 进一步支持经济稳增长",
                "content": "",
                "matched_keywords": '["流动性", "证监会", "货币政策", "逆回购"]',
                "matched_groups": '["liquidity", "policy_market"]',
            },
        )
        data = pd.DataFrame(rows)

        with patch("src.llm.extract_event.read_sql", return_value=data):
            selected = _select_news_candidates(
                start_date="2024-09-24",
                end_date="2024-09-24",
                limit_per_day=20,
                include_unmatched=False,
            )

        self.assertIn("policy-news", selected["news_id"].tolist())
        self.assertEqual(selected.iloc[0]["news_id"], "policy-news")
        self.assertEqual(len(selected), 20)


if __name__ == "__main__":
    unittest.main()
