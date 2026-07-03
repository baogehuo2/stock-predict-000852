from __future__ import annotations

import unittest
from datetime import datetime

import pandas as pd

from src.collectors.collect_news import normalize_akshare_stock_news


class NewsCollectorTests(unittest.TestCase):
    def test_normalize_akshare_stock_news_keeps_real_publish_time_in_window(self) -> None:
        raw = pd.DataFrame(
            [
                {
                    "新闻标题": "流动性改善推动中证1000ETF成交活跃",
                    "新闻内容": "中证1000ETF资金流入，市场流动性改善。",
                    "发布时间": "2026-07-04 07:30:00",
                    "文章来源": "证券时报网",
                    "新闻链接": "https://example.com/a",
                },
                {
                    "新闻标题": "前一日新闻不应进入今日早盘窗口",
                    "新闻内容": "中证1000ETF",
                    "发布时间": "2026-07-03 23:30:00",
                    "文章来源": "证券时报网",
                    "新闻链接": "https://example.com/b",
                },
            ]
        )
        result = normalize_akshare_stock_news(
            raw,
            "159845",
            {"liquidity": ["流动性"], "index_style": ["中证1000"]},
            datetime(2026, 7, 4, 0, 0),
            datetime(2026, 7, 4, 8, 0),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["publish_time"], datetime(2026, 7, 4, 7, 30))
        self.assertEqual(result[0]["trade_date"], datetime(2026, 7, 4, 7, 30).date())
        self.assertIn("AKShare东方财富个股新闻:159845", result[0]["source"])


if __name__ == "__main__":
    unittest.main()
