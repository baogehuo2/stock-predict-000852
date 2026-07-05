from __future__ import annotations

import io
import unittest
from datetime import date

from openpyxl import Workbook

from src.collectors.collect_zz1000_history_probe import (
    _parse_regular_adjustment_text,
    find_cancelled_announcement,
    normalize_announcement,
    parse_adjustment_xlsx,
    resolve_effective_date,
)


def _announcement_payload(
    announcement_id: int,
    publish_date: str,
    title: str,
    content: str,
) -> dict[str, object]:
    return {
        "id": announcement_id,
        "publishDate": publish_date,
        "title": title,
        "content": content,
        "enclosureList": [],
    }


class AnnouncementMetadataTests(unittest.TestCase):
    def test_after_close_requires_trading_calendar_to_resolve(self) -> None:
        announcement = normalize_announcement(
            _announcement_payload(
                1,
                "2025-05-30",
                "关于中证1000指数定期调整结果的公告",
                "<p>于2025年6月13日收市后生效。</p>",
            )
        )
        self.assertEqual(announcement.implementation_after_close_date, date(2025, 6, 13))
        self.assertIsNone(resolve_effective_date(announcement))
        self.assertEqual(
            resolve_effective_date(
                announcement,
                [date(2025, 6, 13), date(2025, 6, 16), date(2025, 6, 17)],
            ),
            date(2025, 6, 16),
        )
        self.assertEqual(str(announcement.available_time), "2025-05-31 00:00:00")

    def test_split_digits_in_official_html_are_normalized(self) -> None:
        announcement = normalize_announcement(
            _announcement_payload(
                3,
                "2025-11-28",
                "关于中证1000指数定期调整结果的公告",
                "<p>于 202 5 年 12 月 12 日收市后生效。</p>",
            )
        )
        self.assertEqual(
            announcement.implementation_after_close_date,
            date(2025, 12, 12),
        )

    def test_direct_effective_date_does_not_shift(self) -> None:
        announcement = normalize_announcement(
            _announcement_payload(
                2,
                "2021-04-02",
                "关于调整中证1000等指数样本的公告",
                "<p>决定自2021年4月12日起调整样本。</p>",
            )
        )
        self.assertEqual(resolve_effective_date(announcement), date(2021, 4, 12))

    def test_cancellation_resolves_nearest_prior_announcement(self) -> None:
        original = normalize_announcement(
            _announcement_payload(
                10,
                "2021-04-02",
                "关于调整中证1000等指数样本的公告",
                "<p>自2021年4月12日起调整。</p>",
            )
        )
        cancellation = normalize_announcement(
            _announcement_payload(
                11,
                "2021-04-09",
                "关于不实施相关样本调整公告的通知",
                "<p>4月2日发布的“关于调整中证1000等指数样本的公告”将不再实施。</p>",
            )
        )
        self.assertTrue(cancellation.is_cancellation)
        self.assertEqual(find_cancelled_announcement(cancellation, [original]), 10)
        self.assertIsNone(resolve_effective_date(cancellation))


class AdjustmentWorkbookTests(unittest.TestCase):
    def test_parse_temporary_adjustment_xlsx(self) -> None:
        workbook = Workbook()
        remove = workbook.active
        remove.title = "换出"
        remove.append(["指数代码", "指数简称", "证券代码", "证券简称"])
        remove.append(["000852", "中证1000", "000546", "ST金圆"])
        add = workbook.create_sheet("换入")
        add.append(["指数代码", "指数简称", "证券代码", "证券简称"])
        add.append(["000852", "中证1000", "688209", "英集芯"])
        add.append(["000300", "沪深300", "600000", "浦发银行"])
        buffer = io.BytesIO()
        workbook.save(buffer)

        result = parse_adjustment_xlsx(buffer.getvalue())

        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["stock_code"]), {"000546", "688209"})
        self.assertEqual(set(result["adjustment_type"]), {"add", "remove"})

    def test_parse_cross_page_regular_adjustment_text(self) -> None:
        text = """
中证500指数样本调整名单：
000001 甲公司 000002 乙公司
中证1000指数样本调整名单：
调出名单 调入名单
证券代码 证券名称 证券代码 证券名称
000546 ST金圆 688209 英集芯
002088 鲁阳节能 002408 齐翔腾达
600169 ST 太重 600105 永鼎股份
中证A50 指数样本调整名单：
000002 万科A 000617 中油资本
"""
        result = _parse_regular_adjustment_text(text)
        self.assertEqual(len(result), 6)
        self.assertEqual(
            set(result["stock_code"]),
            {"000546", "688209", "002088", "002408", "600169", "600105"},
        )
        self.assertEqual(
            result.loc[result["stock_code"] == "600169", "stock_name"].iloc[0],
            "ST太重",
        )


if __name__ == "__main__":
    unittest.main()
