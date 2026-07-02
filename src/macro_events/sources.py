from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote


@dataclass(frozen=True)
class OfficialSource:
    org: str
    home_url: str
    search_url_template: str
    enabled: bool = True

    def search_url(self, keyword: str) -> str:
        return self.search_url_template.format(keyword=quote(keyword))


OFFICIAL_SOURCES = [
    OfficialSource(
        org="中国政府网",
        home_url="https://www.gov.cn/",
        search_url_template="https://sousuo.www.gov.cn/sousuo/search.shtml?searchWord={keyword}",
    ),
    OfficialSource(
        org="新华社",
        home_url="https://www.news.cn/",
        search_url_template="https://so.news.cn/#search/0/{keyword}/1/",
    ),
    OfficialSource(
        org="人民网",
        home_url="http://www.people.com.cn/",
        search_url_template="http://search.people.cn/s/?keyword={keyword}",
    ),
    OfficialSource(
        org="全国人大网",
        home_url="http://www.npc.gov.cn/",
        search_url_template="http://search.npc.gov.cn/search?keyword={keyword}",
        enabled=False,
    ),
    OfficialSource(
        org="中国政协网",
        home_url="http://www.cppcc.gov.cn/",
        search_url_template="http://www.cppcc.gov.cn/zxww/newcppcc/search/index.shtml?keyword={keyword}",
        enabled=False,
    ),
    OfficialSource(
        org="共产党员网",
        home_url="https://www.12371.cn/",
        search_url_template="https://so.12371.cn/dangjian.htm?t=2&keyword={keyword}",
    ),
]


EVENT_KEYWORDS = {
    "TWO_SESSIONS": ["全国人大会议", "全国政协会议", "两会 开幕", "两会 闭幕"],
    "POLITBURO_MEETING": [
        "中共中央政治局会议 分析研究当前经济形势",
        "政治局会议 经济工作",
        "政治局会议 房地产 资本市场",
    ],
    "CEWC": ["中央经济工作会议"],
    "CPC_PLENUM": ["中央全会 公报", "三中全会 公报", "四中全会 公报"],
}


ECONOMIC_POLITBURO_KEYWORDS = [
    "经济形势",
    "经济工作",
    "政府工作报告",
    "资本市场",
    "房地产",
    "宏观政策",
    "财政政策",
    "货币政策",
    "稳增长",
    "防风险",
]
