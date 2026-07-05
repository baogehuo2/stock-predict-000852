# 第四批重大事件原始文本采集说明

本文档面向 `feature/buy-signal-research`，说明模型 2.0 第四批“三类重大事件原始数据”的当前采集状态。

第四批只负责采集新闻、公告、日历和官方声明原文及元数据，不负责事件方向、影响分数、LLM 分类、事件聚类或特征构造。

## 1. 表与字段

第四批继续写入 `news_raw`，不新建结构化事件表。

关键字段：

- `news_id`
- `source`
- `publish_time`
- `trade_date`
- `title`
- `content`
- `url`
- `matched_keywords`
- `matched_groups`
- `language`
- `country_region`
- `source_type`
- `first_seen_time`
- `last_seen_time`
- `canonical_url`
- `content_hash`
- `event_layer_hint`
- `publisher_country`
- `document_type`
- `official_source`
- `source_priority`
- `available_time`
- `raw_hash`

其中 `event_layer_hint` 只是根据来源和关键词做粗分流，可取：

- `cn`
- `geo`
- `macro`
- `unknown`

它不是最终事件分类。

## 2. 采集器

采集器：

```powershell
python -m src.collectors.collect_event_raw_v4 --max-items-per-source 5
```

可按来源单独运行：

```powershell
python -m src.collectors.collect_event_raw_v4 --source pbc_news --max-items-per-source 5
python -m src.collectors.collect_event_raw_v4 --source mfa_press --max-items-per-source 5
python -m src.collectors.collect_event_raw_v4 --source fed_press --max-items-per-source 5
```

主入口已接入：

```powershell
python main_daily_run.py --step collect_event_raw_v4
```

历史回补专用 adapter：

```powershell
python -m src.collectors.collect_event_history_v4 --start-date 2025-01-01 --end-date 2026-06-18 --source fed_fomc --max-items-per-source 50
python -m src.collectors.collect_event_history_v4 --start-date 2025-01-01 --end-date 2026-06-18 --source mfa_press --max-items-per-source 50
python -m src.collectors.collect_event_history_v4 --start-date 2025-01-01 --end-date 2026-06-18 --source pbc_news --max-items-per-source 50
python -m src.collectors.collect_event_history_v4 --start-date 2025-01-01 --end-date 2026-06-18 --source ecb_monetary --max-items-per-source 50
python -m src.collectors.collect_event_history_v4 --start-date 2025-01-01 --end-date 2026-06-18 --source boj_monetary --max-items-per-source 50
```

历史 adapter 当前支持：

- `fed_fomc`
- `mfa_press`
- `pbc_news`
- `ecb_monetary`
- `boj_monetary`

## 3. 当前可用来源

当前已经烟测并写入的来源：

| 来源 | 层级 | 行数 | 时间范围 |
| --- | --- | ---: | --- |
| 中国人民银行 | `cn` | 4 | 2026-06-10 至 2026-06-17 |
| 外交部 | `geo` | 5 | 2026-06-11 至 2026-06-17 |
| Federal Reserve | `macro` | 5 | 2026-06-17 |
| European Central Bank | `macro` | 1 | 2025-03-13 |
| Bank of Japan | `macro` | 1 | 2026-06-17 |

当前第四批合计：

| event_layer_hint | 行数 | 时间范围 |
| --- | ---: | --- |
| `cn` | 4 | 2026-06-10 至 2026-06-17 |
| `geo` | 5 | 2026-06-11 至 2026-06-17 |
| `macro` | 7 | 2025-03-13 至 2026-06-17 |

## 3.1 历史 adapter 试跑结果

试跑区间：2025-01-01 至 2026-06-18。

| source adapter | 层级 | 当前有效入库 | 时间范围 | 结论 |
| --- | --- | ---: | --- | --- |
| `fed_fomc` | `macro` | 37 | 2025-01-29 至 2026-06-17 | 可继续回补，FOMC statement 入口稳定 |
| `mfa_press` | `geo` | 20 | 2026-05-21 至 2026-06-17 | 当前页可采，深历史需要分页/旧页规律 |
| `pbc_news` | `cn` | 4 | 2026-06-10 至 2026-06-17 | 当前入口偏近端，历史页需要继续适配 |
| `ecb_monetary` | `macro` | 3 | 2025-03-13 至 2025-04-04 | 可采但过滤偏窄，需要继续对 ECB 决议页结构专项适配 |
| `boj_monetary` | `macro` | 1 | 2026-06-18 | 当前入口只拿到索引样本，仍需专项适配 |

注意：

- `fed_fomc` 已修正 URL 文件名日期解析，例如 `monetary20250129a.htm` 会写入 `publish_time = 2025-01-29`。
- 对 URL 无法提前判断日期的来源，采集器会在详情页解析 `publish_time` 后再次做区间过滤。
- 区间外样本已按 URL 域名清理，不用中文 source 精确匹配清理，避免控制台编码影响。

## 4. 已跳过或暂缓来源

以下来源当前未纳入正式结果：

- 中国政府网政策页：列表页当前可解析链接多为导航/客户端入口，已从本次样本中清理。
- 证监会：当前列表页容易抓到“政府网站年度报表”等非事件页，已清理，后续需要更精确栏目入口。
- 商务部：当前入口偏专题导航，已清理，后续需要换成新闻发布会或政策发布的稳定列表。

## 5. 质量检查

已运行：

```powershell
python -m src.quality.check_collection_data --dataset news_raw --no-persist
```

结果：

- 必填字段缺失：0
- 重复 `news_id`：0
- `publish_time` 缺失：0
- `available_time` 缺失：0
- `raw_hash` 缺失：0
- `content_hash` 缺失：0

## 6. 建模使用约束

建模分支只能按以下方式使用：

```sql
WHERE available_time <= :signal_time
```

注意：

- 第四批当前只是原始文本样本和采集通道验证，不是完整历史库。
- 不要把 `event_layer_hint` 当作最终分类。
- 不要把 `matched_keywords` 当作事件方向。
- 不要在采集分支生成事件分数、情绪分、风险升级分或宏观冲击分。

## 7. 下一步

后续如果继续第四批，应按以下顺序扩展：

1. 为中国政府网、证监会、商务部找到稳定正文列表入口。
2. 增加历史分页采集，不直接大规模全站爬取。
3. 对 Fed / ECB / BOJ 的公告标题和发布时间做更精确解析。
4. 再补联合国、海外政府和国际组织来源。
