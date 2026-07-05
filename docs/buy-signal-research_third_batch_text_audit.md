# 第三批文本与情绪原始数据审计说明

本文档面向 `feature/buy-signal-research`，说明模型 2.0 第三批“情绪原始数据”的当前数据库状态。

第三批只覆盖原始帖子、评论、新闻和社区文本元数据，不包含情绪打分、事件方向判断、特征工程、模型训练或 LLM 抽取。

## 1. 本次完成内容

- 新增 migration：`sql/migrations/015_third_batch_text_raw_metadata.sql`
- 为旧表补齐模型 2.0 需要的基础元数据字段：
  - `available_time`
  - `raw_hash`
  - `content_hash`
  - `author_id_hash`
  - `news_raw` 的 `language / country_region / source_type / first_seen_time / last_seen_time / canonical_url / is_reprint / event_layer_hint / publisher_country / document_type / official_source / source_priority`
- 新建占位表：`sentiment_social_raw`
- 新增质量检查数据集：
  - `sentiment_guba_raw`
  - `sentiment_guba_comment_raw`
  - `news_raw`
  - `sentiment_social_raw`
- 新增审计脚本：
  - `python -m src.quality.audit_third_batch_text`
- 修正股吧日常采集器：
  - 默认进入详情页获取真实 `publish_time`
  - 新采集行写入 `available_time / raw_hash / content_hash`
- 新增历史修复脚本：
  - `python -m src.collectors.repair_guba_publish_time`

## 2. 当前数据状态

### 2.1 `sentiment_guba_raw`

用途：东方财富股吧帖子原始数据。

当前审计结果：

| 指标 | 值 |
| --- | ---: |
| 行数 | 2573 |
| 发布时间范围 | 2025-08-07 10:24:58 至 2026-06-17 17:34:56 |
| trade_date 范围 | 2025-08-07 至 2026-06-17 |
| 覆盖自然日数 | 223 |
| 重复 `post_id` | 0 |
| `available_time` 缺失率 | 0 |
| `raw_hash` 缺失率 | 0 |
| `publish_time` 缺失率 | 35.64% |

说明：

- 本次尝试用详情页补齐历史缺失发布时间。
- 共检查缺失发布时间旧记录 1495 条，其中补回 577 条，剩余 917 条无法补回。
- 剩余缺失主要因为东方财富详情页返回“身份核实”页面，不能稳定取得真实发布时间。
- 不允许用 `crawl_time` 冒充 `publish_time`。

建模使用约束：

- `publish_time IS NOT NULL` 的记录可以进入日频情绪/热度原始输入。
- `publish_time IS NULL` 的旧浅采样记录只能用于源数据审计，不建议进入日频模型。

### 2.2 `sentiment_guba_comment_raw`

用途：东方财富股吧评论原始数据。

当前审计结果：

| 指标 | 值 |
| --- | ---: |
| 行数 | 926 |
| 发布时间范围 | 2025-07-30 13:04:31 至 2026-05-18 15:10:51 |
| trade_date 范围 | 2025-07-30 至 2026-05-18 |
| 覆盖自然日数 | 167 |
| 重复 `comment_id` | 0 |
| `publish_time` 缺失率 | 0 |
| `available_time` 缺失率 | 0 |
| `raw_hash` 缺失率 | 0 |

风险：

- `like_count` 和 `reply_count` 缺失较高，分别约 78.83% 和 81.21%。
- 评论量目前较小，只能作为补充观察源，不宜单独作为强信号。

### 2.3 `news_raw`

用途：财经新闻、CCTV 新闻联播、经济日历等文章级原始文本。

当前审计结果：

| 指标 | 值 |
| --- | ---: |
| 行数 | 131575 |
| 发布时间范围 | 2016-01-16 至 2026-06-11 |
| trade_date 范围 | 2016-01-16 至 2026-06-11 |
| 覆盖自然日数 | 3786 |
| 重复 `news_id` | 0 |
| `publish_time` 缺失率 | 0 |
| `available_time` 缺失率 | 0 |
| `raw_hash` 缺失率 | 0 |
| `content_hash` 缺失率 | 0 |

来源分布：

| 来源 | 行数 | 范围 |
| --- | ---: | --- |
| 百度股市通经济日历 | 75781 | 2023-01-02 至 2026-06-11 |
| CCTV新闻联播 | 55651 | 2016-01-16 至 2026-06-10 |
| 新浪财经 | 78 | 2026-05-21 |
| 东方财富财经新闻 | 44 | 2026-05-21 |
| 财联社网页 | 21 | 2026-05-21 |

建模使用约束：

- `news_raw` 可以作为第三批里当前最完整的文本原始数据源。
- `百度股市通经济日历` 更接近日历/公告线索，不等同于权威 actual 值来源。
- 当前 `event_layer_hint` 默认填 `unknown`，不能当作已经分类完成的事件层级。

### 2.4 `sentiment_social_raw`

用途：雪球或其他社区原始文本的统一表。

当前状态：

| 指标 | 值 |
| --- | ---: |
| 表是否存在 | 是 |
| 行数 | 0 |

说明：

- 目前只完成 DDL，占位等待稳定合规来源确认。
- 未采集雪球或其他社区数据。
- 不应在建模分支假设该表已有可用数据。

## 3. 质量报告文件

已生成或更新：

- `data/reports/quality_sentiment_guba_raw.json`
- `data/reports/quality_sentiment_guba_comment_raw.json`
- `data/reports/quality_news_raw.json`
- `data/reports/quality_sentiment_social_raw.json`
- `data/reports/third_batch_text_audit.json`

## 4. 当前结论

第三批目前完成的是“旧文本/情绪数据审计与 2.0 元数据补齐”，不是完整采集完成。

可以交给建模侧试用的数据：

- `news_raw`
- `sentiment_guba_comment_raw`
- `sentiment_guba_raw` 中 `publish_time IS NOT NULL` 的记录

暂不建议用于建模的数据：

- `sentiment_guba_raw.publish_time IS NULL` 的旧浅采样记录
- `sentiment_social_raw`，当前为空

## 5. 下一步建议

如果继续推进第三批，应按以下顺序：

1. 只用详情页可稳定返回发布时间的股吧增量采集，不再扩大无发布时间旧数据。
2. 补新闻源的稳定日常增量入口，优先保存文章原文和发布时间。
3. 对雪球/其他社区先做来源探测与合规确认，通过后再写采集器。
4. 不启动情绪分、不做 LLM 事件抽取、不写入模型特征表。
