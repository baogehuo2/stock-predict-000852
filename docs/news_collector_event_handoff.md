# 新闻采集与事件抽取防遗漏交接说明

日期：2026-07-01

来源 worktree：`D:\python workbench\stock_predict\000852_bottom`

目标 collector worktree：`D:\python workbench\stock_predict\data-collectors`

## 目标

把本次在 bottom 分支里修正的新闻采集、事件抽取防遗漏逻辑交接给 collector 分支，并说明 `zz1000_botumn` 到 `zz1000_predict` 的新闻相关数据同步情况。

本文件只放在 collector 分支的 `docs` 下，未直接修改 collector 分支的 `src` 运行代码。可迁移代码副本在：

```text
docs/news_collector_event_handoff_code/
```

## 背景问题

之前发现 `2024-09-24` 有重要政策新闻已经进入 `news_raw`，但没有进入 `event_daily`。

直接原因不是 AkShare 采集阶段没有取到，而是事件抽取阶段默认每日只抽 `limit_per_day=20` 条新闻。当天宏观日历类新闻较多，重要政策/流动性新闻排序靠后，被每日前 20 条截断挤掉。

这会导致模型里的新闻事件特征漏掉关键政策、流动性、风格切换信息。

## 修正原则

1. 采集不应丢信息

历史新闻采集阶段只负责尽可能完整地把匹配关键词的新闻落入 `news_raw`，不在采集侧做每日条数截断。

2. 抽取必须优先保留重点组

事件抽取候选集应优先保留以下类型：

```text
policy_market
liquidity
index_style
growth_industry
macro
```

如果必须设每日上限，也要先按 `matched_groups` 优先级排序，避免重点政策新闻被普通宏观日历挤掉。

3. 补历史数据时优先使用不设每日上限

历史补抽推荐：

```text
--limit-per-day 0
```

表示不按每日条数截断。这样可以最大限度避免遗漏。

4. collector 需要能直接触发抽取

为了避免后续使用者只运行采集命令、忘记正确抽取参数，collector 入口应支持采集完成后直接调用事件抽取，并默认使用不截断的安全口径。

## 已交接代码

以下文件是从 bottom 分支复制过来的完整代码副本，放在 collector 分支 `docs/news_collector_event_handoff_code` 下：

```text
collect_news_history.py
extract_event.py
sync_news_databases.py
test_event_extraction.py
```

用途说明：

```text
collect_news_history.py
```

新增 `--extract-events` 相关参数。采集历史新闻后，可以直接调用事件抽取。核心新增参数：

```text
--extract-events
--event-priority-groups
--event-limit-per-day
--event-batch-size
--llm-retries
--retry-wait
```

其中 `--event-limit-per-day` 默认是 `0`，表示不做每日截断。

```text
extract_event.py
```

事件抽取候选逻辑已支持：

```text
DEFAULT_PRIORITY_GROUPS
_group_priority
priority_groups
audit_missing_priority_news
```

当存在每日上限时，会优先保留 `matched_groups` 命中重点组的新闻。

```text
sync_news_databases.py
```

新增数据库同步工具，用于把新闻相关表从一个 MySQL 库同步到另一个库。

当前支持：

```text
news_raw    唯一键 news_id
event_daily 唯一键 source_ids
```

同步方式是：

```text
INSERT ... SELECT ... ON DUPLICATE KEY UPDATE
```

不会清空目标表，不会删除目标库已有额外数据。

```text
test_event_extraction.py
```

新增回归测试，验证重点政策/流动性新闻不会被普通宏观日历新闻挤出每日候选集。

## 推荐迁移方式

在 collector 分支正式迁移时，不建议直接整文件覆盖。建议按职责拆分：

1. 将 `extract_event.py` 中候选排序和审计函数迁移到 collector 当前版本。
2. 将 `collect_news_history.py` 中 `--extract-events` 参数和调用链迁移到 collector 当前版本。
3. 将 `sync_news_databases.py` 作为独立 collector 工具加入 `src/collectors/` 或 `scripts/`。
4. 加入 `test_event_extraction.py` 对应测试。
5. 运行 py_compile 和 pytest。

建议迁移后测试：

```bash
python -m py_compile src/collectors/collect_news_history.py src/collectors/sync_news_databases.py src/llm/extract_event.py
python -m pytest -q tests/test_event_extraction.py
```

## 推荐以后使用的补历史命令

采集并直接抽取：

```bash
python -m src.collectors.collect_news_history --start-date 2016-01-01 --end-date 2026-06-30 --keyword-mode match --batch-days 10 --sleep 0.5 --extract-events --event-limit-per-day 0 --event-priority-groups policy_market,liquidity,index_style,growth_industry,macro --event-batch-size 20 --llm-retries 3 --retry-wait 10
```

只同步新闻相关数据：

```bash
python -m src.collectors.sync_news_databases --source-database zz1000_botumn --target-database zz1000_predict
```

## 本次数据库同步结果

已执行同步：

```text
source: zz1000_botumn
target: zz1000_predict
tables: news_raw, event_daily
```

同步后统计：

```text
zz1000_botumn.news_raw      132037 行，2016-01-16 到 2026-06-30
zz1000_predict.news_raw     132102 行，2016-01-16 到 2026-06-30

zz1000_botumn.event_daily   66428 行，2016-01-16 到 2026-06-30
zz1000_predict.event_daily  66428 行，2016-01-16 到 2026-06-30
```

源库到目标库反查：

```text
news_raw    source_missing_in_target = 0
event_daily source_missing_in_target = 0
```

说明 `zz1000_botumn` 中新闻相关数据已全部补充到 `zz1000_predict`。

`zz1000_predict.news_raw` 比 `zz1000_botumn.news_raw` 多 65 行，这是目标库原本已有的额外新闻。本次同步没有删除这些记录。

## 已验证

在 bottom worktree 中已验证：

```text
python -m py_compile src/collectors/collect_news_history.py src/collectors/sync_news_databases.py src/llm/extract_event.py
通过

pytest -q tests/test_event_extraction.py
1 passed, 1 warning
```

warning 是 pandas 对 `bottleneck` 版本的提示，不影响本次逻辑。

## 风险和注意事项

1. `zz1000_predict.news_raw` 表比 bottom 版本字段更多。本次同步工具只同步两个库的公共字段，不会写入 predict 独有字段。

2. `event_daily.source_ids` 是唯一键。若未来一条事件对应多个来源拼接策略改变，需要确认唯一键逻辑是否仍稳定。

3. `--event-limit-per-day 0` 能最大限度避免漏抽，但会增加 LLM 调用量和运行时间。历史补抽建议用它；日常增量可按成本选择较小窗口。

4. collector 分支已有大量未提交改动，本次只在 `docs` 下添加交接材料，没有直接改 collector 的运行代码。
