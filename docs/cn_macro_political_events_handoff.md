# 国内宏观政治事件采集入库说明

## 目标

本分支已合入国内宏观政治事件采集、清洗、JSON 落盘和 MySQL 入库能力，用于中证1000量化研究的宏观政治事件原始数据层。

## 采集范围

当前仅覆盖原始事件库，不做特征构建、标签构建、模型训练和策略判断。

事件类型：

- `TWO_SESSIONS`：全国人大会议、全国政协会议。只保留年度大会的开始时间、结束时间、发布日期、来源和摘要，不采集大会期间第几次全体会议、主席团会议、代表团会议等过程性会议。
- `POLITBURO_MEETING`：中共中央政治局经济相关会议，重点保留 4 月、7 月、12 月经济会议。
- `CEWC`：中央经济工作会议。
- `CPC_PLENUM`：中央全会。每次全会只保留一条事件，例如 `十八届三中全会`、`二十届三中全会`。

## 文件

代码：

- `src/macro_events/crawler.py`：官方来源候选页抓取。
- `src/macro_events/parser.py`：文章正文到事件的规则解析。
- `src/macro_events/cleaner.py`：事件清洗、两会口径统一、全会去重、乱码剔除。
- `src/macro_events/predictor.py`：未来 18 个月事件预测。
- `src/macro_events/json_store.py`：JSON 读写和去重合并。
- `src/macro_events/db_store.py`：MySQL upsert。
- `src/macro_events/main_import_existing.py`：导入当前已确认历史数据。
- `src/macro_events/main_history_build.py`：历史回溯构建。
- `src/macro_events/main_daily_update.py`：每日增量更新。

数据：

- `docs/architecture/macro_events_full_clean.csv`：当前最全确认事件表。
- `docs/architecture/macro_events_need_fill_clean.csv`：待人工补充事件表。
- `data/macro_events/macro_events_history.json`：历史事件 JSON。
- `data/macro_events/macro_events_predicted.json`：未来预测事件 JSON。
- `data/macro_events/macro_events_review_queue.json`：待审核事件 JSON。

数据库：

- 迁移文件：`sql/migrations/013_cn_macro_political_event.sql`
- 表名：`cn_macro_political_event`
- 唯一键：`event_id`

## 运行命令

导入当前已确认 CSV，生成 JSON：

```bash
python -m src.macro_events.main_import_existing
```

导入当前已确认 CSV，并写入 MySQL：

```bash
python -m src.macro_events.main_import_existing --write-db --apply-migrations
```

历史回溯构建：

```bash
python -m src.macro_events.main_history_build
```

仅刷新预测事件，不访问网络：

```bash
python -m src.macro_events.main_history_build --no-network
```

每日更新：

```bash
python -m src.macro_events.main_daily_update
```

每日更新并写入 MySQL：

```bash
python -m src.macro_events.main_daily_update --write-db --apply-migrations
```

## 当前入库结果

已执行：

```bash
python -m src.macro_events.main_import_existing --write-db --apply-migrations
```

MySQL 表 `cn_macro_political_event` 当前写入 112 条：

- `TWO_SESSIONS`：38 条
- `CPC_PLENUM`：28 条
- `CEWC`：23 条
- `POLITBURO_MEETING`：23 条

JSON 当前行数：

- `macro_events_history.json`：112 条
- `macro_events_review_queue.json`：25 条
- `macro_events_predicted.json`：8 条

## 待人工补充

待补充清单保存在：

```text
docs/architecture/macro_events_need_fill_clean.csv
```

当前待补 25 条，主要是缺少官方 `source_url`。不再补 `note`。

## 口径说明

- 两会不保留“第几届第几次会议”作为事件名称。
- 两会不保留期间内部会议。
- 中央全会保留“届次+中全会”作为事件名称，因为这是宏观政治事件本体。
- `source_text` 只保留可追溯摘要，不要求完整原文。
- 重复运行导入命令不会重复插入数据库，按 `event_id` 幂等 upsert。
