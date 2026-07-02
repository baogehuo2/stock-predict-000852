# 第二批数据采集交接说明

## 1. 交接目标

本文档面向 `buy-signal-research` 建模分支，说明第二批原始数据采集结果、表结构口径、历史范围、未来数据构建方式和建模使用约束。

第二批只覆盖原始数据采集、清洗、时间戳校验和入库，不包含特征工程、模型训练、事件方向判断、LLM 抽取或情绪打分。

建模侧必须遵守统一约束：

- 只能使用 `available_time <= signal_time` 的数据。
- `actual_value` 是已发布事实值。
- `forecast_value` 和 `previous_value` 是发布前可获得的预期/前值。
- `is_predicted = 1` 的政治事件是未来规则预测窗口，不代表事件已经发生。
- 第三方经济日历只用于 `forecast / previous / release_time` 参考，不作为 actual 权威源。

## 2. 数据总览

当前数据库状态：

| 表名 | 行数 | 历史/未来范围 | 用途 |
| --- | ---: | --- | --- |
| `calendar_daily_raw` | 4748 | 2014-01-01 至 2026-12-31 | A股交易日、休市、节假日 |
| `event_calendar_raw` | 52 | 2024-01-22 14:00:00 至 2027-12-16 14:00:00 | FOMC、ECB、BOJ 固定议程 |
| `macro_release_raw` | 2389 | period 2013-12-31 至 2026-06-30 | 中美宏观 actual / forecast / previous |
| `macro_release_time_reference_raw` | 16 | release 2026-06-17 20:30:00 至 2026-07-16 20:30:00 | 宏观发布准确时间参考 |
| `global_market_daily` | 33833 | 2014-01-01 至 2026-06-17 | 外部风险行情 |
| `cn_macro_political_event` | 111 | 2005-03-03 至 2027-12-08 | 国内宏观政治事件历史与未来预测窗口 |

## 3. 固定交易日历：`calendar_daily_raw`

采集脚本：

```bash
python -m src.collectors.collect_calendar_daily_v2 --start-date 2014-01-01 --end-date 2026-12-31
```

数据源：

- Tushare `trade_cal`：A股交易日历。
- `holiday-cn`：中国法定节假日结构化数据。

当前结果：

| 指标 | 值 |
| --- | --- |
| 行数 | 4748 |
| 日期范围 | 2014-01-01 至 2026-12-31 |
| 交易日数 | 3161 |
| 节假日标记数 | 351 |
| 唯一键 | `(calendar_date, market)` |
| market | `CN_STOCK` |

字段口径：

- `calendar_date`：自然日。
- `is_trading_day`：是否A股交易日。
- `is_exchange_closed`：是否A股休市。
- `is_holiday`：是否中国法定节假日。
- `holiday_name`：节假日名称，非节假日为空。
- `available_time`：该条日历数据对模型可用的时间。

建模使用：

- 可用于生成交易日序列、节假日前后窗口、休市过滤。
- 未来日历可作为预测时已知信息使用。
- 当前 Tushare 未返回 2027 年交易日历，因此本次只补到 2026-12-31。

质量结果：

- 必填字段缺失：0。
- 重复键：0。
- 2015 至 2026 交易日覆盖缺失：0。
- `holiday_name` 空值为正常现象，非节假日为空。

## 4. 海外央行固定议程：`event_calendar_raw`

采集脚本：

```bash
python -m src.collectors.collect_event_calendar_v2 --start-date 2024-01-01 --end-date 2027-12-31
```

数据源：

- Federal Reserve FOMC calendar。
- ECB monetary policy meeting calendar。
- BOJ Monetary Policy Meetings schedule。

当前结果：

| country_region | 行数 | 最早时间 | 最晚时间 |
| --- | ---: | --- | --- |
| EU | 12 | 2026-07-23 14:00:00 | 2027-12-16 14:00:00 |
| JP | 8 | 2024-01-22 14:00:00 | 2024-12-17 14:00:00 |
| US | 32 | 2024-01-31 14:00:00 | 2027-12-08 14:00:00 |

字段口径：

- `event_type = central_bank_meeting`。
- `event_name`：`FOMC`、`ECB_MPC`、`BOJ_MPM`。
- `scheduled_time`：计划议程时间。当前统一使用日期 + `14:00:00` 占位，核心信息是日期。
- `is_confirmed = 1`：来自官方议程页面。
- `available_time`：采集到该官方议程的时间。

建模使用：

- 可用于“未来已知重要节点”窗口，例如 FOMC 前后、ECB/BOJ 议息日前后。
- 不包含会议结果、决议文本或讲话内容。
- `actual_start_time / actual_end_time / announcement_time` 当前为空，正常。

质量结果：

- 必填字段缺失：0。
- 重复键：0。
- 事件表不适用交易日覆盖率检查，相关 warn 不作为阻塞。

## 5. 宏观发布数据：`macro_release_raw`

采集脚本：

```bash
python -m src.collectors.collect_macro_release_v2 --start-date 2014-01-01 --end-date 2026-06-17
python -m src.collectors.collect_macro_forecast_calendar_v2 --start-date 2026-06-17 --future-days 60
```

数据源：

- 国内宏观 actual：Tushare 为主，AKShare 兜底。
- 美国宏观 actual：AKShare。
- 宏观 forecast / previous：AKShare `macro_info_ws` 经济日历。

当前结果：

| 指标 | 值 |
| --- | --- |
| 行数 | 2389 |
| period 范围 | 2013-12-31 至 2026-06-30 |
| actual 行数 | 2374 |
| forecast 非空行数 | 135 |
| previous 非空行数 | 149 |
| 唯一键 | `(indicator_key, period_date, release_time, revision_no)` |

已覆盖指标：

- 国内：CPI、PPI、GDP、M1、M2、PMI、社融、人民币贷款、社零、工业增加值。
- 美国：CPI、PPI、GDP、非农、失业率、零售销售、工业产出、ISM PMI。

字段口径：

- `period_date`：指标所属统计期。
- `release_time`：发布时间；actual 源缺少精确发布时间时部分为保守估算。
- `actual_value`：已发布实际值。
- `forecast_value`：发布前市场预期值。
- `previous_value`：发布前可见前值。
- `available_time`：模型可使用该条数据的时间。
- `data_source = akshare:macro_info_ws` 的行只用于 forecast / previous，不作为 actual 权威源。

建模使用：

- 历史 actual 特征必须按 `available_time` 截断。
- 发布前事件窗口可使用 `forecast_value / previous_value`。
- 同一指标可能既有 actual 行，也有 forecast/previous 行，建模侧应按 `indicator_key + period_date + available_time` 做时点过滤。

质量结果：

- 必填字段缺失：0。
- 重复键：0。
- actual 空值主要来自 forecast 日历行，正常。
- `revised_previous_value` 当前为空，说明尚未形成修订版本链。

## 6. 宏观发布时间参考：`macro_release_time_reference_raw`

采集脚本：

```bash
python -m src.collectors.collect_macro_release_time_reference_v2 --start-date 2026-06-17 --future-days 60
```

数据源：

- AKShare `macro_info_ws` 经济日历。

当前结果：

| 指标 | 值 |
| --- | --- |
| 行数 | 16 |
| release_time 范围 | 2026-06-17 20:30:00 至 2026-07-16 20:30:00 |
| 唯一键 | `(indicator_key, period_date, release_time, source_name)` |
| source_type | `economic_calendar` |
| is_official | 0 |

字段口径：

- `release_time`：经济日历计划发布时间。
- `source_type`：来源类型。
- `is_official = 0`：当前不是官方源。
- `confidence = calendar_scheduled`：经济日历计划时间。

建模使用：

- 用于确认某个宏观指标何时发布，避免 actual 值提前进入训练样本。
- 不直接作为数值特征。
- 后续如接入国家统计局、央行、BLS、BEA 等官方发布日历，可在该表继续落库并标记 `is_official = 1`。

质量结果：

- 必填字段缺失：0。
- 重复键：0。
- 事件/月频表不适用交易日覆盖率检查，相关 warn 不作为阻塞。

## 7. 外部风险行情：`global_market_daily`

采集脚本：

```bash
python -m src.collectors.collect_global_market_v2 --start-date 2014-01-01 --end-date 2026-06-17
```

数据源：

- AKShare 美债收益率、海外指数、海外期货、美国 ETF 代理。
- FRED/Yahoo 当前作为可选参考源，默认不跑。

当前结果：

| symbol | 行数 | 最早日期 | 最新日期 | 说明 |
| --- | ---: | --- | --- | --- |
| CL | 3229 | 2014-01-01 | 2026-06-17 | NYMEX 原油 |
| GC | 2589 | 2016-06-17 | 2026-06-17 | COMEX 黄金 |
| IWM | 3131 | 2014-01-02 | 2026-06-16 | Russell 2000 ETF 代理 |
| NDX | 3100 | 2014-02-18 | 2026-06-16 | Nasdaq 100 |
| SOX | 3139 | 2014-01-02 | 2026-06-16 | 费城半导体指数 |
| SPX | 3133 | 2014-01-02 | 2026-06-16 | S&P 500 |
| US10Y | 3116 | 2014-01-02 | 2026-06-16 | 美国10年期国债收益率 |
| US2Y | 3115 | 2014-01-02 | 2026-06-16 | 美国2年期国债收益率 |
| US30Y | 3083 | 2014-01-02 | 2026-06-16 | 美国30年期国债收益率 |
| UUP | 3131 | 2014-01-02 | 2026-06-16 | 美元指数 ETF 代理 |
| VXX | 3067 | 2014-01-02 | 2026-06-16 | VIX 短期期货 ETN 代理 |

字段口径：

- `trade_date`：海外市场日期。
- `available_time`：按海外市场收盘后可用时间设置。
- `close`：必填，用于外部风险状态。
- `open/high/low/volume`：部分收益率和代理数据源缺失，允许为空。

建模使用：

- 用于外部风险偏好、美元、原油、黄金、美债利率、海外成长风格对照。
- `IWM / UUP / VXX` 是代理资产，不是指数本体；建模侧应按代理口径解释。

质量结果：

- 必填字段缺失：0。
- 重复键：0。
- `ohlc_logic_error_rows = 0`。
- 成交量非负。
- 少量交易日覆盖 warn 主要来自中美市场节假日差异，最长连续缺口为 1 个 A股交易日。

## 8. 国内宏观政治事件：`cn_macro_political_event`

采集/导入脚本：

```bash
python -m src.macro_events.main_import_existing --write-db --apply-migrations
python -m src.macro_events.main_daily_update --no-network --write-db --apply-migrations
```

数据源：

- 已确认历史事件来自整理后的 `docs/architecture/macro_events_full_clean.csv`。
- 待审核事件来自 `docs/architecture/macro_events_need_fill_clean.csv`。
- 未来预测由规则生成，不访问网络。

当前结果：

| 指标 | 值 |
| --- | --- |
| 总行数 | 111 |
| confirmed | 103 |
| predicted | 8 |
| 日期范围 | 2005-03-03 至 2027-12-08 |
| 唯一键 | `event_id` |

按事件类型：

| event_type | is_predicted | 行数 | 最早日期 | 最晚日期 |
| --- | ---: | ---: | --- | --- |
| CEWC | 0 | 21 | 2005-11-29 | 2025-12-10 |
| CEWC | 1 | 2 | 2026-12-08 | 2027-12-08 |
| CPC_PLENUM | 0 | 28 | 2005-10-08 | 2025-10-20 |
| POLITBURO_MEETING | 0 | 18 | 2005-12-01 | 2024-12-09 |
| POLITBURO_MEETING | 1 | 4 | 2026-07-20 | 2027-07-20 |
| TWO_SESSIONS | 0 | 36 | 2005-03-03 | 2026-03-11 |
| TWO_SESSIONS | 1 | 2 | 2027-03-04 | 2027-03-05 |

字段口径：

- `is_predicted = 0`：历史确认事件。
- `is_predicted = 1`：未来规则预测窗口。
- `prediction_confidence`：规则预测置信度，仅用于区分预测强弱，不代表事件实际发生概率校准。
- `status = predicted` 的记录不能当作已发生事件。

建模使用：

- confirmed 历史可用于事件前后窗口回测。
- predicted 未来只能用于“未来可能发生的政策会议窗口”。
- 不包含事件方向、政策强弱、文本情绪或 LLM 分类。

剩余人工项：

- `data/macro_events/macro_events_review_queue.json` 当前仍有 25 条待审核记录，主要缺官方 `source_url`。
- 待审核记录未写入 confirmed 历史库。

## 9. 日常更新建议

预测日前建议按以下顺序刷新第二批数据：

```bash
python -m src.collectors.collect_calendar_daily_v2 --future-days 370
python -m src.collectors.collect_event_calendar_v2 --future-days 370
python -m src.collectors.collect_macro_forecast_calendar_v2 --future-days 60
python -m src.collectors.collect_macro_release_time_reference_v2 --future-days 60
python -m src.collectors.collect_macro_release_v2
python -m src.collectors.collect_global_market_v2
python -m src.macro_events.main_daily_update --no-network --write-db --apply-migrations
```

如需刷新国内政治事件的网络候选页，可去掉 `--no-network`，但该操作会进入规则解析与审核队列，不做 LLM 抽取。

## 10. 质量报告

已生成或更新以下质量报告：

- `data/reports/quality_calendar_daily_raw.json`
- `data/reports/quality_event_calendar_raw.json`
- `data/reports/quality_macro_release_raw.json`
- `data/reports/quality_macro_release_time_reference_raw.json`
- `data/reports/quality_global_market_daily.json`

已确认：

- 所有第二批核心表必填字段缺失为 0。
- 唯一键重复为 0。
- `global_market_daily` 的 OHLC 逻辑错误为 0。
- `calendar_daily_raw` 的交易日覆盖缺失为 0。

注意：

- 事件表、宏观月频表不适用按 A股交易日逐日覆盖的质量指标，因此对应 coverage warn 不作为阻塞。
- 部分外部风险品种在 A股交易日无海外交易，属于市场节假日差异。

## 11. 已知风险

1. A股交易日历当前只补到 2026-12-31，因为 Tushare 当前未返回 2027 年日历。
2. `macro_release_raw` 中部分 actual 行的 `release_time` 来自保守估算；应优先用 `macro_release_time_reference_raw` 做可用时点校验。
3. `macro_release_time_reference_raw` 当前是经济日历源，`is_official = 0`，尚未接官方发布日历。
4. `IWM / UUP / VXX` 是海外风险代理资产，不是 Russell 2000、DXY、VIX 指数本体。
5. 国内政治事件预测为规则窗口，不代表会议已正式公告。

## 12. 建模侧读取建议

建模侧建议统一按时间过滤：

```sql
WHERE available_time <= :signal_time
```

国内政治事件表没有 `available_time` 字段，建议：

- confirmed 事件使用 `publish_date` 或 `start_date` 做可见性约束。
- predicted 事件只在 `start_date >= signal_date` 时作为未来窗口使用。
- 不要把 `is_predicted = 1` 当作历史已发生事件参与标签构造。

第二批数据可以交接给建模侧使用，但建模侧必须保留上述时点约束，避免未来信息泄露。
