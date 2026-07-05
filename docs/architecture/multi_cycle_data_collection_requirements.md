# 多周期顶底模型数据采集需求说明

## 目标

为多周期顶底模型补齐统一的数据底座，支持以下建模任务：

| 模型周期 | 预测窗口 | 主要数据频率 |
| --- | --- | --- |
| 超短期顶底 | 未来 4-6 小时 | 15min / 30min / 60min |
| 短期顶底 | 未来 5-15 个交易日 | 日线 + 周线 |
| 中期顶底 | 未来 20-40 个交易日 | 日线 + 周线 |
| 长期顶底 | 未来 80-120 个交易日 | 日线 + 周线 + 月线 |

本需求只覆盖数据采集、入库、数据质量校验。不包含模型训练、可视化展示、LLM 事件提取。

目标数据库：

```text
zz1000_botumn
```

## 基本原则

所有数据必须能用于滚动训练和历史回测，重点避免未来信息泄露。

必须区分以下时间字段：

| 字段 | 含义 |
| --- | --- |
| `trade_date` / `trade_time` | 行情或数据所属交易日、所属 bar 时间 |
| `release_time` | 数据官方实际发布时间 |
| `available_time` | 模型在实盘中可使用该数据的时间 |
| `crawl_time` | 采集脚本抓取时间 |
| `created_at` / `updated_at` | 入库创建和更新时间 |

约束：

- 日线行情的 `available_time` 不应早于当日收盘后。
- 分时行情的 `available_time` 不应早于该 bar 结束后。
- 融资融券、估值、指数成分等非行情数据必须记录可获得时间。
- 建模时只能使用 `available_time <= signal_time` 的数据。
- 采集脚本必须支持重复运行和增量更新，不能重复插入。

## 一、指数日 K 数据

目标表：

```text
market_index_daily
```

当前库里已有该表，但需要作为长期维护对象持续更新，并尽量补齐历史。

必需字段：

```sql
trade_date DATE NOT NULL,
index_code VARCHAR(20) NOT NULL,
index_name VARCHAR(100),
open DECIMAL(18,6),
high DECIMAL(18,6),
low DECIMAL(18,6),
close DECIMAL(18,6),
pre_close DECIMAL(18,6),
pct_chg DECIMAL(18,6),
volume DECIMAL(24,6),
amount DECIMAL(24,6),
data_source VARCHAR(50),
source_url TEXT,
available_time DATETIME,
crawl_time DATETIME,
created_at DATETIME,
updated_at DATETIME
```

唯一键：

```sql
UNIQUE KEY uk_market_index_daily (trade_date, index_code)
```

必须指数：

| 指数代码 | 名称 |
| --- | --- |
| `000852` | 中证1000 |
| `000905` | 中证500 |
| `000300` | 沪深300 |
| `000001` | 上证指数 |
| `399006` | 创业板指 |
| `399001` | 深证成指 |

建议日期范围：

```text
至少 2005-01-01 至今。
如果数据源只能覆盖 2015 至今，必须在交付说明中写清楚。
```

用途：

- 日线短期、中期、长期顶底模型。
- 周线、月线重采样来源。
- 相对强弱特征。
- 大盘环境过滤。

## 二、指数分时 K 数据

目标新增表：

```text
market_index_intraday
```

这是超短期 4-6 小时顶底模型的核心缺口。

建议先采集 60min，条件允许再采集 30min 和 15min。

必需字段：

```sql
trade_time DATETIME NOT NULL,
trade_date DATE NOT NULL,
index_code VARCHAR(20) NOT NULL,
freq VARCHAR(10) NOT NULL,
open DECIMAL(18,6),
high DECIMAL(18,6),
low DECIMAL(18,6),
close DECIMAL(18,6),
pre_close DECIMAL(18,6),
pct_chg DECIMAL(18,6),
volume DECIMAL(24,6),
amount DECIMAL(24,6),
bar_seq INT,
data_source VARCHAR(50),
source_url TEXT,
available_time DATETIME,
crawl_time DATETIME,
created_at DATETIME,
updated_at DATETIME
```

唯一键：

```sql
UNIQUE KEY uk_market_index_intraday (trade_time, index_code, freq)
```

必须频率：

| 频率 | 优先级 |
| --- | --- |
| `60m` | 必须 |
| `30m` | 建议 |
| `15m` | 可选 |

必须指数：

| 指数代码 | 名称 |
| --- | --- |
| `000852` | 中证1000 |
| `000905` | 中证500 |
| `000300` | 沪深300 |
| `000001` | 上证指数 |
| `399006` | 创业板指 |

建议日期范围：

```text
至少 2020-01-01 至今。
最好 2015-01-01 至今。
```

时间戳要求：

- `trade_time` 统一使用该 bar 的结束时间。
- `available_time` 不得早于该 bar 结束时间。
- `bar_seq` 表示当日第几个 bar，便于区分上午、下午。

用途：

- 超短期顶底模型。
- 日线信号触发后的分时买卖点。
- 盘中回撤、反弹路径、放量缩量特征。

## 三、ETF 日线数据

目标表：

```text
market_etf_daily
```

当前库里已有该表，但需要补齐并持续更新。

建议 ETF：

| ETF代码 | 说明 |
| --- | --- |
| `512100` | 中证1000ETF |
| `159845` | 中证1000ETF |
| `510300` | 沪深300ETF |
| `510500` | 中证500ETF |
| `159915` | 创业板ETF |
| `588000` | 科创50ETF |

必需字段：

```sql
trade_date DATE NOT NULL,
etf_code VARCHAR(20) NOT NULL,
etf_name VARCHAR(100),
open DECIMAL(18,6),
high DECIMAL(18,6),
low DECIMAL(18,6),
close DECIMAL(18,6),
pct_chg DECIMAL(18,6),
volume DECIMAL(24,6),
amount DECIMAL(24,6),
turnover DECIMAL(18,6),
data_source VARCHAR(50),
created_at DATETIME,
updated_at DATETIME
```

唯一键：

```sql
UNIQUE KEY uk_market_etf_daily (trade_date, etf_code)
```

用途：

- ETF 交易活跃度。
- 中证1000情绪代理。
- ETF 成交额放大、萎缩。
- 流动性风险识别。

## 四、股指期货日线数据

目标表：

```text
market_futures_daily
```

当前库里已有该表，但需要确认是否覆盖主力合约和连续合约。

建议品种：

| 品种 | 说明 |
| --- | --- |
| `IM` | 中证1000股指期货 |
| `IC` | 中证500股指期货 |
| `IF` | 沪深300股指期货 |
| `IH` | 上证50股指期货 |

必需字段：

```sql
trade_date DATE NOT NULL,
future_code VARCHAR(20) NOT NULL,
future_name VARCHAR(100),
category VARCHAR(50),
open DECIMAL(18,6),
high DECIMAL(18,6),
low DECIMAL(18,6),
close DECIMAL(18,6),
settle DECIMAL(18,6),
pct_chg DECIMAL(18,6),
volume DECIMAL(24,6),
open_interest DECIMAL(24,6),
amount DECIMAL(24,6),
data_source VARCHAR(50),
created_at DATETIME,
updated_at DATETIME
```

唯一键：

```sql
UNIQUE KEY uk_market_futures_daily (trade_date, future_code)
```

建议新增连续合约表：

```text
market_futures_continuous_daily
```

字段：

```sql
trade_date DATE NOT NULL,
product_code VARCHAR(20) NOT NULL,
continuous_type VARCHAR(20) NOT NULL,
main_contract VARCHAR(20),
open DECIMAL(18,6),
high DECIMAL(18,6),
low DECIMAL(18,6),
close DECIMAL(18,6),
settle DECIMAL(18,6),
basis DECIMAL(18,6),
basis_rate DECIMAL(18,6),
volume DECIMAL(24,6),
open_interest DECIMAL(24,6),
data_source VARCHAR(50),
created_at DATETIME,
updated_at DATETIME
```

唯一键：

```sql
UNIQUE KEY uk_futures_continuous (trade_date, product_code, continuous_type)
```

用途：

- 期货升水、贴水。
- 风险偏好。
- 多空情绪。
- 持仓变化。

## 五、融资融券市场数据

目标表：

```text
margin_market_daily
```

当前库里该表为空，需要补齐。

必需字段：

```sql
trade_date DATE NOT NULL,
exchange VARCHAR(20) NOT NULL,
financing_balance DECIMAL(24,6),
financing_buy_amount DECIMAL(24,6),
securities_lending_balance DECIMAL(24,6),
securities_lending_sell_volume DECIMAL(24,6),
margin_balance DECIMAL(24,6),
currency VARCHAR(10),
data_source VARCHAR(50),
source_url TEXT,
release_time DATETIME,
available_time DATETIME,
crawl_time DATETIME
```

唯一键：

```sql
UNIQUE KEY uk_margin_market_daily (trade_date, exchange)
```

交易所取值：

| 取值 | 含义 |
| --- | --- |
| `SSE` | 上海证券交易所 |
| `SZSE` | 深圳证券交易所 |
| `TOTAL` | 汇总 |

用途：

- 杠杆情绪。
- 顶部过热。
- 底部去杠杆。
- 中长期风险偏好。

## 六、指数估值数据

建议新增表：

```text
market_index_valuation_daily
```

必需字段：

```sql
trade_date DATE NOT NULL,
index_code VARCHAR(20) NOT NULL,
pe_ttm DECIMAL(18,6),
pb_lf DECIMAL(18,6),
ps_ttm DECIMAL(18,6),
dividend_yield DECIMAL(18,6),
pe_percentile_3y DECIMAL(18,6),
pe_percentile_5y DECIMAL(18,6),
pb_percentile_3y DECIMAL(18,6),
pb_percentile_5y DECIMAL(18,6),
data_source VARCHAR(50),
source_url TEXT,
available_time DATETIME,
crawl_time DATETIME,
created_at DATETIME,
updated_at DATETIME
```

唯一键：

```sql
UNIQUE KEY uk_index_valuation_daily (trade_date, index_code)
```

必须指数：

| 指数代码 | 名称 |
| --- | --- |
| `000852` | 中证1000 |
| `000905` | 中证500 |
| `000300` | 沪深300 |
| `000001` | 上证指数 |
| `399006` | 创业板指 |

用途：

- 长期底部、顶部识别。
- 高估、低估分位。
- 长周期仓位判断。

## 七、指数成分与成分股日线

这部分不是第一优先级，但对中证1000顶底模型很有价值。

目标表：

```text
index_constituent_history
```

当前库里该表为空，需要补齐。

必需字段：

```sql
index_code VARCHAR(20) NOT NULL,
stock_code VARCHAR(20) NOT NULL,
effective_date DATE NOT NULL,
end_date DATE,
weight DECIMAL(18,6),
adjustment_type VARCHAR(50),
announcement_time DATETIME,
available_time DATETIME,
data_source VARCHAR(50),
source_url TEXT,
crawl_time DATETIME
```

唯一键：

```sql
UNIQUE KEY uk_index_constituent_history (index_code, stock_code, effective_date)
```

成分股日线表：

```text
market_stock_daily_raw
```

当前库里该表为空，需要补齐。

必需字段：

```sql
trade_date DATE NOT NULL,
stock_code VARCHAR(20) NOT NULL,
exchange VARCHAR(20),
stock_name VARCHAR(100),
open DECIMAL(18,6),
high DECIMAL(18,6),
low DECIMAL(18,6),
close DECIMAL(18,6),
pre_close DECIMAL(18,6),
pct_chg DECIMAL(18,6),
volume DECIMAL(24,6),
amount DECIMAL(24,6),
turnover_rate DECIMAL(18,6),
amplitude DECIMAL(18,6),
currency VARCHAR(10),
data_source VARCHAR(50),
source_url TEXT,
available_time DATETIME,
crawl_time DATETIME,
raw_hash CHAR(64)
```

唯一键：

```sql
UNIQUE KEY uk_stock_daily (trade_date, stock_code)
```

用途：

- 成分股上涨家数比例。
- 新高、新低数量。
- 涨停、跌停数量。
- 成分股扩散宽度。
- 小盘、微盘流动性风险。

## 八、可由日线派生的数据

以下数据不强制采集原始表，可由 `market_index_daily` 重采样生成：

| 数据 | 来源 | 用途 |
| --- | --- | --- |
| 周 K | 指数日 K | 周线 BOLL、KDJ、MACD、均线 |
| 月 K | 指数日 K | 长期趋势和估值周期 |
| 周线特征 | 周 K | 中期、长期模型 |
| 月线特征 | 月 K | 长期模型 |
| 相对强弱 | 多指数日 K | 风格切换 |
| 波动率特征 | 日线、分时 | 风险过滤 |

如果采集分支愿意落库，建议表名：

```text
market_index_weekly
market_index_monthly
market_index_feature_weekly
market_index_feature_monthly
```

## 九、数据质量要求

每个采集任务完成后，需要生成或更新数据质量记录。

目标表：

```text
data_quality_daily
```

建议检查项：

| 检查项 | 含义 |
| --- | --- |
| `row_count` | 行数 |
| `min_date` | 最早日期 |
| `max_date` | 最新日期 |
| `duplicate_key_count` | 唯一键重复数量 |
| `missing_ohlc_count` | OHLC 缺失数量 |
| `missing_volume_count` | 成交量缺失数量 |
| `abnormal_price_count` | 异常价格数量 |
| `calendar_missing_count` | 交易日异常断档数量 |
| `latest_trade_date` | 最新交易日 |

质量标准：

- 唯一键无重复。
- OHLC 不为空。
- `high >= max(open, close, low)`。
- `low <= min(open, close, high)`。
- 成交量、成交额不为负。
- 交易日不能异常断档。
- 最新数据日期要接近最近交易日。

## 十、优先级

第一优先级，必须做：

```text
market_index_daily 持续更新
market_index_intraday 新增 60m
market_index_valuation_daily 新增
margin_market_daily 补齐
```

第二优先级，强烈建议：

```text
market_etf_daily 补齐并持续更新
market_futures_daily 校验主力/连续覆盖
market_futures_continuous_daily 新增
```

第三优先级，增强模型用：

```text
index_constituent_history
market_stock_daily_raw
market_index_weekly
market_index_monthly
market_index_feature_weekly
market_index_feature_monthly
```

## 十一、交付验收

采集分支交付时需要提供：

1. 建表 SQL 或 migration。
2. 采集脚本。
3. 每张表的数据源说明。
4. 每张表最早日期、最新日期、总行数。
5. 每张表唯一键检查结果。
6. 数据质量报告。
7. 可重复运行，不重复插入。
8. 支持增量更新。

验收 SQL 示例：

```sql
SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
FROM market_index_daily
WHERE index_code = '000852';

SELECT MIN(trade_time), MAX(trade_time), COUNT(*)
FROM market_index_intraday
WHERE index_code = '000852' AND freq = '60m';

SELECT index_code, COUNT(*), MIN(trade_date), MAX(trade_date)
FROM market_index_valuation_daily
GROUP BY index_code;

SELECT exchange, COUNT(*), MIN(trade_date), MAX(trade_date)
FROM margin_market_daily
GROUP BY exchange;

SELECT product_code, continuous_type, COUNT(*), MIN(trade_date), MAX(trade_date)
FROM market_futures_continuous_daily
GROUP BY product_code, continuous_type;

SELECT index_code, stock_code, COUNT(*), MIN(effective_date), MAX(effective_date)
FROM index_constituent_history
WHERE index_code = '000852'
GROUP BY index_code, stock_code;
```

## 十二、当前数据库缺口

根据当前 `zz1000_botumn` 检查结果：

| 数据 | 当前状态 | 处理建议 |
| --- | --- | --- |
| 指数日 K | 已有，2015-01-05 到 2026-06-10 | 持续更新，尽量补历史 |
| 日线技术特征 | 已有 | 后续可扩展多周期特征 |
| 人工顶底日频标签 | 已有 | 后续扩展 `cycle` 等多级别字段 |
| 周 K / 月 K | 无独立表 | 可先从日线重采样 |
| 分时 K | 无 | 必须新增 |
| 估值数据 | 无 | 建议新增 |
| 融资融券 | 表有但为空 | 必须补齐 |
| 指数成分 | 表有但为空 | 建议补齐 |
| 成分股日线 | 表有但为空 | 建议补齐 |

## 十三、未来信息泄露风险点

采集分支需要重点避免以下问题：

- 分时 bar 未结束就写入完整 OHLC。
- 日线收盘数据被标记为盘中可用。
- 估值、融资融券、指数成分使用了发布日期之前不可见的数据。
- 后复权、成分股历史权重、连续合约换月规则引入未来信息。
- 数据修订没有保留 `available_time`，导致历史回测用到未来修订值。

建议所有采集脚本都写入 `data_source`、`available_time` 和 `crawl_time`，并在 README 或采集说明中明确字段含义。
