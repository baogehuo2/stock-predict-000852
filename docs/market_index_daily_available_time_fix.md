# market_index_daily available_time 修复需求

## 背景

建模分支在检查 `zz1000_predict` 数据库时发现，核心指数日线行情表 `market_index_daily` 中，部分指数的 `available_time` 为空。

示例：

```text
trade_date = 2026-06-10
index_code = 000852
close = 8198.723
available_time = NULL
```

对模型来说，`available_time` 表示这条数据最早什么时候可以被模型使用。日线 OHLCV 必须在收盘后才可用，否则会造成未来信息泄露。

因此，核心指数日线需要补齐 `available_time`。

## 目标

修复 `zz1000_predict.market_index_daily` 中核心指数日线的 `available_time` 缺失问题，并保证后续采集任务持续写入正确的 `available_time`。

目标口径：

```text
A 股指数日线行情 available_time = trade_date 当日 15:30:00
```

含义：

- 收盘后生成的模型信号可以使用当日完整日 K。
- 盘中模型不能使用当日完整日 K。
- 周线、月线、股票日线、ETF、期货等已有独立 `available_time` 口径，本需求只处理 `market_index_daily`。

## 影响范围

目标数据库：

```text
zz1000_predict
```

目标表：

```text
market_index_daily
```

需要修复的字段：

```text
available_time
```

建议覆盖指数：

```text
000852 中证1000
000905 中证500
000300 沪深300
000001 上证指数
399006 创业板指
399001 深证成指
000016 上证50
000688 科创50
000985 中证全指
399303 国证2000
932000 中证2000
```

如果采集配置里还有其他 A 股指数，也可以按相同口径补齐。

## 当前发现的问题

建模分支检查结果显示：

```text
market_index_daily 中多个核心指数的 available_time 为空。
```

其中包括：

```text
000852
000905
000300
000001
399006
```

同时这些指数的 `open/high/low/close/volume` 是有值的，说明行情数据本身可用，缺的是可用时间字段。

## 建议修复方式

### 1. 历史数据修复

可以新增一个 migration 或修复脚本，对已有空值补齐：

```sql
UPDATE market_index_daily
SET available_time = TIMESTAMP(trade_date, '15:30:00')
WHERE available_time IS NULL
  AND index_code IN (
      '000852',
      '000905',
      '000300',
      '000001',
      '399006',
      '399001',
      '000016',
      '000688',
      '000985',
      '399303',
      '932000'
  );
```

如果希望覆盖所有 A 股指数，也可以按采集配置中的指数列表动态生成，不建议硬编码在多个地方。

### 2. 后续采集逻辑修复

采集 `market_index_daily` 时，写入或 upsert 前应确保：

```python
available_time = datetime.combine(trade_date, time(15, 30))
```

注意：

- `available_time` 不应使用 `crawl_time`。
- `crawl_time` 是实际抓取时间。
- `available_time` 是市场理论上可获得完整日 K 的时间。

推荐字段语义：

| 字段 | 含义 |
| --- | --- |
| `trade_date` | 行情所属交易日 |
| `available_time` | 当日完整日 K 可用于模型的最早时间，A 股日线默认 `trade_date 15:30:00` |
| `crawl_time` | 本系统实际采集时间 |
| `created_at` / `updated_at` | 入库创建和更新时间 |

## 不建议的做法

不要把 `available_time` 设置成采集时间：

```text
available_time = crawl_time
```

原因：

- 历史回填时，`crawl_time` 可能是 2026 年。
- 如果建模按 `available_time <= signal_time` 过滤，历史样本会被错误过滤。
- 这和成分股历史回填问题类似，但日线行情可以明确按收盘后可用时间修复。

不要把 `available_time` 设置成当天 00:00：

```text
available_time = trade_date 00:00:00
```

原因：

- 当天开盘前还不知道当天完整 OHLCV。
- 会造成盘中/开盘前信号使用未来收盘数据。

## 验收 SQL

### 1. 检查核心指数是否还有空值

```sql
SELECT
    index_code,
    COUNT(*) AS rows_count,
    SUM(available_time IS NULL) AS null_available_time_count,
    MIN(trade_date) AS min_trade_date,
    MAX(trade_date) AS max_trade_date
FROM market_index_daily
WHERE index_code IN (
    '000852',
    '000905',
    '000300',
    '000001',
    '399006',
    '399001',
    '000016',
    '000688',
    '000985',
    '399303',
    '932000'
)
GROUP BY index_code
ORDER BY index_code;
```

验收标准：

```text
null_available_time_count = 0
```

### 2. 检查 available_time 是否早于 trade_date

```sql
SELECT
    index_code,
    COUNT(*) AS bad_rows
FROM market_index_daily
WHERE index_code IN (
    '000852',
    '000905',
    '000300',
    '000001',
    '399006',
    '399001',
    '000016',
    '000688',
    '000985',
    '399303',
    '932000'
)
  AND DATE(available_time) < trade_date
GROUP BY index_code;
```

验收标准：

```text
返回 0 行
```

### 3. 检查是否按 15:30 口径写入

```sql
SELECT
    index_code,
    TIME(available_time) AS available_clock,
    COUNT(*) AS rows_count
FROM market_index_daily
WHERE index_code IN (
    '000852',
    '000905',
    '000300',
    '000001',
    '399006',
    '399001',
    '000016',
    '000688',
    '000985',
    '399303',
    '932000'
)
GROUP BY index_code, TIME(available_time)
ORDER BY index_code, available_clock;
```

验收标准：

```text
核心指数应主要为 15:30:00。
```

如果历史中存在其他合理时间，需要在交付说明里解释。

### 4. 抽查最新数据

```sql
SELECT
    trade_date,
    index_code,
    open,
    high,
    low,
    close,
    volume,
    amount,
    available_time,
    crawl_time
FROM market_index_daily
WHERE index_code = '000852'
ORDER BY trade_date DESC
LIMIT 10;
```

验收标准：

```text
available_time = trade_date 15:30:00
crawl_time 保留实际采集时间
```

## 数据质量规则建议

建议在 `data_quality_daily` 检查中增加或更新以下规则：

```text
market_index_daily.available_time 非空率 = 100%
market_index_daily.available_time 日期不早于 trade_date
market_index_daily.available_time 时间默认等于 15:30:00
```

如果后续支持盘中指数数据，盘中数据应进入独立分时表，不应改变日线表口径。

## 建模侧使用口径

修复后，建模侧可以统一使用：

```text
market_index_daily.available_time <= signal_time
```

对于收盘后生成的日线模型：

```text
signal_time = trade_date 15:30:00 或之后
```

即可使用当日完整日 K。

对于盘中模型：

```text
signal_time < trade_date 15:30:00
```

不能使用当日完整日 K，只能使用分时表。

## 优先级

优先级：高。

原因：

- 该字段是防止未来信息泄露的核心字段。
- 当前核心指数日线已被多个模型和特征任务使用。
- 修复成本低，只是补齐时间字段和后续采集逻辑。

## 非目标

本需求不要求：

- 新增分时数据。
- 新增估值数据。
- 修复成交额 `amount` 缺失。
- 修改成分股历史 `available_time` 口径。
- 修改建模代码。

这些问题可以单独开需求处理。
