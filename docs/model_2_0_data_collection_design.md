# 中证1000 Buy Signal 模型 2.0 数据采集与跨分支开发设计

## 1. 文档目的

本文是模型 2.0 的数据契约。它先回答以下问题，再开始编写采集器：

1. 模型 2.0 分批需要哪些原始数据。
2. 每类数据优先从哪里获取，失败时用什么备选来源。
3. 数据写入哪个数据库、哪张表、字段如何定义。
4. 如何保存发布时间、修订版本和采集时间，避免未来数据泄漏。
5. `feature/data-collectors`、`feature/buy-signal-research` 和未来事件抽取分支分别负责什么。
6. 每批数据达到什么标准后，才允许进入模型消融和参数研究。

本文只定义模型 2.0。模型 1.0 的模型文件、冻结特征清单和信号口径不得被覆盖。

## 2. 总体结论

认可并采用以下工作流：

```text
核心研究分支定义问题和数据契约
    -> data-collectors 采集、清洗、校验、入库
    -> 核心研究分支构造因果特征并逐层消融
    -> 证明有增量后进入下一批数据
    -> 需要 LLM 深加工时再创建独立事件抽取分支
    -> 模型 2.0 冻结后再创建正式回测分支
```

不允许一次性将宽度、资金、宏观、情绪和重大事件全部加入模型。每一批必须独立验收、独立消融。

## 3. 数据库与存储边界

### 3.1 数据库

所有结构化数据继续存入现有 MySQL 数据库：

```text
zz1000_predict
```

连接配置沿用：

```text
config/db.yaml
```

不新建第二个业务数据库，避免跨库日期对齐和部署复杂度。大型原始响应可选择落盘到 `data/raw/`，但数据库中必须保存其校验值和本地路径。

### 3.2 分层边界

`feature/data-collectors` 只允许写入以下类型表：

- `*_raw`：尽量保留来源原貌的原始记录。
- `*_daily`：标准化后的行情或官方统计记录，不包含模型判断。
- `data_ingestion_log`：采集运行日志。
- `data_quality_daily`：数据完整性检查结果。

采集分支不得写入：

- `market_feature_daily`
- `model_dataset_daily`
- 模型 2.0 特征表
- 预测概率、方向或策略字段

上述派生特征和模型数据只在 `feature/buy-signal-research` 构造。

### 3.3 命名约定

- 日期字段统一为 `DATE`，名称明确区分 `trade_date`、`period_date` 和 `release_date`。
- 时间字段统一存北京时间的无时区 `DATETIME`；外部 UTC 数据在写库前转换为北京时间。
- 数值字段禁止使用带 `%`、逗号、单位文字的字符串。
- 比率统一保存小数，例如 `1.25%` 保存为 `0.0125`。
- 金额保存来源原始币种和原始单位换算后的数值，并用字段标明币种。
- 每张采集表至少包含 `data_source`、`source_url`、`available_time`、`crawl_time`。
- 无法确认的数据写 `NULL`，不得用 `0` 代替缺失。

## 4. 最重要的因果时间规则

模型预测日为 `T` 时，只能使用 `available_time <= T 日信号生成时刻` 的记录。

必须区分：

| 字段 | 含义 |
|---|---|
| `period_date` | 指标描述的经济或统计周期 |
| `trade_date` | 行情所属交易日 |
| `release_time` | 来源正式发布该数值的时间 |
| `available_time` | 本系统认为市场可以使用该信息的最早时间 |
| `crawl_time` | 本系统实际抓取时间 |
| `revision_no` | 同一周期数据的修订序号 |

示例：5 月 CPI 在 6 月 10 日上午发布。它的 `period_date` 是 5 月末，`release_time` 和 `available_time` 是 6 月 10 日实际发布时间。模型不得把该值合并到 5 月的每日样本。

日终行情默认 `available_time` 为交易日收盘后。若模型信号在收盘后生成，可以用于当日信号；若未来改为盘中预测，必须重新定义可用时刻。

## 5. 数据建设顺序

## 第一批：市场宽度、风格行情和资金行为

这是模型 2.0 的第一优先级。原因是数据结构稳定、解释明确、无需 LLM，最适合先判断是否能改善 2026 年近期失误。

### 5.1 全市场个股日行情

用途：由建模分支计算上涨家数比例、下跌家数比例、涨停/跌停、创新高/新低、放量下跌、小盘扩散度等。

当前工程优先使用 AKShare 作为统一数据适配层：

```python
ak.stock_zh_a_hist(
    symbol="000001",
    period="daily",
    start_date="20150101",
    end_date="20260610",
    adjust="",
)
```

数据由 AKShare 对公开行情接口进行适配，入库时 `data_source` 必须记录实际接口和底层来源。交易所或其他官方数据用于字段定义和抽样核对；若 AKShare 接口失效，再启用经过字段、单位和历史范围验证的备选来源。

建议表：`market_stock_daily_raw`

```sql
CREATE TABLE IF NOT EXISTS market_stock_daily_raw (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trade_date DATE NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    stock_name VARCHAR(100),
    open DECIMAL(18,4),
    high DECIMAL(18,4),
    low DECIMAL(18,4),
    close DECIMAL(18,4),
    pre_close DECIMAL(18,4),
    pct_chg DECIMAL(12,8),
    volume DECIMAL(24,4),
    amount DECIMAL(24,4),
    turnover_rate DECIMAL(12,8),
    amplitude DECIMAL(12,8),
    currency VARCHAR(8) DEFAULT 'CNY',
    data_source VARCHAR(50) NOT NULL,
    source_url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_hash CHAR(64),
    UNIQUE KEY uk_stock_daily (trade_date, stock_code),
    KEY idx_stock_daily_code (stock_code, trade_date),
    KEY idx_stock_daily_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

采集要求：

- 覆盖沪、深、北交所 A 股；建模时是否排除北交所由核心分支决定。
- 使用不复权价格，保证当日真实涨跌、成交额、换手率和创新高/新低计算口径一致。
- 股票列表先通过 AKShare A 股代码接口获取，再逐股调用历史行情接口。
- 首次历史回补数据量较大，采集器必须支持按股票断点续采、失败重试和进度记录；日常更新只补最近日期窗口。
- `amount` 等接口未返回或无法核验的字段必须写 `NULL`，不能补零。
- 涨跌停价、ST、停牌、上市日期和历史市值若后续需要，应从可验证的独立接口采集并使用独立字段或表，不要求 `stock_zh_a_hist` 伪造提供。
- 历史回补目标为 2015 年至今；若接口稳定性不足，先完成 2021 年至今并在质量报告中明确。

### 5.2 指数成分历史

用途：计算中证1000成分内部上涨比例、创新低数量和扩散度，防止全市场宽度被大盘股主导。

AKShare 可以获取当前一期中证1000成分和权重快照：

```python
ak.index_stock_cons_csindex(symbol="000852")
ak.index_stock_cons_weight_csindex(symbol="000852")
```

但这些接口不能直接视为完整的历史成分数据库。返回日期主要描述当前批次或快照日期，不代表每日都有一份新成分名单。中证1000成分通常在指数定期调整或临时调整时变化，两个调整日之间沿用同一份有效名单。

历史成分优先根据中证指数有限公司官网的成分调整公告和历史文件重建。AKShare 用于获取当前快照，并从系统开始运行之日起持续保存每期快照。

建议表：`index_constituent_history`

```sql
CREATE TABLE IF NOT EXISTS index_constituent_history (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    index_code VARCHAR(20) NOT NULL,
    stock_code VARCHAR(20) NOT NULL,
    effective_date DATE NOT NULL,
    end_date DATE,
    weight DECIMAL(12,8),
    adjustment_type VARCHAR(30),
    announcement_time DATETIME,
    available_time DATETIME NOT NULL,
    data_source VARCHAR(50) NOT NULL,
    source_url TEXT,
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_index_member (index_code, stock_code, effective_date),
    KEY idx_index_member_date (index_code, effective_date, end_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

预测日期为 `T` 时，建模分支只能选择：

```text
effective_date <= T
并且 end_date 为空或 end_date >= T
并且 announcement_time / available_time <= T 的信号生成时刻
```

禁止用今天的中证1000成分回填历史，否则会产生幸存者偏差。

指数成分历史不阻塞第一批模型 2.0 研究：

1. 第一轮先使用全市场宽度，以及中证2000、国证2000等小盘风格指数的扩散和相对强弱。
2. AKShare 当前成分只用于当前及未来快照，不用于回填过去。
3. 历史调整公告重建完成后，再单独加入“中证1000成分内部宽度”并做消融。

### 5.3 风格指数行情

现有 `market_index_daily` 继续使用，不另建表。统一通过 AKShare 指数历史行情接口获取：

```python
ak.index_zh_a_hist(
    symbol="932000",
    period="daily",
    start_date="20150101",
    end_date="20260610",
)
```

新增以下指数：

| 指数 | 建议代码 | 用途 |
|---|---|---|
| 中证2000 | `932000` | 更小市值扩散和风险偏好 |
| 上证50 | `000016` | 超大盘风格对照 |
| 中证全指 | `000985` | 全市场基准 |
| 国证2000 | `399303` | 小盘风格补充对照 |

开发前先逐一探测接口能否稳定返回历史数据，并与指数官方名称、代码和日期抽样核对。某一补充指数暂时不可用时，只记录失败并跳过，不阻塞其他指数。新增代码只改 `config/symbols.yaml`，不得在采集器中硬编码。

### 5.4 融资融券市场汇总

用途：观察杠杆资金净流入、融资买入强度、融券压力及变化速度。

优先使用 AKShare 获取沪深交易所融资融券市场汇总：

```python
ak.stock_margin_sse(
    start_date="20260101",
    end_date="20260610",
)

ak.stock_margin_szse(date="20260610")
```

上交所接口支持日期区间；深交所接口按交易日循环获取。AKShare 是工程适配入口，字段定义和抽样数值仍需与上交所、深交所官方页面核对。

建议表：`margin_market_daily`

```sql
CREATE TABLE IF NOT EXISTS margin_market_daily (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trade_date DATE NOT NULL,
    exchange VARCHAR(10) NOT NULL,
    financing_balance DECIMAL(24,4),
    financing_buy_amount DECIMAL(24,4),
    securities_lending_balance DECIMAL(24,4),
    securities_lending_sell_volume DECIMAL(24,4),
    margin_balance DECIMAL(24,4),
    currency VARCHAR(8) DEFAULT 'CNY',
    data_source VARCHAR(50) NOT NULL,
    source_url TEXT,
    release_time DATETIME,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_margin_market (trade_date, exchange),
    KEY idx_margin_market_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

沪深两市先分别存储，建模分支再求和。不要在采集表提前计算融资净买入比例或 z-score。

上交所和深交所接口的金额、数量单位可能不同，写库前必须统一：

```text
金额：元
数量：股或份
比例：小数
```

`financing_repay_amount`、`securities_lending_repay_volume` 等汇总接口不能稳定直接提供的字段不进入第一版表结构，不允许通过余额差倒推后伪装为原始数据。

### 5.5 ETF 份额与规模

现有 `market_etf_daily` 只有行情，不足以代表资金流。ETF 份额和历史净值优先通过 AKShare 获取：

```python
# 上交所 ETF 指定日期份额
ak.fund_etf_scale_sse(date="20260610")

# 深交所 ETF 历史份额；长历史按不超过六个月的窗口分段
ak.fund_scale_daily_szse(
    start_date="20260101",
    end_date="20260610",
    symbol="ETF",
)

# 单只 ETF 历史净值
ak.fund_etf_fund_info_em(
    fund="159845",
    start_date="20260101",
    end_date="20260610",
)
```

上交所与深交所份额接口不同，采集器需要按交易所路由。深交所长历史必须分段回补并去重。

建议表：`etf_fund_daily`

```sql
CREATE TABLE IF NOT EXISTS etf_fund_daily (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trade_date DATE NOT NULL,
    etf_code VARCHAR(20) NOT NULL,
    fund_share DECIMAL(24,4),
    unit_nav DECIMAL(18,8),
    accumulated_nav DECIMAL(18,8),
    subscription_status VARCHAR(30),
    redemption_status VARCHAR(30),
    currency VARCHAR(8) DEFAULT 'CNY',
    share_data_source VARCHAR(50),
    nav_data_source VARCHAR(50),
    data_source VARCHAR(50) NOT NULL,
    source_url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_etf_fund_daily (trade_date, etf_code),
    KEY idx_etf_fund_date (trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

采集范围先沿用配置中的中证1000、沪深300、中证500、创业板和科创50 ETF。

以下字段由建模分支计算，不要求采集器直接提供：

```text
ETF估算规模 = 基金份额 × 单位净值
份额变化率
估算资金净流入
折溢价率 = market_etf_daily.close / unit_nav - 1
```

采集分支不直接接受无法核验的“主力净流入”字段。份额、净值和行情发布时间可能不同，必须分别记录来源，并将 `available_time` 设为该行全部已填字段中最晚的实际可用时间；若需要更精确的时间控制，可拆为份额表和净值表。

### 5.6 股指期货合约级行情

现有主力连续行情无法稳定计算期限结构和真实基差。需要保存 IM、IC、IF 各可交易合约。

优先使用 AKShare 的中金所历史日行情接口：

```python
ak.get_futures_daily(
    start_date="20260101",
    end_date="20260610",
    market="CFFEX",
)
```

接口返回指定日期范围内的真实合约日行情。入库前只保留 `IM`、`IC`、`IF` 产品，并保存各月份真实合约。`futures_zh_daily_sina` 仅作为经过字段和历史范围验证后的备选接口。

建议表：`index_futures_contract_daily`

```sql
CREATE TABLE IF NOT EXISTS index_futures_contract_daily (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trade_date DATE NOT NULL,
    product_code VARCHAR(10) NOT NULL,
    contract_code VARCHAR(20) NOT NULL,
    expiry_date DATE,
    open DECIMAL(18,4),
    high DECIMAL(18,4),
    low DECIMAL(18,4),
    close DECIMAL(18,4),
    settle DECIMAL(18,4),
    pre_settle DECIMAL(18,4),
    volume DECIMAL(24,4),
    open_interest DECIMAL(24,4),
    amount DECIMAL(24,4),
    data_source VARCHAR(50) NOT NULL,
    source_url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_future_contract (trade_date, contract_code),
    KEY idx_future_product (product_code, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

不得使用 `IM0`、`IC0`、`IF0` 等连续合约计算真实基差。建模分支负责从真实合约中选择近月/主力合约，并结合 `market_index_daily` 计算基差、年化基差、展期结构和持仓变化。

### 第一批验收

第一批完成后，collect 分支必须输出：

- 各表最早/最晚日期、总行数、交易日覆盖率。
- 每年缺失率和连续缺口。
- 随机抽取至少 20 个日期与官方来源比对。
- 重复键为 0。
- OHLC 逻辑错误为 0，负成交量/成交额为 0。
- `available_time` 非空率 100%。
- 个股日行情每日股票数量突变超过 10% 时报警。
- 当前指数成分快照与中证指数官方名单抽样核对。
- 已重建的历史指数成分必须检查每次调整前后数量、有效日期和公告时间；历史成分未完成时不阻塞第一轮，但禁止生成伪历史成分宽度。

通过后交给 `feature/buy-signal-research` 做第一批特征消融。第一批没有稳定增量时，不急于开发情绪和 LLM。

## 第二批：固定日历、宏观发布和外部风险映射

### 6.1 固定事件日历

用途：构造事件发生前后的因果窗口，不依赖新闻是否被爬到。

范围：

- 中国法定节假日和交易所休市日。
- 两会、中央经济工作会议、政治局会议等可确认日期。
- 美联储、欧洲央行、日本央行议息日。
- 国家统计局、人民银行和海外重要宏观指标计划发布日期。

建议表：`event_calendar_raw`

```sql
CREATE TABLE IF NOT EXISTS event_calendar_raw (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    event_key VARCHAR(160) NOT NULL,
    event_layer VARCHAR(16) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_name VARCHAR(200) NOT NULL,
    country_region VARCHAR(30),
    scheduled_time DATETIME NOT NULL,
    actual_start_time DATETIME,
    actual_end_time DATETIME,
    is_confirmed TINYINT DEFAULT 0,
    importance INT,
    announcement_time DATETIME,
    available_time DATETIME NOT NULL,
    data_source VARCHAR(50) NOT NULL,
    source_url TEXT,
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_event_calendar (event_key, scheduled_time),
    KEY idx_event_calendar_time (scheduled_time),
    KEY idx_event_calendar_layer (event_layer, scheduled_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

`event_layer` 只允许 `cn / geo / macro`。固定日历由结构化来源决定，不交给 LLM 猜测。

### 6.2 宏观实际值、预期值和前值

优先来源：

1. 中国国家统计局发布日程和正式数据。
2. 中国人民银行金融统计和社会融资规模数据。
3. 各国央行、统计机构和 FRED 等官方数据库。
4. 百度经济日历只能作为补充线索；最终值尽量回到官方来源核对。

建议表：`macro_release_raw`

```sql
CREATE TABLE IF NOT EXISTS macro_release_raw (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    indicator_key VARCHAR(100) NOT NULL,
    indicator_name VARCHAR(160) NOT NULL,
    country_region VARCHAR(30) NOT NULL,
    frequency VARCHAR(20),
    period_date DATE NOT NULL,
    release_time DATETIME NOT NULL,
    actual_value DECIMAL(24,8),
    forecast_value DECIMAL(24,8),
    previous_value DECIMAL(24,8),
    revised_previous_value DECIMAL(24,8),
    unit VARCHAR(30),
    seasonal_adjustment VARCHAR(20),
    revision_no INT DEFAULT 0,
    is_preliminary TINYINT DEFAULT 0,
    available_time DATETIME NOT NULL,
    data_source VARCHAR(50) NOT NULL,
    source_url TEXT,
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_hash CHAR(64),
    UNIQUE KEY uk_macro_vintage (indicator_key, period_date, release_time, revision_no),
    KEY idx_macro_release_time (release_time),
    KEY idx_macro_indicator (indicator_key, period_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

首批指标：

- 中国：PMI、CPI、PPI、工业增加值、社零、固定资产投资、出口、M1、M2、社融增量、人民币贷款。
- 美国：CPI、核心 CPI、非农、失业率、ISM PMI、GDP、零售销售、联邦基金目标区间。
- 全球流动性：主要央行政策利率和会议结果。

`forecast_value` 若没有可靠历史来源可为空，不能用事后调查值回填。模型特征中的超预期值只能在 forecast 和 actual 均可用时计算。

### 6.3 外部风险映射行情

建议表：`global_market_daily`

```sql
CREATE TABLE IF NOT EXISTS global_market_daily (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trade_date DATE NOT NULL,
    symbol VARCHAR(30) NOT NULL,
    symbol_name VARCHAR(100),
    asset_class VARCHAR(30) NOT NULL,
    open DECIMAL(24,8),
    high DECIMAL(24,8),
    low DECIMAL(24,8),
    close DECIMAL(24,8),
    settle DECIMAL(24,8),
    volume DECIMAL(24,4),
    currency VARCHAR(8),
    timezone VARCHAR(40),
    data_source VARCHAR(50) NOT NULL,
    source_url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_global_market (trade_date, symbol),
    KEY idx_global_symbol (symbol, trade_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

首批代码：

| 资产 | 建议标识 | 主要意义 |
|---|---|---|
| VIX | `VIX` | 全球风险厌恶 |
| 美元指数 | `DXY` | 美元流动性和人民币压力 |
| 美国10年期收益率 | `US10Y` | 全球无风险利率 |
| 美国2年期收益率 | `US2Y` | 政策利率预期 |
| WTI/布伦特原油 | `WTI/Brent` | 通胀与地缘冲击 |
| COMEX黄金 | `Gold` | 避险与实际利率 |
| 离岸人民币 | `USDCNH` | 中国风险偏好和汇率压力 |
| 恒生科技/纳斯达克 | 来源正式代码 | 成长风格外部映射 |

对于美国官方时间序列，可优先使用 FRED；对交易所品种优先使用交易所或稳定授权数据。不同市场的收盘时间必须转换为北京时间，并明确该收盘值在 A 股当日收盘前还是之后可用。

### 第二批验收

- 每个宏观指标都有 `period_date`、`release_time` 和 `available_time`。
- 修订数据保留 vintage，不覆盖首次发布值。
- 固定事件日历不得用事后确认日期回填为事前已知。
- 全球行情周末和海外节假日缺失属于正常缺失，建模分支只能向后使用最近已知值，不得向前填充未来值。
- 与官方发布抽样核对日期、值和单位。

通过后，核心分支先分别测试“日历窗口”“宏观超预期”“全球风险映射”，不得直接合成总宏观分数。

## 第三批：情绪原始数据

第三批在前两批完成并消融后启动。

### 7.1 股吧

继续复用现有：

- `sentiment_guba_raw`
- `sentiment_guba_comment_raw`

不删除现有代码。采集分支先解决历史连续性、验证码、发布时间、阅读/评论/点赞字段可靠性，不负责给文本打情绪分。

### 7.2 雪球及其他社区

只有在访问方式、合规性、历史覆盖和稳定性验证通过后才加入。不得把登录 Cookie、账号密码写入代码或仓库。

建议统一表：`sentiment_social_raw`

```sql
CREATE TABLE IF NOT EXISTS sentiment_social_raw (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    content_id VARCHAR(160) NOT NULL,
    source VARCHAR(30) NOT NULL,
    topic VARCHAR(100),
    author_id_hash CHAR(64),
    publish_time DATETIME NOT NULL,
    trade_date DATE NOT NULL,
    title TEXT,
    content MEDIUMTEXT,
    read_count BIGINT,
    comment_count BIGINT,
    like_count BIGINT,
    share_count BIGINT,
    url TEXT,
    available_time DATETIME NOT NULL,
    crawl_time DATETIME DEFAULT CURRENT_TIMESTAMP,
    raw_hash CHAR(64),
    UNIQUE KEY uk_social_content (source, content_id),
    KEY idx_social_trade_date (trade_date, source)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 7.3 财经媒体和热点题材

继续使用 `news_raw` 保存文章级原文，同时补充以下字段：

```text
language
country_region
source_type
first_seen_time
last_seen_time
canonical_url
content_hash
is_reprint
original_news_id
```

跨来源转载去重和事件聚类不在采集分支完成，只保存足够的去重依据。

### 第三批验收

- 每个来源按月报告抓取天数、内容数、互动字段缺失率。
- 同内容重复率和转载率可计算。
- 发布时间不得以抓取时间代替；无法获得发布时间的记录不进入日频模型。
- 异常爆量需要区分真实热点和采集器重复。
- 情绪模型启用前继续保持 `use_guba_sentiment: false`。

## 第四批：三类重大事件原始数据

采集分支只负责获取新闻、公告、日历和官方声明，不负责最终事件方向和影响分数。

### 8.1 中国事件 `cn`

优先来源：国务院、中国政府网、人民银行、证监会、交易所、国家统计局、财政部、发改委及权威政策发布渠道。

重点类型：货币、财政、房地产、资本市场监管、IPO/减持/融券/量化规则、产业政策和重大会议。

### 8.2 地缘政治 `geo`

优先来源：外交部、商务部、联合国、各国政府和国际组织公开发布，以及可合法稳定采集的权威媒体。

重点区域：中美、台海与亚太、俄乌、中东与红海。采集时保存来源国家、发布时间、正文、原始 URL，不在 collect 分支判断“风险升级”。

### 8.3 全球宏观 `macro`

优先来源：各国央行、统计机构、IMF、世界银行、FRED 和结构化经济日历。

### 8.4 统一新闻来源表扩展

建议继续扩展 `news_raw`，新增：

```text
event_layer_hint      cn / geo / macro / unknown
publisher_country
document_type         news / policy / speech / statement / calendar
official_source       0 / 1
source_priority       1-5
```

`event_layer_hint` 只允许根据来源和关键词做粗分流，不是最终 LLM 分类。

## 9. 未来 LLM 事件抽取分支的数据契约

`feature/llm-event-extraction` 尚未创建。需要启动时必须先让用户确认。

该分支读取：

- `news_raw`
- `event_calendar_raw`
- `macro_release_raw`

该分支输出建议表：`event_structured_v2`

```sql
CREATE TABLE IF NOT EXISTS event_structured_v2 (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    event_id VARCHAR(160) NOT NULL,
    event_date DATE NOT NULL,
    event_time DATETIME,
    event_layer VARCHAR(16) NOT NULL,
    event_type VARCHAR(50) NOT NULL,
    event_stage VARCHAR(30),
    country_region VARCHAR(30),
    importance INT,
    direction VARCHAR(16),
    surprise_score DECIMAL(12,8),
    escalation_score DECIMAL(12,8),
    china_relevance DECIMAL(12,8),
    energy_relevance DECIMAL(12,8),
    supply_chain_relevance DECIMAL(12,8),
    affected_assets JSON,
    source_count INT,
    source_ids JSON,
    cluster_key VARCHAR(160),
    model_name VARCHAR(80),
    prompt_version VARCHAR(40),
    confidence DECIMAL(12,8),
    available_time DATETIME NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_event_v2 (event_id),
    KEY idx_event_v2_date (event_date, event_layer),
    KEY idx_event_v2_available (available_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

禁止覆盖旧 `event_daily`。2.0 使用新表，便于对照 1.0 和回滚。

## 10. 通用采集运行日志与质量表

### 10.1 采集日志

```sql
CREATE TABLE IF NOT EXISTS data_ingestion_log (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    run_id VARCHAR(64) NOT NULL,
    dataset_name VARCHAR(80) NOT NULL,
    data_source VARCHAR(50),
    start_time DATETIME NOT NULL,
    end_time DATETIME,
    requested_start_date DATE,
    requested_end_date DATE,
    fetched_rows INT DEFAULT 0,
    inserted_rows INT DEFAULT 0,
    updated_rows INT DEFAULT 0,
    skipped_rows INT DEFAULT 0,
    error_rows INT DEFAULT 0,
    status VARCHAR(20) NOT NULL,
    error_message TEXT,
    code_version VARCHAR(64),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_ingestion_run (run_id, dataset_name),
    KEY idx_ingestion_dataset (dataset_name, start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

### 10.2 数据质量日报

```sql
CREATE TABLE IF NOT EXISTS data_quality_daily (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    check_date DATE NOT NULL,
    dataset_name VARCHAR(80) NOT NULL,
    metric_name VARCHAR(80) NOT NULL,
    metric_value DECIMAL(24,8),
    expected_min DECIMAL(24,8),
    expected_max DECIMAL(24,8),
    status VARCHAR(20) NOT NULL,
    details JSON,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_quality_metric (check_date, dataset_name, metric_name),
    KEY idx_quality_status (check_date, status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

每个采集器必须支持：

- 明确的 `--start-date` 和 `--end-date`。
- 幂等 upsert。
- 单来源失败不写伪造值。
- 原始字段映射校验。
- 运行日志和质量检查。
- 可重复回补，不依赖只能获取“今天”的快照接口来伪造历史。

## 11. 跨分支执行计划

### 步骤 0：数据契约确认

分支：`feature/data-collectors`

工作：

- 用户确认本文的批次、来源和表结构。
- 将新表 DDL 加入独立 SQL migration，不直接破坏旧表。
- 先做来源探测脚本，确认历史起始日期、字段、频率和限流。

交付给核心分支：本文和来源可用性报告。

### 步骤 1：第一批采集开发

分支：`feature/data-collectors`

顺序：

1. 风格指数补充。
2. 全市场个股日行情。
3. 沪深融资融券汇总。
4. ETF 份额与净值。
5. IM/IC/IF 真实合约级行情。
6. AKShare 中证1000当前成分快照保存。
7. 中证1000历史调整公告重建，作为非阻塞并行任务。
8. 质量报告和回补命令。

完成条件：除完整历史指数成分外，第一批验收全部通过，推送分支并向核心分支提供表范围和质量摘要。历史成分未完成时，核心分支先使用全市场宽度和小盘风格指数，不使用伪历史成分名单。若后续取得可靠的历史每日流通市值，再单独增加市值分组扩散特征。

### 步骤 2：第一批建模消融

分支：`feature/buy-signal-research`

工作：

- 从第一批原始表构造宽度、风格、资金和期货基差特征。
- 所有合并使用因果时间。
- 依次测试：1.0 基线、加宽度、加资金、加期货、交叉组合。
- 继续使用 2021-2026 逐年滚动验证、独立信号、同期天然基线。

保留条件：多数年度稳定改善 Buy precision 或平均未来收益，同时不过度压缩 signal count。

输出给 collect 分支：字段有效性、缺失问题、是否需要扩大历史范围。

### 步骤 3：第二批采集开发

分支：`feature/data-collectors`

工作：固定事件日历、宏观 vintage、外部风险行情。第一批建模结果出来后再开始完整开发。

### 步骤 4：第二批建模消融

分支：`feature/buy-signal-research`

工作：分别验证日历窗口、宏观超预期和全球风险，不提前合成单一宏观分数。

### 步骤 5：第三/四批原始数据采集

分支：`feature/data-collectors`

工作：补齐情绪来源和三层事件原始文本。此阶段只负责原文和元数据。

### 步骤 6：事件抽取升级

分支：规划中的 `feature/llm-event-extraction`

前置条件：用户明确确认创建分支。

工作：去重、聚类、`cn/geo/macro` 分类、方向、重要性、阶段、升级程度和影响对象。优先复用已有历史结果，不重新对全历史盲目调用 LLM。

### 步骤 7：模型 2.0 最终研究

分支：`feature/buy-signal-research`

工作：

- 三类事件独立日频特征和衰减窗口。
- 分层消融和交互组合。
- 重新确定 feature manifest、模型参数和信号阈值。
- 冻结新的 `model_version` 和 `feature_version`，保留 1.0 可回滚。

### 步骤 8：正式回测

分支：规划中的 `feature/backtest-engine`

前置条件：模型 2.0 特征和信号格式冻结，且用户明确确认创建分支。

## 12. 建模分支必须遵守的接口规则

核心分支读取采集表时：

- 只按 `available_time` 做 as-of 合并。
- 不修改原始表。
- 不把缺失值一律填 0。
- 不用当前成分股回溯历史。
- 不覆盖宏观初值和修订值。
- 同一新数据组必须能一键开关，便于消融。
- 每个特征记录 `source_table`、计算公式、窗口和最早可用日期。
- 模型 2.0 使用新的特征清单，不能修改 `config/buy_features_v1.json`。

## 13. 来源稳定性与降级原则

当前工程的实现和校验优先级：

```text
AKShare 统一适配入口
    + 官方网页/API用于字段定义和抽样核对
    > 经验证的第二 AKShare 接口或官方直连备选
    > 经验证的公开财经网站接口
```

使用 AKShare 不代表忽略底层来源。每个采集器必须记录 AKShare 函数名、AKShare 版本、底层数据来源、字段单位和抽样核对结果。若 AKShare 升级导致字段变化，质量检查必须阻止静默写入。

若主要来源失效：

1. 记录失败，不写 0。
2. 尝试已配置备选来源。
3. 比对字段单位和定义后才允许切换。
4. 保存 `data_source`，不同来源的数据不能静默混合。
5. 历史与增量来源变化时，在质量报告标记结构断点。

不建议第一阶段采集“主力资金净流入”类无法核验的供应商算法字段。优先采集可复算的成交、份额、融资和期货持仓原始数据。

## 14. 当前明确不做

- 不在 collect 分支训练模型或调整 Buy 阈值。
- 不修改模型 1.0 冻结文件。
- 不启用 short 模型。
- 不创建回测分支。
- 不立即创建 LLM 事件抽取分支。
- 不重复进行全历史大规模 LLM 抽取。
- 不因字段缺失伪造 `amount`、forecast、发布时间或资金流。

## 15. 第一轮实际开发建议

用户进入 collect 分支后，建议只下达以下任务：

```text
按照 docs/model_2_0_data_collection_design.md 开发模型2.0第一批数据采集。
先实现来源探测、DDL migration 和质量检查，再按风格指数、全市场个股日行情、
融资融券、ETF份额与净值、股指期货真实合约级行情的顺序开发。以上数据优先通过
AKShare获取，并与官方来源抽样核对。AKShare中证1000当前成分快照需要保存；完整
历史成分公告重建作为并行非阻塞任务，未完成前禁止用当前名单回填历史。
不要开发特征和模型，不要修改模型1.0文件。每完成一种数据，报告历史范围、缺失率、
字段单位、来源抽样核对和可重复运行结果。
```

第一批采集验证并提交后，再回到 `feature/buy-signal-research` 开始模型 2.0 第一轮消融。

## 16. 参考来源

- 国家统计局发布日程：https://www.stats.gov.cn/sj/fbrc/
- 国家统计局数据：https://data.stats.gov.cn/
- 中国人民银行：https://www.pbc.gov.cn/
- 上海证券交易所融资融券汇总：https://www.sse.com.cn/market/othersdata/margin/sum/
- 深圳证券交易所：https://www.szse.cn/
- 中国金融期货交易所历史数据：https://www.cffex.com.cn/cn/lssjxz.html
- 中证指数有限公司：https://www.csindex.com.cn/
- AKShare 股票数据：https://akshare.akfamily.xyz/data/stock/stock.html
- AKShare 指数数据：https://akshare.akfamily.xyz/data/index/index.html
- AKShare 公募基金和 ETF 数据：https://akshare.akfamily.xyz/data/fund/fund_public.html
- AKShare 期货数据：https://akshare.akfamily.xyz/data/futures/futures.html
- FRED：https://fred.stlouisfed.org/

外部页面和接口可能变化。开发采集器时必须再次验证实际字段、日期范围和使用条件，不能只依据本文中的名称硬编码。
