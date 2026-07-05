# buy-signal-research 数据交接说明

本文档用于把 `feature/data-collectors` 已完成的数据采集结果，交接给 `feature/buy-signal-research` 做模型 2.0 建模开发。

## 1. 交接范围

当前交接的是模型 2.0 第一批已完成的数据采集，包含以下数据域：

1. 来源探测与质量检查基础设施
2. 全市场个股日行情
3. 风格指数行情
4. 沪深融资融券汇总
5. ETF 份额与净值
6. IM / IC / IF 真实合约级行情
7. 中证1000 当前成分快照
8. 中证1000 历史成分月度快照

本批数据已经按“只做原始数据，不做特征和模型”的边界完成。

## 2. 这批数据的使用边界

- `feature/data-collectors` 只负责采集、清洗、时间戳校验、入库和质量检查。
- 不在采集分支里做特征拼接、标签构造、模型训练、阈值策略和信号解释。
- 不把当前成分名单回填成伪历史。
- 不把公告爬虫作为中证1000历史主路径；中证1000历史已经切到 Tushare 月度快照。

## 3. 主数据表

### 3.1 `market_stock_daily_raw`

用途：全市场个股日行情。

关键字段：

- `trade_date`
- `stock_code`
- `exchange`
- `open`
- `high`
- `low`
- `close`
- `pre_close`
- `pct_chg`
- `volume`
- `amount`
- `turnover_rate`
- `amplitude`
- `currency`
- `data_source`
- `source_url`
- `available_time`
- `crawl_time`

说明：

- 使用不复权口径。
- 以 A 股主板、创业板、科创板、北交所等可验证市场为主。
- `amount` 等字段若原始接口缺失，不在建模侧伪造。

### 3.2 `market_index_daily_raw` / 风格指数相关表

用途：沪深300、500、1000、2000、创业板、科创等风格基准行情。

要求：

- 保持原始来源字段
- 明确单位
- 明确 `available_time`
- 用于后续风格相对强弱和市场宽度特征

### 3.3 `market_margin_raw`

用途：融资融券汇总。

主要字段：

- `trade_date`
- `exchange`
- `financing_balance`
- `financing_purchase_amount`
- `securities_lending_balance`
- `securities_lending_sell_volume`
- `margin_balance`
- `data_source`
- `source_url`
- `available_time`

说明：

- 上交所口径和深交所口径单位不同，采集层已按来源做单位保留或换算说明。
- 建模侧只能使用同口径归一后的字段，不能混用原始“亿元/元/股”口径。

### 3.4 `etf_fund_daily_raw`

用途：ETF 份额与净值。

主要字段：

- `trade_date`
- `etf_code`
- `fund_name`
- `fund_share`
- `unit_nav`
- `accumulated_nav`
- `subscription_status`
- `redemption_status`
- `data_source`
- `source_url`
- `available_time`

说明：

- 份额和净值分来源采集，再在采集层合并。
- `available_time` 取份额与净值中较晚者。

### 3.5 `index_futures_contract_daily_raw`

用途：IM / IC / IF 真实合约级行情。

主要字段：

- `trade_date`
- `product_code`
- `contract_code`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `open_interest`
- `amount`
- `contract_multiplier`
- `amount_unit`
- `data_source`
- `source_url`
- `available_time`

说明：

- 真实合约级，不做主力连续合约伪拼接。
- 合约级特征由建模侧自行聚合。

### 3.6 `index_constituent_snapshot_raw`

用途：中证1000 当前成分快照。

来源：

- Tushare `index_weight`

关键字段：

- `snapshot_id`
- `snapshot_date`
- `index_code`
- `stock_code`
- `stock_name`
- `exchange`
- `weight`
- `weight_unit`
- `source_effective_date`
- `constituent_file_date`
- `weight_file_date`
- `data_source`
- `source_url`
- `available_time`
- `crawl_time`
- `raw_hash`

说明：

- 当前成分快照从 Tushare 获取，不再走公告爬虫主路径。
- `weight` 已转成 decimal 小数。
- 当前快照写入后可用于后续校验和抽样核对。

### 3.7 `index_constituent_history`

用途：中证1000 历史成分月度快照。

来源：

- Tushare `index_weight`

关键字段：

- `index_code`
- `stock_code`
- `effective_date`
- `end_date`
- `weight`
- `adjustment_type`
- `announcement_time`
- `available_time`
- `data_source`
- `source_url`
- `crawl_time`

说明：

- 这是建模侧真正要用的历史成分库。
- 按月增量回填。
- 数据已经实际回填到目标历史区间。
- 不再用当前名单回填历史。

## 4. 已完成的数据采集命令

### 4.1 主入口

```powershell
python main_daily_run.py --step collect_zz1000_snapshot
python main_daily_run.py --step collect_zz1000_history_backfill
```

### 4.2 历史成分按区间采集

```powershell
python -m src.collectors.collect_zz1000_history_tushare --start-date 2024-10-01 --end-date 2025-06-30
```

### 4.3 历史成分按月增量回填

```powershell
python -m src.collectors.collect_zz1000_history_tushare --backfill --start-date 2024-10-01 --end-date 2025-06-30
```

## 5. 时间口径

- `available_time`：采集层认为数据“可用于建模”的最早时间。
- `crawl_time`：实际抓取/请求时间。
- 日行情默认以收盘后可用为准。
- 中证1000 历史成分采用月末快照口径，建模侧不要把它误解为逐日变更。

## 6. 质量约束

这批数据已经过基础质量检查，核心约束包括：

- 交易日连续性和覆盖率
- 关键字段非空
- 单位一致性
- 中证1000 当前成分快照 1000 只成分
- 中证1000 月度历史快照每月 1000 只成分
- 同键 upsert，不产生重复脏行

## 7. 建模侧建议

`feature/buy-signal-research` 后续建议优先使用这些数据构造：

- 市场宽度
- 风格相对强弱
- 资金行为
- ETF 流动性和偏好
- 期货基差和风险偏好
- 中证1000 内部成分扩散和内部宽度

不要在采集分支里构造：

- 模型标签
- 特征版本
- 信号阈值
- 回测净值

## 8. 还没做、但不影响这一批采集完成的内容

- 自动跳过已入库月份的增量优化
- 更细的历史抽样核对报告
- 模型 2.0 特征层与消融层

## 9. 结论

模型 2.0 第一批数据采集已经完成，且中证1000 当前成分与历史成分都已切换为 Tushare 主路径。

建模侧可以直接在 `feature/buy-signal-research` 上接这些表做特征开发。
