# Bottom 特征迁入 Buy 主线实验

本文记录从 `feature/bottom-fishing-v1` 迁入底部形态特征、周线底部确认特征和顶部风险标签后的第一轮 Buy 主线实验。

## 迁入内容

新增特征模块：

- `src/features/build_manual_bottom_features.py`

迁入的特征分三类：

1. 日线底部形态特征
   - 回撤：`bf_drawdown_20d`、`bf_drawdown_60d`、`bf_drawdown_120d`
   - 超跌：`bf_rsi6`、`bf_wr14`、`bf_boll_position`
   - 趋势压力：`bf_below_ma20_days`、`bf_ma20_gap`、`bf_ma20_slope_5d`
   - K线压力/修复：`bf_lower_shadow_ratio`、`bf_long_lower_shadow`、`bf_gap_down`
   - 底部候选评分：`bf_candidate_score`、`bf_is_candidate`

2. 周线底部确认特征
   - 周线回撤：`bf_week_drawdown_13w`、`bf_week_drawdown_26w`
   - 周线 BOLL：`bf_week_boll_position`、`bf_week_boll_lower_reclaim`
   - 周线 KDJ：`bf_week_kdj_oversold`、`bf_week_kdj_golden_cross`
   - 周线确认分：`bf_week_bottom_confirmation_score`

3. 顶部风险标签
   - `manual_sell_label`
   - 同步训练 `manual_top_proba`，用于观察顶部风险过滤效果。

所有 bottom 特征使用 `bf_` 前缀，避免覆盖 Buy 1.0 的同名字段。

## 实验设置

- 标签：人工买入窗口 `manual_buy_label`
- 顶部风险：人工顶部/卖出窗口 `manual_sell_label`
- 验证：2021-2026 逐年滚动验证
- 去重：20 个交易日内只保留第一条信号
- 收益评价：未来 20 个交易日收益

测试三种特征组合：

| 方案 | 含义 |
| --- | --- |
| `buy_v1` | 只用 Buy 1.0 的 211 个特征 |
| `bottom_only` | 只用迁入的 bottom 日线 + 周线特征 |
| `buy_plus_bottom_daily` | Buy 1.0 + bottom 日线特征，不含周线 |
| `buy_plus_bottom` | Buy 1.0 + bottom 日线 + 周线特征 |

## 关键结果

按平均未来 20 日收益排序，表现较好的组合如下：

| 方案 | 阈值 | 信号数 | 人工买点命中率 | 顶部误伤率 | 未来20日上涨率 | 平均未来20日收益 | 最差未来20日收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `buy_plus_bottom` | 0.70 | 10 | 10.00% | 30.00% | 60.00% | 2.7016% | -11.0140% |
| `buy_plus_bottom` | 0.50 | 11 | 18.18% | 27.27% | 54.55% | 2.5670% | -11.0140% |
| `buy_plus_bottom_daily` | 0.60 | 12 | 16.67% | 25.00% | 58.33% | 2.0411% | -11.0140% |
| `buy_plus_bottom_daily` | 0.50 | 12 | 25.00% | 25.00% | 50.00% | 2.0187% | -11.0140% |
| `buy_v1` | 0.70 | 10 | 20.00% | 30.00% | 50.00% | 1.1427% | -11.0140% |

按人工买点命中率排序，`bottom_only` 最明显：

| 方案 | 阈值 | 信号数 | 人工买点命中率 | 顶部误伤率 | 未来20日上涨率 | 平均未来20日收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `bottom_only` | 0.50 | 9 | 55.56% | 0.00% | 44.44% | -0.5353% |
| `bottom_only` | 0.40 | 10 | 50.00% | 10.00% | 40.00% | -0.7416% |
| `bottom_only` | 0.60 | 7 | 42.86% | 0.00% | 42.86% | -1.5075% |
| `bottom_only` | 0.70 | 5 | 40.00% | 0.00% | 60.00% | -0.1157% |

## 结论

这次结果说明：

1. Bottom 特征确实学到了人工底部窗口。
   - `bottom_only` 的人工买点命中率最高可到 55.56%。
   - 顶部误伤率也很低。

2. 但 bottom-only 不等于赚钱信号。
   - 虽然更贴近人工底部标签，但平均未来 20 日收益为负。
   - 说明“像底部”不等于“马上能涨”。

3. `buy_plus_bottom` 才是当前更有价值的方向。
   - 它牺牲了一些人工买点命中率，但显著提高了平均未来 20 日收益。
   - 最好组合是 `buy_plus_bottom` 阈值 0.50-0.70。

4. 周线底部确认没有拖累，反而在收益排序里 `buy_plus_bottom` 优于 `buy_plus_bottom_daily`。

5. 顶部风险模型这轮没有明显过滤效果。
   - 在当前阈值 0.40/0.50/0.60 下，大部分信号没有被有效过滤。
   - 后续需要单独做顶部风险阈值敏感性，而不是直接采用。

## 当前判断

不能把 bottom-only 作为 Buy 2.0 主买入模型。

更合理的 2.0 结构是：

- Buy 1.0 / 政治增强：判断“未来是否容易涨”
- Bottom 特征：判断“当前位置是否接近人工认可的低位区”
- 海外风险：判断“普通信号是否需要刹车”
- 顶部风险：后续单独调阈值，作为风险提示或过滤候选

## 下一步

建议下一步做：

1. 固定候选方案：`buy_plus_bottom`。
2. 对 `buy_plus_bottom` 做逐年分析，确认 2.0 改善是否集中在个别年份。
3. 单独测试顶部风险 `manual_top_proba` 的更低阈值，例如 0.10、0.20、0.30。
4. 再把 `buy_plus_bottom` 与政治事件增强、海外风险过滤组合。

## 产物位置

- 底部特征模块：`src/features/build_manual_bottom_features.py`
- 人工标签滚动训练脚本：`src/modeling/walk_forward_manual_label_buy_lgbm.py`
- bottom-only 结果：`data/reports/manual_label_buy_bottom_features_summary.csv`
- Buy+bottom 全量结果：`data/reports/manual_label_buy_plus_bottom_features_summary.csv`
- Buy+bottom 日线结果：`data/reports/manual_label_buy_plus_bottom_daily_features_summary.csv`
