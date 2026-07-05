# XGBoost Top Short-Only Strategy Analysis

本文档记录“只卖做空不做多”策略的第一版验证结果。

## 1. 策略定义

该策略只做空，不做多。

开空条件：

```text
XGBoost 顶部信号 = 1
且 BP 底部信号 = 0
且不在剔除区间
且满足冷却/仓位规则
```

仓位规则：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `first_short_pct` | 0.50 | 首次开空 50% 权益 |
| `add_short_pct` | 0.25 | 加空每次 25% 权益 |
| `max_short_pct` | 0.75 | 最大空头敞口 75% |
| `add_window_days` | 5 | 首次开空后 5 个交易日内可加空 |
| `cooldown_days` | 3 | 平仓后 3 个交易日冷却 |
| `stop_loss` | -0.06 | 空头亏损达到 -6% 平仓 |
| `take_profit` | 0.15 | 空头盈利达到 +15% 平一半 |
| `max_holding_days` | 45 | 最长持仓 45 个交易日 |
| `bottom_cover_pct` | 0.50 | BP 底部首日平一半 |

平空优先级：

1. 空头硬止损：收益小于等于 -6%，全平。
2. 空头硬止盈：收益大于等于 +15%，平一半。
3. indicator_cover：技术指标反弹，全平。
4. BP 底部信号：首日平一半，连续底部全平。
5. 最长持仓：超过 45 个交易日全平。

indicator_cover 条件：

```text
cci_turn_up_above_minus_100
kdj_turn_up
rsi_turn_up
kdj_golden_cross
rsi_oversold
```

## 2. 评估脚本

脚本：

```text
src/analysis/backtest_xgb_top_short_strategy.py
```

默认跑四组口径：

```powershell
python -m src.analysis.backtest_xgb_top_short_strategy --run-default-cases
```

输出：

```text
data/reports/xgb_top_short_strategy_summary_compare.csv
data/reports/xgb_top_short_strategy_summary_all_no_cost.csv
data/reports/xgb_top_short_strategy_trades_all_no_cost.csv
data/reports/xgb_top_short_strategy_equity_all_no_cost.csv
data/reports/xgb_top_short_strategy_summary_exclude_no_cost.csv
data/reports/xgb_top_short_strategy_trades_exclude_no_cost.csv
data/reports/xgb_top_short_strategy_equity_exclude_no_cost.csv
data/reports/xgb_top_short_strategy_summary_all_cost.csv
data/reports/xgb_top_short_strategy_trades_all_cost.csv
data/reports/xgb_top_short_strategy_equity_all_cost.csv
data/reports/xgb_top_short_strategy_summary_exclude_cost.csv
data/reports/xgb_top_short_strategy_trades_exclude_cost.csv
data/reports/xgb_top_short_strategy_equity_exclude_cost.csv
```

## 3. 核心结果

评估区间：`2022-01-04` 到 `2026-06-26`。

| 口径 | 总收益 | 最大回撤 | 已平仓 | 胜率 | 平均单笔 | 中位数单笔 | 最好单笔 | 最差单笔 | 平均持仓 | 平均空头敞口 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 全区间，无成本 | +7.88% | -3.73% | 21 | 57.14% | +0.48% | +0.12% | +6.17% | -2.89% | 3.10 天 | 3.85% |
| 剔除 2023-12~2024-01，无成本 | +7.88% | -3.73% | 21 | 57.14% | +0.48% | +0.12% | +6.17% | -2.89% | 3.10 天 | 3.85% |
| 全区间，含成本 | +5.45% | -4.33% | 21 | 47.62% | +0.32% | -0.05% | +6.02% | -3.05% | 3.10 天 | 3.86% |
| 剔除 2023-12~2024-01，含成本 | +5.45% | -4.33% | 21 | 47.62% | +0.32% | -0.05% | +6.02% | -3.05% | 3.10 天 | 3.86% |

剔除区间前后结果完全相同，说明当前 short-only 策略在 `2023-12-01` 到 `2024-01-31` 没有产生交易。

## 4. 年份拆分

含成本全区间交易按开空年份统计：

| 年份 | 交易数 | 胜率 | 平均单笔 | 简单收益合计 |
| --- | ---: | ---: | ---: | ---: |
| 2022 | 4 | 50.00% | +0.76% | +3.06% |
| 2023 | 7 | 57.14% | +0.85% | +5.98% |
| 2024 | 4 | 50.00% | +0.31% | +1.25% |
| 2025 | 4 | 50.00% | -0.22% | -0.89% |
| 2026 | 2 | 0.00% | -1.38% | -2.75% |

主要贡献来自 2022 和 2023。2025、2026 开始转弱。

## 5. 结果判断

当前 short-only 策略可以作为“顶部风险对冲候选”，不适合作为独立主策略。

理由：

1. 收益为正，但总收益只有 +5.45% 到 +7.88%，显著低于当前多头策略。
2. 回撤很小，说明它没有大规模误伤，但也说明平均空头敞口很低，资金利用率不足。
3. 含成本后胜率从 57.14% 降到 47.62%，说明单笔收益太薄，容易被交易成本吃掉。
4. 平均持仓约 3.1 个交易日，更像短线风险释放策略。
5. 2025 和 2026 表现转弱，说明顶部信号在近期未必适合直接做空，更适合减仓/对冲。

## 6. 后续优化方向

1. 做空入场应提高门槛，例如要求 XGBoost 顶部概率更高、连续顶部确认、或叠加 RSI/KDJ/CCI 高位。
2. indicator_cover 当前过于敏感，导致很多交易 1-5 天内平仓，单笔收益较薄。
3. 可以把 short-only 改成“多头策略的风险对冲模块”，只在持有多头时减少净敞口，而不是单独裸空。
4. 可以增加趋势过滤：只在指数处于 MA20/MA60 下方或周线走弱时允许做空。
5. 需要单独画空头交易图，检查 2024-10、2025-09、2026-01 等亏损交易是否来自顶部模型滞后。
