# 20日机会标签与三阶段模型实验

本文记录 Buy 2.0 按新交易目标重构标签后的第一轮实验。

## 为什么重做标签

原 Buy 1.0 标签是：

```text
future_ret_7d > 0.5%
```

这个标签的问题是：

- 只看第 7 天结果，不看中间有没有可交易上涨。
- 不看买入后是否先大跌。
- 与当前更关注的 20 日交易机会不一致。

因此本轮改成更贴近交易目标的 20 日机会标签。

## 新标签定义

新增三个路径指标：

```text
future_ret_20d = 20 个交易日后的收盘收益
future_mfe_20d = 未来 20 个交易日内最大上涨幅度
future_mae_20d = 未来 20 个交易日内最大下跌幅度
```

机会标签：

```text
opportunity_label_20d = 1
当且仅当：
future_mfe_20d >= 3%
并且
future_mae_20d > -3%
```

风险标签：

```text
risk_label_20d = 1
当 future_mae_20d <= -3%
```

位置标签继续使用人工标记：

```text
manual_buy_label = 人工底部买入窗口
```

## 模型结构

本轮不再用一个模型直接决定买入，而是拆成三层：

| 层 | 学什么 | 输出 |
| --- | --- | --- |
| 机会模型 | 未来 20 日是否有可交易上涨空间且不先大跌 | `opportunity_proba` |
| 位置模型 | 当前是否接近人工认可的底部买点 | `position_proba` |
| 风险模型 | 未来 20 日是否容易先大跌超过 3% | `risk_proba` |

测试规则：

- `opportunity_only`：只看机会概率。
- `opportunity_plus_position`：机会概率 + 位置概率。
- `opportunity_risk_filtered`：机会概率 + 风险过滤。
- `three_stage`：机会概率 + 位置概率 + 风险过滤。

## 核心结果

### 信号数量最多的方向

机会模型单独使用后，信号数量明显增加：

| 特征 | 规则 | 阈值 | 信号数 | 机会命中率 | 20日上涨率 | 平均20日收益 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `buy_v1` | `opportunity_only` | 0.20 | 54 | 31.48% | 51.85% | 0.7120% |
| `buy_v1` | `opportunity_only` | 0.30 | 49 | 32.65% | 53.06% | 0.9945% |
| `buy_v1` | `opportunity_only` | 0.50 | 42 | 38.10% | 57.14% | 1.1070% |
| `buy_plus_bottom` | `opportunity_only` | 0.50 | 42 | 38.10% | 54.76% | 0.8696% |

结论：

- 新机会标签确实解决了一部分“信号太少”的问题。
- 信号可以从原来 10-20 条提升到 40-50 条。
- 但机会模型单独使用，平均收益仍然不高，只有约 0.7%-1.1%。

### 收益最好的方向

加入位置模型后，信号变少，但收益明显提高：

| 特征 | 规则 | 机会阈值 | 位置阈值 | 风险阈值 | 信号数 | 20日上涨率 | 平均20日收益 | 最大有利波动 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `buy_plus_bottom` | `three_stage` | 0.20 | 0.20 | 0.50 | 11 | 63.64% | 4.2211% | 8.5084% |
| `buy_plus_bottom` | `opportunity_plus_position` | 0.30 | 0.20 | 无 | 10 | 60.00% | 4.0350% | 8.8409% |
| `buy_plus_bottom` | `three_stage` | 0.20 | 0.20 | 0.40 | 10 | 60.00% | 3.4634% | 8.0019% |
| `buy_plus_bottom` | `opportunity_plus_position` | 0.20 | 0.20 | 无 | 13 | 61.54% | 3.2766% | 8.0202% |

结论：

- `buy_plus_bottom` 的三阶段结构确实比单纯机会模型更接近可交易信号。
- 平均 20 日收益可以提升到 3%-4%。
- 但代价是信号又变少，回到 10-13 条。

## 逐年稳定性

当前最佳组合：

```text
feature_variant = buy_plus_bottom
rule = three_stage
opportunity_threshold = 0.20
position_threshold = 0.20
risk_threshold = 0.50
```

逐年表现：

| 年份 | 信号数 | 机会命中率 | 20日上涨率 | 平均20日收益 |
| --- | ---: | ---: | ---: | ---: |
| 2021 | 0 | 无 | 无 | 无 |
| 2022 | 2 | 0.00% | 0.00% | -3.4195% |
| 2023 | 1 | 100.00% | 100.00% | 0.1488% |
| 2024 | 4 | 25.00% | 50.00% | 5.6196% |
| 2025 | 2 | 100.00% | 100.00% | 10.4561% |
| 2026 | 2 | 50.00% | 100.00% | 4.8662% |

结论：

- 2024-2026 表现较好。
- 2022 仍然会踩坑。
- 2021 没有信号。
- 所以它还不能封版，只能作为 2.0 候选结构。

## 关键判断

这次实验把问题拆清楚了：

1. `opportunity_only` 能增加信号数量，但收益不够高。
2. `position_proba` 能明显提高收益，但会大幅减少信号。
3. 当前 `risk_proba` 还不够好，很多被选中的信号实际 `risk_rate` 仍然偏高。
4. `buy_plus_bottom` 比单纯 `buy_v1` 更适合做高收益候选。

## 下一步

下一步不应该继续加复杂模型，而应该专门处理两个矛盾：

1. 信号数量和收益的矛盾：
   - 用 `opportunity_only` 做观察池。
   - 用 `position_proba` 做加仓或强信号分层。

2. 风险模型不准的问题：
   - 单独重构 `risk_label_20d`。
   - 不只用 `future_mae_20d <= -3%`，还要区分“先跌后涨”和“持续下跌”。
   - 重点解决 2022 这种环境下的错误信号。

推荐 2.0 暂定结构：

| 层 | 输出 |
| --- | --- |
| 机会池 | `watch_long` |
| 机会 + 位置 | `long` |
| 机会 + 位置 + 低风险 | `strong_long` |
| 高风险 | `risk_warning` 或 `blocked` |

## 产物位置

- 评估脚本：`src/modeling/walk_forward_opportunity_20d_lgbm.py`
- Buy 1.0 特征结果：`data/reports/opportunity_20d_buy_v1_summary.csv`
- Buy + bottom 特征结果：`data/reports/opportunity_20d_buy_plus_bottom_summary.csv`
- Buy + bottom 明细：`data/reports/opportunity_20d_buy_plus_bottom_details.csv`
