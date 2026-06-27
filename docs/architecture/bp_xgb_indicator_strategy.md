# BP + XGBoost + Indicator Strategy

本文档冻结当前抄底模型阶段版本，用于说明建模思路、评估脚本、交易执行策略和复现实验命令。

## 1. 当前版本定位

当前版本不是单纯的“底部概率展示”，而是一个完整的交易策略验证口径：

1. BP 神经网络负责识别底部买入候选。
2. XGBoost 顶部模型负责过滤明显顶部/风险日期。
3. 技术指标卖出规则负责更快退出，不完全等待顶部模型。
4. 策略允许分批买入和分批止盈。
5. 评估默认关注 2022 年至今，并提供剔除 `2023-12-01` 到 `2024-01-31` 特殊股灾区间后的口径。

这个版本暂作为当前 BP + XGBoost + indicator_stop 策略基线。

## 2. 建模思路

### 2.1 标签来源

人工顶底区域标注按弱监督标签使用，属于 `post_hoc_weak` 形态学习，不直接当作无泄漏实时标签。

滚动训练时每一年只使用此前数据训练：

| 预测年份 | 训练截止 |
| --- | --- |
| 2022 | 2021-12-31 |
| 2023 | 2022-12-30 |
| 2024 | 2023-12-29 |
| 2025 | 2024-12-31 |
| 2026 | 2025-12-31 |

该口径的核心是避免模型在预测某一年时直接学习该年份之后的人工标注。

### 2.2 模型组合

当前策略使用两个模型的不同职责：

| 模型 | 职责 | 输入来源 | 输出 |
| --- | --- | --- | --- |
| BP | 底部买入候选 | 滚动训练预测文件 | `bp_bottom_signal` |
| XGBoost | 顶部/风险过滤 | 滚动训练预测文件 | `xgb_top_signal` |

信号阈值来自 `src/modeling/train_manual_turning_models.py`：

```text
weak_combined_bottom_signal = bottom_proba >= 0.6 and top_proba < 0.5
weak_combined_top_signal    = top_proba >= 0.6 and bottom_proba < 0.5
```

策略买入条件是：

```text
BP 底部信号 = 1
且 XGBoost 顶部信号 = 0
且不在剔除区间
且满足仓位/冷却规则
```

### 2.3 为什么图上的买点少

图上展示的是最终执行交易，不是所有模型原始信号。原始信号会经过以下压缩：

1. XGBoost 顶部信号否决。
2. 已持仓时，同一轮底部信号转为加仓，不新开交易。
3. 5 个交易日内只允许加仓，最高仓位 75%。
4. 卖出后冷却 3 个交易日。
5. 剔除 `2023-12-01` 到 `2024-01-31` 的买卖。
6. 分批止盈会让一次买入对应多次卖出。

因此最终交易图应理解为“策略执行后的买卖结果图”，不是“全部底部识别信号图”。

## 3. 交易执行策略

策略参数定义在 `src/analysis/backtest_bp_xgb_indicator_strategy.py`：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `first_entry_pct` | 0.50 | 首次买入 50% 权益 |
| `add_entry_pct` | 0.25 | 加仓每次 25% 权益 |
| `max_position_pct` | 0.75 | 最大仓位 75% |
| `add_window_days` | 5 | 首次买入后 5 个交易日内可加仓 |
| `cooldown_days` | 3 | 清仓后 3 个交易日冷却 |
| `stop_loss` | -0.06 | 持仓亏损达到 -6% 清仓 |
| `take_profit` | 0.15 | 持仓盈利达到 +15% 卖出一半 |
| `max_holding_days` | 45 | 最长持仓 45 个交易日 |
| `top_reduce_pct` | 0.50 | XGBoost 顶部首日卖出 50% |
| `fee_rate` | 0.0003 | 成本版手续费 |
| `slippage_rate` | 0.0005 | 成本版滑点 |

卖出优先级：

1. 硬止损：收益小于等于 -6%，清仓。
2. 硬止盈：收益大于等于 +15%，卖出一半。
3. indicator_stop：技术指标掉头/过热，清仓。
4. XGBoost 顶部：首日卖一半，连续顶部次日清仓。
5. 最长持仓：超过 45 个交易日清仓。

indicator_stop 触发条件：

```text
cci_turn_down_below_100
kdj_turn_down
rsi_turn_down
kdj_dead_cross
rsi_overbought
```

## 4. 评估脚本

### 4.1 滚动训练模型

```powershell
python -m src.modeling.train_manual_turning_models --model-kind bp --walk-forward --tune
python -m src.modeling.train_manual_turning_models --model-kind xgboost --walk-forward --tune
```

主要输出：

```text
data/reports/manual_weak_turning_walk_forward_predictions_bp_wf.csv
data/reports/manual_weak_turning_walk_forward_predictions_xgboost_wf.csv
data/reports/manual_weak_turning_walk_forward_metrics_bp_wf.csv
data/reports/manual_weak_turning_walk_forward_metrics_xgboost_wf.csv
```

### 4.2 执行策略回测

全区间、无成本：

```powershell
python -m src.analysis.backtest_bp_xgb_indicator_strategy --start-date 2022-01-01
```

剔除 `2023-12-01` 到 `2024-01-31`，含手续费和滑点：

```powershell
python -m src.analysis.backtest_bp_xgb_indicator_strategy `
  --start-date 2022-01-01 `
  --exclude-start 2023-12-01 `
  --exclude-end 2024-01-31 `
  --fee-rate 0.0003 `
  --slippage-rate 0.0005
```

输出默认文件：

```text
data/reports/bp_xgb_indicator_strategy_summary.csv
data/reports/bp_xgb_indicator_strategy_trades.csv
data/reports/bp_xgb_indicator_strategy_equity.csv
```

当前讨论中额外保存了以下口径文件：

```text
data/reports/bp_xgb_indicator_strategy_summary_all.csv
data/reports/bp_xgb_indicator_strategy_trades_all.csv
data/reports/bp_xgb_indicator_strategy_equity_all.csv
data/reports/bp_xgb_indicator_strategy_summary_all_cost.csv
data/reports/bp_xgb_indicator_strategy_trades_all_cost.csv
data/reports/bp_xgb_indicator_strategy_equity_all_cost.csv
data/reports/bp_xgb_indicator_strategy_summary_exclude_202312_202401.csv
data/reports/bp_xgb_indicator_strategy_trades_exclude_202312_202401.csv
data/reports/bp_xgb_indicator_strategy_equity_exclude_202312_202401.csv
data/reports/bp_xgb_indicator_strategy_summary_exclude_202312_202401_cost.csv
data/reports/bp_xgb_indicator_strategy_trades_exclude_202312_202401_cost.csv
data/reports/bp_xgb_indicator_strategy_equity_exclude_202312_202401_cost.csv
data/reports/bp_xgb_indicator_strategy_summary_compare.csv
```

### 4.3 画模型顶底信号

```powershell
python -m src.analysis.build_turning_signal_html
```

输出：

```text
data/reports/model_predicted_top_bottom_signals.html
```

该图展示 LightGBM、XGBoost、BP 的模型预测底部/顶部信号。

### 4.4 画策略买卖对应图

```powershell
python -m src.analysis.plot_bp_xgb_indicator_strategy_trades
```

输出：

```text
data/reports/bp_xgb_indicator_strategy_trade_pairs_exclude_202312_202401_cost.html
```

图中：

| 标记 | 含义 |
| --- | --- |
| `B#` | 买入点 |
| `S#` | 对应卖出点 |
| 同一编号 | 同一笔买卖关系 |
| 红色卖点/连线 | 盈利交易 |
| 绿色卖点/连线 | 亏损交易 |
| 灰色背景 | 被剔除区间 |

图表使用 ECharts，缩放方式与 `turning-label-tool` 一致：鼠标滚轮缩放/平移，底部滑块控制显示区间。

## 5. 当前核心评估结果

截至当前版本，策略从 2022 年开始评估，结果如下：

| 口径 | 总收益 | 最大回撤 | 胜率 | 已平仓 | 平均单笔收益 | 平均持仓 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 全区间，无成本 | +26.23% | -13.44% | 67.57% | 37 | +2.31% | 2.92 个交易日 |
| 剔除 2023-12~2024-01，无成本 | +35.47% | -6.61% | 70.59% | 34 | +2.79% | 2.97 个交易日 |
| 全区间，含成本 | +22.54% | -13.76% | 64.86% | 37 | +2.15% | 2.92 个交易日 |
| 剔除 2023-12~2024-01，含成本 | +31.88% | -6.75% | 67.65% | 34 | +2.63% | 2.97 个交易日 |

当前更适合作为阶段性研究版本的口径是：

```text
剔除 2023-12~2024-01，含成本
```

原因：

1. 已经纳入手续费和滑点。
2. 剔除了微盘流动性危机这种对普通模型极端不友好的异常区间。
3. 回撤显著低于全区间。
4. 胜率和收益仍保持可观察优势。

## 6. 当前风险

1. BP 底部信号在 2023 年没有覆盖，说明模型对部分震荡下跌年份仍可能失明。
2. 2024 年底部信号较多，但部分集中在剔除区间，真实适用性仍需要后续标注和新数据验证。
3. indicator_stop 当前较敏感，平均持仓只有约 3 个交易日，更像短波段退出规则。
4. ECharts HTML 图依赖 CDN，离线环境可能无法加载。
5. 当前策略只是中证1000指数级别验证，尚未扩展到成分股、行业宽度和分时数据。

## 7. 后续可继续优化方向

1. 继续完善人工标注，尤其补充 2022-2024 的非极端底部和顶部。
2. 增加市场宽度、成交结构、行业扩散等特征。
3. 对 indicator_stop 做阈值网格搜索，避免过早卖出。
4. 区分短期/中期/长期标签后，分别训练不同持仓周期模型。
5. 在采集完成后增加分时模型，独立验证 4-6 小时级别的超短线信号。
