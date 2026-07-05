# 人工标记买入标签实验

本文记录从 `feature/bottom-fishing-v1` 分支引入人工顶底标记后，在 Buy 主线中训练买入模型的第一轮结果。

## 数据来源

标记文件来自：

- `D:\python workbench\stock_predict\000852_bottom\data\manual_labels\market_turning_regions.csv`

已复制到当前分支：

- `data/manual_labels/market_turning_regions.csv`

标记含义：

- `region_type=bottom`：人工标出的底部区域。
- `entry_start` 到 `entry_end`：人工认为适合尝试买入的窗口。
- `region_type=top`：人工标出的顶部区域。
- `exit_start` 到 `exit_end`：人工认为适合退出或警惕的窗口。

本轮训练使用：

- `manual_buy_label = 1`：交易日在人工底部买入窗口内。
- `manual_buy_label = 0`：不在人工底部买入窗口内。

同时保留：

- `manual_sell_label`：是否落入人工卖出/顶部窗口，用于检查误伤风险。

## 实验方式

- 模型：LightGBM 二分类。
- 特征：当前 Buy 1.0 的 211 个特征。
- 标签：人工买入窗口 `manual_buy_label`。
- 验证：2021-2026 逐年滚动验证。
- 信号去重：20 个交易日内只保留第一条信号。
- 收益评价：未来 20 个交易日收益。

## 汇总结果

| 阈值 | 信号数 | 命中人工买入窗口比例 | 人工买入基础比例 | 提升 | 落入人工卖出窗口比例 | 未来20日上涨率 | 平均未来20日收益 | 最差未来20日收益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 20 | 20.00% | 12.51% | 7.49% | 25.00% | 50.00% | 0.3480% | -14.5641% |
| 0.20 | 17 | 29.41% | 12.51% | 16.90% | 23.53% | 47.06% | 0.2867% | -12.9574% |
| 0.30 | 14 | 21.43% | 12.51% | 8.92% | 21.43% | 42.86% | 0.0004% | -11.6917% |
| 0.40 | 12 | 16.67% | 12.51% | 4.16% | 25.00% | 33.33% | -0.2694% | -11.6917% |
| 0.50 | 11 | 18.18% | 12.51% | 5.67% | 27.27% | 45.45% | 0.8373% | -11.0140% |
| 0.60 | 11 | 9.09% | 12.51% | -3.42% | 27.27% | 45.45% | 0.6364% | -11.0140% |
| 0.70 | 10 | 20.00% | 12.51% | 7.49% | 30.00% | 50.00% | 1.1427% | -11.0140% |

## 结论

人工标记数据已经能接入 Buy 主线训练，但第一轮结果不能替代 1.0。

原因：

- 当前 Buy 1.0 特征更像“趋势/上涨概率”特征，不是专门识别人工底部窗口的特征。
- `manual_buy_label` 是稀疏标签，基础比例只有约 12.51%，训练难度更高。
- 0.20 阈值最能贴近人工买入窗口，但未来 20 日收益很弱。
- 0.70 阈值收益略好，但信号很少，而且落入人工卖出窗口比例达到 30%。
- 说明只换标签、不换特征，效果不够。

## 下一步建议

不要直接把人工标签模型封成 Buy 2.0。

更合理的方向：

1. 保留 `manual_buy_label` 作为新的候选训练标签。
2. 把 bottom 分支的底部形态特征、周线底部确认特征、顶部风险标签一并迁入 Buy 2.0 实验。
3. 训练时同时使用 `manual_buy_label` 和 `manual_sell_label`：
   - 买入模型学习人工买入窗口。
   - 风险/顶部模型过滤落入人工卖出窗口的信号。
4. 再与 1.0 普通信号、政治事件增强、海外风险过滤做组合评估。

## 产物位置

- 标记数据：`data/manual_labels/market_turning_regions.csv`
- 训练评估脚本：`src/modeling/walk_forward_manual_label_buy_lgbm.py`
- 标准阈值汇总：`data/reports/manual_label_buy_walk_forward_summary.csv`
- 标准阈值明细：`data/reports/manual_label_buy_walk_forward_details.csv`
- 低阈值扫描汇总：`data/reports/manual_label_buy_walk_forward_threshold_sweep.csv`
- 低阈值扫描明细：`data/reports/manual_label_buy_walk_forward_threshold_details.csv`
