# 政治事件增强买入信号

本文记录基于第二批国内宏观政治事件特征构造的增强买入信号规则。该规则不替换 1.0 正式模型，而是在 1.0 之上增加高置信度信号分层。

## 信号分层

| 信号层级 | 使用模型 | 规则 | 含义 |
| --- | --- | --- | --- |
| `normal_v1` | 1.0 基线 | 牛市硬门控 + `buy_proba >= 0.60` | 普通买入信号 |
| `political_strong` | 1.0 + 全政治事件特征 | 牛市硬门控 + `buy_proba >= 0.70` | 政治事件增强强买入 |
| `cewc_very_strong` | 1.0 + CEWC 特征 | 牛市硬门控 + `buy_proba >= 0.75` | 中央经济工作会议增强特强买入 |

## 总体结果

验证口径：2021 至 2026 年逐年扩展窗口滚动验证，7 日持有期，7 个交易日去重。

| 信号层级 | 信号数 | Precision | 实际上涨率 | 平均未来7日收益 | 中位数未来7日收益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `normal_v1` | 36 | 69.44% | 72.22% | 1.7693% | 1.3399% |
| `political_strong` | 18 | 77.78% | 88.89% | 3.2336% | 2.8786% |
| `cewc_very_strong` | 17 | 82.35% | 94.12% | 3.0150% | 2.5094% |

结论：

- 政治事件增强信号比普通 1.0 更少，但质量更高。
- `political_strong` 更均衡，信号数 18，平均收益最高。
- `cewc_very_strong` 更稀疏，precision 和实际上涨率最高。

## 逐年表现

| 年份 | 信号层级 | 信号数 | Precision | 平均未来7日收益 |
| --- | --- | ---: | ---: | ---: |
| 2021 | `normal_v1` | 11 | 72.73% | 1.8029% |
| 2021 | `political_strong` | 3 | 100.00% | 4.9650% |
| 2021 | `cewc_very_strong` | 7 | 85.71% | 2.6919% |
| 2022 | `normal_v1` | 7 | 57.14% | 1.0081% |
| 2022 | `political_strong` | 7 | 71.43% | 1.2187% |
| 2022 | `cewc_very_strong` | 4 | 75.00% | 1.9137% |
| 2023 | `normal_v1` | 6 | 66.67% | 0.8392% |
| 2023 | `political_strong` | 0 | - | - |
| 2023 | `cewc_very_strong` | 3 | 100.00% | 1.9486% |
| 2024 | `normal_v1` | 4 | 75.00% | 6.8440% |
| 2024 | `political_strong` | 6 | 66.67% | 4.8637% |
| 2024 | `cewc_very_strong` | 2 | 50.00% | 8.0198% |
| 2025 | `normal_v1` | 3 | 100.00% | 1.7813% |
| 2025 | `political_strong` | 1 | 100.00% | 2.7241% |
| 2025 | `cewc_very_strong` | 0 | - | - |
| 2026 | `normal_v1` | 5 | 60.00% | -0.1896% |
| 2026 | `political_strong` | 1 | 100.00% | 2.8717% |
| 2026 | `cewc_very_strong` | 1 | 100.00% | 2.8717% |

## 与普通信号的重叠

| 对比 | 重叠天数 | 说明 |
| --- | ---: | --- |
| `political_strong` vs `normal_v1` | 7 | 大部分政治强信号不是普通 1.0 的重复 |
| `cewc_very_strong` vs `normal_v1` | 8 | CEWC 特强信号有一半左右是额外信号 |
| `cewc_very_strong` vs `political_strong` | 9 | 两个增强层有部分交叉，但不是完全同一批日期 |

## 当前决策

政治事件增强信号可以进入 2.0 候选规则，但不直接替换正式 1.0。

建议后续采用三档信号输出：

- `neutral`：无买入信号
- `long`：普通 1.0 买入信号
- `strong_long`：政治事件增强强买入
- `very_strong_long`：CEWC 增强特强买入

后续正式落地前还需要解决：

- 当前增强层来自滚动验证研究结果，尚未训练固定正式模型包。
- 正式每日信号输出需要把政治事件特征纳入预测流程。
- 需要决定同一天多档信号同时出现时的优先级，建议 `very_strong_long > strong_long > long > neutral`。

## 产物位置

- 评估脚本：`src/modeling/evaluate_political_enhanced_signal.py`
- 固定模型训练：`src/modeling/train_political_enhanced_models.py`
- 候选信号生成：`src/modeling/generate_political_enhanced_signal.py`
- 政治事件增强模型：`models/buy_lgbm_political_strong_v2_7d.joblib`
- CEWC 增强模型：`models/buy_lgbm_cewc_very_strong_v2_7d.joblib`
- 汇总结果：`data/reports/political_enhanced_signal_summary.csv`
- 信号明细：`data/reports/political_enhanced_signal_details.csv`
- 重叠分析：`data/reports/political_enhanced_signal_overlaps.csv`

## 固定模型包

当前已固化两个 2.0 候选增强模型包：

| 模型包 | 模型标签 | 特征数 | 训练区间 | 阈值 |
| --- | --- | ---: | --- | ---: |
| `models/buy_lgbm_political_strong_v2_7d.joblib` | `buy_v2_political_strong_7d` | 20 | 2016-01-01 至 2026-01-01 前 | 0.70 |
| `models/buy_lgbm_cewc_very_strong_v2_7d.joblib` | `buy_v2_cewc_very_strong_7d` | 5 | 2016-01-01 至 2026-01-01 前 | 0.75 |

训练命令：

```powershell
python -m src.modeling.train_political_enhanced_models --train-end 2026-01-01
```

固定包说明：

- `political_strong` 使用全部国内政治事件特征。
- `cewc_very_strong` 只使用 CEWC 相关特征。
- 标签仍为 `buy_label_7d`。
- 目标仍为未来 7 日收益大于 0.5%。
- 固定包只用于 2.0 候选增强信号，不替换 1.0 正式模型。

## 候选信号生成方式

当前已提供研究版每日输出脚本，只生成 CSV，不写入 `buy_signal_daily` 正式表：

```powershell
python -m src.modeling.generate_political_enhanced_signal `
  --start-date 2026-05-16 `
  --end-date 2026-06-10 `
  --output-csv data/reports/buy_signal_v2_political_enhanced_recent.csv
```

输出字段包括：

- `buy_proba`：1.0 正式模型概率。
- `political_proba`：全政治事件增强模型概率。
- `cewc_proba`：CEWC 增强模型概率。
- `direction`：`neutral`、`long`、`strong_long`、`very_strong_long`。
- `enhanced_reason`：触发原因。

默认优先级：

`very_strong_long > strong_long > long > neutral`

注意：

- 该脚本是 2.0 候选输出，不覆盖 1.0 正式数据库。
- 默认优先加载固定模型包，不再运行时临时训练。
- 如固定模型包缺失，可先运行 `train_political_enhanced_models.py` 训练。
- 最近样例 `2026-05-16` 至 `2026-06-10` 已验证，输出 18 行，其中 1 条普通 `long`，没有 `strong_long` 或 `very_strong_long`。
- 固定包输出与原运行时临时训练输出已校验一致：`buy_proba`、`political_proba`、`cewc_proba` 最大差异均为 0。
