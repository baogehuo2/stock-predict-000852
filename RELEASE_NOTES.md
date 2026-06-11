# buy-signal-v1.0

封版日期：2026-06-11

## 模型口径

- 目标：预测中证1000未来 7 个交易日收益是否超过 0.5%。
- 模型：LightGBM binary Buy classifier。
- 标签：`future_ret_7d > 0.005`。
- 信号门槛：raw `buy_proba >= 0.60`。
- 门控：仅因果状态为 `bull` 时允许输出 `long`，其他情况输出 `neutral`。
- 模型版本：`buy-signal-v1.0`。
- 特征版本：`buy-features-v1.0`。
- 冻结特征：211 个，清单位于 `config/buy_features_v1.json`。

## 样本外结果

2021-2026 逐年扩展窗口、训练尾部净化、因果硬门控及去除持有期重叠后：

- 独立信号：35 个。
- Precision：71.43%。
- 实际上涨率：74.29%。
- 平均未来 7 日收益：1.90%。
- 收益中位数：1.34%。

## 信号输出

固定字段：

```text
trade_date
model_version
feature_version
model_tag
horizon
causal_regime
buy_proba
buy_threshold
direction
signal_strength
pred_ret
future_ret_7d
```

生成命令：

```bash
python main_daily_run.py --step train_buy_final --step generate_buy_signal --stop-on-error
```

CSV 输出到 `data/reports/buy_signal_v1.csv`，数据库写入 `buy_signal_daily`。

## 封版验收

- 数据集重建成功：2759 行。
- 正式发布模型训练成功：2752 行可用标签，211 个特征。
- 统一信号重复执行无重复记录。
- 固定 12 字段顺序验证通过。
- 滚动验证和硬门控结果成功复现。
- 原三分类模型和已有 short 源码保留，未删除。

模型文件、数据库数据、密钥和运行报告不进入 Git；Git 标签冻结代码、配置、特征清单和运行口径。
