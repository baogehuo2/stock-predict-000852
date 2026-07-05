# 抄底分支总览

本分支研究中证1000的独立抄底模型，和 Buy 主线分开。

## 关注点

- 独立标签
- 独立特征
- 独立信号格式
- 独立验证口径

## 约束

- 不改 Buy 1.0 冻结产物
- 不与 Buy 主线共享模型口径
- 不在本分支直接改动主线策略

## 独立入口

抄底分支使用 `bottom_daily_run.py` 作为独立入口，不把抄底训练、walk-forward、blending 和报告步骤塞入主干 `main_daily_run.py`。

常用命令：

```bash
python bottom_daily_run.py --list-steps
python bottom_daily_run.py --steps manual_labels build_dataset walk_forward_sparse_models blend_equal daily_signal_report
python bottom_daily_run.py
```

默认流程保留抄底模型研究主线：人工区间标签、底部数据集、滚动模型、均衡 blending、低召回兜底 blending 和 K 线报告。
