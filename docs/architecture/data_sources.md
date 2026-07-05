# 数据来源与分工

## 数据采集

- `src/collectors/` 负责原始数据采集
- `src/features/` 负责特征构建
- `src/modeling/` 负责训练、评估和预测
- `src/report/` 负责报告输出

## 当前注意点

- `Guba sentiment` 当前保持关闭
- 大规模 LLM 历史抽取不要重复运行
- 缺失字段先在采集分支修复，再进入建模代码

## 运行入口

- `main_daily_run.py`
- `scripts/smoke_test.py`
- `scripts/run_daily.ps1`
- `scripts/run_buy_signal_daily.bat`
