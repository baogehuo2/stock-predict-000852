# Buy 信号看板命令

## 看板入口

当前买入信号看板优先通过主仓库入口输出：

```powershell
python .\main_daily_run.py --step report_buy_signal
```

如果只想跑买入信号看板，也可以直接调用 `src/report/generate_buy_signal_report.py` 对应的封装函数。

## 滚动预测看板

滚动样本外预测结果由研究脚本生成后，再由报告入口读取并展示。相关文件通常是：

- `data/reports/walk_forward_buy_2021_2025_details.csv`
- `data/reports/walk_forward_buy_2021_2025_summary.csv`
- `data/reports/walk_forward_buy_2021_2025_aggregate.csv`

## 说明

本项目的入口拆分原则是：

- 主仓库保留主干共用入口
- 分支保留各自专题入口
- 看板只负责展示和报告，不承载训练和采集
