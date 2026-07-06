# 中证1000 Buy 信号看板

当前分支：`codex/buy-signal-dashboard`。

本分支只负责 Buy 信号展示和报告生成，不负责数据采集、特征构建、模型训练或正式信号生成。模型 1.0 的训练、评估和信号口径仍由 `feature/buy-signal-research` 维护。

## 入口边界

`run_buy_signal_dashboard.py` 是看板专用入口，只保留展示相关 step；`main_daily_run.py` 不承载本分支的看板流程：

```powershell
python .\run_buy_signal_dashboard.py --step report_buy_signal
python .\run_buy_signal_dashboard.py --step report_walk_forward_buy
```

默认运行：

```powershell
python .\run_buy_signal_dashboard.py
```

等价于：

```powershell
python .\run_buy_signal_dashboard.py --step report_buy_signal
```

也可以使用包装脚本：

```powershell
.\scripts\run_daily.ps1 -Step report_buy_signal
.\scripts\run_daily.ps1 -Step report_walk_forward_buy
```

## 日常 Buy 信号看板

命令：

```powershell
python .\run_buy_signal_dashboard.py --step report_buy_signal
```

输出：

```text
data/reports/buy_signal_v1_dashboard.html
```

数据来源：

- MySQL `buy_signal_daily`
- MySQL `market_index_daily`

展示内容：

- 最新方向、因果状态、Buy 评分和阈值
- Buy 评分与阈值交互图
- 中证1000指数走势与 Long 标记
- Long 信号未来 7 日实际收益
- 最近信号表

## 滚动样本外预测看板

命令：

```powershell
python .\run_buy_signal_dashboard.py --step report_walk_forward_buy
```

输出：

```text
data/reports/walk_forward_buy_2021_2025_dashboard.html
```

默认读取：

```text
data/reports/walk_forward_buy_2021_2025_details.csv
data/reports/walk_forward_buy_2021_2025_summary.csv
data/reports/walk_forward_buy_2021_2025_aggregate.csv
```

这些 CSV 由滚动建模研究脚本生成，不在本看板入口中重新训练。

## 依赖

安装依赖：

```powershell
pip install -r requirements.txt
```

本分支看板使用 Plotly 生成离线交互图，HTML 可直接打开查看。

## 合规说明

本系统输出仅用于量化研究和信号展示，不构成投资建议。报告不包含仓位、交易成本、净值或回撤，不属于正式回测。
