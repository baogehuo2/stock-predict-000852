# 看板命令

- `python .\run_buy_signal_dashboard.py --step report_buy_signal`
- `python .\run_buy_signal_dashboard.py --step report_walk_forward_buy`

说明：

- `main_daily_run.py` 保留为主干日常流程入口，不混入本分支看板 step。
- `generate_buy_signal` 属于信号生成流程，不属于本看板入口。
