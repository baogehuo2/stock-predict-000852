# 中证1000 7日 Buy 信号研究系统

当前主线：`buy-signal-v1.0`

这个仓库聚焦中证1000 `000852` 的上涨信号统计价值研究。详细规则和项目背景请看：

- `AGENTS.md`
- `docs/architecture/zz1000_overview.md`
- `docs/architecture/signal_quality.md`
- `docs/architecture/buy_model.md`

## 常用入口

```powershell
python .\scripts\smoke_test.py
python .\main_daily_run.py --step init_db
python .\main_daily_run.py --step collect_index
python .\main_daily_run.py --step build_market_features
python .\main_daily_run.py --step build_dataset
python .\main_daily_run.py --step train_buy_final
python .\main_daily_run.py --step generate_buy_signal
python .\main_daily_run.py --step report_buy_signal
python .\main_daily_run.py
```

## 单步运行

```powershell
.\scripts\run_daily.ps1 -Step collect_index
```

## 每日 Buy 信号

```powershell
.\scripts\run_buy_signal_daily.bat
.\scripts\run_buy_signal_daily.bat 30
```

输出文件：

- `data/reports/buy_signal_v1.csv`
- `data/reports/buy_signal_v1_recent.csv`
- `data/reports/buy_signal_v1_dashboard.html`

## 历史回溯

```powershell
python .\src\collectors\collect_news_history.py --start-date 2024-01-01 --end-date 2024-01-31 --source cctv --keyword-mode all
python .\scripts\check_news_status.py
python .\src\collectors\collect_guba_history_playwright.py --bar-name 中证1000吧 --start-page 1 --end-page 20 --sleep 1 --detail-sleep 0.2
```

## 目录

- `config/`：非敏感配置、目标清单、Prompt 模板
- `sql/create_tables.sql`：MySQL 建表语句
- `src/collectors/`：行情、ETF、期货、股吧、新闻采集
- `src/features/`：技术指标、舆情、事件、模型数据集
- `src/modeling/`：训练、预测、评估、历史相似日
- `src/report/`：HTML 报告
- `scripts/smoke_test.py`：轻量环境检查
