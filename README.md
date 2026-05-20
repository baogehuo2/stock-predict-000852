# 中证1000短期趋势预测系统

本项目按 `开发补充信息确认表.md` 落地第一版完整闭环：行情、ETF、期货、东方财富股吧、新闻、大模型事件抽取、技术/舆情/事件特征、LightGBM 训练、预测、回测评估和 HTML 日报。

## 目录

- `config/`：非敏感配置、标的清单、Prompt 模板、加密后的敏感配置
- `sql/create_tables.sql`：MySQL 建表语句
- `src/collectors/`：行情、ETF、期货、股吧、新闻采集
- `src/features/`：技术指标、舆情热度、事件和模型数据集
- `src/llm/`：OpenAI 兼容大模型客户端和事件抽取
- `src/modeling/`：训练、预测、评估、历史相似日期
- `src/report/`：HTML 日报
- `scripts/smoke_test.py`：轻量级运行环境检查

## 初始化

安装依赖：

```powershell
pip install -r requirements.txt
```

确认已经生成：

```text
config/secrets.key
config/secrets.enc.yaml
```

运行环境检查：

```powershell
python .\scripts\smoke_test.py
```

初始化数据库：

```powershell
python .\main_daily_run.py --step init_db
```

## 单模块运行

```powershell
python .\main_daily_run.py --step collect_index
python .\main_daily_run.py --step build_market_features
python .\main_daily_run.py --step build_dataset
python .\main_daily_run.py --step train
python .\main_daily_run.py --step predict
python .\main_daily_run.py --step report
```

也可以用 PowerShell 包装脚本：

```powershell
.\scripts\run_daily.ps1 -Step collect_index
```

## 一键运行

```powershell
python .\main_daily_run.py
```

默认流程会依次运行：

```text
init_db -> collect_index -> collect_etf -> collect_futures -> collect_guba -> collect_news
-> build_market_features -> build_sentiment_features -> extract_events
-> build_dataset -> train -> predict -> report
```

注意：确认表要求大模型失败时中断流程，所以 `extract_events` 失败会直接停止。

## 报告

HTML 日报输出到：

```text
data/reports/
```

报告包含 3/5/7 日方向预测、预测涨跌幅、置信度、技术/舆情/事件解释、历史相似日期和风险提示。

## 合规说明

本系统输出仅用于量化研究与信息整理，不构成投资建议。报告中避免输出买卖、建仓、清仓等交易建议。

## 历史新闻回溯

历史新闻不要使用每日 `collect_news` 作为回溯来源。请使用独立脚本：

```powershell
python .\src\collectors\collect_news_history.py --start-date 2024-01-01 --end-date 2024-01-31 --source cctv --keyword-mode all
```

当前已支持：

- `cctv`：新闻联播文字稿，按日期回溯。
- `baidu_economic`：百度股市通经济日历，按日期回溯。

参数：

- `--source` 可重复传；不传时默认 `cctv + baidu_economic`。
- `--keyword-mode match` 只保留配置关键词命中的内容。
- `--keyword-mode all` 保留来源返回的全部内容。
- `--sleep` 默认每个日期间隔 `0.5` 秒。

查看入库状态：

```powershell
python .\scripts\check_news_status.py
```

## Playwright 股吧历史回溯

如果东方财富股吧触发验证码，使用 Playwright 版本。首次使用需要安装：

```powershell
pip install playwright
python -m playwright install chromium
```

运行时会打开真实浏览器。若出现验证码，先在浏览器里手动完成，再回到终端按 Enter：

```powershell
python .\src\collectors\collect_guba_history_playwright.py --bar-name 中证1000吧 --start-page 1 --end-page 20 --sleep 1 --detail-sleep 0.2
```

浏览器会话保存在 `data/browser/eastmoney`，通过验证码后的 cookie 会保留，后续可复用。
