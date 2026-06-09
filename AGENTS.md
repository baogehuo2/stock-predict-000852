# stock-predict-000852 Codex Rules

始终使用中文回答。

项目目标：
预测中证1000指数 000852，当前重点不是实盘交易，而是评估模型上涨信号是否有统计价值。

当前优先级：
1. 评估上涨信号质量
2. 比较简单基线
3. 增加滚动新闻事件特征
4. 增加二分类买入模型
5. 股吧舆情暂缓

重要事项：
- 当前建模研究主线分支是 feature/buy-signal-research。
- 先在 feature/buy-signal-research 稳定 buy signal 输出格式，再考虑回测分支。
- buy signal 输出格式应优先稳定字段：trade_date、model_tag、horizon、buy_proba、threshold、signal、pred_ret、future_ret_5d、future_ret_7d。
- 在 buy signal 输出格式稳定前，不要新建 feature/backtest-engine。
- 等信号格式稳定后，再从 feature/buy-signal-research 切出 feature/backtest-engine 开发回测。
- 回测成熟后，再合并回 feature/buy-signal-research 或 main。

硬性规则：
- 修改前必须先运行 git status --short
- 不允许 git reset --hard
- 不允许删除 Guba 代码
- Guba sentiment 当前保持 use_guba_sentiment: false
- 不要重复运行大规模 LLM 历史抽取
- 不要只看三分类总体 accuracy
- 重点看 predicted up 的 hit rate、future return、signal count、coverage
