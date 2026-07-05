# Worktree 约定

## 当前建议分工

- `main`：合并与发布
- `feature/buy-signal-research`：信号质量和买入模型研究
- `feature/data-collectors`：采集与入库
- `feature/backtest-engine`：回测引擎规划
- `feature/llm-event-extraction`：事件抽取规划

## 使用原则

- 每个 worktree 尽量只做一个主题
- 不要在一个会话里让 Codex 同时扫多个主题
- 提交前先检查对应 worktree 是否干净
- 需要跨分支比对时，只比较相关文件
