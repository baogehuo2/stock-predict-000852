# 中证1000 项目总览

项目目标是研究中证1000指数 `000852` 的上涨信号是否具有统计价值，而不是优先做实盘交易。

## 当前重点

1. 评估上涨信号质量
2. 比较简单基线
3. 增加滚动新闻事件特征
4. 增加二分类买入模型
5. 股吧舆情暂缓

## 分支职责

- `feature/buy-signal-research`：主研究分支
- `feature/data-collectors`：数据采集与入库
- `feature/backtest-engine`：后续回测引擎
- `feature/llm-event-extraction`：后续事件抽取

## 评估原则

- 不要只看三分类总体 accuracy
- 重点看 `predicted up` 的 `hit rate`
- 同时看 `future return`
- 同时看 `signal count`
- 同时看 `coverage`

## 推荐提问方式

把任务拆小，直接指定文件和目标，避免让 Codex 先全仓扫描。

示例：

```text
只改 src/modeling/evaluate_signal_quality.py
目标是增加上涨信号分层统计
不要改其他文件
```
