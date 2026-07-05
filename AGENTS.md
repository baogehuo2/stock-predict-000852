# stock-predict-000852 Codex Rules

始终使用中文回答。

## 全局对话规则

- 开头只给：目标、范围、验收
- 中间不汇报过程
- 直接改代码、直接跑测试、直接修复失败
- 最后只汇报：修改了哪些文件、测试结果、是否还有风险

## 工作规则

- 修改前必须先运行 `git status --short`
- 不允许 `git reset --hard`
- 不允许删除 Guba 代码
- `use_guba_sentiment` 保持 `false`
- 不要重复运行大规模 LLM 历史抽取
- 不要只看三分类总体 accuracy
- 重点看 `predicted up` 的 `hit rate`、`future return`、`signal count`、`coverage`

## 当前优先级

1. 评估上涨信号质量
2. 比较简单基线
3. 增加滚动新闻事件特征
4. 增加二分类买入模型
5. 股吧舆情暂缓

## 分支定位

- `feature/buy-signal-research` 是当前主研究分支
- 这个分支负责把握整体项目节奏、信号质量、标签设计、特征取舍和模型结构
- 其他分支只提供原始数据、事件抽取或回测能力，不在别处分支独立决定模型策略

## 工作方式

- 优先复用现有代码
- 先理解现有结构，再修改
- 默认使用 Python 3.10 兼容写法
- 需要时先验证，再汇报最终结果

## 文档入口

- 项目总览：`docs/architecture/zz1000_overview.md`
- 信号质量：`docs/architecture/signal_quality.md`
- 买入模型：`docs/architecture/buy_model.md`
- 数据来源：`docs/architecture/data_sources.md`
- Worktree 约定：`docs/architecture/worktree_map.md`
