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

## 分支定位

- `feature/data-collectors` 只负责原始数据获取、清洗、时间戳校验和入库
- 不训练模型，不决定策略
- 原始数据缺失先在本分支修复，再进入建模分支

## 当前优先级

1. 数据采集完整性
2. 字段缺失修复
3. 时间戳和入库校验
4. 为研究分支提供稳定原始数据

## 文档入口

- 分支说明：`docs/architecture/data_collectors_overview.md`
- 采集清单：`docs/architecture/data_collectors_scope.md`
- 运行命令：`docs/architecture/data_collectors_commands.md`

