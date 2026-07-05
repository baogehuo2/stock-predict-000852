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
- 不允许删除用户代码
- 优先复用现有代码
- 需要时先验证，再汇报最终结果

## 分支定位

- `feature/bottom-fishing-v1` 只负责抄底模型研究、训练、评估和信号输出
- 不得修改或冒用 Buy 1.0 已冻结的版本、特征清单、模型文件和信号输出
- 需要共享能力时，先拆成单一职责修改，再独立验证

## 当前优先级

1. 抄底模型信号质量
2. 简单基线比较
3. 分层特征和标签验证
4. 独立信号格式和输出命令
5. 暂不与 Buy 主线混跑

## 文档入口

- 分支说明：`docs/architecture/bottom_overview.md`
- 抄底模型：`docs/architecture/bottom_model.md`
- 验证口径：`docs/architecture/bottom_validation.md`

