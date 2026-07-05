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

- `codex/buy-signal-dashboard` 只负责 Buy 信号可视化和展示
- 这个分支不改模型策略、不改训练逻辑、不改数据口径
- 只围绕展示、布局和信号查看体验做调整

## 当前优先级

1. Buy 信号可视化
2. 报告和看板展示
3. 路径和入口稳定
4. 不影响主线训练与评估

## 文档入口

- 看板说明：`docs/architecture/dashboard_overview.md`
- 看板入口：`docs/architecture/dashboard_commands.md`

