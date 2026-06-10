# stock-predict-000852 Codex Rules

始终使用中文回答。

项目目标：
预测中证1000指数 000852，当前重点不是实盘交易，而是评估多头、空头信号是否具有样本外统计价值。

当前优先级：
1. 对多头、空头模型做逐年滚动样本外验证
2. 比较同期天然上涨率、天然下跌率等简单基线
3. 增加因果市场状态门控，稳定双向信号格式
4. 验证通过后再增加市场宽度、风格、新闻事件等特征
5. 信号格式稳定后再开发正式回测
6. 股吧舆情暂缓

重要事项：
- 当前建模研究主线分支是 feature/buy-signal-research。
- 当前已保留分支：main、feature/buy-signal-research、feature/data-collectors。
- 规划但暂不创建的分支：feature/llm-event-extraction、feature/backtest-engine。
- 如果后续修改发现需要优化 LLM 事件抽取，再提示用户确认是否创建 feature/llm-event-extraction。
- 如果后续修改发现 buy signal 输出格式已稳定且需要开发回测，再提示用户确认是否创建 feature/backtest-engine。
- 先在 feature/buy-signal-research 稳定 buy signal 输出格式，再考虑回测分支。
- buy signal 输出格式应优先稳定字段：trade_date、model_tag、horizon、buy_proba、threshold、signal、pred_ret、future_ret_5d、future_ret_7d。
- 在 buy signal 输出格式稳定前，不要新建 feature/backtest-engine。
- 等信号格式稳定后，再从 feature/buy-signal-research 切出 feature/backtest-engine 开发回测。
- 回测成熟后，再合并回 feature/buy-signal-research 或 main。

已完成的研究步骤：
1. 已增加技术、量价、趋势和因果牛熊状态特征。
2. 已建立独立的 binary buy model 和 binary short model，保留原三分类模型，不允许删除原三分类模型。
3. 已按牛熊市场评估多空信号，并比较 0.60、0.80 等概率阈值。

牛熊研究口径：
- 2021-07-01 至 2024-08-24 作为事后熊市标签，其余 2021 年区间和 2024-08-25 以后作为事后牛市标签。
- 上述事后牛熊日期只能用于分组评估，不能作为模型输入，避免未来信息泄漏。
- 模型输入只能使用当日及以前可知的因果趋势状态。
- 评估必须区分 train、valid、test；训练期高分不能解释为预测能力。
- 评价模型增量价值时，必须与同一时间区间的天然上涨率或天然下跌率比较，不能跨区间比较。

后续执行路线（按顺序开展）：

第 4 步：逐年滚动验证，继续在 feature/buy-signal-research 开展。
- 按年度扩展训练窗口：只使用预测年度以前的数据训练，再预测下一年度。
- 分别输出牛市做多、熊市做空的 precision、signal count、coverage、average future return、median future return。
- 比较 0.60、0.70、0.80 阈值及同期天然涨跌基线。
- 同时输出原始逐日信号和去除持有期重叠后的独立信号结果。
- 验收标准：效果不能只集中在 2024 熊市或 2025 牛市，多数样本外年度应具有同方向增量价值。

第 5 步：市场状态门控，继续在 feature/buy-signal-research 开展。
- 比较无门控、硬门控、软门控三种方案。
- 硬门控：因果牛市状态只允许做多，因果熊市状态只允许做空。
- 软门控：逆势方向使用更高阈值，例如 0.85；中性状态提高阈值或不交易。
- 增加状态确认天数，降低牛熊状态频繁切换造成的噪声。
- 门控规则必须完全由当时可知的数据计算，不允许使用事后牛熊标签。

第 6 步：稳定双向信号输出格式，继续在 feature/buy-signal-research 开展。
- 目标字段：trade_date、model_tag、horizon、market_regime、buy_proba、short_proba、buy_threshold、short_threshold、direction、signal_strength、pred_ret、future_ret_5d、future_ret_7d。
- direction 统一为 long、short、neutral；signal_strength 统一为 normal、strong。
- 同一天、同一 horizon 最终只能输出一个交易方向，但必须保留原始 buy_proba 和 short_proba。
- 概率阈值只负责筛选信号，不应被解释为真实命中率。
- 完成滚动验证、门控比较和字段稳定后，才视为信号格式稳定。

第 7 步：补充外部原始数据与衍生特征。
- 原始数据采集必须在 feature/data-collectors 开展：市场上涨/下跌家数、创新高/新低家数、中证 300/500/1000/2000 行情、ETF、期指基差、新闻和宏观政策原始数据。
- 采集代码只负责可靠获取、清洗和入库，不在采集分支训练模型。
- 数据合并回建模主线后，在 feature/buy-signal-research 计算市场宽度、小盘股扩散度、中证1000相对大盘强弱、风格持续走弱天数、政策及负面事件滚动累积等特征。
- amount 相关原始字段缺失时，应先在 feature/data-collectors 修复，不要在建模代码中伪造或填充为有效数据。

第 8 步：优化 LLM 事件抽取，仅在现有事件结果不足时开展。
- 需要开展前，必须先提示用户确认是否创建 feature/llm-event-extraction。
- 该分支负责新闻去重与聚类、事件方向和强度、是否超预期、利好兑现风险、负面事件连续累积和事件影响衰减。
- 优先复用现有历史抽取结果，不得重复运行大规模 LLM 历史抽取。
- 原始新闻爬取仍属于 feature/data-collectors，LLM 结构化抽取才属于 feature/llm-event-extraction。

第 9 步：特征消融与模型优化，回到 feature/buy-signal-research 开展。
- 依次比较：基础行情；基础行情加技术指标；再加市场宽度和风格；再加新闻事件；再加情绪。
- 新增特征只有在逐年滚动样本外验证中提高 precision、average future return，并维持合理 signal count 时才保留。
- 不允许仅凭训练集特征重要性、总体 accuracy 或单一年份结果判断特征有效。

第 10 步：正式回测。
- 只有第 6 步信号格式稳定后，才提示用户确认创建 feature/backtest-engine。
- 回测分支负责持有 5/7 日、信号重叠处理、多空仓位、手续费、滑点、融券成本、最大持仓、净值、回撤、胜率和盈亏比。
- 回测必须与始终持有、始终做空、同期天然方向和简单趋势规则比较。
- 模型评估回答信号有没有预测价值；回测回答这些信号按可执行交易规则组合后是否仍有收益。

当前下一步：
- 优先执行第 4 步逐年滚动验证。
- 在滚动验证完成前，不急于增加更多复杂特征，也不创建 feature/backtest-engine。
- 如果滚动验证证明现有信号只在少数年份有效，应先修正标签、特征或门控，不进入正式回测。

硬性规则：
- 修改前必须先运行 git status --short
- 不允许 git reset --hard
- 不允许删除 Guba 代码
- Guba sentiment 当前保持 use_guba_sentiment: false
- 不要重复运行大规模 LLM 历史抽取
- 不要只看三分类总体 accuracy
- 重点看 long/short precision、future return、signal count、coverage 及相对同期天然基线的提升
