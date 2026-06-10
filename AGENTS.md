# stock-predict-000852 Codex Rules

始终使用中文回答。

## 项目定位

预测中证1000指数 000852。当前重点不是实盘交易，而是先证明 7 日 buy 信号具有稳定的样本外统计价值，再逐步升级为多源事件驱动预测系统。

- `feature/buy-signal-research` 是核心研究主线，负责决定预测任务、策略、所需数据、特征取舍、模型结构和信号标准。
- 其他分支只提供原始数据、事件抽取或回测能力，不能在其他分支独立决定模型策略。
- 原三分类模型和已有 short 源码必须保留，不允许删除；但模型 1.0 当前只训练和评估 7 日 binary buy model。

## 版本路线

### 模型 1.0：现有数据定稿

目标：不先扩张数据源，使用当前已经具备的数据，把模型做成无泄漏、可复现、跨年度相对稳定的基础版本。

1. 完成逐年滚动样本外验证和同期天然基线比较。
2. 完成因果市场状态门控。
3. 优化收益标签、概率校准和概率分桶。
4. 对现有特征做分组消融。
5. 稳定统一信号输出格式。
6. 冻结 `model_version=buy-signal-v1.0` 和对应 `feature_version`。

模型 1.0 的当前定位：

- 唯一当前模型：未来 7 日 binary buy model。
- 5 日 binary buy model 退出模型 1.0 当前训练和评估范围，历史研究结果保留。
- short model 暂停训练和评估，后续使用非对称标签和独立目标单独重构。
- 三分类模型：保留为对照基线。
- 不为了匹配外部方案立即改成 T+1/T+3/T+5；新周期放到 2.0 单独验证。
- 模型 1.0 阶段不重新进行大规模数据爬取，不开发正式回测。

模型 1.0 验收条件：

- 滚动训练严格只使用预测日期以前可知的数据，并净化训练窗口尾部，防止未来收益标签跨期泄漏。
- 独立信号在多数样本外年度优于同期天然上涨率或下跌率。
- 完成无门控、硬门控、软门控比较。
- 完成概率校准和实际命中率分桶。
- 完成现有特征消融，不保留只改善单一年份的特征。
- 信号字段、阈值解释和生成命令固定且可重复运行。

### 模型 2.0：多源事件驱动升级

目标：在 1.0 基础上，分批增加市场宽度、资金行为、情绪行为和重大事件，不允许一次性混入全部新特征。

建设顺序：

1. 市场宽度和资金行为。
2. 固定事件日历、宏观日历和外部风险映射行情。
3. 情绪数据。
4. 中国事件、地缘政治、全球宏观三类重大事件。
5. 统一事件窗口和衰减机制。
6. 逐层特征消融和滚动验证。
7. 稳定模型 2.0 信号格式。
8. 模型 2.0 完成后才开发正式回测。

## 已完成研究

1. 已增加技术、量价、趋势和因果市场状态特征。
2. 历史上已建立 5/7 日 binary buy model 和 binary short model；当前研究范围已收敛为 7 日 binary buy model。
3. 已按事后牛熊市场评估多空信号，并比较 0.60、0.70、0.80 阈值。
4. 已完成首轮 2021-2026 逐年扩展窗口验证，并净化训练窗口末尾 5/7 个交易日。
5. 已完成无门控、硬门控、软门控和 short 风险过滤的信号质量比较；只统计 precision、future return、signal count 和不重叠信号，不涉及净值、仓位、执行价格或交易成本。

首轮滚动验证结论：

- 牛市 7 日做多相对稳定：阈值 0.60、去除持有期重叠后共 36 个信号，precision 80.56%，同期天然上涨率 62.60%，平均收益 2.14%，同期天然平均收益 1.06%。
- 牛市 5 日做多次优：阈值 0.70、去除重叠后共 40 个信号，precision 67.50%，同期天然上涨率 59.51%，平均收益 1.83%。
- 熊市 7 日做空增量较弱：阈值 0.60、去除重叠后共 46 个信号，precision 56.52%，同期天然下跌率 53.52%，平均策略收益 0.64%，优势主要集中在 2024。
- 5 日空头模型没有稳定的滚动样本外 precision 增量，当前不作为主策略。
- 7 日多头全市场独立信号：无门控 126 个、precision 60.32%、平均收益 0.60%；硬门控 55 个、precision 72.73%、平均收益 1.34%；软门控 107 个、precision 61.68%、平均收益 0.73%。硬门控提升最明显，但信号数量减少约 56%。
- 5 日多头全市场独立信号：无门控 134 个、precision 55.22%、平均收益 0.56%；硬门控 45 个、precision 62.22%、平均收益 1.06%；软门控 103 个、precision 60.19%、平均收益 0.78%。
- 硬门控并非逐年都改善：7 日模型主要改善 2022 和 2024，2023 仍弱，2026 独立信号平均收益为负。门控不能替代后续标签、概率和特征优化。
- 当前 short label 是 buy label 的严格补集，buy/short 模型使用相同特征与参数，滚动预测满足 `short_proba = 1 - buy_proba`。因此 short 概率否决和 buy-short 概率差不会过滤任何新增信号，当前 short model 不能作为独立 `risk_veto`。

## 牛熊研究口径

- 2021-07-01 至 2024-08-24 作为事后熊市标签，其余 2021 年区间和 2024-08-25 以后作为事后牛市标签。
- 事后牛熊日期只能用于分组评估，不能作为模型输入。
- 模型输入和市场门控只能使用当日及以前可知的因果状态。
- 评估必须区分 train、valid、test；训练期高分不能解释为预测能力。
- 模型必须与同一时间区间的天然上涨率、天然下跌率和平均收益比较。
- 同时报告逐日信号和去除持有期重叠后的独立信号，策略判断优先看独立信号。

## 模型 1.0 后续步骤

### 1. 因果市场状态门控

在 `feature/buy-signal-research` 开展：

- 比较无门控、硬门控和软门控。
- 硬门控：因果牛市状态允许做多，因果熊市状态禁止普通多头。
- 软门控：逆势或中性状态提高阈值，例如 0.80/0.85，或不发信号。
- 测试状态确认 3/5 日，降低频繁切换噪声。
- short model 已退出模型 1.0 当前流程，不再训练、评估或用作 `risk_veto`。后续另行使用非对称下跌标签、独立特征或独立模型目标重构。

### 2. 标签与概率优化

在 `feature/buy-signal-research` 开展：

- 比较未来收益 `>0%`、`>0.5%`、`>1%` 的上涨标签。
- 测试剔除或降低接近零收益样本的权重。
- 使用时间序列验证进行 sigmoid/isotonic 概率校准。
- 输出 0.50-0.60、0.60-0.70、0.70-0.80、0.80 以上概率分桶的真实命中率和平均收益。
- 概率阈值只表示筛选门槛，不得直接解释为真实命中率。

### 3. 现有特征消融

在 `feature/buy-signal-research` 开展，按顺序比较：

1. 基础行情。
2. 基础行情加传统技术指标。
3. 再加趋势、量价、波动和风险特征。
4. 再加当前已有 ETF、期货特征。
5. 再加当前已有新闻事件特征。

新增特征必须在逐年滚动样本外验证中提高 precision、average future return，并维持合理 signal count 才能保留。不能仅凭训练集特征重要性、总体 accuracy 或单一年份结果判断有效。

### 4. 模型 1.0 信号格式

在 `feature/buy-signal-research` 稳定以下字段：

```text
trade_date
model_version
feature_version
model_tag
horizon
causal_regime
buy_proba
buy_threshold
direction
signal_strength
pred_ret
future_ret_7d
```

- `direction` 统一为 `long / neutral`。
- `signal_strength` 统一为 `normal / strong`。
- 同一天最终只能输出一个 7 日 buy 方向结果。
- 连续日期信号可以保留原始概率，模型评估必须另行输出不重叠信号。

## 模型 2.0 数据与分支职责

### feature/data-collectors

只负责原始数据可靠获取、清洗、时间戳校验和入库，不训练模型、不决定策略。

优先采集：

- 市场宽度：上涨/下跌家数、涨停/跌停家数、创新高/新低家数、炸板率、连板高度、全市场成交额和换手率。
- 风格行情：中证 300/500/1000/2000、科创和创业板等可用指数。
- 资金行为：ETF 资金、融资融券、期指基差、可验证的主力资金数据。
- 固定日历：中国节假日、重要会议、国内外宏观数据发布日期。
- 风险映射行情：原油、黄金、美元指数、美债收益率、VIX。
- 情绪原始数据：股吧、雪球、财经媒体和热点题材。
- 重大事件原始新闻：中国政策、地缘政治和全球宏观新闻。

amount 等原始字段缺失时，必须先在该分支修复，不能在建模代码中伪造有效数据。

### feature/buy-signal-research

原始数据合并后，在核心分支负责：

- 市场宽度、小盘扩散度和风格相对强弱特征。
- 资金趋势和流动性风险特征。
- 三类重大事件的日频滚动特征。
- 分层特征消融、标签选择、模型训练和策略决策。
- 判断每类数据是否进入模型 2.0。

### feature/llm-event-extraction

规划但暂不创建。需要优化 LLM 事件抽取时，必须先提示用户确认创建。

负责：

- 新闻去重、事件聚类和跨来源合并。
- 事件类型、方向、重要性、阶段和影响对象。
- 是否超预期、风险是否升级、利好兑现和负面累积。
- 事件影响衰减和主题轮动识别。

原始新闻爬取不属于该分支。优先复用已有历史抽取结果，不得重复进行大规模 LLM 历史抽取。

## 模型 2.0 三类重大事件

三类事件必须独立存储、独立生成特征、独立做消融验证。不要过早合并成一个 `event_score`。

统一基础字段：

```text
event_id
event_date
event_layer
event_type
event_stage
importance
direction
surprise_score
days_to_event
days_after_event
decay_score
source_count
```

`event_layer` 统一为 `cn / geo / macro`。

统一事件窗口可测试：

- T-10 至 T-6：预热期。
- T-5 至 T-1：预期交易期。
- T：事件发生。
- T+1 至 T+3：解读期。
- T+4 至 T+10：兑现期。

### 1. 中国事件层 cn

优先级最高，描述直接影响A股制度、流动性、风险偏好和产业预期的事件。

包括：

- 固定事件：两会、中央经济工作会议、政治局会议、节假日和重要纪念活动。
- 临时政策：货币、财政、房地产、资本市场监管、IPO、减持、融券和量化交易政策。
- 产业事件：国家规划、产业政策、人工智能、科技和制造业战略。

建议独立字段：

```text
cn_event_type
cn_event_stage
cn_event_importance
cn_event_direction
cn_event_score
days_to_cn_event
days_after_cn_event
cn_policy_positive_count
cn_policy_negative_count
```

固定日历优先使用结构化数据，不需要每次交给 LLM 判断。

### 2. 地缘政治层 geo

描述通过风险偏好、能源、供应链和国际关系影响A股的外部冲击。

包括：

- 中美贸易、科技限制和芯片制裁。
- 台海、朝鲜半岛和亚太安全事件。
- 俄乌冲突。
- 中东、红海和霍尔木兹海峡风险。
- 战争升级、制裁和外交摩擦。

建议独立字段：

```text
geo_event_region
geo_event_type
geo_event_stage
geo_event_importance
geo_risk_score
geo_risk_change
war_news_count
china_us_score
middle_east_score
russia_ukraine_score
asia_pacific_score
```

必须重点识别风险是否升级、是否涉及中国、是否影响能源/供应链以及市场是否已经提前反应。

### 3. 全球宏观层 macro

描述全球流动性、利率、通胀、汇率和经济周期变化。

包括：

- 美联储、欧洲央行和日本央行政策。
- CPI、PPI、非农、GDP、PMI和零售数据。
- 中国 CPI、PPI、PMI、社融、M1/M2等宏观数据。
- 美债收益率、美元指数、原油、黄金和 VIX 等风险映射行情。

建议独立字段：

```text
macro_region
macro_indicator
macro_event_stage
actual_value
forecast_value
previous_value
surprise_score
macro_direction
macro_importance
global_liquidity_score
fed_risk_score
china_macro_score
```

宏观层优先使用结构化经济日历和 `actual - forecast` 超预期信息，LLM只负责解释，不负责替代真实数值。

建议特征表保持独立：

```text
cn_event_feature_daily
geo_event_feature_daily
macro_event_feature_daily
```

进入模型时仍保留独立维度：

```text
f_cn_event_score
f_geo_risk_score
f_macro_score
```

最终只有在分层消融证明有效后，才测试综合事件总分。

## 回测安排

- `feature/backtest-engine` 规划但暂不创建。
- 模型 1.0 完成后也不立即开发正式回测。
- 只有模型 2.0 完成新增数据分层消融、信号格式稳定后，才提示用户确认创建 `feature/backtest-engine`。
- 回测分支负责持有期、信号重叠、多空仓位、手续费、滑点、融券成本、净值、回撤、胜率和盈亏比。
- 模型评估回答信号是否具有预测价值；回测回答这些信号按可执行规则组合后是否仍有收益。

## 当前下一步

1. 继续在 `feature/buy-signal-research` 完成模型 1.0。
2. 因果市场状态门控首轮比较已完成，不开始新数据爬取。
3. 下一项工作是标签与概率校准，然后做现有特征消融和信号格式稳定。
4. 冻结模型 1.0 后，再进入模型 2.0 的数据采集阶段。
5. 正式回测推迟到模型 2.0 完成以后。

## 硬性规则

- 修改前必须先运行 `git status --short`。
- 不允许 `git reset --hard`。
- 不允许删除 Guba 代码。
- Guba sentiment 当前保持 `use_guba_sentiment: false`。
- 不要重复运行大规模 LLM 历史抽取。
- 不要只看三分类总体 accuracy。
- 模型 1.0 只看 7 日 buy precision、future return、signal count、coverage、独立信号结果及相对同期天然上涨基线的提升。
- short 模型后续单独重构，在用户明确启动前不训练、不评估、不加入风险过滤。
- 需要创建 `feature/llm-event-extraction` 或 `feature/backtest-engine` 时，必须先提示用户确认。
