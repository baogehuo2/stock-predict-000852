# 顶底标记工具需求说明

## 目标

本分支用于开发顶底标记工具。工具的目标是让人工在 K 线图上标注不同周期、不同级别的顶部和底部区域，并输出统一 CSV，供后续多周期顶底模型训练使用。

本说明只定义标记工具需要支持的信息、字段含义、交互规则、校验规则和输出格式。不包含模型训练、数据采集、信号生成逻辑。

## 标记对象

标记对象不是单个交易日，而是一段区域。

一个完整标记通常包含：

- 顶部或底部区域：行情已经走出来后，人工认为属于顶部或底部的 K 线区域。
- 交易窗口：在该区域内，模型未来应该学习的可买入窗口或可卖出窗口。
- 周期级别：短期、中期、长期、分时。
- 重要程度：小级别、波段级别、大级别。
- 置信度：标记者对该区域是否可靠的判断。

工具需要支持同一个行情区域被标成多个周期。例如 2024 年 2 月低点既可以是短期底，也可以是中期底，还可能是长期底。这种情况不要强行合并成一条，应该拆成多条标注记录。

## 输出文件

推荐输出文件：

```text
data/manual_labels/market_turning_regions.csv
```

标记工具也可以先输出到临时文件，再由建模分支导入。

最终 CSV 字段顺序固定为：

```csv
region_id,start_date,end_date,region_type,cycle,level,label_freq,confidence,usable_for_signal,entry_start,entry_end,exit_start,exit_end,reason,notes
```

## 字段定义

| 字段 | 必填 | 类型 | 含义 |
| --- | --- | --- | --- |
| `region_id` | 是 | string | 标记区域唯一 ID |
| `start_date` | 是 | date | 顶底区域开始日期 |
| `end_date` | 是 | date | 顶底区域结束日期 |
| `region_type` | 是 | enum | `bottom` 或 `top` |
| `cycle` | 是 | enum | `short`、`medium`、`long`、`intraday` |
| `level` | 是 | enum | `minor`、`swing`、`major` |
| `label_freq` | 是 | enum | 标记主要来自哪个观察频率：`intraday`、`daily`、`weekly`、`monthly` |
| `confidence` | 是 | int | 置信度，1 到 3 |
| `usable_for_signal` | 是 | int | 是否用于训练信号，0 或 1 |
| `entry_start` | 条件必填 | date | 底部买入窗口开始日期 |
| `entry_end` | 条件必填 | date | 底部买入窗口结束日期 |
| `exit_start` | 条件必填 | date | 顶部卖出窗口开始日期 |
| `exit_end` | 条件必填 | date | 顶部卖出窗口结束日期 |
| `reason` | 否 | string | 标记原因，可多选，用 `|` 分隔 |
| `notes` | 否 | string | 人工备注 |

## region_id 规则

`region_id` 必须唯一，推荐格式：

```text
B202402_daily_short_01
T202410_weekly_medium_01
B202402_monthly_long_01
```

命名规则：

| 部分 | 说明 |
| --- | --- |
| `B` / `T` | `B` 表示 bottom，`T` 表示 top |
| `YYYYMM` | 区域核心月份 |
| `label_freq` | `intraday`、`daily`、`weekly`、`monthly` |
| `cycle` | `short`、`medium`、`long`、`intraday` |
| 序号 | 同月多个区域时用 `01`、`02` |

工具可以自动生成 `region_id`，但要允许人工修改。

## region_type 含义

| 取值 | 含义 |
| --- | --- |
| `bottom` | 底部区域，用于学习买入或抄底 |
| `top` | 顶部区域，用于学习卖出或逃顶 |

`bottom` 记录必须填写 `entry_start` 和 `entry_end`。

`top` 记录必须填写 `exit_start` 和 `exit_end`。

## cycle 含义

`cycle` 是本轮重构中最重要的新字段，用于区分不同预测周期的顶底。

| cycle | 中文含义 | 预测窗口 | 主要用途 |
| --- | --- | --- | --- |
| `intraday` | 分时级别 | 未来 4-6 小时 | 日线信号后的进出场择时 |
| `short` | 短期级别 | 未来 5-15 个交易日 | 反弹、短线减仓、试仓 |
| `medium` | 中期级别 | 未来 20-40 个交易日 | 波段买卖点 |
| `long` | 长期级别 | 未来 80-120 个交易日 | 大底、大顶、仓位区 |

标记原则：

- `short`：只要求未来 5-15 个交易日有明显可交易反弹或回落。
- `medium`：要求未来 1-2 个月有较明显波段空间。
- `long`：要求未来 4-6 个月属于重要大级别拐点或仓位区域。
- `intraday`：只用于分时 K 线，日线工具暂时可以不启用。

## level 含义

`level` 表示顶底重要程度，不等同于 `cycle`，但二者有关联。

| level | 中文含义 | 典型特征 |
| --- | --- | --- |
| `minor` | 小级别顶底 | 短暂反弹、短暂调整，持续时间短 |
| `swing` | 波段顶底 | 有明确波段空间，通常持续数周到数月 |
| `major` | 大级别顶底 | 市场重要拐点，通常伴随极端情绪、估值、政策或流动性变化 |

建议对应关系：

| level | 常见 cycle |
| --- | --- |
| `minor` | `short` |
| `swing` | `short`、`medium` |
| `major` | `medium`、`long` |

注意：这只是默认建议，不是强制规则。工具可以给出提示，但不要禁止人工选择。

## label_freq 含义

`label_freq` 表示这条顶底标注主要是从哪个 K 线频率上观察出来的。

周线和月线顶底需要标记，尤其是中期和长期模型。但不建议单独维护周线标注文件、月线标注文件。更推荐所有顶底仍然写入同一套 CSV，通过 `label_freq` 区分观察频率。

| label_freq | 中文含义 | 典型用途 |
| --- | --- | --- |
| `intraday` | 分时图观察得到 | 超短期进出场点 |
| `daily` | 日 K 观察得到 | 短期顶底、波段细节 |
| `weekly` | 周 K 观察得到 | 中期波段顶底、趋势切换 |
| `monthly` | 月 K 观察得到 | 长期大底、大顶、仓位区 |

推荐对应关系：

| cycle | 推荐 label_freq |
| --- | --- |
| `intraday` | `intraday` |
| `short` | `daily`，必要时 `weekly` |
| `medium` | `daily`、`weekly` |
| `long` | `weekly`、`monthly` |

标记原则：

- 日线标注负责更细的买卖窗口和短期交易机会。
- 周线标注负责波段结构和中期趋势切换。
- 月线标注负责长期顶部、长期底部和战略仓位区域。
- 同一个区域如果日线、周线、月线都能构成不同级别判断，应拆成多条记录。

示例：

```csv
B202402_daily_short_01,2024-02-05,2024-02-23,bottom,short,swing,daily,3,1,2024-02-05,2024-02-23,,,panic_liquidation|lower_shadow,日线短期反弹底
B202402_weekly_medium_01,2024-01-22,2024-02-23,bottom,medium,swing,weekly,3,1,2024-02-05,2024-02-23,,,weekly_boll_lower|weekly_kdj_golden_cross,周线波段底
B202402_monthly_long_01,2024-01-22,2024-02-23,bottom,long,major,monthly,3,1,2024-02-05,2024-02-23,,,monthly_oversold|valuation_low,月线长期大底
```

注意：周线、月线标记的是较大的区域判断，但后续建模仍会把区域展开到日频训练样本。也就是说，周线和月线标注最终仍会落到日线日期上，让日频模型学习“中长期状态下哪些日线位置适合行动”。

## confidence 含义

`confidence` 表示标记者对该区域的确定程度。

| confidence | 含义 | 是否建议训练 |
| --- | --- | --- |
| `1` | 低置信，只是观察或待验证 | 默认不建议 |
| `2` | 中等置信，比较像有效顶底 | 可以用于训练 |
| `3` | 高置信，典型顶底区域 | 强烈建议用于训练 |

工具默认逻辑：

- `confidence = 1` 时，`usable_for_signal` 默认填 `0`。
- `confidence = 2` 或 `3` 时，`usable_for_signal` 默认填 `1`。
- 用户可以手动覆盖 `usable_for_signal`。

## usable_for_signal 含义

| 取值 | 含义 |
| --- | --- |
| `1` | 用于模型训练和信号评估 |
| `0` | 只保留为观察标注，不参与训练 |

常见 `usable_for_signal = 0` 的情况：

- 区域还没完全走出来。
- 只是主观猜测。
- 数据异常或停牌影响明显。
- 区域和其他标注冲突严重。
- 未来信息依赖太强，不适合训练。

## start_date / end_date 含义

`start_date` 和 `end_date` 表示完整顶底区域，不表示买卖点。

底部例子：

```text
start_date = 2024-01-22
end_date   = 2024-02-23
```

含义是：这段 K 线整体属于底部区域。

顶部例子：

```text
start_date = 2024-10-08
end_date   = 2024-10-18
```

含义是：这段 K 线整体属于顶部区域。

工具交互上建议支持拖拽选择区域，也支持手动输入日期。

## entry_start / entry_end 含义

`entry_start` 和 `entry_end` 只用于 `bottom`。

它表示在底部区域内，人工认为更适合作为买入或抄底训练样本的窗口。

要求：

- `entry_start` 必须大于等于 `start_date`。
- `entry_end` 必须小于等于 `end_date`。
- `entry_start` 必须小于等于 `entry_end`。
- `bottom` 记录必须填写。
- `top` 记录必须为空。

示例：

```csv
B202402_daily_short_01,2024-01-22,2024-02-23,bottom,short,swing,daily,3,1,2024-02-05,2024-02-23,,,panic_liquidation|policy_support,短期反弹底
```

含义：

- 2024-01-22 到 2024-02-23 是底部区域。
- 2024-02-05 到 2024-02-23 是更适合模型学习的买入窗口。

## exit_start / exit_end 含义

`exit_start` 和 `exit_end` 只用于 `top`。

它表示在顶部区域内，人工认为更适合作为卖出或逃顶训练样本的窗口。

要求：

- `exit_start` 必须大于等于 `start_date`。
- `exit_end` 必须小于等于 `end_date`。
- `exit_start` 必须小于等于 `exit_end`。
- `top` 记录必须填写。
- `bottom` 记录必须为空。

示例：

```csv
T202410_daily_short_01,2024-10-08,2024-10-18,top,short,swing,daily,3,1,,,2024-10-08,2024-10-10,euphoria|volume_expand|policy_trade,924政策行情后的短期亢奋顶
```

含义：

- 2024-10-08 到 2024-10-18 是顶部区域。
- 2024-10-08 到 2024-10-10 是更适合模型学习的卖出窗口。

## reason 建议枚举

`reason` 是多选字段，用 `|` 分隔。

工具建议提供复选项，同时允许人工输入自定义原因。

### 通用原因

```text
panic_selloff
panic_liquidation
liquidity_crisis
policy_support
policy_expectation
policy_tightening
reopening_trade
trade_war
covid
global_risk
fed_hike_expect
valuation_low
valuation_high
growth_overvalued
high_valuation
uncertain
```

### 技术类底部原因

```text
weekly_boll_lower
monthly_boll_lower
daily_boll_lower
oversold_rebound
kdj_oversold
weekly_kdj_golden_cross
macd_bottom_divergence
volume_shrink
volume_expand
down_with_volume
lower_shadow
technical_pullback
```

### 技术类顶部原因

```text
weekly_boll_upper
monthly_boll_upper
daily_boll_upper
kdj_overbought
weekly_kdj_dead_cross
macd_top_divergence
euphoria
volume_expand
rebound_exhaustion
upper_shadow
new_high_area
```

### 分时原因

```text
intraday_v_reversal
intraday_volume_spike
intraday_boll_break
intraday_kdj_cross
intraday_macd_cross
intraday_failed_breakout
intraday_panic_drop
intraday_exhaustion
```

## notes 含义

`notes` 是自由文本，记录人工判断背景。

建议写法：

```text
2024年2月微盘股流动性危机后形成的重要底部
2024年10月924政策行情后短期亢奋顶部
```

工具可以不强制填写，但建议提供输入框。

## 多周期拆分规则

同一个行情区域如果对应多个周期，必须拆成多行。

示例：

```csv
B202402_daily_short_01,2024-01-22,2024-02-23,bottom,short,swing,daily,3,1,2024-02-05,2024-02-23,,,panic_liquidation|policy_support,日线短期反弹底
B202402_weekly_medium_01,2024-01-22,2024-02-23,bottom,medium,swing,weekly,3,1,2024-02-05,2024-02-23,,,weekly_boll_lower|policy_support,周线中期波段底
B202402_monthly_long_01,2024-01-22,2024-02-23,bottom,long,major,monthly,3,1,2024-02-05,2024-02-23,,,panic_liquidation|liquidity_crisis|national_team,月线长期大底
```

不要这样写：

```csv
B202402_01,2024-01-22,2024-02-23,bottom,short|medium|long,major,daily|weekly|monthly,3,1,2024-02-05,2024-02-23,,,panic_liquidation,错误示例
```

原因：模型训练时需要按 `cycle` 分别训练，混合周期会导致标签含义不清。

## 推荐工具交互

标记工具建议至少支持以下能力：

1. 加载 K 线图。
2. 支持日线、周线、月线切换。
3. 后续支持 60min 分时图。
4. 鼠标拖拽选择 `start_date` 到 `end_date`。
5. 在区域内二次选择 `entry_start` 到 `entry_end` 或 `exit_start` 到 `exit_end`。
6. 选择 `region_type`。
7. 选择 `cycle`。
8. 选择 `level`。
9. 选择 `confidence`。
10. 自动建议 `usable_for_signal`。
11. 多选 `reason`。
12. 填写 `notes`。
13. 自动生成 `region_id`。
14. 保存前做字段校验。
15. 支持编辑和删除已有标注。
16. 支持导入现有 CSV 并继续编辑。
17. 支持导出标准 CSV。

## 可视化建议

不同标注建议使用不同颜色：

| 标注 | 建议颜色 |
| --- | --- |
| short bottom | 浅蓝 |
| medium bottom | 蓝色 |
| long bottom | 深蓝 |
| short top | 浅橙 |
| medium top | 橙色 |
| long top | 红色 |
| usable_for_signal = 0 | 灰色虚线 |

图上建议展示：

- 完整 `start_date` 到 `end_date` 区域。
- 买入窗口或卖出窗口。
- `region_id`。
- `cycle`、`level`、`confidence`。

## 保存前校验规则

工具保存前必须做以下校验：

### 通用校验

- `region_id` 非空。
- `region_id` 唯一。
- `start_date` 非空。
- `end_date` 非空。
- `start_date <= end_date`。
- `region_type` 只能是 `bottom` 或 `top`。
- `cycle` 只能是 `intraday`、`short`、`medium`、`long`。
- `level` 只能是 `minor`、`swing`、`major`。
- `label_freq` 只能是 `intraday`、`daily`、`weekly`、`monthly`。
- `confidence` 只能是 `1`、`2`、`3`。
- `usable_for_signal` 只能是 `0` 或 `1`。

### label_freq 校验

- `cycle = intraday` 时，`label_freq` 必须是 `intraday`。
- `label_freq = intraday` 时，`cycle` 必须是 `intraday`。
- `cycle = short` 时，`label_freq` 推荐为 `daily`，允许 `weekly`。
- `cycle = medium` 时，`label_freq` 推荐为 `daily` 或 `weekly`。
- `cycle = long` 时，`label_freq` 推荐为 `weekly` 或 `monthly`。
- 如果用户选择了不推荐组合，工具可以提示，但不强制禁止。

### bottom 校验

- `entry_start` 非空。
- `entry_end` 非空。
- `exit_start` 必须为空。
- `exit_end` 必须为空。
- `start_date <= entry_start <= entry_end <= end_date`。

### top 校验

- `exit_start` 非空。
- `exit_end` 非空。
- `entry_start` 必须为空。
- `entry_end` 必须为空。
- `start_date <= exit_start <= exit_end <= end_date`。

### cycle 和日期校验

日线工具暂时只允许：

```text
short
medium
long
```

分时工具启用后才允许：

```text
intraday
```

如果 `cycle = intraday`，后续需要使用 `start_time`、`end_time`、`entry_start_time`、`entry_end_time`、`exit_start_time`、`exit_end_time` 这一类分时字段。当前 CSV 先不强制启用分时字段。

## 建议默认值

新增一条标注时，工具可以按以下默认值填充：

| 字段 | 默认值 |
| --- | --- |
| `region_type` | 用户当前选择 |
| `cycle` | `short` |
| `level` | `swing` |
| `label_freq` | 当前图表频率，日线图默认为 `daily`，周线图默认为 `weekly`，月线图默认为 `monthly` |
| `confidence` | `2` |
| `usable_for_signal` | `1` |
| `reason` | 空 |
| `notes` | 空 |

如果 `confidence = 1`，工具应提示是否把 `usable_for_signal` 改为 `0`。

## 示例 CSV

```csv
region_id,start_date,end_date,region_type,cycle,level,label_freq,confidence,usable_for_signal,entry_start,entry_end,exit_start,exit_end,reason,notes
B202402_daily_short_01,2024-02-05,2024-02-23,bottom,short,swing,daily,3,1,2024-02-05,2024-02-23,,,panic_liquidation|lower_shadow,日线短期反弹底
B202402_weekly_medium_01,2024-01-22,2024-02-23,bottom,medium,swing,weekly,3,1,2024-02-05,2024-02-23,,,weekly_boll_lower|weekly_kdj_golden_cross,周线中期波段底
B202402_monthly_long_01,2024-01-22,2024-02-23,bottom,long,major,monthly,3,1,2024-02-05,2024-02-23,,,panic_liquidation|liquidity_crisis|valuation_low,月线长期大底
T202410_daily_short_01,2024-10-08,2024-10-18,top,short,swing,daily,3,1,,,2024-10-08,2024-10-10,euphoria|volume_expand|policy_trade,日线短期亢奋顶
T202410_weekly_medium_01,2024-10-08,2024-10-18,top,medium,swing,weekly,2,1,,,2024-10-08,2024-10-18,weekly_boll_upper|euphoria|volume_expand,周线中期观察顶
```

## 和建模的关系

后续建模会按以下方式使用标注：

| cycle | side | 标签窗口 |
| --- | --- | --- |
| short | bottom | 未来 5、10、15 日 |
| short | top | 未来 5、10、15 日 |
| medium | bottom | 未来 20、40 日 |
| medium | top | 未来 20、40 日 |
| long | bottom | 未来 80、120 日 |
| long | top | 未来 80、120 日 |
| intraday | bottom | 未来 4-6 小时 |
| intraday | top | 未来 4-6 小时 |

标注工具只负责输出人工标签，不负责判断未来收益。

## 未来扩展字段

如果后续要支持分时标注，可以新增独立文件或扩展字段：

```csv
region_id,start_time,end_time,region_type,cycle,level,confidence,usable_for_signal,entry_start_time,entry_end_time,exit_start_time,exit_end_time,reason,notes
```

也可以新增：

```text
source_freq
index_code
created_by
created_at
updated_at
review_status
```

当前第一版标注工具先以日线 CSV 为主。

## 非目标

标记工具第一版不需要做：

- 模型训练。
- 信号预测。
- 数据采集。
- LLM 事件提取。
- 自动识别顶底。
- 复杂权限系统。

第一版只要把人工标注做准确、可编辑、可导出、可校验，就已经足够支撑下一阶段建模。
