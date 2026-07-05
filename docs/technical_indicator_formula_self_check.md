# 技术指标公式自查与重构规范

适用分支：`feature/bottom-fishing-v1`

对应代码重点：

- `src/features/build_bottom_dataset.py`
- 如从 buy 分支迁移技术指标工具函数，必须同步更新本文档。

## 目标

重新构建技术指标前，必须先把行情数据口径、K线频率、公式实现、预热窗口和历史回归样本逐项自查，避免模型使用与真实行情软件不一致的技术指标。

本文档的自查原则以 AkShare 文档为准：

- AkShare 负责定义行情接口、字段、频率、复权和数据来源。
- 技术指标公式必须基于 AkShare 返回的 `open/high/low/close/volume/amount` 等字段复算。
- 如果 AkShare 文档没有给出指标公式，则用东方财富/通达信常见指标公式作为显示口径，并用 AkShare 取回的行情数据做抽样复核。
- 不能只用 pandas 默认实现替代行情软件公式；递推类指标必须有足够历史预热。

参考入口：

- AkShare 指数数据文档：`https://akshare.akfamily.xyz/data/index/index.html`
- AkShare 股票数据文档：`https://akshare.akfamily.xyz/data/stock/stock.html`
- AkShare 指数分钟接口：`index_zh_a_hist_min_em`
- AkShare 东方财富指数历史接口：`stock_zh_index_daily_em`

## 总体自查原则

### 1. 数据源口径

每个指标重构前先确认输入行情表来自哪个 AkShare 接口。

日K优先：

- 中证1000指数：优先用东方财富指数历史行情口径。
- 字段必须包含：`trade_date/open/high/low/close/volume/amount`。
- 指数不做股票复权逻辑；不要把股票 `adjust` 口径误用于指数。

分钟K优先：

- 使用 AkShare 东方财富指数分钟接口口径。
- 1小时K线需要确认日内切分点。常见东方财富1小时K线为每日4根：`10:30/11:30/14:00/15:00`。
- 不要直接相信数据库中已有 `freq='60min'` 表就是东方财富1小时K线；先抽样核对每日根数和结束时间。

### 2. 时间与泄漏

- 日K技术指标只能使用当日及以前已经形成的完整K线。
- 分钟K技术指标只能使用该分钟/该周期收盘及以前已经形成的K线。
- 周K特征如果是日内滚动形成的“当周进行中K线”，必须标记为 causal weekly，不得误认为完整周K。
- 所有递推指标必须用足够长历史预热，再截取研究区间；不能从研究起始日才开始计算。

### 3. 单位与缩放

必须记录每个字段单位：

- `ret_*`、`gap`、`drawdown`、`volatility` 使用小数收益率，例如 `0.01` 表示 `1%`。
- RSI、KDJ、CCI、WR 使用指标自身数值范围。
- `volume/amount` 使用数据库原始单位，不得混用手、股、万元、元。

### 4. 缺失值与窗口

- 普通滚动均线默认需要完整窗口，除非明确写明 `min_periods=1` 是为了复现行情软件初始显示。
- 递推指标必须记录初始值。
- 分母为0时必须输出 `NaN`，不能静默填0。
- 指标用于训练前再统一由模型 pipeline 做缺失值处理。

## 必须自查的指标清单

### 收益、回撤与均线

当前字段：

- `ret_1d/ret_3d/ret_5d/ret_10d`
- `downside_ret_3d/downside_ret_5d`
- `drawdown_20d/drawdown_60d/drawdown_120d`
- `distance_low_5d/distance_low_10d/distance_low_20d`
- `ma5_gap/ma10_gap/ma20_gap/ma60_gap`
- `ma20_slope_5d/ma60_slope_10d`
- `below_ma20_days/below_ma60_days`

公式口径：

```text
ret_N = close / close.shift(N) - 1
downside_ret_N = SUM(MIN(ret_1d, 0), N)
drawdown_N = close / rolling_max(close 或 high, N) - 1
distance_low_N = close / rolling_min(low, N) - 1
MA_N = mean(close, N)
maN_gap = close / MA_N - 1
```

自查要求：

- `drawdown_N` 必须明确使用 close 高点还是 high 高点。
- 抄底模型如果强调最大下跌深度，建议评估 `close / HHV(high,N) - 1`。
- 连续天数只按交易日计数。

### RSI

当前风险：历史代码中 `_rsi()` 使用简单滚动均值，和东方财富显示口径不一致。

必须改为东方财富/通达信常见平滑公式：

```text
LC = REF(CLOSE, 1)
UP = MAX(CLOSE - LC, 0)
ABS_DIFF = ABS(CLOSE - LC)
SMA(X, N, 1) = (X + (N - 1) * REF(SMA(X, N, 1), 1)) / N
RSI_N = SMA(UP, N, 1) / SMA(ABS_DIFF, N, 1) * 100
```

自查样本：

- 中证1000日K `2025-03-24`
- AkShare/数据库收盘价：`6360.341`
- RSI6 应约等于 `29.14`
- 如果只从 `2024-09-24` 开始预热，会得到偏差值；必须用完整可得历史预热。

实现要求：

- 新增统一 `_sma_cn(values, window, weight=1)`。
- `rsi6/rsi14` 全部调用统一实现。
- 抄底候选阈值 `rsi6 <= 30` 依赖该口径，修正后必须重新评估候选覆盖率。

### WR

当前字段：

- `wr14`
- 候选阈值 `wr14 <= -80`

当前代码使用负数口径：

```text
wr14 = (HHV(HIGH,14) - CLOSE) / (HHV(HIGH,14) - LLV(LOW,14)) * -100
```

常见行情软件也可能显示正数口径：

```text
WR14_POSITIVE = (HHV(HIGH,14) - CLOSE) / (HHV(HIGH,14) - LLV(LOW,14)) * 100
```

自查要求：

- 如果保留负数口径，阈值仍是 `<= -80`。
- 如果改正数口径，阈值应同步调整为 `>= 80`。
- 不能只改公式不改阈值。

### CCI

当前字段：

- `cci14`

常见公式：

```text
TYP = (HIGH + LOW + CLOSE) / 3
CCI_N = (TYP - MA(TYP, N)) / (0.015 * AVEDEV(TYP, N))
AVEDEV(TYP, N) = MA(ABS(TYP - MA(TYP, N)), N)
```

自查要求：

- 当前实现接近常见公式，但必须确认 `AVEDEV` 的滚动窗口和 `MA(TYP,N)` 对齐。
- 分母为0输出 `NaN`。

### BOLL

当前字段：

- `boll_width`
- `boll_position`
- `boll_lower_break`
- 周K `week_boll_*`

常见公式：

```text
MID = MA(CLOSE, 20)
STD20 = STD(CLOSE, 20)
UPPER = MID + 2 * STD20
LOWER = MID - 2 * STD20
boll_width = (UPPER - LOWER) / MID
boll_position = (CLOSE - LOWER) / (UPPER - LOWER)
```

自查要求：

- pandas `rolling.std()` 默认 `ddof=1`，部分行情软件可能使用总体标准差 `ddof=0`。必须用样本对齐后固定。
- 候选阈值 `boll_position <= 0.15` 依赖 BOLL 口径，修正后必须重新统计。

### MACD

当前字段：

- `macd`
- `macd_signal`
- `macd_hist`
- `macd_hist_delta_1d/macd_hist_delta_3d`

建议公式：

```text
EMA12 = EMA(CLOSE, 12)
EMA26 = EMA(CLOSE, 26)
DIF = EMA12 - EMA26
DEA = EMA(DIF, 9)
MACD_BAR_DISPLAY = 2 * (DIF - DEA)
MACD_HIST_MODEL = DIF - DEA
```

自查要求：

- 东方财富图上通常显示 `MACD = 2 * (DIF - DEA)`。
- 如果模型字段继续使用 `DIF - DEA`，必须在字段说明里写清楚它是半幅柱。

### KDJ

当前字段：

- `kdj_k/kdj_d/kdj_j`
- `kdj_k_minus_d`
- `kdj_golden_cross`
- 周K `week_kdj_*`

常见公式：

```text
RSV = (CLOSE - LLV(LOW, 9)) / (HHV(HIGH, 9) - LLV(LOW, 9)) * 100
K = SMA(RSV, 3, 1)
D = SMA(K, 3, 1)
J = 3 * K - 2 * D
```

自查要求：

- 初始 `K/D` 建议按行情软件口径设为 `50`，或用足够长历史预热后再截取。
- 当前 `ewm(alpha=1/3, adjust=False)` 数学上接近 `SMA(X,3,1)`，但初值不同；必须通过样本核对。
- 当 `HHV == LLV` 时，RSV 不得产生无穷值。

### ATR 与波动

当前字段：

- `atr14`
- `volatility_5d/10d/20d`
- `volatility_expand`

常见公式：

```text
TR = MAX(HIGH - LOW, ABS(HIGH - REF(CLOSE,1)), ABS(LOW - REF(CLOSE,1)))
ATR14 = MA(TR, 14)
atr14_model = ATR14 / CLOSE
volatility_N = STD(close.pct_change(), N)
volatility_expand = volatility_5d / volatility_20d
```

自查要求：

- 行情软件 ATR 通常是不除以 close 的点数值；模型当前使用归一化 `ATR / close`。必须在字段说明中写清。
- `volatility` 是收益率标准差，不是价格标准差。

### 影线、缺口、振幅与连续涨跌

当前字段：

- `intraday_range`
- `body_return`
- `lower_shadow_ratio`
- `long_lower_shadow`
- `gap_down`
- `down_streak/up_streak`

公式：

```text
range = HIGH - LOW
body_return = CLOSE / OPEN - 1
lower_shadow = MIN(OPEN, CLOSE) - LOW
lower_shadow_ratio = lower_shadow / range
intraday_range = range / previous_close
gap_down = OPEN < REF(CLOSE,1) * 0.995
```

自查要求：

- buy 分支 `intraday_range = range / close`，bottom 分支当前是 `range / previous_close`。如果要共用指标函数，必须统一或明确保留差异。
- `range=0` 时输出 `NaN`。

### 成交量与相对强弱

当前字段：

- `volume_zscore_20d`
- `volume_ratio_5d_20d`
- `relative_hs300_1d/5d`
- `relative_zz500_1d/5d`
- `relative_cyb_1d/5d`

公式：

```text
volume_zscore_20d = (volume - MA(volume,20)) / STD(volume,20)
volume_ratio_5d_20d = MA(volume,5) / MA(volume,20)
relative_index_1d = ret_000852_1d - ret_benchmark_1d
relative_index_5d = ret_000852_5d - ret_benchmark_5d
```

自查要求：

- AkShare 不同接口的 `volume/amount` 单位可能不同，必须记录数据库单位。
- 相对强弱必须保证目标指数和基准指数在同一交易日对齐。

### 周K与因果周K特征

当前字段：

- `week_*`

自查要求：

- 当前周K是“因果进行中周K”：周内每日使用当周截至当天的数据，不是完整周五K线。
- 文档和字段名必须明确“进行中周K”。
- 周K KDJ/BOLL/MA 也要使用同一套指标公式自查。
- 不能在周二使用周五收盘后才知道的完整周K。

### 候选规则

当前字段：

- `candidate_depth`
- `candidate_oversold`
- `candidate_trend`
- `candidate_stress`
- `candidate_score`
- `is_candidate`

自查要求：

- RSI、WR、BOLL 任一口径变化，都必须重新统计 `candidate_oversold`。
- `candidate_score >= min_conditions` 的覆盖率要按年度输出。
- 重构后必须重新评估 `quality_bottom_label/path_rebound_label/continuation_risk_label` 的样本分布。

## 重构实施要求

### 统一工具函数

建议从 buy 分支迁移或共同维护统一模块：

```text
src/features/technical_indicators.py
```

至少包含：

- `sma_cn(values, window, weight=1)`
- `rsi_cn(close, window)`
- `ema(series, span)`
- `macd(close, fast=12, slow=26, signal=9, bar_multiplier=2)`
- `kdj(high, low, close, n=9, m1=3, m2=3)`
- `boll(close, window=20, multiplier=2, ddof=0 or 1)`
- `cci(high, low, close, window=14)`
- `wr(high, low, close, window=14, sign="negative")`
- `atr(high, low, close, window=14, normalize=True)`

buy 和 bottom 分支不能各自复制一套不一致公式。

### 回归样本

至少固定以下样本：

```text
000852 日K 2025-03-24 RSI6 ~= 29.14
000852 日K 2025-03-24 close = 6360.341
000852 60分钟K 2025-03-24 必须为 10:30/11:30/14:00/15:00 四根
000852 120分钟K 2025-03-24 必须为 11:30/15:00 两根
```

每次重构后输出一份 CSV：

```text
data/reports/technical_indicator_self_check.csv
```

建议字段：

```text
freq
trade_time
open
high
low
close
rsi6
rsi14
macd_dif
macd_dea
macd_bar
kdj_k
kdj_d
kdj_j
boll_mid
boll_upper
boll_lower
cci14
wr14
atr14
source
check_status
note
```

### 禁止项

- 禁止只因为 pandas 代码能运行就认为指标正确。
- 禁止从测试区间起点才开始计算递推指标。
- 禁止分钟K直接使用未核对过的数据库聚合表。
- 禁止只改指标公式、不重跑特征分布、候选覆盖率和滚动验证。

## 当前已知高风险点

1. `rsi6/rsi14` 当前简单 rolling RSI 与东方财富 RSI 不一致，必须优先修正。
2. `WR` 当前为负数口径，阈值依赖此口径，不能随意改为正数。
3. `BOLL` 标准差 `ddof` 未固定，需要抽样决定。
4. `KDJ` 初值和 `SMA` 递推口径需要抽样核对。
5. `MACD hist` 如果对齐行情软件显示，可能需要乘2；如果不乘2，必须明确是模型半幅柱。
6. `intraday_range` 与 buy 分支分母不同，需要统一或写清差异。
7. 60分钟K线必须从1分钟数据按东方财富1h切分重新合成后再计算指标。

## 验收条件

重构完成后必须满足：

1. 生成并提交指标自查 CSV。
2. 关键样本值对齐 AkShare/东方财富行情口径。
3. RSI6 在 `2025-03-24` 日K样本上约等于 `29.14`。
4. 所有指标字段有公式说明、单位说明和缺失值说明。
5. buy 与 bottom 分支使用同一套核心指标函数。
6. 重跑 bottom 模型前先说明哪些指标发生口径变化，并重新做滚动样本外验证。
