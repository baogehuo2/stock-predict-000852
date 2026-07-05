# 顶底标注工具使用说明

## 启动

在项目根目录运行：

```powershell
python -m src.label_tool.app --host 127.0.0.1 --port 8765
```

浏览器打开：

```text
http://127.0.0.1:8765
```

默认标注文件：

```text
data/manual_labels/market_turning_regions.csv
```

如需指定其他标注文件：

```powershell
python -m src.label_tool.app --label-path data/manual_labels/market_turning_regions_test.csv
```

## 行情数据来源

日 K 从 MySQL 表 `market_index_daily` 读取，使用 `config/db.yaml` 中的数据库配置和
`config/config.yaml` 中的 `project.target_index`，当前默认是 `000852`。

周 K、月 K 由日 K 聚合生成：

- 周 K：按自然交易周聚合。
- 月 K：按自然月聚合。
- 开盘取区间第一条，收盘取区间最后一条，最高/最低取极值，成交量和成交额求和。

小时 K 当前会依次尝试以下表：

```text
market_index_hourly
market_index_60m
market_index_intraday
```

小时表需要包含：

```text
trade_time,index_code,open,high,low,close,volume,amount
```

如果暂时没有小时表，日 K、周 K、月 K 不受影响。

## 操作

- 顶部切换 `1H / 日K / 周K / 月K`。
- 主图指标支持 `MA / BOLL / 无`，BOLL 支持 `2.0 / 3.0` 倍标准差切换。
- 副图指标支持 `MACD / KDJ / RSI / CCI / WR / ATR / 无`，成交量固定显示。
- 日K、周K、月K和小时K指标均先用完整可得历史预热，再截取当前显示区间，避免从页面起始日期重新计算导致递推指标偏差。
- RSI 使用东方财富/通达信常见 `SMA(X,N,1)` 平滑公式；KDJ 使用 `RSV + SMA(3,1)` 口径；MACD 柱按行情软件显示口径使用 `2 * (DIF - DEA)`；BOLL 标准差固定使用 `ddof=0`。
- 技术指标自检报告输出到：

```text
data/reports/technical_indicator_self_check.csv
```
- 点击 `生成候选` 会基于历史行情生成短期、中期、长期顶底候选，写入：

```text
data/manual_labels/turning_region_candidates.csv
```

- 勾选 `候选` 时，候选区域会以虚线显示在 K 线上。
- 点击右侧候选列表中的候选项，会把候选内容预填到右侧标注编辑面板。
- 候选不会自动进入正式标注，只有点击 `保存` 后才会写入正式 CSV。
- 可以使用 ECharts 自带框选工具选择一段区域，右侧会打开标注编辑面板。
- 也可以点击 `手动画线`，然后在主图 K 线区域按住鼠标拖出范围，松开后右侧会打开标注编辑面板。
- 标注编辑面板固定在右侧，不遮挡 K 线，便于一边看图一边调整 `entry_start`、`entry_end`、`exit_start`、`exit_end`。
- 标注窗口里的日期可以手写修改。
- `bottom` 标注填写 `entry_start / entry_end`。
- `top` 标注填写 `exit_start / exit_end`。
- 保存前会校验字段、日期范围、`region_id` 唯一性和 `label_freq` 规则。

## CSV 字段

工具输出字段顺序固定为：

```csv
region_id,start_date,end_date,region_type,cycle,level,label_freq,confidence,usable_for_signal,entry_start,entry_end,exit_start,exit_end,reason,notes
```
