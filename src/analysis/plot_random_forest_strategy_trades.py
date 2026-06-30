from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass

import pandas as pd

from src.analysis.backtest_bp_xgb_indicator_strategy import _load_market, _load_target_index
from src.common.config import project_path


DEFAULT_TRADES = "data/reports/manual_weak_turning_trade_strategy_trades_random_forest.csv"
DEFAULT_SUMMARY = "data/reports/manual_weak_turning_trade_strategy_summary_random_forest.csv"
DEFAULT_OUTPUT = "data/reports/random_forest_compressed_regions_trade_pairs.html"
DEFAULT_STRATEGY = "compressed_regions"


@dataclass(frozen=True)
class TradePair:
    trade_id: int
    strategy: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    trade_return: float
    status: str
    exit_reason: str
    entry_count: int
    holding_days: int


def _pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _fmt(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _load_trades(path_value: str, strategy: str) -> pd.DataFrame:
    path = project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Trade file not found: {path}")
    trades = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    trades = trades[trades["strategy"] == strategy].copy()
    if trades.empty:
        raise RuntimeError(f"No random forest trades found for strategy={strategy}.")
    for column in ["entry_price", "exit_price", "trade_return"]:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")
    trades["entry_count"] = pd.to_numeric(trades["entry_count"], errors="coerce").fillna(1).astype(int)
    trades["holding_days"] = pd.to_numeric(trades["holding_days"], errors="coerce").fillna(0).astype(int)
    return trades.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)


def _load_summary(path_value: str, strategy: str) -> dict:
    path = project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    summary = pd.read_csv(path)
    summary = summary[summary["strategy"] == strategy].copy()
    if summary.empty:
        raise RuntimeError(f"No random forest summary found for strategy={strategy}.")
    row = summary.iloc[0]
    return {
        "strategy_total_return": float(row["strategy_total_return"]),
        "max_drawdown": float(row["max_drawdown"]),
        "win_rate": float(row["win_rate"]),
        "closed_trades": int(row["closed_trades"]),
        "open_trades": int(row["open_trades"]),
        "avg_holding_days": float(row["avg_holding_days"]),
        "bottom_signal_count": int(row["bottom_signal_count"]),
        "top_signal_count": int(row["top_signal_count"]),
    }


def _to_pairs(trades: pd.DataFrame) -> list[TradePair]:
    pairs: list[TradePair] = []
    for idx, row in enumerate(trades.itertuples(index=False), start=1):
        pairs.append(
            TradePair(
                trade_id=idx,
                strategy=str(row.strategy),
                entry_date=pd.Timestamp(row.entry_date),
                exit_date=pd.Timestamp(row.exit_date),
                entry_price=float(row.entry_price),
                exit_price=float(row.exit_price),
                trade_return=float(row.trade_return),
                status=str(row.status),
                exit_reason=str(row.exit_reason),
                entry_count=int(row.entry_count),
                holding_days=int(row.holding_days),
            )
        )
    return pairs


def _render_trade_table(pairs: list[TradePair]) -> str:
    headers = ["#", "buy", "sell", "return", "holding_days", "entry_count", "status", "reason"]
    header_html = "".join(f"<th>{html.escape(col)}</th>" for col in headers)
    body = []
    for pair in pairs:
        css_class = "win" if pair.trade_return > 0 else "loss"
        body.append(
            f'<tr class="{css_class}">'
            f"<td>{pair.trade_id}</td>"
            f"<td>{pair.entry_date:%Y-%m-%d}<br>{_fmt(pair.entry_price)}</td>"
            f"<td>{pair.exit_date:%Y-%m-%d}<br>{_fmt(pair.exit_price)}</td>"
            f"<td>{_pct(pair.trade_return)}</td>"
            f"<td>{pair.holding_days}</td>"
            f"<td>{pair.entry_count}</td>"
            f"<td>{html.escape(pair.status)}</td>"
            f"<td>{html.escape(pair.exit_reason)}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _render_chart_payload(market: pd.DataFrame, pairs: list[TradePair]) -> dict:
    market = market.sort_values("trade_date").copy()
    dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in market["trade_date"]]
    date_to_index = {trade_date: index for index, trade_date in enumerate(dates)}
    candles = [
        [_round(row.open), _round(row.close), _round(row.low), _round(row.high)]
        for row in market.itertuples(index=False)
    ]
    closes = [_round(row.close) for row in market.itertuples(index=False)]

    buy_points = []
    sell_profit_points = []
    sell_loss_points = []
    profit_lines = []
    loss_lines = []
    for pair in pairs:
        entry_date = pair.entry_date.strftime("%Y-%m-%d")
        exit_date = pair.exit_date.strftime("%Y-%m-%d")
        entry_index = date_to_index.get(entry_date)
        exit_index = date_to_index.get(exit_date)
        if entry_index is None or exit_index is None:
            continue
        trade_label = f"#{pair.trade_id} {_pct(pair.trade_return)} {pair.exit_reason}"
        buy_points.append(
            {
                "name": f"B{pair.trade_id}",
                "value": [entry_index, _round(pair.entry_price)],
                "label": f"B{pair.trade_id}",
                "tooltip": f"B{pair.trade_id} {entry_date} {trade_label}",
            }
        )
        sell_point = {
            "name": f"S{pair.trade_id}",
            "value": [exit_index, _round(pair.exit_price)],
            "label": f"S{pair.trade_id}",
            "tooltip": f"S{pair.trade_id} {exit_date} {trade_label}",
        }
        line = [
            entry_index,
            _round(pair.entry_price),
            exit_index,
            _round(pair.exit_price),
            pair.trade_id,
            _round(pair.trade_return, 6),
        ]
        if pair.trade_return > 0:
            sell_profit_points.append(sell_point)
            profit_lines.append(line)
        else:
            sell_loss_points.append(sell_point)
            loss_lines.append(line)

    return {
        "dates": dates,
        "candles": candles,
        "closes": closes,
        "buyPoints": buy_points,
        "sellProfitPoints": sell_profit_points,
        "sellLossPoints": sell_loss_points,
        "profitLines": profit_lines,
        "lossLines": loss_lines,
    }


def _render_html(title: str, summary: dict, chart_payload: dict, table: str) -> str:
    chart_json = json.dumps(chart_payload, ensure_ascii=False, allow_nan=False)
    replacements = {
        "__TITLE__": html.escape(title),
        "__TOTAL_RETURN__": _pct(summary["strategy_total_return"]),
        "__MAX_DRAWDOWN__": _pct(summary["max_drawdown"]),
        "__WIN_RATE__": _pct(summary["win_rate"]),
        "__CLOSED_TRADES__": str(summary["closed_trades"]),
        "__OPEN_TRADES__": str(summary["open_trades"]),
        "__AVG_HOLDING__": f'{summary["avg_holding_days"]:.2f}d',
        "__BOTTOM_COUNT__": str(summary["bottom_signal_count"]),
        "__TOP_COUNT__": str(summary["top_signal_count"]),
        "__CHART_JSON__": chart_json,
        "__TABLE__": table,
    }
    html_text = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
    body { font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; color: #111827; background: #f6f7f9; }
    main { max-width: 1500px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 8px; font-size: 26px; }
    h2 { margin: 20px 0 10px; font-size: 20px; }
    .muted { color: #6b7280; font-size: 13px; }
    .cards { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin: 16px 0; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; }
    .value { font-size: 20px; font-weight: 700; margin-top: 4px; }
    .chart-wrap { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px; }
    #tradeChart { width: 100%; height: 860px; }
    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; vertical-align: top; }
    th { background: #f3f4f6; }
    tr.win td:nth-child(4) { color: #dc2626; font-weight: 700; }
    tr.loss td:nth-child(4) { color: #16a34a; font-weight: 700; }
    .legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 10px 0; font-size: 13px; color: #374151; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
  </style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <div class="muted">B# 是买点，S# 是对应卖点；同一编号代表同一笔交易。支持鼠标滚轮缩放、拖动平移、底部滑块缩放。</div>
  <section class="cards">
    <div class="card"><div class="muted">total return</div><div class="value">__TOTAL_RETURN__</div></div>
    <div class="card"><div class="muted">max drawdown</div><div class="value">__MAX_DRAWDOWN__</div></div>
    <div class="card"><div class="muted">win rate</div><div class="value">__WIN_RATE__</div></div>
    <div class="card"><div class="muted">closed/open</div><div class="value">__CLOSED_TRADES__/__OPEN_TRADES__</div></div>
    <div class="card"><div class="muted">avg holding</div><div class="value">__AVG_HOLDING__</div></div>
    <div class="card"><div class="muted">bottom/top signals</div><div class="value">__BOTTOM_COUNT__/__TOP_COUNT__</div></div>
  </section>
  <div class="legend">
    <span><span class="dot" style="background:#2563eb"></span>buy B#</span>
    <span><span class="dot" style="background:#dc2626"></span>profitable sell S#</span>
    <span><span class="dot" style="background:#16a34a"></span>losing sell S#</span>
  </div>
  <section class="chart-wrap"><div id="tradeChart"></div></section>
  <h2>Trade pairs</h2>
  __TABLE__
</main>
<script>
(() => {
  const chartData = __CHART_JSON__;
  const winColor = "#dc2626";
  const lossColor = "#16a34a";
  const buyColor = "#2563eb";
  const chart = echarts.init(document.getElementById("tradeChart"));

  function tradeLineSeries(name, color, data) {
    return {
      name,
      type: "custom",
      coordinateSystem: "cartesian2d",
      data,
      silent: true,
      z: 4,
      renderItem: function (_params, api) {
        const p1 = api.coord([api.value(0), api.value(1)]);
        const p2 = api.coord([api.value(2), api.value(3)]);
        return {
          type: "line",
          shape: { x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1] },
          style: { stroke: color, lineWidth: 1.8, opacity: 0.72 },
        };
      },
    };
  }

  function pointSeries(name, color, data, symbol) {
    return {
      name,
      type: "scatter",
      data,
      symbol,
      symbolSize: 15,
      itemStyle: { color, borderColor: "#fff", borderWidth: 1.5 },
      label: {
        show: true,
        formatter: p => p.data.label,
        position: "top",
        color,
        fontWeight: "bold",
        fontSize: 12,
      },
      tooltip: {
        formatter: p => p.data.tooltip,
      },
      z: 8,
    };
  }

  const option = {
    animation: false,
    legend: { top: 8 },
    tooltip: {
      trigger: "axis",
      axisPointer: { type: "cross" },
    },
    grid: [
      { left: 60, right: 36, top: 52, height: 640 },
      { left: 60, right: 36, top: 720, height: 60 },
    ],
    xAxis: [
      { type: "category", data: chartData.dates, boundaryGap: false, axisLine: { onZero: false }, min: "dataMin", max: "dataMax" },
      { type: "category", data: chartData.dates, gridIndex: 1, boundaryGap: false, axisLabel: { show: false }, axisTick: { show: false } },
    ],
    yAxis: [
      { scale: true, splitArea: { show: true } },
      { scale: true, gridIndex: 1, splitNumber: 2, axisLabel: { show: false }, axisTick: { show: false } },
    ],
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1], start: 0, end: 100, zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false },
      { type: "slider", xAxisIndex: [0, 1], top: 792, height: 28, start: 0, end: 100 },
    ],
    series: [
      { name: "K线", type: "candlestick", data: chartData.candles, itemStyle: { color: "#ef4444", color0: "#10b981", borderColor: "#ef4444", borderColor0: "#10b981" } },
      { name: "收盘", type: "line", data: chartData.closes, smooth: true, showSymbol: false, lineStyle: { width: 1.2, color: "#64748b" }, z: 2 },
      tradeLineSeries("profit pair", winColor, chartData.profitLines),
      tradeLineSeries("loss pair", lossColor, chartData.lossLines),
      pointSeries("buy", buyColor, chartData.buyPoints, "triangle"),
      pointSeries("profitable sell", winColor, chartData.sellProfitPoints, "circle"),
      pointSeries("losing sell", lossColor, chartData.sellLossPoints, "circle"),
      { name: "overview", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: chartData.closes, showSymbol: false, lineStyle: { color: "#94a3b8", width: 1 } },
    ],
  };
  chart.setOption(option);
  window.addEventListener("resize", () => chart.resize());
})();
</script>
</body>
</html>
"""
    for key, value in replacements.items():
        html_text = html_text.replace(key, value)
    return html_text


def build_html(
    strategy: str = DEFAULT_STRATEGY,
    trades_path: str = DEFAULT_TRADES,
    summary_path: str = DEFAULT_SUMMARY,
    output_path: str = DEFAULT_OUTPUT,
    start_date: str = "2022-01-01",
) -> str:
    trades = _load_trades(trades_path, strategy)
    pairs = _to_pairs(trades)
    summary = _load_summary(summary_path, strategy)
    target_index = _load_target_index()
    market = _load_market(target_index, start_date)
    payload = _render_chart_payload(market, pairs)
    table = _render_trade_table(pairs)
    title = f"Random Forest {strategy} Trade Pairs"
    html_text = _render_html(title, summary, payload, table)
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot random forest strategy trade pairs.")
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--trades", default=DEFAULT_TRADES)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="2022-01-01")
    args = parser.parse_args()

    print(build_html(args.strategy, args.trades, args.summary, args.output, args.start_date))


if __name__ == "__main__":
    main()
