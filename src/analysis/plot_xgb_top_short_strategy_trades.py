from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.analysis.backtest_bp_xgb_indicator_strategy import _load_market, _load_target_index
from src.common.config import project_path


DEFAULT_TRADES = "data/reports/xgb_top_short_strategy_trades_all_cost.csv"
DEFAULT_SUMMARY = "data/reports/xgb_top_short_strategy_summary_all_cost.csv"
DEFAULT_OUTPUT = "data/reports/xgb_top_short_strategy_trade_pairs_all_cost.html"


@dataclass(frozen=True)
class ShortTradePair:
    trade_id: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    trade_return: float
    status: str
    exit_reason: str
    entry_count: int
    cover_fraction: float


def _pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _fmt(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _load_trades(path_value: str) -> pd.DataFrame:
    path = project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Trade file not found: {path}")
    trades = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    for column in ["entry_price", "exit_price", "trade_return", "cover_fraction"]:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")
    trades["entry_count"] = pd.to_numeric(trades["entry_count"], errors="coerce").fillna(1).astype(int)
    return trades.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)


def _to_pairs(trades: pd.DataFrame) -> list[ShortTradePair]:
    pairs: list[ShortTradePair] = []
    for idx, row in enumerate(trades.itertuples(index=False), start=1):
        pairs.append(
            ShortTradePair(
                trade_id=idx,
                entry_date=pd.Timestamp(row.entry_date),
                exit_date=pd.Timestamp(row.exit_date),
                entry_price=float(row.entry_price),
                exit_price=float(row.exit_price),
                trade_return=float(row.trade_return),
                status=str(row.status),
                exit_reason=str(row.exit_reason),
                entry_count=int(row.entry_count),
                cover_fraction=float(row.cover_fraction),
            )
        )
    return pairs


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _render_trade_table(pairs: list[ShortTradePair]) -> str:
    headers = [
        "#",
        "short entry",
        "cover",
        "return",
        "entry_count",
        "cover_fraction",
        "status",
        "reason",
    ]
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
            f"<td>{pair.entry_count}</td>"
            f"<td>{_pct(pair.cover_fraction)}</td>"
            f"<td>{html.escape(pair.status)}</td>"
            f"<td>{html.escape(pair.exit_reason)}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _render_chart_payload(market: pd.DataFrame, pairs: list[ShortTradePair]) -> dict:
    market = market.sort_values("trade_date").copy()
    dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in market["trade_date"]]
    date_to_index = {trade_date: index for index, trade_date in enumerate(dates)}
    candles = [
        [_round(row.open), _round(row.close), _round(row.low), _round(row.high)]
        for row in market.itertuples(index=False)
    ]
    closes = [_round(row.close) for row in market.itertuples(index=False)]

    short_points = []
    cover_profit_points = []
    cover_loss_points = []
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
        short_points.append(
            {
                "name": f"S{pair.trade_id}",
                "value": [entry_index, _round(pair.entry_price)],
                "label": f"S{pair.trade_id}",
                "tooltip": f"S{pair.trade_id} short {entry_date} {trade_label}",
            }
        )
        cover_point = {
            "name": f"C{pair.trade_id}",
            "value": [exit_index, _round(pair.exit_price)],
            "label": f"C{pair.trade_id}",
            "tooltip": f"C{pair.trade_id} cover {exit_date} {trade_label}",
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
            cover_profit_points.append(cover_point)
            profit_lines.append(line)
        else:
            cover_loss_points.append(cover_point)
            loss_lines.append(line)

    return {
        "dates": dates,
        "candles": candles,
        "closes": closes,
        "shortPoints": short_points,
        "coverProfitPoints": cover_profit_points,
        "coverLossPoints": cover_loss_points,
        "profitLines": profit_lines,
        "lossLines": loss_lines,
    }


def _render_html(title: str, summary: dict, chart_payload: dict, table: str) -> str:
    chart_json = json.dumps(chart_payload, ensure_ascii=False, allow_nan=False)
    replacements = {
        "__TITLE__": html.escape(title),
        "__TOTAL_RETURN__": _pct(summary["total_return"]),
        "__MAX_DRAWDOWN__": _pct(summary["max_drawdown"]),
        "__WIN_RATE__": _pct(summary["win_rate"]),
        "__CLOSED_TRADES__": str(int(summary["closed_trades"])),
        "__AVG_HOLDING__": f'{float(summary["avg_holding_trade_days"]):.2f}d',
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
    .cards { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 16px 0; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; }
    .value { font-size: 20px; font-weight: 700; margin-top: 4px; }
    .chart-wrap { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px; }
    #tradeChart { width: 100%; height: 820px; }
    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; vertical-align: top; }
    th { background: #f3f4f6; }
    tr.win td:nth-child(4) { color: #dc2626; font-weight: 700; }
    tr.loss td:nth-child(4) { color: #16a34a; font-weight: 700; }
    .legend { display: flex; gap: 16px; flex-wrap: wrap; margin: 10px 0 10px; font-size: 13px; color: #374151; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
  </style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <div class="muted">S# is short entry, C# is the matching cover. Same number means one short trade pair.</div>
  <section class="cards">
    <div class="card"><div class="muted">total return</div><div class="value">__TOTAL_RETURN__</div></div>
    <div class="card"><div class="muted">max drawdown</div><div class="value">__MAX_DRAWDOWN__</div></div>
    <div class="card"><div class="muted">win rate</div><div class="value">__WIN_RATE__</div></div>
    <div class="card"><div class="muted">closed trades</div><div class="value">__CLOSED_TRADES__</div></div>
    <div class="card"><div class="muted">avg holding</div><div class="value">__AVG_HOLDING__</div></div>
  </section>
  <div class="legend">
    <span><span class="dot" style="background:#7c3aed"></span>short entry S#</span>
    <span><span class="dot" style="background:#dc2626"></span>profitable cover C#</span>
    <span><span class="dot" style="background:#16a34a"></span>losing cover C#</span>
  </div>
  <div class="muted">缩放方式同 turning-label-tool：鼠标滚轮缩放/平移，底部滑块控制显示区间。</div>
  <section class="chart-wrap"><div id="tradeChart"></div></section>
  <h2>Short trade pairs</h2>
  __TABLE__
</main>
<script>
(() => {
  const chartData = __CHART_JSON__;
  const winColor = "#dc2626";
  const lossColor = "#16a34a";
  const shortColor = "#7c3aed";
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

  function signalTooltip(params) {
    return params.data?.tooltip || params.name || "";
  }

  const option = {
    animation: false,
    legend: { top: 8, left: 8, type: "scroll" },
    tooltip: { trigger: "axis", axisPointer: { type: "cross" }, confine: true },
    axisPointer: { link: [{ xAxisIndex: "all" }] },
    grid: [{ left: 60, right: 28, top: 48, bottom: 58 }],
    xAxis: [{ type: "category", data: chartData.dates, boundaryGap: true, axisLine: { onZero: false }, splitLine: { show: false } }],
    yAxis: [{ scale: true, splitArea: { show: true } }],
    dataZoom: [
      { type: "inside", xAxisIndex: [0], start: 0, end: 100, minSpan: 8, filterMode: "none", zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false },
      { type: "slider", xAxisIndex: [0], bottom: 12, height: 22, start: 0, end: 100, minSpan: 8, filterMode: "none" },
    ],
    series: [
      {
        name: "K线",
        type: "candlestick",
        data: chartData.candles,
        itemStyle: { color: "#ef4444", color0: "#16a34a", borderColor: "#dc2626", borderColor0: "#16a34a" },
      },
      { name: "收盘线", type: "line", data: chartData.closes, symbol: "none", lineStyle: { color: "#1d4ed8", width: 1.3, opacity: 0.58 } },
      tradeLineSeries("盈利连线", winColor, chartData.profitLines),
      tradeLineSeries("亏损连线", lossColor, chartData.lossLines),
      {
        name: "开空",
        type: "scatter",
        data: chartData.shortPoints,
        symbol: "triangle",
        symbolSize: 15,
        itemStyle: { color: shortColor, borderColor: "#ffffff", borderWidth: 1.5 },
        label: { show: true, formatter: (p) => p.data.label, position: "right", color: shortColor, fontWeight: 700 },
        tooltip: { trigger: "item", formatter: signalTooltip },
        z: 10,
      },
      {
        name: "盈利平空",
        type: "scatter",
        data: chartData.coverProfitPoints,
        symbolSize: 14,
        itemStyle: { color: winColor, borderColor: "#ffffff", borderWidth: 1.5 },
        label: { show: true, formatter: (p) => p.data.label, position: "right", color: winColor, fontWeight: 700 },
        tooltip: { trigger: "item", formatter: signalTooltip },
        z: 10,
      },
      {
        name: "亏损平空",
        type: "scatter",
        data: chartData.coverLossPoints,
        symbolSize: 14,
        itemStyle: { color: lossColor, borderColor: "#ffffff", borderWidth: 1.5 },
        label: { show: true, formatter: (p) => p.data.label, position: "right", color: lossColor, fontWeight: 700 },
        tooltip: { trigger: "item", formatter: signalTooltip },
        z: 10,
      },
    ],
  };
  chart.setOption(option, true);
  window.addEventListener("resize", () => chart.resize());
})();
</script>
</body>
</html>
"""
    for key, value in replacements.items():
        html_text = html_text.replace(key, value)
    return html_text


def build_short_trade_pair_html(
    trades_path: str = DEFAULT_TRADES,
    summary_path: str = DEFAULT_SUMMARY,
    output: str = DEFAULT_OUTPUT,
    start_date: str = "2022-01-01",
) -> Path:
    trades = _load_trades(trades_path)
    pairs = _to_pairs(trades)
    summary_file = project_path(summary_path)
    summary = pd.read_csv(summary_file).iloc[0].to_dict()
    target_index = _load_target_index()
    market = _load_market(target_index, start_date)
    chart_payload = _render_chart_payload(market=market, pairs=pairs)
    html_text = _render_html(
        title="XGBoost Top Short-Only Strategy Trade Pairs",
        summary=summary,
        chart_payload=chart_payload,
        table=_render_trade_table(pairs),
    )
    output_path = project_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot numbered short trade pairs for XGBoost top short-only strategy.")
    parser.add_argument("--trades", default=DEFAULT_TRADES)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default="2022-01-01")
    args = parser.parse_args()
    print(
        build_short_trade_pair_html(
            trades_path=args.trades,
            summary_path=args.summary,
            output=args.output,
            start_date=args.start_date,
        )
    )


if __name__ == "__main__":
    main()
