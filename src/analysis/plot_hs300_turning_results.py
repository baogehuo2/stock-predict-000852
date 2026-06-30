from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.config import project_path
from src.common.db import read_sql


DEFAULT_TARGET_INDEX = "000300"
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_MODELS = ["lightgbm", "xgboost", "bp", "random_forest"]
DEFAULT_OUTPUT = "data/reports/hs300_turning_results_dashboard.html"
PREDICTION_TEMPLATE = "data/reports/manual_weak_turning_walk_forward_predictions_hs300_{model_kind}_wf.csv"
TRADE_SUMMARY_PATH = "data/reports/manual_weak_turning_trade_strategy_summary_hs300.csv"
TRADE_PATH = "data/reports/manual_weak_turning_trade_strategy_trades_hs300.csv"


@dataclass(frozen=True)
class TradePair:
    trade_id: int
    model_kind: str
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


def _fmt(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _normalize_model_kind(model_kind: str) -> str:
    normalized = model_kind.lower().strip()
    if normalized == "xboost":
        return "xgboost"
    if normalized in {"bpnn", "bp_neural_network", "neural_network", "mlp"}:
        return "bp"
    if normalized in {"rf", "randomforest", "random-forest"}:
        return "random_forest"
    return normalized


def _load_market(target_index: str, start_date: str) -> pd.DataFrame:
    market = read_sql(
        """
        SELECT trade_date, index_code, open, high, low, close
        FROM market_index_daily
        WHERE index_code = :target_index
          AND trade_date >= :start_date
        ORDER BY trade_date
        """,
        {"target_index": target_index, "start_date": start_date},
    )
    if market.empty:
        raise RuntimeError(f"No market_index_daily rows found for {target_index} after {start_date}.")
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    for column in ["open", "high", "low", "close"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    return market.dropna(subset=["trade_date", "open", "high", "low", "close"]).reset_index(drop=True)


def _load_prediction(model_kind: str, start_date: str) -> pd.DataFrame:
    model_kind = _normalize_model_kind(model_kind)
    path = project_path(PREDICTION_TEMPLATE.format(model_kind=model_kind))
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    prediction = pd.read_csv(path, parse_dates=["trade_date"])
    prediction = prediction[prediction["trade_date"] >= pd.Timestamp(start_date)].copy()
    for column in ["weak_combined_bottom_signal", "weak_combined_top_signal"]:
        prediction[column] = pd.to_numeric(prediction[column], errors="coerce").fillna(0).astype(int)
    for column in ["manual_weak_bottom_proba", "manual_weak_top_proba", "future_ret_15d"]:
        if column in prediction.columns:
            prediction[column] = pd.to_numeric(prediction[column], errors="coerce")
    if "future_ret_15d" not in prediction.columns:
        prediction["future_ret_15d"] = pd.NA
    if "train_end" not in prediction.columns:
        prediction["train_end"] = ""
    prediction["model_kind"] = model_kind
    return prediction.sort_values("trade_date").reset_index(drop=True)


def _load_trade_summary() -> pd.DataFrame:
    path = project_path(TRADE_SUMMARY_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Trade summary file not found: {path}")
    summary = pd.read_csv(path)
    numeric_columns = [
        "strategy_total_return",
        "buy_hold_return",
        "excess_vs_buy_hold",
        "max_drawdown",
        "win_rate",
        "avg_holding_days",
        "bottom_signal_count",
        "top_signal_count",
        "closed_trades",
        "open_trades",
    ]
    for column in numeric_columns:
        if column in summary.columns:
            summary[column] = pd.to_numeric(summary[column], errors="coerce")
    return summary


def _load_trade_pairs(model_kind: str, strategy: str) -> list[TradePair]:
    path = project_path(TRADE_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Trade file not found: {path}")
    trades = pd.read_csv(path, parse_dates=["entry_date", "exit_date"])
    trades["model_kind"] = trades["model_kind"].map(_normalize_model_kind)
    trades = trades[
        (trades["model_kind"] == _normalize_model_kind(model_kind))
        & (trades["strategy"] == strategy)
    ].copy()
    if trades.empty:
        raise RuntimeError(f"No trades found for model={model_kind}, strategy={strategy}.")
    for column in ["entry_price", "exit_price", "trade_return"]:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")
    trades["entry_count"] = pd.to_numeric(trades["entry_count"], errors="coerce").fillna(1).astype(int)
    trades["holding_days"] = pd.to_numeric(trades["holding_days"], errors="coerce").fillna(0).astype(int)
    pairs: list[TradePair] = []
    for idx, row in enumerate(trades.sort_values(["entry_date", "exit_date"]).itertuples(index=False), start=1):
        pairs.append(
            TradePair(
                trade_id=idx,
                model_kind=str(row.model_kind),
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


def _market_payload(market: pd.DataFrame) -> dict:
    return {
        "dates": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in market["trade_date"]],
        "candles": [
            [_round(row.open), _round(row.close), _round(row.low), _round(row.high)]
            for row in market.itertuples(index=False)
        ],
        "closes": [_round(row.close) for row in market.itertuples(index=False)],
    }


def _signal_payload(market: pd.DataFrame, prediction: pd.DataFrame) -> dict:
    data = market.merge(
        prediction[
            [
                "trade_date",
                "manual_weak_bottom_proba",
                "manual_weak_top_proba",
                "weak_combined_bottom_signal",
                "weak_combined_top_signal",
                "future_ret_15d",
                "train_end",
            ]
        ],
        on="trade_date",
        how="left",
    ).sort_values("trade_date")
    for column in ["weak_combined_bottom_signal", "weak_combined_top_signal"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int)
    for column in ["manual_weak_bottom_proba", "manual_weak_top_proba", "future_ret_15d"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["train_end"] = data["train_end"].fillna("")

    payload = _market_payload(data)
    payload["bottomProb"] = [_round(value, 6) for value in data["manual_weak_bottom_proba"]]
    payload["topProb"] = [_round(value, 6) for value in data["manual_weak_top_proba"]]
    payload["bottomPoints"] = []
    payload["topPoints"] = []
    bottom_id = 1
    top_id = 1
    for index, row in enumerate(data.itertuples(index=False)):
        trade_date = pd.Timestamp(row.trade_date).strftime("%Y-%m-%d")
        bottom_proba = None if pd.isna(row.manual_weak_bottom_proba) else float(row.manual_weak_bottom_proba)
        top_proba = None if pd.isna(row.manual_weak_top_proba) else float(row.manual_weak_top_proba)
        future_ret_15d = None if pd.isna(row.future_ret_15d) else float(row.future_ret_15d)
        train_end = html.escape(str(row.train_end))
        if int(row.weak_combined_bottom_signal) == 1:
            payload["bottomPoints"].append(
                {
                    "name": f"B{bottom_id}",
                    "value": [index, _round(row.low)],
                    "label": f"B{bottom_id}",
                    "tooltip": (
                        f"B{bottom_id} {trade_date}<br>"
                        f"close={_fmt(row.close)}<br>"
                        f"bottom_prob={_pct(bottom_proba)}<br>"
                        f"top_prob={_pct(top_proba)}<br>"
                        f"future_ret_15d={_pct(future_ret_15d)}<br>"
                        f"train_end={train_end}"
                    ),
                }
            )
            bottom_id += 1
        if int(row.weak_combined_top_signal) == 1:
            payload["topPoints"].append(
                {
                    "name": f"T{top_id}",
                    "value": [index, _round(row.high)],
                    "label": f"T{top_id}",
                    "tooltip": (
                        f"T{top_id} {trade_date}<br>"
                        f"close={_fmt(row.close)}<br>"
                        f"top_prob={_pct(top_proba)}<br>"
                        f"bottom_prob={_pct(bottom_proba)}<br>"
                        f"future_ret_15d={_pct(future_ret_15d)}<br>"
                        f"train_end={train_end}"
                    ),
                }
            )
            top_id += 1
    return payload


def _trade_payload(market: pd.DataFrame, pairs: list[TradePair]) -> dict:
    payload = _market_payload(market)
    date_to_index = {trade_date: index for index, trade_date in enumerate(payload["dates"])}
    payload.update(
        {
            "buyPoints": [],
            "sellProfitPoints": [],
            "sellLossPoints": [],
            "profitLines": [],
            "lossLines": [],
        }
    )
    for pair in pairs:
        entry_date = pair.entry_date.strftime("%Y-%m-%d")
        exit_date = pair.exit_date.strftime("%Y-%m-%d")
        entry_index = date_to_index.get(entry_date)
        exit_index = date_to_index.get(exit_date)
        if entry_index is None or exit_index is None:
            continue
        label = f"#{pair.trade_id} {_pct(pair.trade_return)} {pair.exit_reason}"
        payload["buyPoints"].append(
            {
                "name": f"B{pair.trade_id}",
                "value": [entry_index, _round(pair.entry_price)],
                "label": f"B{pair.trade_id}",
                "tooltip": f"B{pair.trade_id} {entry_date}<br>{label}",
            }
        )
        sell_point = {
            "name": f"S{pair.trade_id}",
            "value": [exit_index, _round(pair.exit_price)],
            "label": f"S{pair.trade_id}",
            "tooltip": f"S{pair.trade_id} {exit_date}<br>{label}",
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
            payload["sellProfitPoints"].append(sell_point)
            payload["profitLines"].append(line)
        else:
            payload["sellLossPoints"].append(sell_point)
            payload["lossLines"].append(line)
    return payload


def _render_trade_table(pairs: list[TradePair]) -> str:
    headers = ["#", "buy", "sell", "return", "holding_days", "entry_count", "status", "reason"]
    header_html = "".join(f"<th>{html.escape(column)}</th>" for column in headers)
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


def _render_strategy_rank(summary: pd.DataFrame) -> str:
    columns = [
        "model_kind",
        "strategy",
        "strategy_total_return",
        "buy_hold_return",
        "excess_vs_buy_hold",
        "max_drawdown",
        "win_rate",
        "closed_trades",
        "open_trades",
        "avg_holding_days",
    ]
    headers = [
        "model",
        "strategy",
        "total_return",
        "buy_hold",
        "excess",
        "max_dd",
        "win_rate",
        "closed",
        "open",
        "avg_days",
    ]
    data = summary[columns].sort_values("strategy_total_return", ascending=False)
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    body = []
    for row in data.itertuples(index=False):
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.model_kind))}</td>"
            f"<td>{html.escape(str(row.strategy))}</td>"
            f"<td>{_pct(row.strategy_total_return)}</td>"
            f"<td>{_pct(row.buy_hold_return)}</td>"
            f"<td>{_pct(row.excess_vs_buy_hold)}</td>"
            f"<td>{_pct(row.max_drawdown)}</td>"
            f"<td>{_pct(row.win_rate)}</td>"
            f"<td>{int(row.closed_trades)}</td>"
            f"<td>{int(row.open_trades)}</td>"
            f"<td>{_fmt(row.avg_holding_days, 1)}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _summary_cards(row: pd.Series) -> str:
    items = [
        ("best model", str(row["model_kind"])),
        ("best strategy", str(row["strategy"])),
        ("total return", _pct(row["strategy_total_return"])),
        ("max drawdown", _pct(row["max_drawdown"])),
        ("win rate", _pct(row["win_rate"])),
        ("closed/open", f"{int(row['closed_trades'])}/{int(row['open_trades'])}"),
    ]
    return "".join(
        f'<div class="card"><div class="muted">{html.escape(name)}</div><div class="value">{html.escape(value)}</div></div>'
        for name, value in items
    )


def _render_html(
    target_index: str,
    start_date: str,
    end_date: str,
    summary: pd.DataFrame,
    trade_payload: dict,
    trade_pairs: list[TradePair],
    signal_payloads: dict[str, dict],
) -> str:
    best = summary.sort_values("strategy_total_return", ascending=False).iloc[0]
    charts_json = json.dumps(
        {"trade": trade_payload, "signals": signal_payloads},
        ensure_ascii=False,
        allow_nan=False,
    )
    signal_sections = []
    for model_kind, payload in signal_payloads.items():
        signal_sections.append(
            f"""
  <section class="panel">
    <h2>{html.escape(model_kind)} top/bottom signals</h2>
    <div class="legend">
      <span><span class="dot" style="background:#2563eb"></span>bottom B#</span>
      <span><span class="dot" style="background:#dc2626"></span>top T#</span>
      <span><span class="dot line" style="background:#64748b"></span>close and probabilities</span>
    </div>
    <div id="signalChart_{html.escape(model_kind)}" class="chart signal-chart"></div>
  </section>
"""
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HS300 Turning Model Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; color: #111827; background: #f6f7f9; }}
    main {{ max-width: 1520px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin: 16px 0; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; }}
    .value {{ font-size: 20px; font-weight: 700; margin-top: 4px; }}
    .panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; margin: 16px 0; padding: 14px; }}
    .chart {{ width: 100%; height: 860px; }}
    .signal-chart {{ height: 880px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; z-index: 1; }}
    tr.win td:nth-child(4), td:nth-child(3) {{ color: #dc2626; font-weight: 700; }}
    tr.loss td:nth-child(4) {{ color: #16a34a; font-weight: 700; }}
    .table-wrap {{ max-height: 520px; overflow: auto; border-radius: 10px; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0 10px; font-size: 13px; color: #374151; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
    .dot.line {{ border-radius: 2px; height: 3px; }}
  </style>
</head>
<body>
<main>
  <h1>HS300 Turning Model Dashboard</h1>
  <div class="muted">target={html.escape(target_index)}, range={html.escape(start_date)} to {html.escape(end_date)}. Charts support mouse wheel zoom, drag pan, and bottom slider zoom.</div>
  <section class="cards">{_summary_cards(best)}</section>

  <section class="panel">
    <h2>Strategy ranking</h2>
    <div class="table-wrap">{_render_strategy_rank(summary)}</div>
  </section>

  <section class="panel">
    <h2>{html.escape(str(best["model_kind"]))} / {html.escape(str(best["strategy"]))} trade pairs</h2>
    <div class="legend">
      <span><span class="dot" style="background:#2563eb"></span>buy B#</span>
      <span><span class="dot" style="background:#dc2626"></span>profit sell S#</span>
      <span><span class="dot" style="background:#16a34a"></span>loss sell S#</span>
    </div>
    <div id="tradeChart" class="chart"></div>
    <h2>Trade table</h2>
    <div class="table-wrap">{_render_trade_table(trade_pairs)}</div>
  </section>

  {"".join(signal_sections)}
</main>
<script>
(() => {{
  const chartData = {charts_json};
  const winColor = "#dc2626";
  const lossColor = "#16a34a";
  const buyColor = "#2563eb";
  const topColor = "#dc2626";
  const closeColor = "#64748b";

  function tradeLineSeries(name, color, data) {{
    return {{
      name,
      type: "custom",
      coordinateSystem: "cartesian2d",
      data,
      silent: true,
      z: 4,
      renderItem: function (_params, api) {{
        const p1 = api.coord([api.value(0), api.value(1)]);
        const p2 = api.coord([api.value(2), api.value(3)]);
        return {{
          type: "line",
          shape: {{ x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1] }},
          style: {{ stroke: color, lineWidth: 1.8, opacity: 0.72 }},
        }};
      }},
    }};
  }}

  function pointSeries(name, color, data, symbol, position) {{
    return {{
      name,
      type: "scatter",
      data,
      symbol,
      symbolSize: 15,
      itemStyle: {{ color, borderColor: "#fff", borderWidth: 1.5 }},
      label: {{
        show: true,
        formatter: p => p.data.label,
        position,
        color,
        fontWeight: "bold",
        fontSize: 12,
      }},
      tooltip: {{ trigger: "item", formatter: p => p.data.tooltip }},
      z: 8,
    }};
  }}

  function baseZoom() {{
    return [
      {{ type: "inside", xAxisIndex: [0, 1], start: 0, end: 100, zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false }},
      {{ type: "slider", xAxisIndex: [0, 1], top: 806, height: 28, start: 0, end: 100 }},
    ];
  }}

  function renderTradeChart() {{
    const data = chartData.trade;
    const chart = echarts.init(document.getElementById("tradeChart"));
    chart.setOption({{
      animation: false,
      legend: {{ top: 8 }},
      tooltip: {{ trigger: "axis", axisPointer: {{ type: "cross" }} }},
      grid: [
        {{ left: 64, right: 42, top: 52, height: 650 }},
        {{ left: 64, right: 42, top: 730, height: 58 }},
      ],
      xAxis: [
        {{ type: "category", data: data.dates, boundaryGap: false, axisLine: {{ onZero: false }}, min: "dataMin", max: "dataMax" }},
        {{ type: "category", data: data.dates, gridIndex: 1, boundaryGap: false, axisLabel: {{ show: false }}, axisTick: {{ show: false }} }},
      ],
      yAxis: [
        {{ scale: true, splitArea: {{ show: true }} }},
        {{ scale: true, gridIndex: 1, splitNumber: 2, axisLabel: {{ show: false }}, axisTick: {{ show: false }} }},
      ],
      dataZoom: baseZoom(),
      series: [
        {{ name: "K line", type: "candlestick", data: data.candles, itemStyle: {{ color: "#ef4444", color0: "#10b981", borderColor: "#ef4444", borderColor0: "#10b981" }} }},
        {{ name: "close", type: "line", data: data.closes, smooth: true, showSymbol: false, lineStyle: {{ width: 1.2, color: closeColor }}, z: 2 }},
        tradeLineSeries("profit pair", winColor, data.profitLines),
        tradeLineSeries("loss pair", lossColor, data.lossLines),
        pointSeries("buy", buyColor, data.buyPoints, "triangle", "bottom"),
        pointSeries("profit sell", winColor, data.sellProfitPoints, "circle", "top"),
        pointSeries("loss sell", lossColor, data.sellLossPoints, "circle", "top"),
        {{ name: "overview", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: data.closes, showSymbol: false, lineStyle: {{ color: "#94a3b8", width: 1 }} }},
      ],
    }});
    window.addEventListener("resize", () => chart.resize());
  }}

  function renderSignalChart(modelKind, data) {{
    const chart = echarts.init(document.getElementById(`signalChart_${{modelKind}}`));
    chart.setOption({{
      animation: false,
      legend: {{ top: 8 }},
      tooltip: {{ trigger: "axis", axisPointer: {{ type: "cross" }} }},
      grid: [
        {{ left: 64, right: 42, top: 54, height: 620 }},
        {{ left: 64, right: 42, top: 712, height: 110 }},
      ],
      xAxis: [
        {{ type: "category", data: data.dates, boundaryGap: false, axisLine: {{ onZero: false }}, min: "dataMin", max: "dataMax" }},
        {{ type: "category", data: data.dates, gridIndex: 1, boundaryGap: false }},
      ],
      yAxis: [
        {{ scale: true, splitArea: {{ show: true }} }},
        {{ type: "value", min: 0, max: 1, gridIndex: 1, splitNumber: 4, axisLabel: {{ formatter: value => `${{Math.round(value * 100)}}%` }} }},
      ],
      dataZoom: [
        {{ type: "inside", xAxisIndex: [0, 1], start: 0, end: 100, zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false }},
        {{ type: "slider", xAxisIndex: [0, 1], top: 842, height: 30, start: 0, end: 100 }},
      ],
      series: [
        {{ name: "K line", type: "candlestick", data: data.candles, itemStyle: {{ color: "#ef4444", color0: "#10b981", borderColor: "#ef4444", borderColor0: "#10b981" }} }},
        {{ name: "close", type: "line", data: data.closes, smooth: true, showSymbol: false, lineStyle: {{ width: 1.2, color: closeColor }}, z: 2 }},
        pointSeries("bottom signal", buyColor, data.bottomPoints, "triangle", "bottom"),
        pointSeries("top signal", topColor, data.topPoints, "triangle", "top"),
        {{ name: "bottom probability", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: data.bottomProb, showSymbol: false, lineStyle: {{ color: buyColor, width: 1.4 }} }},
        {{ name: "top probability", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: data.topProb, showSymbol: false, lineStyle: {{ color: topColor, width: 1.4 }} }},
      ],
    }});
    window.addEventListener("resize", () => chart.resize());
  }}

  renderTradeChart();
  for (const [modelKind, data] of Object.entries(chartData.signals)) {{
    renderSignalChart(modelKind, data);
  }}
}})();
</script>
</body>
</html>
"""


def build_dashboard(
    target_index: str = DEFAULT_TARGET_INDEX,
    start_date: str = DEFAULT_START_DATE,
    models: list[str] | None = None,
    trade_model: str = "xgboost",
    trade_strategy: str = "risk_exit",
    output_path: str = DEFAULT_OUTPUT,
) -> str:
    market = _load_market(target_index, start_date)
    summary = _load_trade_summary()
    pairs = _load_trade_pairs(trade_model, trade_strategy)
    signal_payloads = {
        model_kind: _signal_payload(market, _load_prediction(model_kind, start_date))
        for model_kind in [_normalize_model_kind(model) for model in (models or DEFAULT_MODELS)]
    }
    html_text = _render_html(
        target_index=target_index,
        start_date=start_date,
        end_date=market["trade_date"].max().strftime("%Y-%m-%d"),
        summary=summary,
        trade_payload=_trade_payload(market, pairs),
        trade_pairs=pairs,
        signal_payloads=signal_payloads,
    )
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot HS300 turning model results into one ECharts HTML.")
    parser.add_argument("--target-index", default=DEFAULT_TARGET_INDEX)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--trade-model", default="xgboost")
    parser.add_argument("--trade-strategy", default="risk_exit")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        build_dashboard(
            target_index=args.target_index,
            start_date=args.start_date,
            models=args.models,
            trade_model=args.trade_model,
            trade_strategy=args.trade_strategy,
            output_path=args.output,
        )
    )


if __name__ == "__main__":
    main()
