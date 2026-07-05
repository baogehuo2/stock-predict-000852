from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from src.analysis.backtest_bp_xgb_indicator_strategy import (
    BP_PREDICTION_PATH,
    XGBOOST_PREDICTION_PATH,
    StrategyParams,
    _load_market,
    _load_prediction,
    _load_target_index,
    run_backtest,
)
from src.common.config import project_path


RF_PREDICTION_PATH = "data/reports/manual_weak_turning_walk_forward_predictions_random_forest_wf.csv"
SUMMARY_OUTPUT = "data/reports/bp_bottom_top_model_strategy_compare.csv"
TRADES_OUTPUT = "data/reports/bp_bottom_top_model_strategy_trades.csv"
EQUITY_OUTPUT = "data/reports/bp_bottom_top_model_strategy_equity.csv"
HTML_OUTPUT = "data/reports/bp_bottom_xgb_vs_rf_top_strategy_compare.html"

TOP_MODEL_PATHS = {
    "xgboost": XGBOOST_PREDICTION_PATH,
    "random_forest": RF_PREDICTION_PATH,
}

TOP_MODEL_LABELS = {
    "xgboost": "BP bottom + XGBoost top",
    "random_forest": "BP bottom + RF top",
}


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


def _normalize_top_model(top_model: str) -> str:
    normalized = top_model.lower().strip().replace("-", "_")
    aliases = {
        "xgb": "xgboost",
        "xboost": "xgboost",
        "rf": "random_forest",
        "randomforest": "random_forest",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TOP_MODEL_PATHS:
        raise ValueError(f"Unsupported top model: {top_model}")
    return normalized


def _case_params() -> list[tuple[str, StrategyParams]]:
    return [
        ("all_no_cost", StrategyParams()),
        ("exclude_no_cost", StrategyParams(exclude_start="2023-12-01", exclude_end="2024-01-31")),
        ("all_cost", StrategyParams(fee_rate=0.0003, slippage_rate=0.0005)),
        (
            "exclude_cost",
            StrategyParams(
                fee_rate=0.0003,
                slippage_rate=0.0005,
                exclude_start="2023-12-01",
                exclude_end="2024-01-31",
            ),
        ),
    ]


def _load_backtest_frame(start_date: str, top_model: str) -> pd.DataFrame:
    top_model = _normalize_top_model(top_model)
    target_index = _load_target_index()
    market = _load_market(target_index, start_date)
    bp = _load_prediction(BP_PREDICTION_PATH, "bp", start_date)
    top = _load_prediction(TOP_MODEL_PATHS[top_model], "xgb", start_date)
    data = market.merge(bp, on="trade_date", how="left").merge(top, on="trade_date", how="left")
    for column in ["bp_bottom_signal", "bp_top_signal", "xgb_bottom_signal", "xgb_top_signal"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int)
    for column in ["bp_bottom_proba", "bp_top_proba", "xgb_bottom_proba", "xgb_top_proba"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values("trade_date").reset_index(drop=True)


def _append_variant_columns(frame: pd.DataFrame, top_model: str, case_name: str) -> pd.DataFrame:
    result = frame.copy()
    result["top_model_kind"] = top_model
    result["variant"] = TOP_MODEL_LABELS[top_model]
    result["case"] = case_name
    return result


def run_comparison(start_date: str = "2022-01-01") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    equity_parts: list[pd.DataFrame] = []

    for top_model in ["xgboost", "random_forest"]:
        data = _load_backtest_frame(start_date, top_model)
        for case_name, params in _case_params():
            summary, trades, equity = run_backtest(data, params)
            summary = _append_variant_columns(summary, top_model, case_name)
            trades = _append_variant_columns(trades, top_model, case_name)
            equity = _append_variant_columns(equity, top_model, case_name)
            summary["top_signal_count"] = summary["xgb_top_signal_count"]
            summary["strategy"] = "bp_bottom_top_model_indicator_strategy"
            summary_parts.append(summary)
            trade_parts.append(trades)
            equity_parts.append(equity)

    summary_df = pd.concat(summary_parts, ignore_index=True)
    trades_df = pd.concat(trade_parts, ignore_index=True)
    equity_df = pd.concat(equity_parts, ignore_index=True)

    for frame, output in [
        (summary_df, SUMMARY_OUTPUT),
        (trades_df, TRADES_OUTPUT),
        (equity_df, EQUITY_OUTPUT),
    ]:
        path = project_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return summary_df, trades_df, equity_df


def _summary_table(summary: pd.DataFrame) -> str:
    columns = [
        "case",
        "variant",
        "total_return",
        "max_drawdown",
        "win_rate",
        "closed_trades",
        "open_trades",
        "avg_trade_return",
        "avg_holding_trade_days",
        "bp_bottom_signal_count",
        "top_signal_count",
    ]
    headers = [
        "case",
        "strategy",
        "return",
        "max_dd",
        "win_rate",
        "closed",
        "open",
        "avg_trade",
        "avg_hold",
        "bp_bottom",
        "top_signal",
    ]
    header_html = "".join(f"<th>{html.escape(column)}</th>" for column in headers)
    body = []
    data = summary[[*columns, "top_model_kind"]].sort_values(["case", "top_model_kind"])
    for row in data.itertuples(index=False):
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.case))}</td>"
            f"<td>{html.escape(str(row.variant))}</td>"
            f"<td>{_pct(row.total_return)}</td>"
            f"<td>{_pct(row.max_drawdown)}</td>"
            f"<td>{_pct(row.win_rate)}</td>"
            f"<td>{int(row.closed_trades)}</td>"
            f"<td>{int(row.open_trades)}</td>"
            f"<td>{_pct(row.avg_trade_return)}</td>"
            f"<td>{_fmt(row.avg_holding_trade_days, 2)}</td>"
            f"<td>{int(row.bp_bottom_signal_count)}</td>"
            f"<td>{int(row.top_signal_count)}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _diff_table(summary: pd.DataFrame, case_name: str) -> str:
    part = summary[summary["case"] == case_name].copy()
    rows = []
    by_model = part.set_index("top_model_kind")
    if {"xgboost", "random_forest"}.issubset(by_model.index):
        xgb = by_model.loc["xgboost"]
        rf = by_model.loc["random_forest"]
        metrics = [
            ("total_return", "return", True),
            ("max_drawdown", "max_dd", True),
            ("win_rate", "win_rate", True),
            ("closed_trades", "closed", False),
            ("avg_trade_return", "avg_trade", True),
            ("avg_holding_trade_days", "avg_hold", False),
            ("top_signal_count", "top_signals", False),
        ]
        for column, label, is_pct in metrics:
            xgb_value = xgb[column]
            rf_value = rf[column]
            diff = rf_value - xgb_value
            formatter = _pct if is_pct else lambda value: _fmt(value, 2)
            rows.append(
                "<tr>"
                f"<td>{html.escape(label)}</td>"
                f"<td>{formatter(xgb_value)}</td>"
                f"<td>{formatter(rf_value)}</td>"
                f"<td>{formatter(diff)}</td>"
                "</tr>"
            )
    header = "<tr><th>metric</th><th>BP+XGB top</th><th>BP+RF top</th><th>RF - XGB</th></tr>"
    return f"<table><thead>{header}</thead><tbody>{''.join(rows)}</tbody></table>"


def _trade_table(trades: pd.DataFrame, case_name: str, top_model: str) -> str:
    data = trades[
        (trades["case"] == case_name)
        & (trades["top_model_kind"] == top_model)
    ].copy()
    data = data.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)
    headers = ["#", "entry", "exit", "return", "sell", "status", "reason"]
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    body = []
    for idx, row in enumerate(data.itertuples(index=False), start=1):
        css_class = "win" if float(row.trade_return) > 0 else "loss"
        body.append(
            f'<tr class="{css_class}">'
            f"<td>{idx}</td>"
            f"<td>{pd.Timestamp(row.entry_date):%Y-%m-%d}<br>{_fmt(row.entry_price)}</td>"
            f"<td>{pd.Timestamp(row.exit_date):%Y-%m-%d}<br>{_fmt(row.exit_price)}</td>"
            f"<td>{_pct(row.trade_return)}</td>"
            f"<td>{_pct(row.sell_fraction)}</td>"
            f"<td>{html.escape(str(row.status))}</td>"
            f"<td>{html.escape(str(row.exit_reason))}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _market_payload(market: pd.DataFrame) -> dict:
    market = market.sort_values("trade_date").copy()
    return {
        "dates": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in market["trade_date"]],
        "candles": [
            [_round(row.open), _round(row.close), _round(row.low), _round(row.high)]
            for row in market.itertuples(index=False)
        ],
        "closes": [_round(row.close) for row in market.itertuples(index=False)],
    }


def _chart_payload(
    market: pd.DataFrame,
    trades: pd.DataFrame,
    case_name: str,
    top_model: str,
    exclude_start: str | None,
    exclude_end: str | None,
) -> dict:
    payload = _market_payload(market)
    date_to_index = {trade_date: index for index, trade_date in enumerate(payload["dates"])}
    market_dates = [pd.Timestamp(value) for value in payload["dates"]]
    payload.update(
        {
            "buyPoints": [],
            "sellProfitPoints": [],
            "sellLossPoints": [],
            "profitLines": [],
            "lossLines": [],
            "excludeStartIndex": None,
            "excludeEndIndex": None,
        }
    )
    if exclude_start and exclude_end:
        start_ts = pd.Timestamp(exclude_start)
        end_ts = pd.Timestamp(exclude_end)
        start_candidates = [idx for idx, value in enumerate(market_dates) if value >= start_ts]
        end_candidates = [idx for idx, value in enumerate(market_dates) if value <= end_ts]
        if start_candidates and end_candidates:
            payload["excludeStartIndex"] = start_candidates[0]
            payload["excludeEndIndex"] = end_candidates[-1]

    data = trades[
        (trades["case"] == case_name)
        & (trades["top_model_kind"] == top_model)
    ].copy()
    data = data.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)
    for idx, row in enumerate(data.itertuples(index=False), start=1):
        entry_date = pd.Timestamp(row.entry_date).strftime("%Y-%m-%d")
        exit_date = pd.Timestamp(row.exit_date).strftime("%Y-%m-%d")
        entry_index = date_to_index.get(entry_date)
        exit_index = date_to_index.get(exit_date)
        if entry_index is None or exit_index is None:
            continue
        label = f"#{idx} {_pct(row.trade_return)} {row.exit_reason}"
        payload["buyPoints"].append(
            {
                "name": f"B{idx}",
                "value": [entry_index, _round(row.entry_price)],
                "label": f"B{idx}",
                "tooltip": f"B{idx} {entry_date}<br>{html.escape(label)}",
            }
        )
        sell_point = {
            "name": f"S{idx}",
            "value": [exit_index, _round(row.exit_price)],
            "label": f"S{idx}",
            "tooltip": f"S{idx} {exit_date}<br>{html.escape(label)}",
        }
        line = [
            entry_index,
            _round(row.entry_price),
            exit_index,
            _round(row.exit_price),
            idx,
            _round(row.trade_return, 6),
        ]
        if float(row.trade_return) > 0:
            payload["sellProfitPoints"].append(sell_point)
            payload["profitLines"].append(line)
        else:
            payload["sellLossPoints"].append(sell_point)
            payload["lossLines"].append(line)
    return payload


def _render_html(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    chart_payloads: dict[str, dict],
    case_name: str,
) -> str:
    chart_json = json.dumps(chart_payloads, ensure_ascii=False, allow_nan=False)
    sections = []
    for top_model in ["xgboost", "random_forest"]:
        sections.append(
            f"""
  <section class="panel">
    <h2>{html.escape(TOP_MODEL_LABELS[top_model])} / {html.escape(case_name)}</h2>
    <div class="legend">
      <span><span class="dot" style="background:#2563eb"></span>buy B#</span>
      <span><span class="dot" style="background:#dc2626"></span>profit sell S#</span>
      <span><span class="dot" style="background:#16a34a"></span>loss sell S#</span>
      <span><span class="dot" style="background:#9ca3af"></span>excluded period</span>
    </div>
    <div id="chart_{html.escape(top_model)}" class="chart"></div>
    <h3>Trades</h3>
    <div class="table-wrap">{_trade_table(trades, case_name, top_model)}</div>
  </section>
"""
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BP bottom with XGBoost vs RF top</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; color: #111827; background: #f6f7f9; }}
    main {{ max-width: 1520px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    h3 {{ margin: 18px 0 10px; font-size: 16px; }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    .panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; margin: 16px 0; padding: 14px; }}
    .chart {{ width: 100%; height: 840px; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0 10px; font-size: 13px; color: #374151; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; z-index: 1; }}
    tr.win td:nth-child(4), td:nth-child(3) {{ color: #dc2626; font-weight: 700; }}
    tr.loss td:nth-child(4) {{ color: #16a34a; font-weight: 700; }}
    .table-wrap {{ max-height: 460px; overflow: auto; border-radius: 10px; }}
  </style>
</head>
<body>
<main>
  <h1>BP bottom with XGBoost top vs RF top</h1>
  <div class="muted">Same BP bottom entry and same indicator/risk execution logic. Only the top model is switched between XGBoost and RandomForest. The primary chart case is {html.escape(case_name)}.</div>
  <section class="panel">
    <h2>All cases summary</h2>
    <div class="table-wrap">{_summary_table(summary)}</div>
  </section>
  <section class="panel">
    <h2>Primary case difference: {html.escape(case_name)}</h2>
    <div class="table-wrap">{_diff_table(summary, case_name)}</div>
  </section>
  {"".join(sections)}
</main>
<script>
(() => {{
  const allCharts = {chart_json};
  const winColor = "#dc2626";
  const lossColor = "#16a34a";
  const buyColor = "#2563eb";
  const closeColor = "#64748b";

  function lineSeries(name, color, data) {{
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
      symbolSize: 14,
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
      z: 9,
    }};
  }}

  function renderChart(domId, data) {{
    const chart = echarts.init(document.getElementById(domId));
    chart.setOption({{
      animation: false,
      legend: {{ top: 8, type: "scroll" }},
      tooltip: {{ trigger: "axis", axisPointer: {{ type: "cross" }}, confine: true }},
      axisPointer: {{ link: [{{ xAxisIndex: "all" }}] }},
      grid: [
        {{ left: 64, right: 42, top: 54, height: 650 }},
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
      dataZoom: [
        {{ type: "inside", xAxisIndex: [0, 1], start: 0, end: 100, minSpan: 8, filterMode: "none", zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false }},
        {{ type: "slider", xAxisIndex: [0, 1], top: 806, height: 28, start: 0, end: 100, minSpan: 8, filterMode: "none" }},
      ],
      series: [
        {{
          name: "K line",
          type: "candlestick",
          data: data.candles,
          itemStyle: {{ color: "#ef4444", color0: "#16a34a", borderColor: "#dc2626", borderColor0: "#16a34a" }},
          markArea: {{
            silent: true,
            itemStyle: {{ color: "rgba(156, 163, 175, 0.16)" }},
            data: data.excludeStartIndex !== null && data.excludeEndIndex !== null ? [[{{ name: "excluded period", xAxis: data.excludeStartIndex }}, {{ xAxis: data.excludeEndIndex }}]] : [],
          }},
        }},
        {{ name: "close", type: "line", data: data.closes, smooth: true, showSymbol: false, lineStyle: {{ width: 1.2, color: closeColor }}, z: 2 }},
        lineSeries("profit pair", winColor, data.profitLines),
        lineSeries("loss pair", lossColor, data.lossLines),
        pointSeries("buy", buyColor, data.buyPoints, "triangle", "bottom"),
        pointSeries("profit sell", winColor, data.sellProfitPoints, "circle", "top"),
        pointSeries("loss sell", lossColor, data.sellLossPoints, "circle", "top"),
        {{ name: "overview", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: data.closes, showSymbol: false, lineStyle: {{ color: "#94a3b8", width: 1 }} }},
      ],
    }});
    window.addEventListener("resize", () => chart.resize());
  }}

  renderChart("chart_xgboost", allCharts.xgboost);
  renderChart("chart_random_forest", allCharts.random_forest);
}})();
</script>
</body>
</html>
"""


def build_html(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    start_date: str,
    chart_case: str,
    output: str = HTML_OUTPUT,
) -> Path:
    target_index = _load_target_index()
    market = _load_market(target_index, start_date)
    case_lookup = {name: params for name, params in _case_params()}
    params = case_lookup[chart_case]
    chart_payloads = {
        top_model: _chart_payload(
            market=market,
            trades=trades,
            case_name=chart_case,
            top_model=top_model,
            exclude_start=params.exclude_start,
            exclude_end=params.exclude_end,
        )
        for top_model in ["xgboost", "random_forest"]
    }
    html_text = _render_html(summary, trades, chart_payloads, chart_case)
    output_path = project_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BP bottom with XGBoost top vs RF top.")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument(
        "--chart-case",
        choices=["all_no_cost", "exclude_no_cost", "all_cost", "exclude_cost"],
        default="exclude_cost",
    )
    parser.add_argument("--output", default=HTML_OUTPUT)
    args = parser.parse_args()

    summary, trades, _ = run_comparison(args.start_date)
    output_path = build_html(summary, trades, args.start_date, args.chart_case, args.output)
    pd.set_option("display.max_columns", None)
    print("[summary]")
    print(summary.sort_values(["case", "top_model_kind"]).to_string(index=False))
    print(f"\n[html]\n{output_path}")


if __name__ == "__main__":
    main()
