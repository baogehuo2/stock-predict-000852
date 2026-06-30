from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.config import load_yaml, project_path
from src.common.db import read_sql


DEFAULT_PREDICTION = "data/reports/manual_weak_turning_walk_forward_predictions_random_forest_wf.csv"
DEFAULT_OUTPUT = "data/reports/random_forest_top_bottom_signals.html"
DEFAULT_START_DATE = "2022-01-01"


@dataclass(frozen=True)
class SignalRow:
    signal_id: int
    trade_date: pd.Timestamp
    signal_type: str
    close: float
    bottom_proba: float
    top_proba: float
    future_ret_15d: float | None
    train_end: str


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


def _load_target_index() -> str:
    cfg = load_yaml(project_path("config", "bottom_model.yaml"))
    return str(cfg.get("model", {}).get("target_index", "000852"))


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


def _load_prediction(path_value: str, start_date: str) -> pd.DataFrame:
    path = project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    prediction = pd.read_csv(path, parse_dates=["trade_date"])
    prediction = prediction[prediction["trade_date"] >= pd.Timestamp(start_date)].copy()
    required = [
        "trade_date",
        "manual_weak_bottom_proba",
        "manual_weak_top_proba",
        "weak_combined_bottom_signal",
        "weak_combined_top_signal",
    ]
    missing = [column for column in required if column not in prediction.columns]
    if missing:
        raise RuntimeError(f"Prediction file missing columns: {missing}")
    for column in ["manual_weak_bottom_proba", "manual_weak_top_proba", "future_ret_15d"]:
        if column in prediction.columns:
            prediction[column] = pd.to_numeric(prediction[column], errors="coerce")
    for column in ["weak_combined_bottom_signal", "weak_combined_top_signal"]:
        prediction[column] = pd.to_numeric(prediction[column], errors="coerce").fillna(0).astype(int)
    if "train_end" not in prediction.columns:
        prediction["train_end"] = ""
    if "future_ret_15d" not in prediction.columns:
        prediction["future_ret_15d"] = pd.NA
    return prediction.sort_values("trade_date").reset_index(drop=True)


def _signal_rows(data: pd.DataFrame) -> list[SignalRow]:
    rows: list[SignalRow] = []
    signal_id = 1
    signal_data = data[
        (data["weak_combined_bottom_signal"].eq(1)) | (data["weak_combined_top_signal"].eq(1))
    ].copy()
    for row in signal_data.sort_values("trade_date").itertuples(index=False):
        signal_types: list[str] = []
        if int(row.weak_combined_bottom_signal) == 1:
            signal_types.append("bottom")
        if int(row.weak_combined_top_signal) == 1:
            signal_types.append("top")
        for signal_type in signal_types:
            rows.append(
                SignalRow(
                    signal_id=signal_id,
                    trade_date=pd.Timestamp(row.trade_date),
                    signal_type=signal_type,
                    close=float(row.close),
                    bottom_proba=float(row.manual_weak_bottom_proba),
                    top_proba=float(row.manual_weak_top_proba),
                    future_ret_15d=None
                    if pd.isna(row.future_ret_15d)
                    else float(row.future_ret_15d),
                    train_end=str(row.train_end),
                )
            )
            signal_id += 1
    return rows


def _render_signal_table(rows: list[SignalRow]) -> str:
    headers = ["#", "type", "date", "close", "bottom_prob", "top_prob", "future_ret_15d", "train_end"]
    header_html = "".join(f"<th>{html.escape(column)}</th>" for column in headers)
    body: list[str] = []
    for row in rows:
        css_class = "bottom-row" if row.signal_type == "bottom" else "top-row"
        body.append(
            f'<tr class="{css_class}">'
            f"<td>{row.signal_id}</td>"
            f"<td>{html.escape(row.signal_type)}</td>"
            f"<td>{row.trade_date:%Y-%m-%d}</td>"
            f"<td>{_fmt(row.close)}</td>"
            f"<td>{_pct(row.bottom_proba)}</td>"
            f"<td>{_pct(row.top_proba)}</td>"
            f"<td>{_pct(row.future_ret_15d)}</td>"
            f"<td>{html.escape(row.train_end)}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _build_chart_payload(market: pd.DataFrame, prediction: pd.DataFrame) -> dict:
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

    dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in data["trade_date"]]
    candles = [
        [_round(row.open), _round(row.close), _round(row.low), _round(row.high)]
        for row in data.itertuples(index=False)
    ]
    closes = [_round(row.close) for row in data.itertuples(index=False)]
    bottom_prob = [_round(value, 6) for value in data["manual_weak_bottom_proba"]]
    top_prob = [_round(value, 6) for value in data["manual_weak_top_proba"]]

    bottom_points: list[dict] = []
    top_points: list[dict] = []
    bottom_id = 1
    top_id = 1
    for index, row in enumerate(data.itertuples(index=False)):
        trade_date = pd.Timestamp(row.trade_date).strftime("%Y-%m-%d")
        bottom_proba = None if pd.isna(row.manual_weak_bottom_proba) else float(row.manual_weak_bottom_proba)
        top_proba = None if pd.isna(row.manual_weak_top_proba) else float(row.manual_weak_top_proba)
        future_ret_15d = None if pd.isna(row.future_ret_15d) else float(row.future_ret_15d)
        if int(row.weak_combined_bottom_signal) == 1:
            bottom_points.append(
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
                        f"train_end={html.escape(str(row.train_end))}"
                    ),
                }
            )
            bottom_id += 1
        if int(row.weak_combined_top_signal) == 1:
            top_points.append(
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
                        f"train_end={html.escape(str(row.train_end))}"
                    ),
                }
            )
            top_id += 1

    signal_rows = _signal_rows(data)
    return {
        "dates": dates,
        "candles": candles,
        "closes": closes,
        "bottomProb": bottom_prob,
        "topProb": top_prob,
        "bottomPoints": bottom_points,
        "topPoints": top_points,
        "signalRows": signal_rows,
        "firstDate": dates[0] if dates else "",
        "lastDate": dates[-1] if dates else "",
    }


def _render_html(title: str, payload: dict, table: str) -> str:
    chart_json = json.dumps(
        {key: value for key, value in payload.items() if key != "signalRows"},
        ensure_ascii=False,
        allow_nan=False,
    )
    bottom_count = len(payload["bottomPoints"])
    top_count = len(payload["topPoints"])
    replacements = {
        "__TITLE__": html.escape(title),
        "__DATE_RANGE__": f'{html.escape(payload["firstDate"])} 至 {html.escape(payload["lastDate"])}',
        "__BOTTOM_COUNT__": str(bottom_count),
        "__TOP_COUNT__": str(top_count),
        "__TOTAL_COUNT__": str(bottom_count + top_count),
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
    h2 { margin: 22px 0 10px; font-size: 20px; }
    .muted { color: #6b7280; font-size: 13px; }
    .cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 16px 0; }
    .card { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; }
    .value { font-size: 22px; font-weight: 700; margin-top: 4px; }
    .chart-wrap { background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px; }
    #signalChart { width: 100%; height: 920px; }
    .legend { display: flex; gap: 18px; flex-wrap: wrap; margin: 10px 0; font-size: 13px; color: #374151; }
    .dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }
    table { width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; vertical-align: top; }
    th { background: #f3f4f6; position: sticky; top: 0; z-index: 1; }
    tr.bottom-row td:nth-child(2), tr.bottom-row td:nth-child(5) { color: #2563eb; font-weight: 700; }
    tr.top-row td:nth-child(2), tr.top-row td:nth-child(6) { color: #dc2626; font-weight: 700; }
    .table-wrap { max-height: 560px; overflow: auto; border-radius: 10px; }
  </style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <div class="muted">同一张图展示 RF 滚动模型识别到的底部 B# 和顶部 T#。支持鼠标滚轮缩放、拖动平移、底部滑块缩放。</div>
  <section class="cards">
    <div class="card"><div class="muted">date range</div><div class="value">__DATE_RANGE__</div></div>
    <div class="card"><div class="muted">bottom signals</div><div class="value">__BOTTOM_COUNT__</div></div>
    <div class="card"><div class="muted">top signals</div><div class="value">__TOP_COUNT__</div></div>
    <div class="card"><div class="muted">total signals</div><div class="value">__TOTAL_COUNT__</div></div>
  </section>
  <div class="legend">
    <span><span class="dot" style="background:#2563eb"></span>底部信号 B#，标在当日低点</span>
    <span><span class="dot" style="background:#dc2626"></span>顶部信号 T#，标在当日高点</span>
    <span><span class="dot" style="background:#64748b"></span>下方曲线为 bottom/top probability</span>
  </div>
  <section class="chart-wrap"><div id="signalChart"></div></section>
  <h2>RF 顶底信号明细</h2>
  <section class="table-wrap">__TABLE__</section>
</main>
<script>
(() => {
  const chartData = __CHART_JSON__;
  const bottomColor = "#2563eb";
  const topColor = "#dc2626";
  const chart = echarts.init(document.getElementById("signalChart"));

  function markerSeries(name, color, data, position, rotate) {
    return {
      name,
      type: "scatter",
      data,
      symbol: "triangle",
      symbolRotate: rotate,
      symbolSize: 15,
      itemStyle: { color, borderColor: "#fff", borderWidth: 1.5 },
      label: {
        show: true,
        formatter: p => p.data.label,
        position,
        color,
        fontWeight: "bold",
        fontSize: 12,
      },
      tooltip: { trigger: "item", formatter: p => p.data.tooltip },
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
      { left: 64, right: 42, top: 54, height: 620 },
      { left: 64, right: 42, top: 712, height: 110 },
    ],
    xAxis: [
      { type: "category", data: chartData.dates, boundaryGap: false, axisLine: { onZero: false }, min: "dataMin", max: "dataMax" },
      { type: "category", data: chartData.dates, gridIndex: 1, boundaryGap: false },
    ],
    yAxis: [
      { scale: true, splitArea: { show: true } },
      { type: "value", min: 0, max: 1, gridIndex: 1, splitNumber: 4, axisLabel: { formatter: value => `${Math.round(value * 100)}%` } },
    ],
    dataZoom: [
      { type: "inside", xAxisIndex: [0, 1], start: 0, end: 100, zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false },
      { type: "slider", xAxisIndex: [0, 1], top: 842, height: 30, start: 0, end: 100 },
    ],
    series: [
      {
        name: "K线",
        type: "candlestick",
        data: chartData.candles,
        itemStyle: { color: "#ef4444", color0: "#10b981", borderColor: "#ef4444", borderColor0: "#10b981" },
      },
      {
        name: "收盘价",
        type: "line",
        data: chartData.closes,
        smooth: true,
        showSymbol: false,
        lineStyle: { width: 1.2, color: "#64748b" },
        z: 2,
      },
      markerSeries("RF底部信号", bottomColor, chartData.bottomPoints, "bottom", 0),
      markerSeries("RF顶部信号", topColor, chartData.topPoints, "top", 180),
      {
        name: "bottom probability",
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: chartData.bottomProb,
        showSymbol: false,
        lineStyle: { color: bottomColor, width: 1.4 },
      },
      {
        name: "top probability",
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: chartData.topProb,
        showSymbol: false,
        lineStyle: { color: topColor, width: 1.4 },
      },
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
    prediction_path: str = DEFAULT_PREDICTION,
    output_path: str = DEFAULT_OUTPUT,
    start_date: str = DEFAULT_START_DATE,
) -> str:
    target_index = _load_target_index()
    market = _load_market(target_index, start_date)
    prediction = _load_prediction(prediction_path, start_date)
    payload = _build_chart_payload(market, prediction)
    table = _render_signal_table(payload["signalRows"])
    title = "Random Forest 顶底识别信号"
    html_text = _render_html(title, payload, table)
    output = project_path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return str(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot random forest predicted top/bottom signals.")
    parser.add_argument("--prediction", default=DEFAULT_PREDICTION)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    args = parser.parse_args()
    print(build_html(args.prediction, args.output, args.start_date))


if __name__ == "__main__":
    main()
