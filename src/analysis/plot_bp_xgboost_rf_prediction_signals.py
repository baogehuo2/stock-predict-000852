from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import pandas as pd

from src.common.config import load_yaml, project_path
from src.common.db import read_sql


DEFAULT_MODELS = ["bp", "xgboost", "random_forest"]
SUPPORTED_MODELS = {
    "bp",
    "xgboost",
    "xgboost_small",
    "random_forest",
    "gam",
    "logistic",
    "l1_logistic",
    "l1_logistic_var",
    "l1_logistic_kbest",
    "blend_equal",
    "blend_equal_rescue",
    "blend_equal_low_recall",
    "lda",
    "qda",
    "svm",
}
PREDICTION_TEMPLATE = "data/reports/manual_weak_turning_walk_forward_predictions_{model_kind}_wf.csv"
DEFAULT_OUTPUT = "data/reports/bp_xgboost_rf_prediction_signals.html"
DEFAULT_START_DATE = "2022-01-01"

MODEL_LABELS = {
    "bp": "BP prediction signals",
    "xgboost": "XGBoost prediction signals",
    "xgboost_small": "Conservative XGBoost prediction signals",
    "random_forest": "Random Forest prediction signals",
    "gam": "GAM prediction signals",
    "logistic": "Logistic Regression prediction signals",
    "l1_logistic": "L1 Logistic Regression prediction signals",
    "l1_logistic_var": "L1 Logistic + Variance Filter prediction signals",
    "l1_logistic_kbest": "L1 Logistic + KBest prediction signals",
    "blend_equal": "Equal Blend prediction signals",
    "blend_equal_rescue": "Equal Blend + Strong Signal Rescue",
    "blend_equal_low_recall": "Equal Blend + Low-Level Bottom Recall",
    "lda": "LDA prediction signals",
    "qda": "QDA prediction signals",
    "svm": "SVM prediction signals",
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


def _normalize_model_kind(model_kind: str) -> str:
    normalized = model_kind.lower().strip().replace("-", "_")
    aliases = {
        "xgb": "xgboost",
        "xboost": "xgboost",
        "rf": "random_forest",
        "randomforest": "random_forest",
        "bpnn": "bp",
        "bp_neural_network": "bp",
        "mlp": "bp",
        "spline_gam": "gam",
        "linear_discriminant_analysis": "lda",
        "quadratic_discriminant_analysis": "qda",
        "svc": "svm",
        "support_vector_machine": "svm",
        "l1_logit": "l1_logistic",
        "l1_logistic_variance": "l1_logistic_var",
        "l1_logistic_selectkbest": "l1_logistic_kbest",
        "xgb_small": "xgboost_small",
        "equal_blend": "blend_equal",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model kind: {model_kind}")
    return normalized


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


def _load_prediction(model_kind: str, start_date: str, target_index: str) -> pd.DataFrame:
    model_kind = _normalize_model_kind(model_kind)
    path = project_path(PREDICTION_TEMPLATE.format(model_kind=model_kind))
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    prediction = pd.read_csv(path, parse_dates=["trade_date"])
    prediction = prediction[prediction["trade_date"] >= pd.Timestamp(start_date)].copy()
    if "index_code" in prediction.columns:
        prediction = prediction[prediction["index_code"].astype(str).str.zfill(6) == str(target_index).zfill(6)].copy()
        if prediction.empty:
            raise RuntimeError(f"{path} has no prediction rows for index_code={target_index} after {start_date}.")
    required = [
        "trade_date",
        "manual_weak_bottom_proba",
        "manual_weak_top_proba",
        "weak_combined_bottom_signal",
        "weak_combined_top_signal",
    ]
    missing = [column for column in required if column not in prediction.columns]
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")
    for column in ["manual_weak_bottom_proba", "manual_weak_top_proba", "future_ret_15d"]:
        if column in prediction.columns:
            prediction[column] = pd.to_numeric(prediction[column], errors="coerce")
    for column in ["weak_combined_bottom_signal", "weak_combined_top_signal"]:
        prediction[column] = pd.to_numeric(prediction[column], errors="coerce").fillna(0).astype(int)
    if "future_ret_15d" not in prediction.columns:
        prediction["future_ret_15d"] = pd.NA
    if "train_end" not in prediction.columns:
        prediction["train_end"] = ""
    prediction["model_kind"] = model_kind
    return prediction.sort_values("trade_date").reset_index(drop=True)


def _base_market_payload(data: pd.DataFrame) -> dict:
    return {
        "dates": [pd.Timestamp(value).strftime("%Y-%m-%d") for value in data["trade_date"]],
        "candles": [
            [_round(row.open), _round(row.close), _round(row.low), _round(row.high)]
            for row in data.itertuples(index=False)
        ],
        "closes": [_round(row.close) for row in data.itertuples(index=False)],
    }


def _build_model_payload(
    market: pd.DataFrame,
    prediction: pd.DataFrame,
    model_kind: str,
    bottom_threshold: float,
    top_threshold: float,
    bottom_top_veto_threshold: float,
    top_bottom_veto_threshold: float,
    compress_signals: bool,
    max_gap_days: int,
    compressed_min_proba: float | None,
    min_daily_proba: float | None,
    use_file_signals: bool,
) -> dict:
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
    if not use_file_signals:
        data["weak_combined_bottom_signal"] = (
            (data["manual_weak_bottom_proba"] >= bottom_threshold)
            & (data["manual_weak_top_proba"] < bottom_top_veto_threshold)
        ).astype(int)
        data["weak_combined_top_signal"] = (
            (data["manual_weak_top_proba"] >= top_threshold)
            & (data["manual_weak_bottom_proba"] < top_bottom_veto_threshold)
        ).astype(int)
    if min_daily_proba is not None and not use_file_signals:
        data.loc[data["manual_weak_bottom_proba"] <= min_daily_proba, "weak_combined_bottom_signal"] = 0
        data.loc[data["manual_weak_top_proba"] <= min_daily_proba, "weak_combined_top_signal"] = 0
    if compress_signals:
        data["weak_combined_bottom_signal"] = _representative_signal(
            data,
            signal_col="weak_combined_bottom_signal",
            proba_col="manual_weak_bottom_proba",
            max_gap_days=max_gap_days,
            min_proba=compressed_min_proba,
        )
        data["weak_combined_top_signal"] = _representative_signal(
            data,
            signal_col="weak_combined_top_signal",
            proba_col="manual_weak_top_proba",
            max_gap_days=max_gap_days,
            min_proba=compressed_min_proba,
        )

    payload = _base_market_payload(data)
    payload.update(
        {
            "modelKind": model_kind,
            "modelLabel": MODEL_LABELS[model_kind],
            "bottomProb": [_round(value, 6) for value in data["manual_weak_bottom_proba"]],
            "topProb": [_round(value, 6) for value in data["manual_weak_top_proba"]],
            "bottomPoints": [],
            "topPoints": [],
        }
    )
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
                    "label": f"B{bottom_id}\n{_pct(bottom_proba)}",
                    "tooltip": (
                        f"{MODEL_LABELS[model_kind]}<br>"
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
                    "label": f"T{top_id}\n{_pct(top_proba)}",
                    "tooltip": (
                        f"{MODEL_LABELS[model_kind]}<br>"
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


def _representative_signal(
    data: pd.DataFrame,
    signal_col: str,
    proba_col: str,
    max_gap_days: int,
    min_proba: float | None = None,
) -> pd.Series:
    result = pd.Series(0, index=data.index, dtype=int)
    signals = data[data[signal_col] == 1].sort_values("trade_date").copy()
    if signals.empty:
        return result
    gaps = signals["trade_date"].diff().dt.days.fillna(max_gap_days + 1)
    signals["_region_id"] = (gaps > max_gap_days).cumsum()
    for _, part in signals.groupby("_region_id", sort=True):
        representative_idx = pd.to_numeric(part[proba_col], errors="coerce").idxmax()
        representative_proba = pd.to_numeric(pd.Series([data.loc[representative_idx, proba_col]]), errors="coerce").iloc[0]
        if min_proba is None or representative_proba >= min_proba:
            result.loc[representative_idx] = 1
    return result


def _summary_table(payloads: dict[str, dict]) -> str:
    headers = ["model", "bottom B count", "top T count", "date range"]
    header_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    rows = []
    for model_kind in payloads:
        payload = payloads[model_kind]
        rows.append(
            "<tr>"
            f"<td>{html.escape(payload['modelLabel'])}</td>"
            f"<td>{len(payload['bottomPoints'])}</td>"
            f"<td>{len(payload['topPoints'])}</td>"
            f"<td>{html.escape(payload['dates'][0])} to {html.escape(payload['dates'][-1])}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _render_html(
    payloads: dict[str, dict],
    target_index: str,
    start_date: str,
    bottom_threshold: float,
    top_threshold: float,
    bottom_top_veto_threshold: float,
    top_bottom_veto_threshold: float,
    compress_signals: bool,
    max_gap_days: int,
    compressed_min_proba: float | None,
    min_daily_proba: float | None,
    use_file_signals: bool,
) -> str:
    chart_json = json.dumps(payloads, ensure_ascii=False, allow_nan=False)
    sections = []
    ordered_models = list(payloads.keys())
    for index, model_kind in enumerate(ordered_models, start=1):
        payload = payloads[model_kind]
        sections.append(
            f"""
  <section class="panel">
    <h2>{index}. {html.escape(payload["modelLabel"])}</h2>
    <div class="cards">
      <div class="card"><div class="muted">bottom B count</div><div class="value">{len(payload["bottomPoints"])}</div></div>
      <div class="card"><div class="muted">top T count</div><div class="value">{len(payload["topPoints"])}</div></div>
      <div class="card"><div class="muted">start</div><div class="value">{html.escape(payload["dates"][0])}</div></div>
      <div class="card"><div class="muted">end</div><div class="value">{html.escape(payload["dates"][-1])}</div></div>
    </div>
    <div class="legend">
      <span><span class="dot" style="background:#2563eb"></span>bottom prediction B# / buy candidate</span>
      <span><span class="dot" style="background:#dc2626"></span>top prediction T# / sell candidate</span>
      <span><span class="dot" style="background:#64748b"></span>bottom/top probability lines</span>
    </div>
    <div id="chart_{html.escape(model_kind)}" class="chart"></div>
  </section>
"""
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BP XGBoost RF Prediction Signals</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; color: #111827; background: #f6f7f9; }}
    main {{ max-width: 1540px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    .panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; margin: 18px 0; padding: 14px; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; margin: 14px 0; }}
    .card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; }}
    .value {{ font-size: 18px; font-weight: 700; margin-top: 3px; }}
    .chart {{ width: 100%; height: 900px; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0 10px; font-size: 13px; color: #374151; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #f3f4f6; }}
  </style>
</head>
<body>
<main>
  <h1>BP / XGBoost / Random Forest Prediction Signals</h1>
  <div class="muted">target={html.escape(target_index)}, range starts at {html.escape(start_date)}. Signal source: {"prediction file weak_combined_*_signal columns" if use_file_signals else "probability thresholds"}. Bottom signal: bottom probability &gt;= {bottom_threshold:.2f} and top probability &lt; {bottom_top_veto_threshold:.2f}. Top signal: top probability &gt;= {top_threshold:.2f} and bottom probability &lt; {top_bottom_veto_threshold:.2f}. Daily probability filter: {"&gt; " + f"{min_daily_proba:.2f}" if min_daily_proba is not None else "off"}. Signal compression: {"on, max gap " + str(max_gap_days) + " days, min representative probability " + f"{compressed_min_proba:.2f}" if compress_signals and compressed_min_proba is not None else ("on, max gap " + str(max_gap_days) + " days" if compress_signals else "off")}.</div>
  <section class="panel">
    <h2>Summary</h2>
    {_summary_table(payloads)}
  </section>
  {"".join(sections)}
</main>
<script>
(() => {{
  const allCharts = {chart_json};
  const bottomColor = "#2563eb";
  const topColor = "#dc2626";
  const closeColor = "#64748b";

  function markerSeries(name, color, data, position, rotate) {{
    return {{
      name,
      type: "scatter",
      data,
      symbol: "triangle",
      symbolRotate: rotate,
      symbolSize: 15,
      itemStyle: {{ color, borderColor: "#fff", borderWidth: 1.5 }},
      label: {{
        show: true,
        formatter: p => p.data.label,
        position,
        color,
        fontWeight: "bold",
        fontSize: 11,
        lineHeight: 14,
      }},
      tooltip: {{ trigger: "item", formatter: p => p.data.tooltip }},
      z: 8,
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
        {{ left: 64, right: 42, top: 54, height: 620 }},
        {{ left: 64, right: 42, top: 708, height: 110 }},
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
        {{ type: "inside", xAxisIndex: [0, 1], start: 0, end: 100, minSpan: 8, filterMode: "none", zoomOnMouseWheel: true, moveOnMouseMove: true, moveOnMouseWheel: false }},
        {{ type: "slider", xAxisIndex: [0, 1], top: 842, height: 28, start: 0, end: 100, minSpan: 8, filterMode: "none" }},
      ],
      series: [
        {{
          name: "K line",
          type: "candlestick",
          data: data.candles,
          itemStyle: {{ color: "#ef4444", color0: "#16a34a", borderColor: "#dc2626", borderColor0: "#16a34a" }},
        }},
        {{ name: "close", type: "line", data: data.closes, smooth: true, showSymbol: false, lineStyle: {{ width: 1.2, color: closeColor }}, z: 2 }},
        markerSeries("bottom B", bottomColor, data.bottomPoints, "bottom", 0),
        markerSeries("top T", topColor, data.topPoints, "top", 180),
        {{ name: "bottom probability", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: data.bottomProb, showSymbol: false, lineStyle: {{ color: bottomColor, width: 1.4 }} }},
        {{ name: "top probability", type: "line", xAxisIndex: 1, yAxisIndex: 1, data: data.topProb, showSymbol: false, lineStyle: {{ color: topColor, width: 1.4 }} }},
      ],
    }});
    window.addEventListener("resize", () => chart.resize());
  }}

  for (const modelKind of Object.keys(allCharts)) {{
    renderChart(`chart_${{modelKind}}`, allCharts[modelKind]);
  }}
}})();
</script>
</body>
</html>
"""


def build_html(
    models: list[str] | None = None,
    start_date: str = DEFAULT_START_DATE,
    output: str = DEFAULT_OUTPUT,
    bottom_threshold: float = 0.60,
    top_threshold: float = 0.60,
    bottom_top_veto_threshold: float = 0.50,
    top_bottom_veto_threshold: float = 0.50,
    compress_signals: bool = False,
    max_gap_days: int = 5,
    compressed_min_proba: float | None = None,
    min_daily_proba: float | None = None,
    use_file_signals: bool = False,
) -> Path:
    model_list = [_normalize_model_kind(model) for model in (models or DEFAULT_MODELS)]
    target_index = _load_target_index()
    market = _load_market(target_index, start_date)
    payloads = {}
    for model_kind in model_list:
        prediction = _load_prediction(model_kind, start_date, target_index)
        payloads[model_kind] = _build_model_payload(
            market,
            prediction,
            model_kind,
            bottom_threshold,
            top_threshold,
            bottom_top_veto_threshold,
            top_bottom_veto_threshold,
            compress_signals,
            max_gap_days,
            compressed_min_proba,
            min_daily_proba,
            use_file_signals,
        )
    output_path = project_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_html(
            payloads,
            target_index,
            start_date,
            bottom_threshold,
            top_threshold,
            bottom_top_veto_threshold,
            top_bottom_veto_threshold,
            compress_signals,
            max_gap_days,
            compressed_min_proba,
            min_daily_proba,
            use_file_signals,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BP, XGBoost, and RF raw prediction signals.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bottom-threshold", type=float, default=0.60)
    parser.add_argument("--top-threshold", type=float, default=0.60)
    parser.add_argument("--bottom-top-veto-threshold", type=float, default=0.50)
    parser.add_argument("--top-bottom-veto-threshold", type=float, default=0.50)
    parser.add_argument("--compress-signals", action="store_true")
    parser.add_argument("--max-gap-days", type=int, default=5)
    parser.add_argument("--compressed-min-proba", type=float)
    parser.add_argument("--min-daily-proba", type=float)
    parser.add_argument("--use-file-signals", action="store_true")
    args = parser.parse_args()
    print(
        build_html(
            args.models,
            args.start_date,
            args.output,
            args.bottom_threshold,
            args.top_threshold,
            args.bottom_top_veto_threshold,
            args.top_bottom_veto_threshold,
            args.compress_signals,
            args.max_gap_days,
            args.compressed_min_proba,
            args.min_daily_proba,
            args.use_file_signals,
        )
    )


if __name__ == "__main__":
    main()
