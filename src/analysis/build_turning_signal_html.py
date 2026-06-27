from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.config import load_yaml, project_path
from src.common.db import read_sql


DEFAULT_MODELS = ["lightgbm", "xgboost", "bp"]
PREDICTION_TEMPLATE = "data/reports/manual_weak_turning_walk_forward_predictions_{model_kind}_wf.csv"
MODEL_TOP_BOTTOM_HTML = "data/reports/model_predicted_top_bottom_signals.html"


@dataclass(frozen=True)
class ChartMarker:
    trade_date: pd.Timestamp
    price: float
    label: str
    marker_type: str
    color: str
    tooltip: str


def _pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _fmt_price(value: float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _normalize_model_kind(model_kind: str) -> str:
    normalized = model_kind.lower().strip()
    if normalized == "xboost":
        return "xgboost"
    if normalized in {"bpnn", "bp_neural_network", "neural_network", "mlp"}:
        return "bp"
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


def _load_prediction(model_kind: str, start_date: str) -> pd.DataFrame:
    model_kind = _normalize_model_kind(model_kind)
    path = project_path(PREDICTION_TEMPLATE.format(model_kind=model_kind))
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    prediction = pd.read_csv(path, parse_dates=["trade_date"])
    prediction = prediction[prediction["trade_date"] >= pd.Timestamp(start_date)].copy()
    for column in ["weak_combined_bottom_signal", "weak_combined_top_signal"]:
        prediction[column] = pd.to_numeric(prediction[column], errors="coerce").fillna(0).astype(int)
    for column in ["manual_weak_bottom_proba", "manual_weak_top_proba"]:
        prediction[column] = pd.to_numeric(prediction[column], errors="coerce")
    prediction["model_kind"] = model_kind
    return prediction.sort_values("trade_date").reset_index(drop=True)


def _market_lookup(market: pd.DataFrame) -> pd.DataFrame:
    return market.set_index("trade_date")[["open", "high", "low", "close"]]


def _prediction_markers(prediction: pd.DataFrame, market_by_date: pd.DataFrame) -> list[ChartMarker]:
    markers: list[ChartMarker] = []
    for row in prediction.itertuples(index=False):
        trade_date = pd.Timestamp(row.trade_date)
        if trade_date not in market_by_date.index:
            continue
        market_row = market_by_date.loc[trade_date]
        if int(row.weak_combined_bottom_signal) == 1:
            markers.append(
                ChartMarker(
                    trade_date=trade_date,
                    price=float(market_row["low"]),
                    label="B",
                    marker_type="bottom",
                    color="#2563eb",
                    tooltip=(
                        f"{trade_date:%Y-%m-%d} bottom "
                        f"bottom={_pct(row.manual_weak_bottom_proba)} "
                        f"top={_pct(row.manual_weak_top_proba)}"
                    ),
                )
            )
        if int(row.weak_combined_top_signal) == 1:
            markers.append(
                ChartMarker(
                    trade_date=trade_date,
                    price=float(market_row["high"]),
                    label="T",
                    marker_type="top",
                    color="#dc2626",
                    tooltip=(
                        f"{trade_date:%Y-%m-%d} top "
                        f"top={_pct(row.manual_weak_top_proba)} "
                        f"bottom={_pct(row.manual_weak_bottom_proba)}"
                    ),
                )
            )
    return markers


def _x_positions(market: pd.DataFrame, inner_width: float, left: float) -> dict[pd.Timestamp, float]:
    count = len(market)
    if count <= 1:
        return {pd.Timestamp(market.iloc[0]["trade_date"]): left}
    step = inner_width / (count - 1)
    return {
        pd.Timestamp(row.trade_date): left + index * step
        for index, row in enumerate(market.itertuples(index=False))
    }


def _axis_ticks(min_price: float, max_price: float, tick_count: int = 6) -> list[float]:
    if max_price <= min_price:
        return [min_price]
    return [min_price + (max_price - min_price) * i / (tick_count - 1) for i in range(tick_count)]


def _render_svg_chart(
    market: pd.DataFrame,
    markers: list[ChartMarker],
    title: str,
    subtitle: str,
    width: int = 2200,
    height: int = 620,
) -> str:
    left, right, top, bottom = 72, 36, 44, 64
    inner_width = width - left - right
    inner_height = height - top - bottom
    low = float(market["low"].min())
    high = float(market["high"].max())
    padding = (high - low) * 0.08 if high > low else 1.0
    min_price = low - padding
    max_price = high + padding

    def y(price: float) -> float:
        return top + (max_price - price) / (max_price - min_price) * inner_height

    x_by_date = _x_positions(market, inner_width, left)
    candle_width = max(1.1, min(6.0, inner_width / max(len(market), 1) * 0.62))
    elements: list[str] = []

    elements.append(
        f'<text x="{left}" y="26" font-size="20" font-weight="700" fill="#111827">{html.escape(title)}</text>'
    )
    elements.append(
        f'<text x="{left}" y="43" font-size="12" fill="#6b7280">{html.escape(subtitle)}</text>'
    )

    for tick in _axis_ticks(min_price, max_price):
        yy = y(tick)
        elements.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#e5e7eb" />')
        elements.append(
            f'<text x="10" y="{yy + 4:.2f}" font-size="11" fill="#6b7280">{_fmt_price(tick)}</text>'
        )

    market = market.copy()
    market["year"] = market["trade_date"].dt.year
    for row in market.drop_duplicates("year", keep="first").itertuples(index=False):
        xx = x_by_date[pd.Timestamp(row.trade_date)]
        elements.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}" stroke="#f3f4f6" />')
        elements.append(
            f'<text x="{xx + 3:.2f}" y="{height - 22}" font-size="11" fill="#6b7280">{int(row.year)}</text>'
        )

    close_points = []
    for row in market.itertuples(index=False):
        trade_date = pd.Timestamp(row.trade_date)
        xx = x_by_date[trade_date]
        open_y = y(float(row.open))
        close_y = y(float(row.close))
        high_y = y(float(row.high))
        low_y = y(float(row.low))
        color = "#dc2626" if float(row.close) >= float(row.open) else "#16a34a"
        body_top = min(open_y, close_y)
        body_height = max(abs(close_y - open_y), 1.0)
        elements.append(
            f'<line x1="{xx:.2f}" y1="{high_y:.2f}" x2="{xx:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="1" />'
        )
        elements.append(
            f'<rect x="{xx - candle_width / 2:.2f}" y="{body_top:.2f}" '
            f'width="{candle_width:.2f}" height="{body_height:.2f}" fill="{color}" opacity="0.72" />'
        )
        close_points.append(f"{xx:.2f},{close_y:.2f}")
    elements.append(
        f'<polyline points="{" ".join(close_points)}" fill="none" stroke="#1d4ed8" stroke-width="1.4" opacity="0.62" />'
    )

    for marker in markers:
        if marker.trade_date not in x_by_date:
            continue
        xx = x_by_date[marker.trade_date]
        yy = y(marker.price)
        escaped_tooltip = html.escape(marker.tooltip)
        if marker.marker_type == "bottom":
            yy += 14
            points = f"{xx:.2f},{yy + 8:.2f} {xx - 8:.2f},{yy - 6:.2f} {xx + 8:.2f},{yy - 6:.2f}"
        else:
            yy -= 14
            points = f"{xx:.2f},{yy - 8:.2f} {xx - 8:.2f},{yy + 6:.2f} {xx + 8:.2f},{yy + 6:.2f}"
        elements.append(
            f'<g><title>{escaped_tooltip}</title>'
            f'<polygon points="{points}" fill="{marker.color}" opacity="0.92" stroke="white" stroke-width="1" />'
            f'<text x="{xx + 9:.2f}" y="{yy + 4:.2f}" font-size="10" fill="{marker.color}">{marker.label}</text>'
            f'</g>'
        )

    elements.append(
        f'<rect x="{left}" y="{top}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#d1d5db" />'
    )
    return (
        f'<svg class="kline-svg" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{html.escape(title)}">{"".join(elements)}</svg>'
    )


def _summary_table(rows: list[dict]) -> str:
    headers = ["model", "bottom signals", "top signals", "note"]
    header_html = "".join(f"<th>{column}</th>" for column in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{int(row['bottom_count'])}</td>"
            f"<td>{int(row['top_count'])}</td>"
            f"<td>{html.escape(str(row['note']))}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _section(model_kind: str, svg: str) -> str:
    return f"""
  <section class="card">
    <h2>{html.escape(model_kind)}</h2>
    <div class="chart-wrap">{svg}</div>
    <div class="legend">
      <span><span class="dot" style="background:#2563eb"></span>model bottom</span>
      <span><span class="dot" style="background:#dc2626"></span>model top</span>
    </div>
  </section>
"""


def _render_html(title: str, description: str, summary_rows: list[dict], sections: list[str]) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; color: #111827; background: #f6f7f9; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    .card {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 16px; margin: 16px 0; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); }}
    .chart-wrap {{ overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; }}
    .kline-svg {{ display: block; min-width: 1280px; width: 100%; height: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #f3f4f6; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin-top: 8px; font-size: 13px; color: #374151; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <div class="muted">{html.escape(description)}</div>
  <section class="card">
    <h2>Summary</h2>
    {_summary_table(summary_rows)}
  </section>
  {"".join(sections)}
</main>
</body>
</html>
"""


def build_html_report(
    models: list[str] | None = None,
    start_date: str = "2022-01-01",
    output: str = MODEL_TOP_BOTTOM_HTML,
) -> Path:
    model_list = [_normalize_model_kind(model) for model in (models or DEFAULT_MODELS)]
    target_index = _load_target_index()
    market = _load_market(target_index, start_date)
    market_by_date = _market_lookup(market)

    sections: list[str] = []
    summary_rows: list[dict] = []
    for model_kind in model_list:
        prediction = _load_prediction(model_kind, start_date)
        markers = _prediction_markers(prediction, market_by_date)
        bottom_count = int(prediction["weak_combined_bottom_signal"].sum())
        top_count = int(prediction["weak_combined_top_signal"].sum())
        summary_rows.append(
            {
                "model": model_kind,
                "bottom_count": bottom_count,
                "top_count": top_count,
                "note": "blue=B/model bottom, red=T/model top",
            }
        )
        sections.append(
            _section(
                model_kind,
                _render_svg_chart(
                    market,
                    markers,
                    f"{model_kind} predicted top/bottom signals",
                    f"{target_index}, {start_date} to {market['trade_date'].max():%Y-%m-%d}",
                ),
            )
        )

    output_path = project_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_html(
            "ZZ1000 Model Predicted Top/Bottom Signals",
            "Default models: lightgbm, xgboost, bp. Blue markers are model bottoms; red markers are model tops.",
            summary_rows,
            sections,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reusable HTML chart for turning model signals.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=MODEL_TOP_BOTTOM_HTML)
    args = parser.parse_args()

    print(build_html_report(models=args.models, start_date=args.start_date, output=args.output))


if __name__ == "__main__":
    main()
