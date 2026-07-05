from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import read_sql
from src.features.build_bottom_weekly_dataset import build_weekly_market_bars


DEFAULT_START_DATE = "2021-01-01"


@dataclass(frozen=True)
class ChartMarker:
    week_end_date: pd.Timestamp
    price: float
    label: str
    marker_type: str
    color: str
    tooltip: str


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _fmt_price(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2f}"


def _load_weekly_market(target_index: str, start_date: str) -> pd.DataFrame:
    daily = read_sql(
        """
        SELECT trade_date, index_code, open, high, low, close, volume
        FROM market_index_daily
        WHERE index_code = :target_index
          AND trade_date >= :start_date
        ORDER BY trade_date
        """,
        {"target_index": target_index, "start_date": start_date},
    )
    if daily.empty:
        raise RuntimeError(f"No market_index_daily rows found for {target_index} after {start_date}.")
    daily["trade_date"] = pd.to_datetime(daily["trade_date"])
    for column in ["open", "high", "low", "close", "volume"]:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")
    weekly = build_weekly_market_bars(daily)
    return weekly.rename(columns={"week_end_date": "trade_date"}).sort_values("trade_date").reset_index(drop=True)


def _load_details(path_value: str, start_date: str) -> pd.DataFrame:
    path = project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Weekly walk-forward detail report not found: {path}")
    details = pd.read_csv(path, parse_dates=["week_start_date", "week_end_date"])
    details = details[details["week_end_date"] >= pd.Timestamp(start_date)].copy()
    details["turning_proba"] = pd.to_numeric(details["turning_proba"], errors="coerce")
    details["future_ret_3w"] = pd.to_numeric(details["future_ret_3w"], errors="coerce")
    details["week_pos"] = pd.to_numeric(details["week_pos"], errors="coerce").astype(int)
    return details.sort_values(["model", "side", "week_pos"]).reset_index(drop=True)


def _non_overlapping(signal: pd.DataFrame, horizon_weeks: int) -> pd.DataFrame:
    if signal.empty:
        return signal
    selected = []
    next_position = -1
    for idx, row in signal.sort_values("week_pos").iterrows():
        position = int(row["week_pos"])
        if position >= next_position:
            selected.append(idx)
            next_position = position + horizon_weeks
    return signal.loc[selected].copy()


def _signals_for_model(
    details: pd.DataFrame,
    model: str,
    threshold: float,
    horizon_weeks: int,
) -> pd.DataFrame:
    parts = []
    for side in ["bottom", "top"]:
        signal = details[
            (details["model"] == model)
            & (details["side"] == side)
            & (details["turning_proba"] >= threshold)
        ].copy()
        parts.append(_non_overlapping(signal, horizon_weeks))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).sort_values("week_pos").reset_index(drop=True)


def _market_lookup(market: pd.DataFrame) -> pd.DataFrame:
    return market.set_index("trade_date")[["open", "high", "low", "close"]]


def _markers(signals: pd.DataFrame, market_by_date: pd.DataFrame) -> list[ChartMarker]:
    result: list[ChartMarker] = []
    for row in signals.itertuples(index=False):
        trade_date = pd.Timestamp(row.week_end_date)
        if trade_date not in market_by_date.index:
            continue
        market_row = market_by_date.loc[trade_date]
        side = str(row.side)
        is_buy = side == "bottom"
        result.append(
            ChartMarker(
                week_end_date=trade_date,
                price=float(market_row["low"] if is_buy else market_row["high"]),
                label="底" if is_buy else "顶",
                marker_type="buy" if is_buy else "sell",
                color="#2563eb" if is_buy else "#dc2626",
                tooltip=(
                    f"{trade_date:%Y-%m-%d} "
                    f"{'预测底部' if is_buy else '预测顶部'} "
                    f"model={row.model} proba={_pct(row.turning_proba)} "
                    f"future_ret_3w={_pct(row.future_ret_3w)}"
                ),
            )
        )
    return result


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
    height: int = 640,
) -> str:
    left, right, top, bottom = 76, 42, 48, 68
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
    candle_width = max(3.0, min(10.0, inner_width / max(len(market), 1) * 0.55))
    elements: list[str] = []
    elements.append(
        f'<text x="{left}" y="27" font-size="20" font-weight="700" fill="#111827">{html.escape(title)}</text>'
    )
    elements.append(
        f'<text x="{left}" y="44" font-size="12" fill="#6b7280">{html.escape(subtitle)}</text>'
    )

    for tick in _axis_ticks(min_price, max_price):
        yy = y(tick)
        elements.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#e5e7eb" />')
        elements.append(f'<text x="12" y="{yy + 4:.2f}" font-size="11" fill="#6b7280">{_fmt_price(tick)}</text>')

    market = market.copy()
    market["year"] = market["trade_date"].dt.year
    for row in market.drop_duplicates("year", keep="first").itertuples(index=False):
        xx = x_by_date[pd.Timestamp(row.trade_date)]
        elements.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height-bottom}" stroke="#f3f4f6" />')
        elements.append(f'<text x="{xx + 4:.2f}" y="{height - 24}" font-size="11" fill="#6b7280">{int(row.year)}</text>')

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
            f'<line x1="{xx:.2f}" y1="{high_y:.2f}" x2="{xx:.2f}" y2="{low_y:.2f}" stroke="{color}" stroke-width="1.1" />'
        )
        elements.append(
            f'<rect x="{xx - candle_width / 2:.2f}" y="{body_top:.2f}" '
            f'width="{candle_width:.2f}" height="{body_height:.2f}" fill="{color}" opacity="0.76" />'
        )
        close_points.append(f"{xx:.2f},{close_y:.2f}")
    elements.append(
        f'<polyline points="{" ".join(close_points)}" fill="none" stroke="#111827" stroke-width="1.2" opacity="0.52" />'
    )

    for marker in markers:
        if marker.week_end_date not in x_by_date:
            continue
        xx = x_by_date[marker.week_end_date]
        yy = y(marker.price)
        escaped_tooltip = html.escape(marker.tooltip)
        if marker.marker_type == "buy":
            yy += 18
            points = f"{xx:.2f},{yy + 10:.2f} {xx - 10:.2f},{yy - 8:.2f} {xx + 10:.2f},{yy - 8:.2f}"
        else:
            yy -= 18
            points = f"{xx:.2f},{yy - 10:.2f} {xx - 10:.2f},{yy + 8:.2f} {xx + 10:.2f},{yy + 8:.2f}"
        elements.append(
            f'<g><title>{escaped_tooltip}</title>'
            f'<polygon points="{points}" fill="{marker.color}" opacity="0.94" stroke="white" stroke-width="1.2" />'
            f'<text x="{xx + 11:.2f}" y="{yy + 4:.2f}" font-size="12" font-weight="700" fill="{marker.color}">{html.escape(marker.label)}</text>'
            f'</g>'
        )

    elements.append(
        f'<rect x="{left}" y="{top}" width="{inner_width}" height="{inner_height}" fill="none" stroke="#d1d5db" />'
    )
    return f'<svg class="kline-svg" viewBox="0 0 {width} {height}" role="img">{"".join(elements)}</svg>'


def _summary_table(rows: list[dict]) -> str:
    headers = ["模型", "阈值", "预测底", "预测顶", "最近预测"]
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    body = []
    for row in rows:
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['model']))}</td>"
            f"<td>{float(row['threshold']):.2f}</td>"
            f"<td>{int(row['buy_count'])}</td>"
            f"<td>{int(row['sell_count'])}</td>"
            f"<td>{html.escape(str(row['latest_signal']))}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _section(model: str, svg: str) -> str:
    return f"""
  <section class="panel">
    <h2>{html.escape(model)}</h2>
    <div class="chart-wrap">{svg}</div>
    <div class="legend">
      <span><span class="dot" style="background:#2563eb"></span>预测底部：bottom 信号</span>
      <span><span class="dot" style="background:#dc2626"></span>预测顶部：top 信号</span>
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
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; color: #111827; background: #f7f8fa; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    .panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; margin: 16px 0; }}
    .chart-wrap {{ overflow-x: auto; border: 1px solid #e5e7eb; border-radius: 8px; background: #fff; }}
    .kline-svg {{ display: block; min-width: 1280px; width: 100%; height: auto; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #f3f4f6; }}
    .legend {{ display: flex; gap: 18px; flex-wrap: wrap; margin-top: 8px; font-size: 13px; color: #374151; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
  </style>
</head>
<body>
<main>
  <h1>{html.escape(title)}</h1>
  <div class="muted">{html.escape(description)}</div>
  <section class="panel">
    <h2>信号汇总</h2>
    {_summary_table(summary_rows)}
  </section>
  {"".join(sections)}
</main>
</body>
</html>
"""


def build_bottom_weekly_signal_html(
    threshold: float | None = None,
    start_date: str = DEFAULT_START_DATE,
    output: str | None = None,
) -> Path:
    cfg = get_config("bottom_weekly_model.yaml")
    model_cfg = cfg["model"]
    default_threshold = model_cfg.get("signal_html_threshold", min(model_cfg["probability_thresholds"]))
    threshold = float(threshold if threshold is not None else default_threshold)
    horizon_weeks = int(model_cfg["horizon_weeks"])
    target_index = str(model_cfg["target_index"])
    models = [str(model) for model in model_cfg.get("model_kinds", ["logistic", "lightgbm"])]
    details = _load_details(str(cfg["outputs"]["walk_forward_detail"]), start_date)
    market = _load_weekly_market(target_index, start_date)
    market_by_date = _market_lookup(market)

    sections = []
    summary_rows = []
    for model in models:
        signals = _signals_for_model(details, model, threshold, horizon_weeks)
        markers = _markers(signals, market_by_date)
        buy_count = int((signals["side"] == "bottom").sum()) if not signals.empty else 0
        sell_count = int((signals["side"] == "top").sum()) if not signals.empty else 0
        latest_signal = ""
        if not signals.empty:
            last = signals.sort_values("week_end_date").iloc[-1]
            latest_signal = f"{pd.Timestamp(last['week_end_date']):%Y-%m-%d} {last['side']} {_pct(last['turning_proba'])}"
        summary_rows.append(
            {
                "model": model,
                "threshold": threshold,
                "buy_count": buy_count,
                "sell_count": sell_count,
                "latest_signal": latest_signal,
            }
        )
        sections.append(
            _section(
                model,
                _render_svg_chart(
                    market,
                    markers,
                    f"{model} 周K预测顶底点",
                    f"{target_index}，{start_date} 至 {market['trade_date'].max():%Y-%m-%d}，阈值 {threshold:.2f}，非重叠间隔 {horizon_weeks} 周",
                ),
            )
        )

    output_path = project_path(output or str(cfg["outputs"]["signal_html"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        _render_html(
            "中证1000周K预测顶底点",
            "预测底部来自周线 bottom 信号，预测顶部来自周线 top 信号；同一方向按未来 3 周窗口做非重叠压缩。",
            summary_rows,
            sections,
        ),
        encoding="utf-8",
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weekly K-line predicted top/bottom signal HTML.")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(build_bottom_weekly_signal_html(args.threshold, args.start_date, args.output))


if __name__ == "__main__":
    main()
