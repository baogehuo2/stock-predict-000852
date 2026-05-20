from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from jinja2 import Template

from src.common.config import get_config, project_path
from src.common.db import read_sql
from src.common.logger import get_logger
from src.modeling.similar_event import find_similar_dates


logger = get_logger(__name__)

HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>中证1000短期趋势预测日报 {{ trade_date }}</title>
  <style>
    body { font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; color: #1f2937; background: #f6f7f9; }
    main { max-width: 1180px; margin: 0 auto; padding: 28px; }
    h1 { font-size: 26px; margin: 0 0 8px; }
    h2 { font-size: 18px; margin: 28px 0 12px; }
    .muted { color: #6b7280; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    .card { background: white; border: 1px solid #e5e7eb; border-radius: 8px; padding: 16px; }
    .direction { font-size: 24px; font-weight: 700; margin: 8px 0; }
    table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #e5e7eb; }
    th, td { padding: 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 14px; }
    th { background: #f3f4f6; }
    .risk { background: #fff7ed; border-color: #fed7aa; }
    .disclaimer { margin-top: 24px; font-size: 13px; color: #6b7280; }
  </style>
</head>
<body>
<main>
  <h1>中证1000短期趋势预测日报</h1>
  <div class="muted">交易日：{{ trade_date }}</div>

  <h2>预测结论</h2>
  <div class="grid">
    {% for row in predictions %}
    <section class="card">
      <div class="muted">未来 {{ row.horizon }} 个交易日</div>
      <div class="direction">{{ row.direction_pred }}</div>
      <div>预测涨跌幅：{{ "%.2f%%"|format(row.return_pred * 100) }}</div>
      <div>区间：{{ "%.2f%%"|format(row.return_low * 100) }} 至 {{ "%.2f%%"|format(row.return_high * 100) }}</div>
      <div>置信度：{{ "%.1f%%"|format(row.confidence * 100) }}</div>
    </section>
    {% endfor %}
  </div>

  <h2>解释</h2>
  {% for row in predictions %}
  <section class="card" style="margin-bottom: 12px;">
    <strong>{{ row.horizon }}日：</strong>{{ row.explanation }}
  </section>
  {% endfor %}

  <h2>历史相似日期</h2>
  <table>
    <thead><tr><th>日期</th><th>相似度</th><th>后3日</th><th>后5日</th><th>后7日</th></tr></thead>
    <tbody>
    {% for row in similar %}
      <tr>
        <td>{{ row.trade_date }}</td>
        <td>{{ "%.3f"|format(row.similarity) }}</td>
        <td>{{ "%.2f%%"|format(row.future_ret_3d * 100) }}</td>
        <td>{{ "%.2f%%"|format(row.future_ret_5d * 100) }}</td>
        <td>{{ "%.2f%%"|format(row.future_ret_7d * 100) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>

  <h2>风险提示</h2>
  <section class="card risk">{{ risk_warning }}</section>

  <div class="disclaimer">{{ disclaimer }}</div>
</main>
</body>
</html>
"""


def generate_html_report(trade_date: str | None = None) -> Path:
    cfg = get_config()
    if trade_date is None:
        latest = read_sql("SELECT MAX(trade_date) trade_date FROM prediction_result")
        if latest.empty or pd.isna(latest.iloc[0]["trade_date"]):
            raise RuntimeError("No prediction_result rows available.")
        trade_date = pd.to_datetime(latest.iloc[0]["trade_date"]).strftime("%Y-%m-%d")

    predictions = read_sql(
        "SELECT * FROM prediction_result WHERE trade_date=:trade_date ORDER BY horizon",
        {"trade_date": trade_date},
    )
    if predictions.empty:
        raise RuntimeError(f"No predictions found for {trade_date}")
    similar = find_similar_dates(trade_date, top_n=10)
    similar_records = similar.to_dict(orient="records") if not similar.empty else []
    for row in similar_records:
        row["trade_date"] = pd.to_datetime(row["trade_date"]).strftime("%Y-%m-%d")
    risk_warning = predictions.iloc[0]["risk_warning"]
    html = Template(HTML_TEMPLATE).render(
        trade_date=trade_date,
        predictions=predictions.to_dict(orient="records"),
        similar=similar_records,
        risk_warning=risk_warning,
        disclaimer=cfg["project"]["output_disclaimer"],
    )
    report_dir = project_path(cfg["paths"]["reports"])
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"zz1000_report_{trade_date}.html"
    out.write_text(html, encoding="utf-8")
    logger.info("generated html report %s", out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trade-date")
    args = parser.parse_args()
    print(generate_html_report(args.trade_date))


if __name__ == "__main__":
    main()

