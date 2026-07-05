from __future__ import annotations

import argparse
import html
from pathlib import Path

import pandas as pd

from src.common.config import get_config, project_path


def _pct(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.2%}"


def _num(value: object, digits: int = 0) -> str:
    if value is None or pd.isna(value):
        return ""
    if digits <= 0:
        return str(int(float(value)))
    return f"{float(value):.{digits}f}"


def _read_report(path_value: str) -> pd.DataFrame:
    path = project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    return pd.read_csv(path)


def _metric_cards(aggregate: pd.DataFrame) -> str:
    focus = aggregate[aggregate["sample_mode"] == "non_overlapping"].copy()
    if focus.empty:
        focus = aggregate.copy()
    best_precision = focus.sort_values(["precision", "signal_count"], ascending=False).head(1)
    best_return = focus.sort_values(["avg_return_lift", "signal_count"], ascending=False).head(1)
    largest_sample = focus.sort_values(["signal_count", "precision"], ascending=False).head(1)
    rows = [
        ("最高 precision", best_precision),
        ("最高收益提升", best_return),
        ("最多非重叠信号", largest_sample),
    ]
    cards = []
    for title, frame in rows:
        if frame.empty:
            continue
        row = frame.iloc[0]
        cards.append(
            "<section class='metric-card'>"
            f"<div class='muted'>{html.escape(title)}</div>"
            f"<div class='metric-main'>{html.escape(str(row['model']))} / {html.escape(str(row['side']))} / {float(row['threshold']):.2f}</div>"
            f"<div>precision {_pct(row['precision'])}，信号 {_num(row['signal_count'])}，收益提升 {_pct(row['avg_return_lift'])}</div>"
            "</section>"
        )
    return "".join(cards)


def _table(frame: pd.DataFrame, columns: list[str], max_rows: int = 120) -> str:
    headers = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in frame.head(max_rows).itertuples(index=False):
        values = row._asdict()
        cells = []
        for column in columns:
            value = values[column]
            if column in {
                "precision", "natural_precision", "precision_lift", "actual_up_rate",
                "actual_down_rate", "avg_future_ret_3w", "avg_return_lift",
                "natural_avg_future_ret_3w", "avg_future_mfe_3w", "avg_future_mae_3w",
            }:
                text = _pct(value)
            elif column in {"threshold"}:
                text = f"{float(value):.2f}" if not pd.isna(value) else ""
            elif column in {"candidate_rows", "signal_count", "test_year", "train_rows"}:
                text = _num(value)
            else:
                text = "" if pd.isna(value) else str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _bar_chart(frame: pd.DataFrame) -> str:
    focus = frame[frame["sample_mode"] == "non_overlapping"].copy()
    if focus.empty:
        focus = frame.copy()
    focus["name"] = (
        focus["model"].astype(str)
        + " / "
        + focus["side"].astype(str)
        + " / "
        + focus["threshold"].astype(float).map(lambda value: f"{value:.2f}")
    )
    focus = focus.sort_values(["side", "model", "threshold"]).reset_index(drop=True)
    width = 1160
    row_height = 24
    left = 230
    right = 90
    top = 28
    height = top + row_height * len(focus) + 24
    max_value = max(float(focus["precision"].max() or 0), float(focus["natural_precision"].max() or 0), 0.01)
    chart_width = width - left - right
    elements = [
        f'<text x="12" y="18" font-size="14" font-weight="700" fill="#111827">非重叠 precision 对比</text>'
    ]
    for index, row in enumerate(focus.itertuples(index=False)):
        y = top + index * row_height
        precision_width = float(row.precision or 0) / max_value * chart_width
        natural_width = float(row.natural_precision or 0) / max_value * chart_width
        elements.append(f'<text x="12" y="{y + 13}" font-size="11" fill="#374151">{html.escape(row.name)}</text>')
        elements.append(f'<rect x="{left}" y="{y + 3}" width="{natural_width:.2f}" height="7" fill="#cbd5e1" />')
        elements.append(f'<rect x="{left}" y="{y + 12}" width="{precision_width:.2f}" height="7" fill="#2563eb" />')
        elements.append(f'<text x="{left + max(precision_width, natural_width) + 6:.2f}" y="{y + 17}" font-size="10" fill="#4b5563">{_pct(row.precision)}</text>')
    return f'<svg class="bar-svg" viewBox="0 0 {width} {height}" role="img">{"".join(elements)}</svg>'


def _render_html(aggregate: pd.DataFrame, summary: pd.DataFrame, detail: pd.DataFrame) -> str:
    aggregate_focus = aggregate.sort_values(
        ["sample_mode", "side", "model", "threshold"],
        ascending=[False, True, True, True],
    )
    summary_focus = summary[summary["sample_mode"] == "non_overlapping"].copy()
    summary_focus = summary_focus.sort_values(["test_year", "side", "model", "threshold"])
    detail_focus = detail.sort_values(["week_end_date", "side", "model"]).tail(80)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>中证1000周线顶底模型评估</title>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; color: #111827; background: #f7f8fa; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 24px 0 10px; font-size: 18px; }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 16px; }}
    .metric-card, .panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px; }}
    .metric-main {{ font-size: 18px; font-weight: 700; margin: 6px 0; }}
    .table-wrap {{ overflow-x: auto; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; min-width: 980px; }}
    th, td {{ padding: 8px 9px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 12px; white-space: nowrap; }}
    th {{ background: #f3f4f6; color: #374151; }}
    .chart-wrap {{ overflow-x: auto; background: #fff; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
    .bar-svg {{ min-width: 1000px; width: 100%; height: auto; display: block; }}
  </style>
</head>
<body>
<main>
  <h1>中证1000周线顶底模型评估</h1>
  <div class="muted">模型：logistic / lightgbm / bp / svm；样本：每周最后交易日；标签：人工周线弱标记；horizon：未来 3 周。</div>
  <div class="grid">{_metric_cards(aggregate)}</div>

  <h2>非重叠 Precision 图</h2>
  <section class="chart-wrap">{_bar_chart(aggregate)}</section>

  <h2>聚合评估</h2>
  <section class="table-wrap">
    {_table(aggregate_focus, ["model", "side", "threshold", "sample_mode", "candidate_rows", "signal_count", "precision", "natural_precision", "precision_lift", "avg_future_ret_3w", "avg_return_lift"])}
  </section>

  <h2>逐年非重叠评估</h2>
  <section class="table-wrap">
    {_table(summary_focus, ["test_year", "model", "side", "threshold", "candidate_rows", "signal_count", "precision", "natural_precision", "precision_lift", "avg_future_ret_3w", "avg_return_lift"], max_rows=260)}
  </section>

  <h2>最近预测明细</h2>
  <section class="table-wrap">
    {_table(detail_focus, ["week_end_date", "model", "side", "turning_proba", "tuning_status", "tuned_params_json", "manual_weak_bottom_label", "manual_weak_top_label", "future_ret_3w"], max_rows=100)}
  </section>
</main>
</body>
</html>
"""


def _tuning_section(cfg: dict) -> str:
    path_value = cfg["outputs"].get("tuning_report")
    if not path_value:
        return ""
    path = project_path(str(path_value))
    if not path.exists():
        return ""
    tuning = pd.read_csv(path)
    if tuning.empty:
        return ""
    selected = tuning[tuning.get("selected", 0) == 1].copy()
    if selected.empty:
        selected = tuning.copy()
    selected = selected.sort_values(["test_year", "side", "model"])
    return f"""
  <h2>调参选择</h2>
  <section class="table-wrap">
    {_table(selected, ["test_year", "model", "side", "tuning_status", "threshold", "validation_rows", "validation_signal_count", "validation_precision", "validation_avg_return_lift", "params_json"], max_rows=260)}
  </section>
"""


def _optimized_policy_section(cfg: dict) -> str:
    path_value = cfg["outputs"].get("optimized_policy_report")
    if not path_value:
        return ""
    path = project_path(str(path_value))
    if not path.exists():
        return ""
    policy = pd.read_csv(path)
    if policy.empty:
        return ""
    policy = policy.sort_values(["objective_score", "signal_count"], ascending=False)
    return f"""
  <h2>组合优化规则</h2>
  <section class="table-wrap">
    {_table(policy, ["side", "enabled_models", "thresholds_json", "min_votes", "signal_count", "precision", "natural_precision", "precision_lift", "avg_future_ret_3w", "avg_return_lift", "objective_score"], max_rows=80)}
  </section>
"""


def build_bottom_weekly_evaluation_html(output: str | None = None) -> Path:
    cfg = get_config("bottom_weekly_model.yaml")
    aggregate = _read_report(str(cfg["outputs"]["walk_forward_aggregate"]))
    summary = _read_report(str(cfg["outputs"]["walk_forward_summary"]))
    detail = _read_report(str(cfg["outputs"]["walk_forward_detail"]))
    output_path = project_path(output or str(cfg["outputs"]["walk_forward_html"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_text = _render_html(aggregate, summary, detail)
    tuning_html = _tuning_section(cfg)
    if tuning_html:
        html_text = html_text.replace("</main>", f"{tuning_html}</main>")
    policy_html = _optimized_policy_section(cfg)
    if policy_html:
        html_text = html_text.replace("</main>", f"{policy_html}</main>")
    output_path.write_text(html_text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build weekly top/bottom evaluation HTML.")
    parser.add_argument("--output")
    args = parser.parse_args()
    print(build_bottom_weekly_evaluation_html(args.output))


if __name__ == "__main__":
    main()
