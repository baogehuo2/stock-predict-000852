from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template
from plotly.offline.offline import get_plotlyjs

from src.common.config import project_path
from src.common.config import get_config
from src.common.db import read_sql
from src.common.logger import get_logger


logger = get_logger(__name__)

UP_COLOR = "#b42318"
DOWN_COLOR = "#067647"
NEUTRAL_COLOR = "#98a2b3"
INFO_COLOR = "#175cd3"
THRESHOLD_COLOR = "#f79009"

DEFAULT_DETAILS_CSV = "data/reports/walk_forward_buy_2021_2025_details.csv"
DEFAULT_SUMMARY_CSV = "data/reports/walk_forward_buy_2021_2025_summary.csv"
DEFAULT_AGGREGATE_CSV = "data/reports/walk_forward_buy_2021_2025_aggregate.csv"
DEFAULT_OUTPUT_FILE = "data/reports/walk_forward_buy_2021_2025_dashboard.html"

PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>2021-2025 滚动样本外 Buy 预测看板</title>
  <script>{{ plotly_js }}</script>
  <style>
    :root {
      color-scheme: light;
      --bg: #f5f6f8;
      --surface: #ffffff;
      --line: #dfe3e8;
      --text: #171a1f;
      --muted: #68707c;
      --long: #b42318;
      --neutral: #475467;
      --bull: #fff0ee;
      --bear: #e9f7ef;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
    }
    header { background: var(--surface); border-bottom: 1px solid var(--line); }
    .header-inner, main { width: min(1240px, calc(100% - 32px)); margin: 0 auto; }
    .header-inner { padding: 24px 0 20px; }
    h1 { margin: 0; font-size: 28px; }
    h2 { margin: 0 0 14px; font-size: 18px; }
    .subtitle { margin-top: 8px; color: var(--muted); font-size: 14px; line-height: 1.6; }
    main { padding: 24px 0 40px; }
    .metrics {
      display: grid;
      grid-template-columns: repeat(5, minmax(130px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }
    .metric {
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-height: 104px;
    }
    .metric-label { color: var(--muted); font-size: 13px; }
    .metric-value { margin-top: 10px; font-size: 24px; font-weight: 700; }
    .metric-note { margin-top: 6px; color: var(--muted); font-size: 12px; }
    section { margin-top: 26px; }
    .chart {
      width: 100%;
      min-height: 420px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
    }
    .table-wrap {
      overflow: auto;
      max-height: 620px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
    }
    table { width: 100%; min-width: 1120px; border-collapse: collapse; background: var(--surface); }
    th, td { padding: 9px 11px; border-bottom: 1px solid var(--line); text-align: left; font-size: 13px; white-space: nowrap; }
    th { color: var(--muted); background: #f9fafb; font-weight: 600; position: sticky; top: 0; z-index: 1; }
    tr:last-child td { border-bottom: 0; }
    .tag { display: inline-block; padding: 3px 7px; border-radius: 4px; font-weight: 600; }
    .tag-long { color: var(--long); background: #fff0ee; }
    .tag-neutral { color: var(--neutral); background: #f2f4f7; }
    .tag-bull { color: #b42318; background: var(--bull); }
    .tag-bear { color: #067647; background: var(--bear); }
    .tag-neutral-regime { color: var(--neutral); background: #f2f4f7; }
    .note { color: var(--muted); font-size: 13px; line-height: 1.7; margin: 8px 0 0; }
    footer { margin-top: 28px; color: var(--muted); font-size: 12px; line-height: 1.7; }
    @media (max-width: 1000px) {
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
      .header-inner, main { width: min(100% - 20px, 1240px); }
      .metrics { grid-template-columns: 1fr; }
      h1 { font-size: 23px; }
    }
  </style>
</head>
<body>
<header>
  <div class="header-inner">
    <h1>2021-2025 滚动样本外 Buy 预测看板</h1>
    <div class="subtitle">
      逐年扩展窗口训练 · 7日 binary Buy · 因果牛市硬门控 · raw Buy评分门槛 {{ threshold }} · 生成时间 {{ generated_at }}
    </div>
  </div>
</header>
<main>
  <div class="metrics">
    <div class="metric">
      <div class="metric-label">预测区间</div>
      <div class="metric-value">{{ metrics.date_range }}</div>
      <div class="metric-note">{{ metrics.rows_count }} 个样本外交易日</div>
    </div>
    <div class="metric">
      <div class="metric-label">Long 信号数</div>
      <div class="metric-value">{{ metrics.long_count }}</div>
      <div class="metric-note">因果牛市且评分达标</div>
    </div>
    <div class="metric">
      <div class="metric-label">Long Precision</div>
      <div class="metric-value">{{ metrics.long_precision }}</div>
      <div class="metric-note">未来7日收益 &gt; {{ label_threshold_text }}</div>
    </div>
    <div class="metric">
      <div class="metric-label">Long 平均收益</div>
      <div class="metric-value">{{ metrics.long_avg_ret }}</div>
      <div class="metric-note">逐日 Long 信号口径</div>
    </div>
    <div class="metric">
      <div class="metric-label">独立 Long 平均收益</div>
      <div class="metric-value">{{ metrics.independent_avg_ret }}</div>
      <div class="metric-note">{{ metrics.independent_count }} 个不重叠信号</div>
    </div>
  </div>

  <section>
    <h2>滚动样本外 Buy 评分与硬门控 Long</h2>
    <div class="chart">{{ score_chart }}</div>
    <p class="note">图中每个点都是对应年度模型对该交易日的样本外评分；红色三角表示按模型 1.0 规则输出 Long。</p>
  </section>

  <section>
    <h2>指数走势与 Long 信号</h2>
    {% if price_chart %}<div class="chart">{{ price_chart }}</div>{% else %}<div class="chart">当前缺少可匹配的指数行情。</div>{% endif %}
  </section>

  <section>
    <h2>Long 信号未来 7 日实际收益</h2>
    <div class="chart">{{ return_chart }}</div>
  </section>

  <section>
    <h2>年度样本外表现</h2>
    <div class="chart">{{ yearly_chart }}</div>
  </section>

  <section>
    <h2>年度汇总表</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>年份</th><th>样本</th><th>信号数</th><th>覆盖率</th><th>Precision</th><th>实际上涨率</th><th>平均收益</th><th>天然Precision</th><th>天然平均收益</th><th>Precision提升</th><th>平均收益提升</th><th>训练结束</th></tr>
        </thead>
        <tbody>
        {% for row in yearly_rows %}
          <tr>
            <td>{{ row.test_year }}</td>
            <td>{{ row.sample_mode }}</td>
            <td>{{ row.signal_count }}</td>
            <td>{{ row.coverage }}</td>
            <td>{{ row.precision }}</td>
            <td>{{ row.actual_up_rate }}</td>
            <td>{{ row.avg_strategy_ret }}</td>
            <td>{{ row.natural_precision }}</td>
            <td>{{ row.natural_avg_strategy_ret }}</td>
            <td>{{ row.precision_lift }}</td>
            <td>{{ row.avg_strategy_ret_lift }}</td>
            <td>{{ row.train_end }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>滚动 Long 信号明细</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>日期</th><th>年份</th><th>因果状态</th><th>Buy评分</th><th>方向</th><th>实际7日收益</th><th>是否达标</th><th>训练结束</th></tr>
        </thead>
        <tbody>
        {% for row in long_rows %}
          <tr>
            <td>{{ row.trade_date }}</td>
            <td>{{ row.test_year }}</td>
            <td><span class="tag tag-{{ row.regime_class }}">{{ row.causal_regime }}</span></td>
            <td>{{ row.buy_proba }}</td>
            <td><span class="tag tag-long">Long</span></td>
            <td>{{ row.future_ret }}</td>
            <td>{{ row.outcome }}</td>
            <td>{{ row.train_end }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </section>

  <section>
    <h2>全部逐日滚动预测表</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr><th>日期</th><th>年份</th><th>因果状态</th><th>Buy评分</th><th>门槛</th><th>方向</th><th>实际7日收益</th><th>是否达标</th><th>训练区间</th></tr>
        </thead>
        <tbody>
        {% for row in detail_rows %}
          <tr>
            <td>{{ row.trade_date }}</td>
            <td>{{ row.test_year }}</td>
            <td><span class="tag tag-{{ row.regime_class }}">{{ row.causal_regime }}</span></td>
            <td>{{ row.buy_proba }}</td>
            <td>{{ row.threshold }}</td>
            <td><span class="tag tag-{{ row.direction }}">{{ row.direction_text }}</span></td>
            <td>{{ row.future_ret }}</td>
            <td>{{ row.outcome }}</td>
            <td>{{ row.train_start }} 至 {{ row.train_end }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </section>

  <footer>
    本报告展示滚动样本外研究结果，不是日常正式信号表，也不是正式回测。Buy评分是模型相对评分，不可解释为真实成功概率。未来7日收益为事后验证数据。
  </footer>
</main>
</body>
</html>
"""


def _pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "无"
    return f"{float(value) * 100:.{digits}f}%"


def _num(value: float | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "无"
    return f"{float(value):.{digits}f}"


def _date_axis(values: pd.Series) -> list[str]:
    return pd.to_datetime(values).dt.strftime("%Y-%m-%d").tolist()


def _chart_html(fig: go.Figure) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config=PLOT_CONFIG,
        default_width="100%",
        default_height="420px",
    )


def _apply_chart_layout(fig: go.Figure, yaxis_title: str) -> go.Figure:
    fig.update_layout(
        autosize=True,
        height=420,
        margin={"l": 54, "r": 24, "t": 22, "b": 44},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        dragmode="zoom",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        font={"family": "Microsoft YaHei, PingFang SC, Arial, sans-serif", "size": 12},
    )
    fig.update_xaxes(
        showgrid=False,
        rangeslider={"visible": True, "thickness": 0.08},
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1月", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6月", "step": "month", "stepmode": "backward"},
                {"count": 1, "label": "1年", "step": "year", "stepmode": "backward"},
                {"step": "all", "label": "全部"},
            ],
            "font": {"size": 11},
        },
    )
    fig.update_yaxes(title=yaxis_title, gridcolor="#e5e7eb", zerolinecolor=NEUTRAL_COLOR)
    return fig


def _non_overlapping(data: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if data.empty:
        return data
    selected = []
    next_allowed_pos_by_year: dict[int, int] = {}
    for idx, row in data.sort_values("trade_date").iterrows():
        year = int(row["test_year"])
        pos = int(row["year_pos"])
        next_allowed_pos = next_allowed_pos_by_year.get(year, -1)
        if pos >= next_allowed_pos:
            selected.append(idx)
            next_allowed_pos_by_year[year] = pos + horizon
    return data.loc[selected]


def _load_data(details_csv: str, summary_csv: str, aggregate_csv: str, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    details = pd.read_csv(project_path(details_csv))
    summary = pd.read_csv(project_path(summary_csv))
    aggregate = pd.read_csv(project_path(aggregate_csv))
    details["trade_date"] = pd.to_datetime(details["trade_date"])
    details["future_ret"] = pd.to_numeric(details["future_ret"], errors="coerce")
    details["buy_proba"] = pd.to_numeric(details["buy_proba"], errors="coerce")
    details["label_threshold"] = pd.to_numeric(details["label_threshold"], errors="coerce")
    details["test_year"] = details["test_year"].astype(int)
    details["year_pos"] = details.groupby("test_year").cumcount()
    details["causal_regime"] = "neutral"
    if "f_is_bull_trend" in details:
        details.loc[details["f_is_bull_trend"].fillna(0).astype(int) == 1, "causal_regime"] = "bull"
    if "f_is_bear_trend" in details:
        details.loc[details["f_is_bear_trend"].fillna(0).astype(int) == 1, "causal_regime"] = "bear"
    details["direction"] = "neutral"
    details.loc[(details["causal_regime"] == "bull") & (details["buy_proba"] >= threshold), "direction"] = "long"
    details["threshold"] = threshold
    details["is_success"] = details["future_ret"] > details["label_threshold"]
    return details, summary, aggregate


def _load_prices(details: pd.DataFrame) -> pd.DataFrame:
    cfg = get_config()
    prices = read_sql(
        "SELECT trade_date, close FROM market_index_daily "
        "WHERE index_code=:index_code AND trade_date BETWEEN :start_date AND :end_date "
        "ORDER BY trade_date",
        {
            "index_code": cfg["project"]["target_index"],
            "start_date": details["trade_date"].min().strftime("%Y-%m-%d"),
            "end_date": details["trade_date"].max().strftime("%Y-%m-%d"),
        },
    )
    if prices.empty:
        return prices
    prices["trade_date"] = pd.to_datetime(prices["trade_date"])
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    return prices


def _score_chart(details: pd.DataFrame, threshold: float) -> str:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_date_axis(details["trade_date"]),
            y=details["buy_proba"],
            mode="lines+markers",
            name="Rolling Buy score",
            line={"color": INFO_COLOR, "width": 1.8},
            marker={"size": 4},
            customdata=details[["test_year", "causal_regime", "future_ret"]].to_numpy(),
            hovertemplate=(
                "%{x}<br>year=%{customdata[0]}<br>regime=%{customdata[1]}"
                "<br>score=%{y:.4f}<br>future_ret=%{customdata[2]:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=_date_axis(details["trade_date"]),
            y=[threshold] * len(details),
            mode="lines",
            name="Threshold",
            line={"color": THRESHOLD_COLOR, "width": 2, "dash": "dash"},
            hovertemplate="%{x}<br>threshold=%{y:.2f}<extra></extra>",
        )
    )
    long_rows = details[details["direction"] == "long"]
    if not long_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=_date_axis(long_rows["trade_date"]),
                y=long_rows["buy_proba"],
                mode="markers",
                name="Hard-gated Long",
                marker={"color": UP_COLOR, "size": 10, "symbol": "triangle-up"},
                customdata=long_rows[["test_year", "future_ret"]].to_numpy(),
                hovertemplate="%{x}<br>year=%{customdata[0]}<br>score=%{y:.4f}<br>future_ret=%{customdata[1]:.2%}<extra></extra>",
            )
        )
    _apply_chart_layout(fig, "Score")
    fig.update_yaxes(range=[0, 1])
    return _chart_html(fig)


def _price_chart(details: pd.DataFrame, prices: pd.DataFrame) -> str | None:
    if details.empty or prices.empty:
        return None
    merged = prices.merge(
        details[["trade_date", "direction", "buy_proba", "test_year", "causal_regime"]],
        on="trade_date",
        how="left",
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_date_axis(merged["trade_date"]),
            y=merged["close"],
            mode="lines",
            name="CSI 1000 close",
            line={"color": "#344054", "width": 2},
            hovertemplate="%{x}<br>close=%{y:.2f}<extra></extra>",
        )
    )
    long_rows = merged[merged["direction"] == "long"].copy()
    if not long_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=_date_axis(long_rows["trade_date"]),
                y=long_rows["close"],
                mode="markers",
                name="Hard-gated Long",
                marker={"color": UP_COLOR, "size": 10, "symbol": "triangle-up"},
                customdata=long_rows[["test_year", "buy_proba", "causal_regime"]].to_numpy(),
                hovertemplate=(
                    "%{x}<br>year=%{customdata[0]}<br>score=%{customdata[1]:.4f}"
                    "<br>regime=%{customdata[2]}<br>close=%{y:.2f}<extra></extra>"
                ),
            )
        )
    _apply_chart_layout(fig, "Close")
    return _chart_html(fig)


def _return_chart(details: pd.DataFrame) -> str:
    long_rows = details[details["direction"] == "long"].copy()
    if long_rows.empty:
        fig = go.Figure()
        _apply_chart_layout(fig, "Future 7D return (%)")
        return _chart_html(fig)
    long_rows["future_ret_pct"] = long_rows["future_ret"] * 100
    colors = [
        UP_COLOR if value > 0 else DOWN_COLOR if value < 0 else NEUTRAL_COLOR
        for value in long_rows["future_ret"]
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=_date_axis(long_rows["trade_date"]),
            y=long_rows["future_ret_pct"],
            name="Long future 7D return",
            marker={"color": colors},
            customdata=long_rows[["test_year", "buy_proba", "causal_regime"]].to_numpy(),
            hovertemplate=(
                "%{x}<br>year=%{customdata[0]}<br>score=%{customdata[1]:.4f}"
                "<br>regime=%{customdata[2]}<br>future_ret=%{y:.2f}%<extra></extra>"
            ),
        )
    )
    label_threshold_pct = float(long_rows["label_threshold"].iloc[0]) * 100
    fig.add_hline(
        y=label_threshold_pct,
        line={"color": INFO_COLOR, "width": 2, "dash": "dash"},
        annotation_text=f"Target {label_threshold_pct:.1f}%",
        annotation_position="top left",
    )
    fig.add_hline(y=0, line={"color": NEUTRAL_COLOR, "width": 1})
    _apply_chart_layout(fig, "Future 7D return (%)")
    return _chart_html(fig)


def _yearly_chart(summary: pd.DataFrame) -> str:
    yearly = summary[
        (summary["regime"] == "all")
        & (summary["sample_mode"] == "non_overlapping")
    ].sort_values("test_year")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=yearly["test_year"].astype(str),
            y=yearly["precision"] * 100,
            name="Signal precision",
            marker={"color": UP_COLOR},
            hovertemplate="year=%{x}<br>precision=%{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=yearly["test_year"].astype(str),
            y=yearly["natural_precision"] * 100,
            mode="lines+markers",
            name="Natural precision",
            line={"color": INFO_COLOR, "width": 2},
            hovertemplate="year=%{x}<br>natural=%{y:.2f}%<extra></extra>",
        )
    )
    fig.update_layout(
        autosize=True,
        height=420,
        margin={"l": 54, "r": 24, "t": 22, "b": 44},
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        font={"family": "Microsoft YaHei, PingFang SC, Arial, sans-serif", "size": 12},
    )
    fig.update_xaxes(title="Year", showgrid=False)
    fig.update_yaxes(title="Precision (%)", gridcolor="#e5e7eb", zerolinecolor=NEUTRAL_COLOR)
    return _chart_html(fig)


def _metrics(details: pd.DataFrame, independent: pd.DataFrame) -> dict:
    long_rows = details[details["direction"] == "long"]
    if long_rows.empty:
        long_precision = None
        long_avg_ret = None
    else:
        long_precision = float(long_rows["is_success"].mean())
        long_avg_ret = float(long_rows["future_ret"].mean())
    independent_avg_ret = float(independent["future_ret"].mean()) if len(independent) else None
    return {
        "date_range": f"{details['trade_date'].min():%Y-%m-%d} 至 {details['trade_date'].max():%Y-%m-%d}",
        "rows_count": len(details),
        "long_count": len(long_rows),
        "long_precision": _pct(long_precision),
        "long_avg_ret": _pct(long_avg_ret),
        "independent_count": len(independent),
        "independent_avg_ret": _pct(independent_avg_ret),
    }


def _summary_rows(summary: pd.DataFrame) -> list[dict]:
    rows = summary[
        (summary["regime"] == "all")
        & (summary["sample_mode"].isin(["all_signals", "non_overlapping"]))
    ].sort_values(["test_year", "sample_mode"]).copy()
    result = []
    for item in rows.itertuples(index=False):
        result.append(
            {
                "test_year": int(item.test_year),
                "sample_mode": "全部信号" if item.sample_mode == "all_signals" else "不重叠信号",
                "signal_count": int(item.signal_count),
                "coverage": _pct(item.coverage),
                "precision": _pct(item.precision),
                "actual_up_rate": _pct(item.actual_up_rate),
                "avg_strategy_ret": _pct(item.avg_strategy_ret),
                "natural_precision": _pct(item.natural_precision),
                "natural_avg_strategy_ret": _pct(item.natural_avg_strategy_ret),
                "precision_lift": _pct(item.precision_lift),
                "avg_strategy_ret_lift": _pct(item.avg_strategy_ret_lift),
                "train_end": item.train_end,
            }
        )
    return result


def _detail_rows(details: pd.DataFrame) -> list[dict]:
    regime_text = {"bull": "牛市", "bear": "熊市", "neutral": "中性"}
    direction_text = {"long": "Long", "neutral": "Neutral"}
    rows = []
    for item in details.sort_values("trade_date").itertuples(index=False):
        rows.append(
            {
                "trade_date": item.trade_date.strftime("%Y-%m-%d"),
                "test_year": int(item.test_year),
                "causal_regime": regime_text.get(item.causal_regime, item.causal_regime),
                "regime_class": item.causal_regime if item.causal_regime in regime_text else "neutral-regime",
                "buy_proba": _num(item.buy_proba),
                "threshold": _num(item.threshold, 2),
                "direction": item.direction,
                "direction_text": direction_text.get(item.direction, item.direction),
                "future_ret": _pct(item.future_ret),
                "outcome": "达标" if item.is_success else "未达标",
                "train_start": item.train_start,
                "train_end": item.train_end,
            }
        )
    return rows


def generate_walk_forward_buy_report(
    details_csv: str = DEFAULT_DETAILS_CSV,
    summary_csv: str = DEFAULT_SUMMARY_CSV,
    aggregate_csv: str = DEFAULT_AGGREGATE_CSV,
    output_file: str = DEFAULT_OUTPUT_FILE,
    threshold: float = 0.60,
) -> Path:
    details, summary, _aggregate = _load_data(details_csv, summary_csv, aggregate_csv, threshold)
    prices = _load_prices(details)
    horizon = int(details["horizon"].iloc[0])
    independent = _non_overlapping(details[details["direction"] == "long"], horizon)
    label_threshold = float(details["label_threshold"].iloc[0])
    detail_rows = _detail_rows(details)
    long_rows = _detail_rows(details[details["direction"] == "long"])
    html = Template(HTML_TEMPLATE).render(
        plotly_js=get_plotlyjs(),
        generated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        threshold=f"{threshold:.2f}",
        label_threshold_text=_pct(label_threshold, 1),
        metrics=_metrics(details, independent),
        score_chart=_score_chart(details, threshold),
        price_chart=_price_chart(details, prices),
        return_chart=_return_chart(details),
        yearly_chart=_yearly_chart(summary),
        yearly_rows=_summary_rows(summary),
        long_rows=long_rows,
        detail_rows=detail_rows,
    )
    output_path = project_path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("generated walk-forward Buy report path=%s rows=%s", output_path, len(details))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate walk-forward Buy prediction dashboard.")
    parser.add_argument("--details-csv", default=DEFAULT_DETAILS_CSV)
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--aggregate-csv", default=DEFAULT_AGGREGATE_CSV)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--threshold", type=float, default=0.60)
    args = parser.parse_args()
    print(
        generate_walk_forward_buy_report(
            details_csv=args.details_csv,
            summary_csv=args.summary_csv,
            aggregate_csv=args.aggregate_csv,
            output_file=args.output_file,
            threshold=args.threshold,
        )
    )


if __name__ == "__main__":
    main()
