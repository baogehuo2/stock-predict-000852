from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Template
from plotly.offline.offline import get_plotlyjs

from src.common.config import get_config, project_path
from src.common.db import read_sql
from src.common.logger import get_logger


logger = get_logger(__name__)

UP_COLOR = "#b42318"
DOWN_COLOR = "#067647"
NEUTRAL_COLOR = "#98a2b3"
INFO_COLOR = "#175cd3"
THRESHOLD_COLOR = "#f79009"

HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>中证1000 Buy 信号看板</title>
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
      --accent: #175cd3;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "PingFang SC", Arial, sans-serif;
    }
    header { background: var(--surface); border-bottom: 1px solid var(--line); }
    .header-inner, main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; }
    .header-inner { padding: 24px 0 20px; }
    h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
    h2 { margin: 0 0 14px; font-size: 18px; letter-spacing: 0; }
    .subtitle { margin-top: 8px; color: var(--muted); font-size: 14px; }
    main { padding: 24px 0 40px; }
    .latest {
      display: grid;
      grid-template-columns: 1.3fr repeat(4, minmax(120px, 1fr));
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
    .direction-long { color: var(--long); }
    .direction-neutral { color: var(--neutral); }
    section { margin-top: 26px; }
    .chart {
      display: block;
      width: 100%;
      min-height: 420px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
    }
    .empty {
      padding: 30px;
      background: var(--surface);
      border: 1px solid var(--line);
      color: var(--muted);
    }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
    table { width: 100%; min-width: 920px; border-collapse: collapse; background: var(--surface); }
    th, td { padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; font-size: 13px; white-space: nowrap; }
    th { color: var(--muted); background: #f9fafb; font-weight: 600; }
    tr:last-child td { border-bottom: 0; }
    .tag { display: inline-block; padding: 3px 7px; border-radius: 4px; font-weight: 600; }
    .tag-long { color: var(--long); background: #fff0ee; }
    .tag-neutral { color: var(--neutral); background: #f2f4f7; }
    .tag-bull { color: #b42318; background: var(--bull); }
    .tag-bear { color: #067647; background: var(--bear); }
    .tag-neutral-regime { color: var(--neutral); background: #f2f4f7; }
    footer { margin-top: 28px; color: var(--muted); font-size: 12px; line-height: 1.7; }
    @media (max-width: 900px) {
      .latest { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .latest .metric:first-child { grid-column: 1 / -1; }
    }
    @media (max-width: 560px) {
      .header-inner, main { width: min(100% - 20px, 1180px); }
      .latest { grid-template-columns: 1fr; }
      .latest .metric:first-child { grid-column: auto; }
      h1 { font-size: 23px; }
    }
  </style>
</head>
<body>
<header>
  <div class="header-inner">
    <h1>中证1000 Buy 信号看板</h1>
    <div class="subtitle">{{ model_version }} · {{ feature_version }} · 更新时间 {{ generated_at }}</div>
  </div>
</header>
<main>
  <div class="latest">
    <div class="metric">
      <div class="metric-label">最新方向 · {{ latest.trade_date }}</div>
      <div class="metric-value direction-{{ latest.direction }}">{{ latest.direction_text }}</div>
      <div class="metric-note">因果状态：{{ latest.regime_text }} · 未来 {{ latest.horizon }} 个交易日</div>
    </div>
    <div class="metric">
      <div class="metric-label">Buy 评分</div>
      <div class="metric-value">{{ latest.buy_proba }}</div>
      <div class="metric-note">模型相对评分，不是真实概率</div>
    </div>
    <div class="metric">
      <div class="metric-label">信号门槛</div>
      <div class="metric-value">{{ latest.buy_threshold }}</div>
      <div class="metric-note">仅 bull 状态允许 Long</div>
    </div>
    <div class="metric">
      <div class="metric-label">已兑现 Long Precision</div>
      <div class="metric-value">{{ metrics.precision }}</div>
      <div class="metric-note">实际未来7日收益 &gt; 0.5%</div>
    </div>
    <div class="metric">
      <div class="metric-label">已兑现 Long 平均收益</div>
      <div class="metric-value">{{ metrics.avg_future_ret }}</div>
      <div class="metric-note">{{ metrics.realized_long_count }} 个已兑现 Long 信号</div>
    </div>
  </div>

  <section>
    <h2>Buy 评分与信号门槛</h2>
    {% if score_chart %}<div class="chart">{{ score_chart }}</div>{% else %}<div class="empty">当前信号记录不足，生成更多日期信号后会显示评分曲线。</div>{% endif %}
  </section>

  <section>
    <h2>指数走势与 Long 信号</h2>
    {% if price_chart %}<div class="chart">{{ price_chart }}</div>{% else %}<div class="empty">当前缺少可匹配的指数行情。</div>{% endif %}
  </section>

  <section>
    <h2>Long 信号兑现结果</h2>
    {% if return_chart %}<div class="chart">{{ return_chart }}</div>{% else %}<div class="empty">暂时没有已兑现的 Long 信号。满 7 个交易日并重建数据集后会自动回填。</div>{% endif %}
  </section>

  <section>
    <h2>最近信号</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>日期</th><th>状态</th><th>Buy评分</th><th>门槛</th><th>方向</th><th>实际7日收益</th><th>验证结果</th></tr></thead>
        <tbody>
        {% for row in recent_rows %}
          <tr>
            <td>{{ row.trade_date }}</td>
            <td><span class="tag tag-{{ row.regime_class }}">{{ row.regime_text }}</span></td>
            <td>{{ row.buy_proba }}</td>
            <td>{{ row.buy_threshold }}</td>
            <td><span class="tag tag-{{ row.direction }}">{{ row.direction_text }}</span></td>
            <td>{{ row.future_ret_7d }}</td>
            <td>{{ row.outcome }}</td>
          </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
  </section>

  <footer>
    Buy评分仅表示模型相对评分，不能解释为真实上涨概率。future_ret_7d 是事后实际收益；显示“等待验证”的记录尚未获得完整未来7个交易日行情。本页面只展示预测结果，不包含仓位、交易成本、净值或回撤，不属于正式回测。
  </footer>
</main>
</body>
</html>
"""


PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


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
                {"count": 3, "label": "3月", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6月", "step": "month", "stepmode": "backward"},
                {"step": "all", "label": "全部"},
            ],
            "font": {"size": 11},
        },
    )
    fig.update_yaxes(
        title=yaxis_title,
        gridcolor="#e5e7eb",
        zerolinecolor="#98a2b3",
    )
    return fig


def _chart_html(fig: go.Figure) -> str:
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config=PLOT_CONFIG,
        default_width="100%",
        default_height="420px",
    )


def _date_axis(values: pd.Series) -> list[str]:
    return pd.to_datetime(values).dt.strftime("%Y-%m-%d").tolist()


def _score_chart(signals: pd.DataFrame) -> str | None:
    if signals.empty:
        return None
    x_values = _date_axis(signals["trade_date"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=signals["buy_proba"],
            mode="lines+markers",
            name="Buy score",
            line={"color": INFO_COLOR, "width": 2},
            marker={"size": 6},
            hovertemplate="%{x|%Y-%m-%d}<br>Buy score=%{y:.4f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=signals["buy_threshold"],
            mode="lines",
            name="Threshold",
            line={"color": THRESHOLD_COLOR, "width": 2, "dash": "dash"},
            hovertemplate="%{x|%Y-%m-%d}<br>Threshold=%{y:.2f}<extra></extra>",
        )
    )
    long_rows = signals[signals["direction"] == "long"]
    if not long_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=_date_axis(long_rows["trade_date"]),
                y=long_rows["buy_proba"],
                mode="markers",
                name="Long",
                marker={"color": UP_COLOR, "size": 11, "symbol": "triangle-up"},
                hovertemplate="%{x|%Y-%m-%d}<br>Long score=%{y:.4f}<extra></extra>",
            )
        )
    _apply_chart_layout(fig, "Score")
    fig.update_yaxes(range=[0, 1])
    return _chart_html(fig)


def _price_chart(signals: pd.DataFrame, prices: pd.DataFrame) -> str | None:
    if signals.empty or prices.empty:
        return None
    merged = prices.merge(signals[["trade_date", "direction"]], on="trade_date", how="left")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_date_axis(merged["trade_date"]),
            y=merged["close"],
            mode="lines",
            name="CSI 1000 close",
            line={"color": "#344054", "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>Close=%{y:.2f}<extra></extra>",
        )
    )
    long_rows = merged[merged["direction"] == "long"]
    if not long_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=_date_axis(long_rows["trade_date"]),
                y=long_rows["close"],
                mode="markers",
                name="Long",
                marker={"color": UP_COLOR, "size": 11, "symbol": "triangle-up"},
                hovertemplate="%{x|%Y-%m-%d}<br>Long close=%{y:.2f}<extra></extra>",
            )
        )
    _apply_chart_layout(fig, "Close")
    return _chart_html(fig)


def _return_chart(signals: pd.DataFrame) -> str | None:
    realized = signals[(signals["direction"] == "long") & signals["future_ret_7d"].notna()].copy()
    if realized.empty:
        return None
    realized["future_ret_pct"] = realized["future_ret_7d"] * 100
    colors = [
        UP_COLOR if value > 0 else DOWN_COLOR if value < 0 else NEUTRAL_COLOR
        for value in realized["future_ret_7d"]
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=_date_axis(realized["trade_date"]),
            y=realized["future_ret_pct"],
            name="Future 7D return",
            marker={"color": colors},
            hovertemplate="%{x|%Y-%m-%d}<br>Future 7D return=%{y:.2f}%<extra></extra>",
        )
    )
    fig.add_hline(
        y=0.5,
        line={"color": INFO_COLOR, "width": 2, "dash": "dash"},
        annotation_text="Target 0.5%",
        annotation_position="top left",
    )
    fig.add_hline(y=0, line={"color": NEUTRAL_COLOR, "width": 1})
    _apply_chart_layout(fig, "Future 7D return (%)")
    return _chart_html(fig)


def _percent(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "等待验证"
    return f"{float(value) * 100:.{digits}f}%"


def _prepare_rows(signals: pd.DataFrame, limit: int = 30) -> list[dict]:
    regime_text = {"bull": "牛市", "bear": "熊市", "neutral": "中性"}
    direction_text = {"long": "做多", "neutral": "观望"}
    rows = []
    for item in signals.sort_values("trade_date", ascending=False).head(limit).itertuples(index=False):
        future_ret = item.future_ret_7d
        if pd.isna(future_ret):
            outcome = "等待验证"
        elif item.direction != "long":
            outcome = "非Long信号"
        elif float(future_ret) > 0.005:
            outcome = "达到目标"
        elif float(future_ret) > 0:
            outcome = "上涨但未达0.5%"
        else:
            outcome = "未命中"
        rows.append(
            {
                "trade_date": item.trade_date.strftime("%Y-%m-%d"),
                "regime_text": regime_text.get(item.causal_regime, item.causal_regime),
                "regime_class": item.causal_regime if item.causal_regime in regime_text else "neutral-regime",
                "buy_proba": f"{float(item.buy_proba):.4f}",
                "buy_threshold": f"{float(item.buy_threshold):.2f}",
                "direction": item.direction,
                "direction_text": direction_text.get(item.direction, item.direction),
                "future_ret_7d": _percent(future_ret),
                "outcome": outcome,
            }
        )
    return rows


def generate_buy_signal_report(
    model_version: str | None = None,
    output_file: str = "data/reports/buy_signal_v1_dashboard.html",
) -> Path:
    cfg = get_config()
    buy_cfg = cfg.get("binary_buy_model", {})
    model_version = model_version or str(buy_cfg.get("model_version", "buy-signal-v1.0"))
    signals = read_sql(
        "SELECT trade_date, model_version, feature_version, model_tag, horizon, causal_regime, "
        "buy_proba, buy_threshold, direction, signal_strength, pred_ret, future_ret_7d "
        "FROM buy_signal_daily WHERE model_version=:model_version ORDER BY trade_date",
        {"model_version": model_version},
    )
    if signals.empty:
        raise RuntimeError(
            f"No Buy signals found for {model_version}. Run generate_buy_signal first."
        )
    signals["trade_date"] = pd.to_datetime(signals["trade_date"])
    for column in ["buy_proba", "buy_threshold", "future_ret_7d"]:
        signals[column] = pd.to_numeric(signals[column], errors="coerce")
    start_date = signals["trade_date"].min().strftime("%Y-%m-%d")
    end_date = signals["trade_date"].max().strftime("%Y-%m-%d")
    prices = read_sql(
        "SELECT trade_date, close FROM market_index_daily "
        "WHERE index_code=:index_code AND trade_date BETWEEN :start_date AND :end_date "
        "ORDER BY trade_date",
        {
            "index_code": cfg["project"]["target_index"],
            "start_date": start_date,
            "end_date": end_date,
        },
    )
    if not prices.empty:
        prices["trade_date"] = pd.to_datetime(prices["trade_date"])
        prices["close"] = pd.to_numeric(prices["close"], errors="coerce")

    latest_row = signals.iloc[-1]
    realized_long = signals[(signals["direction"] == "long") & signals["future_ret_7d"].notna()]
    metrics = {
        "realized_long_count": int(len(realized_long)),
        "precision": _percent((realized_long["future_ret_7d"] > 0.005).mean()) if len(realized_long) else "等待验证",
        "avg_future_ret": _percent(realized_long["future_ret_7d"].mean()) if len(realized_long) else "等待验证",
    }
    latest = {
        "trade_date": latest_row["trade_date"].strftime("%Y-%m-%d"),
        "horizon": int(latest_row["horizon"]),
        "direction": latest_row["direction"],
        "direction_text": "做多" if latest_row["direction"] == "long" else "观望",
        "regime_text": {"bull": "牛市", "bear": "熊市", "neutral": "中性"}.get(
            latest_row["causal_regime"], latest_row["causal_regime"]
        ),
        "buy_proba": f"{float(latest_row['buy_proba']):.4f}",
        "buy_threshold": f"{float(latest_row['buy_threshold']):.2f}",
    }
    html = Template(HTML_TEMPLATE).render(
        plotly_js=get_plotlyjs(),
        model_version=model_version,
        feature_version=latest_row["feature_version"],
        generated_at=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        latest=latest,
        metrics=metrics,
        score_chart=_score_chart(signals),
        price_chart=_price_chart(signals, prices),
        return_chart=_return_chart(signals),
        recent_rows=_prepare_rows(signals),
    )
    output_path = project_path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    logger.info("generated Buy signal dashboard path=%s rows=%s", output_path, len(signals))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the Buy signal visualization dashboard.")
    parser.add_argument("--model-version")
    parser.add_argument("--output-file", default="data/reports/buy_signal_v1_dashboard.html")
    args = parser.parse_args()
    print(generate_buy_signal_report(args.model_version, args.output_file))


if __name__ == "__main__":
    main()
