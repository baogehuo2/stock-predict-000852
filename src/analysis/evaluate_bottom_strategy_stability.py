from __future__ import annotations

import argparse
import html
import json
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
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

SUMMARY_OUTPUT = "data/reports/bottom_strategy_stability_summary.csv"
YEARLY_OUTPUT = "data/reports/bottom_strategy_stability_yearly.csv"
TRADES_OUTPUT = "data/reports/bottom_strategy_stability_trades.csv"
HTML_OUTPUT = "data/reports/bottom_strategy_stability_report.html"
CHART_HTML_OUTPUT = "data/reports/bottom_strategy_three_way_trade_pairs.html"

CHART_CASES = [
    (
        "BP direct bottom + XGBoost top",
        "strict_xgboost_slm0p06_tp0p15_mh45",
        "BP bottom signal buys directly; XGBoost top signal and indicator/risk rules sell.",
    ),
    (
        "BP direct bottom + RF top",
        "strict_random_forest_slm0p06_tp0p15_mh45",
        "BP bottom signal buys directly; RandomForest top signal and indicator/risk rules sell.",
    ),
    (
        "BP watch confirm + RF top",
        "watch_random_forest_w0p35_d10_c2_slm0p06_tp0p15_mh45",
        "BP bottom probability enters a watch window; right-side indicator confirmation buys; RandomForest top signal and indicator/risk rules sell.",
    ),
]


@dataclass(frozen=True)
class StabilityCase:
    strategy_family: str
    top_model_kind: str
    watch_threshold: float | None
    watch_days: int | None
    confirm_min_score: int | None
    stop_loss: float
    take_profit: float
    max_holding_days: int
    exclude_start: str | None
    exclude_end: str | None
    fee_rate: float
    slippage_rate: float

    @property
    def case_id(self) -> str:
        if self.strategy_family == "strict_bp":
            return (
                f"strict_{self.top_model_kind}_sl{self.stop_loss:.2f}_tp{self.take_profit:.2f}"
                f"_mh{self.max_holding_days}"
            ).replace("-", "m").replace(".", "p")
        return (
            f"watch_{self.top_model_kind}_w{self.watch_threshold:.2f}_d{self.watch_days}"
            f"_c{self.confirm_min_score}_sl{self.stop_loss:.2f}_tp{self.take_profit:.2f}"
            f"_mh{self.max_holding_days}"
        ).replace("-", "m").replace(".", "p")


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


def _top_prediction_path(top_model_kind: str) -> str:
    normalized = top_model_kind.lower().strip().replace("-", "_")
    if normalized in {"xgb", "xboost", "xgboost"}:
        return XGBOOST_PREDICTION_PATH
    if normalized in {"rf", "randomforest", "random_forest"}:
        return RF_PREDICTION_PATH
    raise ValueError(f"Unsupported top model: {top_model_kind}")


def _top_model_name(top_model_kind: str) -> str:
    normalized = top_model_kind.lower().strip().replace("-", "_")
    if normalized in {"xgb", "xboost", "xgboost"}:
        return "xgboost"
    if normalized in {"rf", "randomforest", "random_forest"}:
        return "random_forest"
    raise ValueError(f"Unsupported top model: {top_model_kind}")


def _load_base_frame(start_date: str, top_model_kind: str) -> pd.DataFrame:
    target_index = _load_target_index()
    market = _load_market(target_index, start_date)
    bp = _load_prediction(BP_PREDICTION_PATH, "bp", start_date)
    top = _load_prediction(_top_prediction_path(top_model_kind), "xgb", start_date)
    data = market.merge(bp, on="trade_date", how="left").merge(top, on="trade_date", how="left")
    for column in ["bp_bottom_signal", "bp_top_signal", "xgb_bottom_signal", "xgb_top_signal"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int)
    for column in ["bp_bottom_proba", "bp_top_proba", "xgb_bottom_proba", "xgb_top_proba"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values("trade_date").reset_index(drop=True)


def _add_confirmation_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["ma5"] = pd.to_numeric(result["close"], errors="coerce").rolling(5).mean()
    result["rsi_turn_up"] = (result["rsi6"] > result["rsi6"].shift(1)).astype(int)
    result["kdj_turn_up"] = (
        (result["kdj_j"] > result["kdj_j"].shift(1))
        & (result["kdj_k"] > result["kdj_k"].shift(1))
    ).astype(int)
    result["kdj_golden_cross"] = (
        (result["kdj_k_minus_d"] > 0) & (result["kdj_k_minus_d"].shift(1) <= 0)
    ).astype(int)
    result["cci_turn_up"] = (result["cci14"] > result["cci14"].shift(1)).astype(int)
    result["close_reclaim_ma5"] = (
        (result["close"] > result["ma5"]) & (result["close"].shift(1) <= result["ma5"].shift(1))
    ).astype(int)
    result["right_confirm_score"] = result[
        ["rsi_turn_up", "kdj_turn_up", "kdj_golden_cross", "cci_turn_up", "close_reclaim_ma5"]
    ].sum(axis=1)
    return result


def _watch_entry_signal(
    data: pd.DataFrame,
    watch_threshold: float,
    watch_days: int,
    confirm_min_score: int,
    exclude_start: str | None,
    exclude_end: str | None,
) -> pd.Series:
    data = _add_confirmation_features(data)
    signal = pd.Series(0, index=data.index, dtype=int)
    watch_until = -1
    for idx, row in enumerate(data.itertuples(index=False)):
        trade_date = pd.Timestamp(row.trade_date)
        excluded = (
            bool(exclude_start and exclude_end)
            and pd.Timestamp(exclude_start) <= trade_date <= pd.Timestamp(exclude_end)
        )
        if excluded:
            continue

        watch_condition = (
            float(row.bp_bottom_proba) >= watch_threshold
            and float(row.bp_top_proba) < 0.5
            and int(row.xgb_top_signal) == 0
        )
        if watch_condition:
            watch_until = max(watch_until, idx + watch_days)

        in_watch = idx <= watch_until
        confirmed = int(row.right_confirm_score) >= confirm_min_score
        if in_watch and confirmed and int(row.xgb_top_signal) == 0:
            signal.iloc[idx] = 1
    return signal


def _strategy_params(case: StabilityCase) -> StrategyParams:
    return StrategyParams(
        first_entry_pct=0.50,
        add_entry_pct=0.25,
        max_position_pct=0.75,
        add_window_days=5,
        cooldown_days=3,
        stop_loss=case.stop_loss,
        take_profit=case.take_profit,
        max_holding_days=case.max_holding_days,
        top_reduce_pct=0.50,
        fee_rate=case.fee_rate,
        slippage_rate=case.slippage_rate,
        exclude_start=case.exclude_start,
        exclude_end=case.exclude_end,
    )


def _prepare_case_frame(base_frame: pd.DataFrame, case: StabilityCase) -> pd.DataFrame:
    data = base_frame.copy()
    if case.strategy_family == "watch_confirm":
        data["bp_bottom_signal"] = _watch_entry_signal(
            data=data,
            watch_threshold=float(case.watch_threshold),
            watch_days=int(case.watch_days),
            confirm_min_score=int(case.confirm_min_score),
            exclude_start=case.exclude_start,
            exclude_end=case.exclude_end,
        )
    elif case.strategy_family != "strict_bp":
        raise ValueError(f"Unsupported strategy family: {case.strategy_family}")
    return data


def _max_drawdown(series: pd.Series, initial_equity: float | None = None) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return np.nan
    if initial_equity is not None:
        values = pd.concat([pd.Series([initial_equity]), values], ignore_index=True)
    return float((values / values.cummax() - 1.0).min())


def _yearly_rows(case: StabilityCase, equity: pd.DataFrame, trades: pd.DataFrame) -> list[dict]:
    equity = equity.copy()
    trades = trades.copy()
    equity["trade_date"] = pd.to_datetime(equity["trade_date"])
    if not trades.empty:
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    rows = []
    previous_equity = 1.0
    for year, part in equity.groupby(equity["trade_date"].dt.year, sort=True):
        end_equity = float(part["equity"].iloc[-1])
        year_trades = trades[
            (trades["status"] == "closed")
            & (trades["exit_date"].dt.year == int(year))
        ].copy() if not trades.empty else pd.DataFrame()
        rows.append(
            {
                **asdict(case),
                "case_id": case.case_id,
                "year": int(year),
                "year_return": end_equity / previous_equity - 1.0 if previous_equity else np.nan,
                "year_max_drawdown": _max_drawdown(part["equity"], previous_equity),
                "year_closed_trades": int(len(year_trades)),
                "year_win_rate": float(year_trades["win"].mean()) if not year_trades.empty else np.nan,
                "year_avg_trade_return": float(year_trades["trade_return"].mean()) if not year_trades.empty else np.nan,
                "year_avg_position_ratio": float(part["position_ratio"].mean()),
            }
        )
        previous_equity = end_equity
    return rows


def _score_row(row: pd.Series, yearly: pd.DataFrame) -> dict:
    case_years = yearly[yearly["case_id"] == row["case_id"]].copy()
    positive_years = int((case_years["year_return"] > 0).sum())
    active_years = int(case_years["year"].nunique())
    worst_year_return = float(case_years["year_return"].min()) if not case_years.empty else np.nan
    median_year_return = float(case_years["year_return"].median()) if not case_years.empty else np.nan
    return_to_drawdown = (
        float(row["total_return"]) / abs(float(row["max_drawdown"]))
        if pd.notna(row["max_drawdown"]) and float(row["max_drawdown"]) < 0
        else np.nan
    )
    stability_score = (
        float(row["total_return"])
        + median_year_return
        + 0.02 * positive_years
        - abs(float(row["max_drawdown"]))
        - abs(worst_year_return)
    )
    return {
        "positive_years": positive_years,
        "active_years": active_years,
        "worst_year_return": worst_year_return,
        "median_year_return": median_year_return,
        "return_to_drawdown": return_to_drawdown,
        "stability_score": stability_score,
    }


def _default_cases() -> list[StabilityCase]:
    stop_losses = [-0.05, -0.06, -0.08]
    take_profits = [0.10, 0.15, 0.20]
    max_holding_days = [45]
    watch_thresholds = [0.35, 0.40, 0.45]
    watch_days_values = [5, 10, 15]
    confirm_scores = [2, 3]
    exclude_start = "2023-12-01"
    exclude_end = "2024-01-31"
    fee_rate = 0.0003
    slippage_rate = 0.0005

    cases: list[StabilityCase] = []
    for top_model_kind, stop_loss, take_profit, max_holding in product(
        ["xgboost", "random_forest"], stop_losses, take_profits, max_holding_days
    ):
        cases.append(
            StabilityCase(
                strategy_family="strict_bp",
                top_model_kind=top_model_kind,
                watch_threshold=None,
                watch_days=None,
                confirm_min_score=None,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_holding_days=max_holding,
                exclude_start=exclude_start,
                exclude_end=exclude_end,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
        )

    for watch_threshold, watch_days, confirm_min, stop_loss, take_profit, max_holding in product(
        watch_thresholds,
        watch_days_values,
        confirm_scores,
        stop_losses,
        take_profits,
        max_holding_days,
    ):
        cases.append(
            StabilityCase(
                strategy_family="watch_confirm",
                top_model_kind="random_forest",
                watch_threshold=watch_threshold,
                watch_days=watch_days,
                confirm_min_score=confirm_min,
                stop_loss=stop_loss,
                take_profit=take_profit,
                max_holding_days=max_holding,
                exclude_start=exclude_start,
                exclude_end=exclude_end,
                fee_rate=fee_rate,
                slippage_rate=slippage_rate,
            )
        )
    return cases


def evaluate_stability(start_date: str = "2022-01-01") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cases = _default_cases()
    base_frames = {
        "xgboost": _load_base_frame(start_date, "xgboost"),
        "random_forest": _load_base_frame(start_date, "random_forest"),
    }
    summary_parts: list[pd.DataFrame] = []
    trade_parts: list[pd.DataFrame] = []
    yearly_rows: list[dict] = []

    for case in cases:
        base = base_frames[_top_model_name(case.top_model_kind)]
        data = _prepare_case_frame(base, case)
        summary, trades, equity = run_backtest(data, _strategy_params(case))
        metadata = {
            **asdict(case),
            "case_id": case.case_id,
            "entry_signal_count": int(data["bp_bottom_signal"].sum()),
            "top_signal_count": int(data["xgb_top_signal"].sum()),
        }
        for key, value in metadata.items():
            summary[key] = value
            trades[key] = value
        summary_parts.append(summary)
        trade_parts.append(trades)
        yearly_rows.extend(_yearly_rows(case, equity, trades))

    summary_df = pd.concat(summary_parts, ignore_index=True)
    trades_df = pd.concat(trade_parts, ignore_index=True)
    yearly_df = pd.DataFrame(yearly_rows)
    score_rows = [_score_row(row, yearly_df) for _, row in summary_df.iterrows()]
    summary_df = pd.concat([summary_df, pd.DataFrame(score_rows)], axis=1)
    summary_df = summary_df.sort_values("stability_score", ascending=False).reset_index(drop=True)

    for frame, output in [
        (summary_df, SUMMARY_OUTPUT),
        (yearly_df, YEARLY_OUTPUT),
        (trades_df, TRADES_OUTPUT),
    ]:
        path = project_path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return summary_df, yearly_df, trades_df


def _summary_table(summary: pd.DataFrame, limit: int = 30) -> str:
    columns = [
        "strategy_family",
        "top_model_kind",
        "watch_threshold",
        "watch_days",
        "confirm_min_score",
        "stop_loss",
        "take_profit",
        "total_return",
        "max_drawdown",
        "closed_trades",
        "win_rate",
        "positive_years",
        "active_years",
        "worst_year_return",
        "return_to_drawdown",
        "stability_score",
    ]
    headers = [
        "family",
        "top",
        "watch",
        "days",
        "confirm",
        "stop",
        "take",
        "return",
        "max_dd",
        "trades",
        "win",
        "pos_years",
        "worst_year",
        "ret/dd",
        "score",
    ]
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    body = []
    for row in summary[columns].head(limit).itertuples(index=False):
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row.strategy_family))}</td>"
            f"<td>{html.escape(str(row.top_model_kind))}</td>"
            f"<td>{_fmt(row.watch_threshold, 2)}</td>"
            f"<td>{'' if pd.isna(row.watch_days) else int(row.watch_days)}</td>"
            f"<td>{'' if pd.isna(row.confirm_min_score) else int(row.confirm_min_score)}</td>"
            f"<td>{_pct(row.stop_loss)}</td>"
            f"<td>{_pct(row.take_profit)}</td>"
            f"<td>{_pct(row.total_return)}</td>"
            f"<td>{_pct(row.max_drawdown)}</td>"
            f"<td>{int(row.closed_trades)}</td>"
            f"<td>{_pct(row.win_rate)}</td>"
            f"<td>{int(row.positive_years)}/{int(row.active_years)}</td>"
            f"<td>{_pct(row.worst_year_return)}</td>"
            f"<td>{_fmt(row.return_to_drawdown, 2)}</td>"
            f"<td>{_fmt(row.stability_score, 4)}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _yearly_table(yearly: pd.DataFrame, best_case_id: str) -> str:
    part = yearly[yearly["case_id"] == best_case_id].sort_values("year")
    headers = ["year", "return", "max_dd", "trades", "win", "avg_trade", "avg_position"]
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    body = []
    for row in part.itertuples(index=False):
        body.append(
            "<tr>"
            f"<td>{int(row.year)}</td>"
            f"<td>{_pct(row.year_return)}</td>"
            f"<td>{_pct(row.year_max_drawdown)}</td>"
            f"<td>{int(row.year_closed_trades)}</td>"
            f"<td>{_pct(row.year_win_rate)}</td>"
            f"<td>{_pct(row.year_avg_trade_return)}</td>"
            f"<td>{_pct(row.year_avg_position_ratio)}</td>"
            "</tr>"
        )
    return f"<table><thead><tr>{header_html}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _heatmap_payload(summary: pd.DataFrame, metric: str, confirm_min_score: int = 2) -> dict:
    watch = summary[
        (summary["strategy_family"] == "watch_confirm")
        & (summary["top_model_kind"] == "random_forest")
        & (summary["confirm_min_score"] == confirm_min_score)
    ].copy()
    grouped = watch.groupby(["watch_threshold", "watch_days"], dropna=False)[metric].median().reset_index()
    thresholds = sorted(grouped["watch_threshold"].dropna().unique().tolist())
    days = sorted(grouped["watch_days"].dropna().unique().astype(int).tolist())
    value_by_key = {
        (float(row.watch_threshold), int(row.watch_days)): float(getattr(row, metric))
        for row in grouped.itertuples(index=False)
        if pd.notna(getattr(row, metric))
    }
    data = []
    values = []
    for y_idx, threshold in enumerate(thresholds):
        for x_idx, day in enumerate(days):
            value = value_by_key.get((float(threshold), int(day)))
            if value is None:
                continue
            values.append(value)
            data.append([x_idx, y_idx, round(value, 6), f"{value:.2%}"])
    return {
        "x": [str(day) for day in days],
        "y": [f"{threshold:.2f}" for threshold in thresholds],
        "data": data,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
        "metric": metric,
    }


def _render_html(summary: pd.DataFrame, yearly: pd.DataFrame) -> str:
    best = summary.iloc[0]
    best_case_id = str(best["case_id"])
    payload = {
        "return": _heatmap_payload(summary, "total_return", confirm_min_score=2),
        "drawdown": _heatmap_payload(summary, "max_drawdown", confirm_min_score=2),
        "score": _heatmap_payload(summary, "stability_score", confirm_min_score=2),
    }
    payload_json = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bottom Strategy Stability Report</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; background: #f6f7f9; color: #111827; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0 0 10px; font-size: 20px; }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    .cards {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; margin: 16px 0; }}
    .card, .panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; }}
    .panel {{ margin: 16px 0; }}
    .value {{ font-size: 19px; font-weight: 700; margin-top: 4px; }}
    .heatmaps {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .heatmap {{ height: 360px; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; }}
    .table-wrap {{ max-height: 560px; overflow: auto; border-radius: 10px; }}
  </style>
</head>
<body>
<main>
  <h1>Bottom Strategy Stability Report</h1>
  <div class="muted">Primary universe: CSI1000. Evaluation uses cost and excludes 2023-12-01 to 2024-01-31. Watch strategy uses BP bottom probability, right-side confirmation, and RF top exit.</div>
  <section class="cards">
    <div class="card"><div class="muted">best family</div><div class="value">{html.escape(str(best["strategy_family"]))}</div></div>
    <div class="card"><div class="muted">watch threshold</div><div class="value">{_fmt(best["watch_threshold"], 2)}</div></div>
    <div class="card"><div class="muted">watch days</div><div class="value">{"" if pd.isna(best["watch_days"]) else int(best["watch_days"])}</div></div>
    <div class="card"><div class="muted">total return</div><div class="value">{_pct(best["total_return"])}</div></div>
    <div class="card"><div class="muted">max drawdown</div><div class="value">{_pct(best["max_drawdown"])}</div></div>
    <div class="card"><div class="muted">positive years</div><div class="value">{int(best["positive_years"])}/{int(best["active_years"])}</div></div>
  </section>
  <section class="panel">
    <h2>Watch strategy heatmaps, confirm_min_score=2, median over stop/take grid</h2>
    <div class="heatmaps">
      <div id="heat_return" class="heatmap"></div>
      <div id="heat_drawdown" class="heatmap"></div>
      <div id="heat_score" class="heatmap"></div>
    </div>
  </section>
  <section class="panel">
    <h2>Top 30 parameter sets</h2>
    <div class="table-wrap">{_summary_table(summary, 30)}</div>
  </section>
  <section class="panel">
    <h2>Best case yearly breakdown</h2>
    <div class="table-wrap">{_yearly_table(yearly, best_case_id)}</div>
  </section>
</main>
<script>
(() => {{
  const payload = {payload_json};
  function renderHeatmap(domId, title, item) {{
    const chart = echarts.init(document.getElementById(domId));
    const min = item.min;
    const max = item.max;
    chart.setOption({{
      title: {{ text: title, left: "center", textStyle: {{ fontSize: 14 }} }},
      tooltip: {{
        formatter: p => `watch_days=${{item.x[p.data[0]]}}<br>threshold=${{item.y[p.data[1]]}}<br>${{title}}=${{p.data[3]}}`
      }},
      grid: {{ left: 64, right: 24, top: 48, bottom: 44 }},
      xAxis: {{ type: "category", data: item.x, name: "watch days" }},
      yAxis: {{ type: "category", data: item.y, name: "threshold" }},
      visualMap: {{ min, max, calculable: true, orient: "horizontal", left: "center", bottom: 0 }},
      series: [{{
        type: "heatmap",
        data: item.data,
        label: {{ show: true, formatter: p => p.data[3] }},
        emphasis: {{ itemStyle: {{ shadowBlur: 10, shadowColor: "rgba(0,0,0,0.25)" }} }}
      }}]
    }});
    window.addEventListener("resize", () => chart.resize());
  }}
  renderHeatmap("heat_return", "median total return", payload.return);
  renderHeatmap("heat_drawdown", "median max drawdown", payload.drawdown);
  renderHeatmap("heat_score", "median stability score", payload.score);
}})();
</script>
</body>
</html>
"""


def build_html(summary: pd.DataFrame, yearly: pd.DataFrame, output: str = HTML_OUTPUT) -> Path:
    output_path = project_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_html(summary, yearly), encoding="utf-8")
    return output_path


def _market_chart_payload(market: pd.DataFrame) -> dict:
    market = market.sort_values("trade_date").copy()
    dates = [pd.Timestamp(value).strftime("%Y-%m-%d") for value in market["trade_date"]]
    return {
        "dates": dates,
        "candles": [
            [_round(row.open), _round(row.close), _round(row.low), _round(row.high)]
            for row in market.itertuples(index=False)
        ],
        "closes": [_round(row.close) for row in market.itertuples(index=False)],
    }


def _excluded_index_range(market: pd.DataFrame, exclude_start: str | None, exclude_end: str | None) -> tuple[int | None, int | None]:
    if not exclude_start or not exclude_end:
        return None, None
    dates = [pd.Timestamp(value) for value in market.sort_values("trade_date")["trade_date"]]
    start_ts = pd.Timestamp(exclude_start)
    end_ts = pd.Timestamp(exclude_end)
    start_candidates = [idx for idx, value in enumerate(dates) if value >= start_ts]
    end_candidates = [idx for idx, value in enumerate(dates) if value <= end_ts]
    if not start_candidates or not end_candidates:
        return None, None
    return start_candidates[0], end_candidates[-1]


def _case_row(summary: pd.DataFrame, case_id: str) -> pd.Series:
    part = summary[summary["case_id"] == case_id].copy()
    if part.empty:
        raise ValueError(f"Case not found in stability summary: {case_id}")
    return part.iloc[0]


def _case_trades(trades: pd.DataFrame, case_id: str) -> pd.DataFrame:
    part = trades[trades["case_id"] == case_id].copy()
    if part.empty:
        return part
    part["entry_date"] = pd.to_datetime(part["entry_date"])
    part["exit_date"] = pd.to_datetime(part["exit_date"])
    for column in ["entry_price", "exit_price", "trade_return", "sell_fraction"]:
        part[column] = pd.to_numeric(part[column], errors="coerce")
    return part.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)


def _trade_chart_payload(
    market: pd.DataFrame,
    trades: pd.DataFrame,
    case_id: str,
    exclude_start: str | None,
    exclude_end: str | None,
) -> dict:
    payload = _market_chart_payload(market)
    date_to_index = {trade_date: index for index, trade_date in enumerate(payload["dates"])}
    exclude_start_index, exclude_end_index = _excluded_index_range(market, exclude_start, exclude_end)
    payload.update(
        {
            "buyPoints": [],
            "sellProfitPoints": [],
            "sellLossPoints": [],
            "profitLines": [],
            "lossLines": [],
            "excludeStartIndex": exclude_start_index,
            "excludeEndIndex": exclude_end_index,
        }
    )

    case_trades = _case_trades(trades, case_id)
    for idx, row in enumerate(case_trades.itertuples(index=False), start=1):
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


def _trade_table(trades: pd.DataFrame, case_id: str) -> str:
    case_trades = _case_trades(trades, case_id)
    headers = ["#", "entry", "exit", "return", "sell", "status", "reason"]
    header_html = "".join(f"<th>{header}</th>" for header in headers)
    body = []
    for idx, row in enumerate(case_trades.itertuples(index=False), start=1):
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


def _summary_cards(row: pd.Series) -> str:
    cards = [
        ("return", _pct(row["total_return"])),
        ("max drawdown", _pct(row["max_drawdown"])),
        ("win rate", _pct(row["win_rate"])),
        ("closed trades", str(int(row["closed_trades"]))),
        ("entry signals", str(int(row["entry_signal_count"]))),
        ("top signals", str(int(row["top_signal_count"]))),
    ]
    return "".join(
        f'<div class="card"><div class="muted">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )


def _render_trade_chart_html(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    chart_payloads: dict[str, dict],
) -> str:
    payload_json = json.dumps(chart_payloads, ensure_ascii=False, allow_nan=False)
    sections = []
    for index, (title, case_id, description) in enumerate(CHART_CASES, start=1):
        row = _case_row(summary, case_id)
        sections.append(
            f"""
  <section class="panel">
    <h2>{index}. {html.escape(title)}</h2>
    <div class="muted">{html.escape(description)} case_id={html.escape(case_id)}</div>
    <div class="cards">{_summary_cards(row)}</div>
    <div class="legend">
      <span><span class="dot" style="background:#2563eb"></span>buy B#</span>
      <span><span class="dot" style="background:#dc2626"></span>profit sell S#</span>
      <span><span class="dot" style="background:#16a34a"></span>loss sell S#</span>
      <span><span class="dot" style="background:#9ca3af"></span>excluded period</span>
    </div>
    <div id="chart_{index}" class="chart"></div>
    <h3>Trade pairs</h3>
    <div class="table-wrap">{_trade_table(trades, case_id)}</div>
  </section>
"""
        )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bottom Strategy Three Way Trade Pairs</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.1/dist/echarts.min.js"></script>
  <style>
    body {{ font-family: "Microsoft YaHei", Arial, sans-serif; margin: 0; color: #111827; background: #f6f7f9; }}
    main {{ max-width: 1540px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 26px; }}
    h2 {{ margin: 0 0 8px; font-size: 20px; }}
    h3 {{ margin: 16px 0 10px; font-size: 16px; }}
    .muted {{ color: #6b7280; font-size: 13px; }}
    .panel {{ background: #fff; border: 1px solid #e5e7eb; border-radius: 10px; margin: 18px 0; padding: 14px; }}
    .cards {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 10px; margin: 14px 0; }}
    .card {{ background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 12px; }}
    .value {{ font-size: 18px; font-weight: 700; margin-top: 3px; }}
    .chart {{ width: 100%; height: 820px; }}
    .legend {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0 10px; font-size: 13px; color: #374151; }}
    .dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; vertical-align: middle; }}
    table {{ width: 100%; border-collapse: collapse; background: #fff; border: 1px solid #e5e7eb; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #f3f4f6; position: sticky; top: 0; z-index: 1; }}
    tr.win td:nth-child(4) {{ color: #dc2626; font-weight: 700; }}
    tr.loss td:nth-child(4) {{ color: #16a34a; font-weight: 700; }}
    .table-wrap {{ max-height: 360px; overflow: auto; border-radius: 10px; }}
  </style>
</head>
<body>
<main>
  <h1>Bottom Strategy Three Way Trade Pairs</h1>
  <div class="muted">B# is the buy point and S# is the matching sell point. The same number means one trade pair. Mouse wheel zoom, drag pan, and the bottom slider are enabled.</div>
  {"".join(sections)}
</main>
<script>
(() => {{
  const allCharts = {payload_json};
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
        {{ left: 64, right: 42, top: 54, height: 640 }},
        {{ left: 64, right: 42, top: 722, height: 52 }},
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
        {{ type: "slider", xAxisIndex: [0, 1], top: 792, height: 24, start: 0, end: 100, minSpan: 8, filterMode: "none" }},
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

  renderChart("chart_1", allCharts["chart_1"]);
  renderChart("chart_2", allCharts["chart_2"]);
  renderChart("chart_3", allCharts["chart_3"]);
}})();
</script>
</body>
</html>
"""


def build_trade_chart_html(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    start_date: str,
    output: str = CHART_HTML_OUTPUT,
) -> Path:
    target_index = _load_target_index()
    market = _load_market(target_index, start_date)
    chart_payloads = {}
    for index, (_, case_id, _) in enumerate(CHART_CASES, start=1):
        row = _case_row(summary, case_id)
        chart_payloads[f"chart_{index}"] = _trade_chart_payload(
            market=market,
            trades=trades,
            case_id=case_id,
            exclude_start=str(row["exclude_start"]) if pd.notna(row["exclude_start"]) else None,
            exclude_end=str(row["exclude_end"]) if pd.notna(row["exclude_end"]) else None,
        )
    output_path = project_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_trade_chart_html(summary, trades, chart_payloads), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate bottom strategy parameter stability.")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--output", default=HTML_OUTPUT)
    parser.add_argument("--chart-output", default=CHART_HTML_OUTPUT)
    args = parser.parse_args()
    summary, yearly, _ = evaluate_stability(args.start_date)
    output_path = build_html(summary, yearly, args.output)
    chart_output_path = build_trade_chart_html(
        summary=summary,
        trades=pd.read_csv(project_path(TRADES_OUTPUT)),
        start_date=args.start_date,
        output=args.chart_output,
    )
    pd.set_option("display.max_columns", None)
    print("[top 20]")
    print(summary.head(20).to_string(index=False))
    print(f"\n[html]\n{output_path}")
    print(f"\n[chart html]\n{chart_output_path}")


if __name__ == "__main__":
    main()
