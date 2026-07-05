from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import read_sql
from src.modeling.bottom_data import load_bottom_features
from src.modeling.train_manual_turning_models import (
    _fit_model,
    _positive_probability,
    load_manual_model_frame,
)
from src.modeling.turning_signal_postprocess import (
    assign_bottom_signal_layers,
    compress_top_signal_regions,
)


matplotlib.use("Agg")

BASE_COLUMNS = [
    "trade_date",
    "index_code",
    "is_manual_buy_window",
    "is_manual_sell_window",
    "manual_state",
    "label_mode",
    "future_ret_15d",
    "future_mfe_15d",
    "future_mae_15d",
    "quality_bottom_label",
    "continuation_risk_label",
]


def _metrics(signal: pd.DataFrame) -> dict:
    complete = signal[signal["future_ret_15d"].notna()]
    count = int(len(complete))
    return {
        "signal_count": count,
        "avg_future_ret_15d": float(complete["future_ret_15d"].mean()) if count else None,
        "median_future_ret_15d": float(complete["future_ret_15d"].median()) if count else None,
        "down_rate_15d": float((complete["future_ret_15d"] < 0).mean()) if count else None,
        "quality_bottom_rate": float(complete["quality_bottom_label"].mean()) if count else None,
        "continuation_risk_rate": float(complete["continuation_risk_label"].mean()) if count else None,
        "avg_future_mfe_15d": float(complete["future_mfe_15d"].mean()) if count else None,
        "avg_future_mae_15d": float(complete["future_mae_15d"].mean()) if count else None,
    }


def _summarize(signals: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["signal_side", "signal_layer", "signal_source"]
    for keys, part in signals.groupby(groups, sort=True):
        rows.append({**dict(zip(groups, keys)), "test_year": "all", **_metrics(part)})
    for keys, part in signals.groupby(groups + ["test_year"], sort=True):
        rows.append({**dict(zip(groups + ["test_year"], keys)), **_metrics(part)})
    return pd.DataFrame(rows)


def _summarize_regions(regions: pd.DataFrame, signal_source: str) -> pd.DataFrame:
    if regions.empty:
        return pd.DataFrame()
    complete = regions[regions["avg_future_ret_15d"].notna()]
    return pd.DataFrame(
        [
            {
                "signal_side": "top_region",
                "signal_layer": "top_region",
                "signal_source": signal_source,
                "test_year": "all",
                "signal_count": int(len(complete)),
                "avg_future_ret_15d": float(complete["avg_future_ret_15d"].mean())
                if len(complete)
                else None,
                "median_future_ret_15d": float(complete["avg_future_ret_15d"].median())
                if len(complete)
                else None,
                "down_rate_15d": float((complete["avg_future_ret_15d"] < 0).mean())
                if len(complete)
                else None,
                "quality_bottom_rate": None,
                "continuation_risk_rate": float(complete["continuation_risk_rate"].mean())
                if len(complete)
                else None,
                "avg_future_mfe_15d": None,
                "avg_future_mae_15d": None,
            }
        ]
    )


def _build_previous_bottom(
    data: pd.DataFrame,
    features: list[str],
    start_year: int,
    random_state: int,
    train_start: pd.Timestamp,
) -> pd.DataFrame:
    parts = []
    years = sorted(int(year) for year in data["trade_date"].dt.year.unique() if int(year) >= start_year)
    for year in years:
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        train = data[(data["trade_date"] >= train_start) & (data["trade_date"] < test_start)].copy()
        test = data[(data["trade_date"] >= test_start) & (data["trade_date"] <= test_end)].copy()
        if train.empty or test.empty:
            continue
        bottom_model, bottom_features = _fit_model(
            "lightgbm", train, features, "is_manual_buy_window", random_state
        )
        top_veto_model, top_veto_features = _fit_model(
            "lightgbm", train, features, "is_manual_sell_window", random_state
        )
        fold = test[BASE_COLUMNS].copy()
        fold["test_year"] = year
        fold["train_start"] = train["trade_date"].min().strftime("%Y-%m-%d")
        fold["train_end"] = train["trade_date"].max().strftime("%Y-%m-%d")
        fold["train_rows"] = int(len(train))
        fold["manual_weak_bottom_proba"] = _positive_probability(bottom_model, test, bottom_features)
        fold["manual_weak_top_proba"] = _positive_probability(top_veto_model, test, top_veto_features)
        fold["manual_weak_risk_proba"] = np.nan
        fold["signal_side"] = "bottom"
        fold["signal_source"] = "previous_walk_forward_lightgbm_no_risk_gate"
        parts.append(
            assign_bottom_signal_layers(
                fold,
                watch_threshold=0.50,
                action_threshold=0.70,
                top_veto_threshold=0.50,
            )
        )
    if not parts:
        return pd.DataFrame(columns=BASE_COLUMNS)
    return pd.concat(parts, ignore_index=True)


def _build_previous_top(
    data: pd.DataFrame,
    features: list[str],
    random_state: int,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    oot_start: pd.Timestamp,
) -> pd.DataFrame:
    train = data[(data["trade_date"] >= train_start) & (data["trade_date"] <= train_end)].copy()
    test = data[data["trade_date"] >= oot_start].copy()
    top_model, top_features = _fit_model("lightgbm", train, features, "is_manual_sell_window", random_state)
    bottom_veto_model, bottom_veto_features = _fit_model(
        "lightgbm", train, features, "is_manual_buy_window", random_state
    )
    prediction = test[BASE_COLUMNS].copy()
    prediction["test_year"] = prediction["trade_date"].dt.year.astype(int)
    prediction["train_start"] = train["trade_date"].min().strftime("%Y-%m-%d")
    prediction["train_end"] = train["trade_date"].max().strftime("%Y-%m-%d")
    prediction["train_rows"] = int(len(train))
    prediction["manual_weak_bottom_proba"] = _positive_probability(
        bottom_veto_model, test, bottom_veto_features
    )
    prediction["manual_weak_top_proba"] = _positive_probability(top_model, test, top_features)
    prediction["manual_weak_risk_proba"] = np.nan
    prediction["signal_side"] = "top"
    prediction["signal_source"] = "previous_anchored_lightgbm"
    result = prediction[
        (prediction["manual_weak_top_proba"] >= 0.60)
        & (prediction["manual_weak_bottom_proba"] < 0.30)
    ].copy()
    result["signal_layer"] = "top_action"
    return result


def _resolve_same_day_conflicts(signals: pd.DataFrame) -> pd.DataFrame:
    signals = signals.copy()
    signals["bottom_excess"] = signals["manual_weak_bottom_proba"] - 0.70
    signals["top_excess"] = signals["manual_weak_top_proba"] - 0.60
    signals["signal_strength"] = np.where(
        signals["signal_side"] == "bottom",
        signals["bottom_excess"],
        signals["top_excess"],
    )
    signals = signals.sort_values(
        ["trade_date", "index_code", "signal_strength"],
        ascending=[True, True, False],
    )
    return signals.drop_duplicates(["trade_date", "index_code"], keep="first").drop(
        columns=["bottom_excess", "top_excess", "signal_strength"]
    )


def _load_market() -> pd.DataFrame:
    market = read_sql(
        """
        SELECT trade_date, index_code, open, high, low, close
        FROM market_index_daily
        WHERE index_code='000852' AND trade_date >= '2022-01-01'
        ORDER BY trade_date
        """
    )
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["index_code"] = market["index_code"].astype(str).str.zfill(6)
    for column in ["open", "high", "low", "close"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    return market.dropna(subset=["close"]).sort_values("trade_date")


def _make_chart(market: pd.DataFrame, signals: pd.DataFrame, side: str, title: str, path: Path) -> None:
    side_signals = signals[signals["signal_side"] == side].sort_values("trade_date").copy()
    plot_signals = side_signals.merge(
        market[["trade_date", "index_code", "close"]],
        on=["trade_date", "index_code"],
        how="left",
    )
    fig, ax = plt.subplots(figsize=(20, 11))
    ax.plot(
        market["trade_date"],
        market["close"],
        color="#1f77b4",
        linewidth=1.9,
        label="000852 close",
        zorder=2,
    )
    layer_styles = {
        "bottom_watch": ("#9ecae1", 34),
        "bottom_candidate": ("#3182bd", 50),
        "bottom_action": ("#0057ff", 76),
        "top_action": ("#0057ff", 76),
    }
    if "signal_layer" not in plot_signals.columns:
        plot_signals["signal_layer"] = f"{side}_action"
    for layer, layer_data in plot_signals.groupby("signal_layer", sort=True):
        color, size = layer_styles.get(layer, ("#0057ff", 72))
        ax.scatter(
            layer_data["trade_date"],
            layer_data["close"],
            s=size,
            color=color,
            edgecolors="white",
            linewidths=1.0,
            label=f"{layer} ({len(layer_data)})",
            zorder=6,
        )
    ax.set_title(title, fontsize=19, pad=18)
    ax.set_xlabel("Trade date", fontsize=12)
    ax.set_ylabel("Close", fontsize=12)
    ax.grid(True, color="#d0d7de", linewidth=0.8, alpha=0.72)
    ax.legend(loc="best", frameon=True)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    y_min = market["close"].min()
    y_max = market["close"].max()
    pad = (y_max - y_min) * 0.10
    ax.set_ylim(y_min - pad, y_max + pad)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)


def build_previous_turning_signals_no_risk_gate(start_year: int = 2022) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = get_config("bottom_model.yaml")
    manual_cfg = cfg["manual_models"]
    random_state = int(cfg["model"]["random_state"])
    train_start = pd.Timestamp(str(manual_cfg["train_start"]))
    train_end = pd.Timestamp(str(manual_cfg["train_end"]))
    oot_start = pd.Timestamp(str(manual_cfg["oot_start"]))
    data = load_manual_model_frame()
    features = load_bottom_features(data)

    bottom = _build_previous_bottom(data, features, start_year, random_state, train_start)
    top = _build_previous_top(data, features, random_state, train_start, train_end, oot_start)
    signals = _resolve_same_day_conflicts(pd.concat([bottom, top], ignore_index=True))
    signals = signals.sort_values(["trade_date", "signal_side"])
    top_representatives, top_regions = compress_top_signal_regions(
        signals[signals["signal_side"] == "top"]
    )
    if not top_representatives.empty:
        top_representatives["signal_layer"] = "top_action"
    signals = pd.concat(
        [signals[signals["signal_side"] != "top"], top_representatives],
        ignore_index=True,
    ).sort_values(["trade_date", "signal_side"])
    signals["optimized_signal"] = np.where(signals["signal_side"] == "bottom", 1, -1)
    summary = pd.concat(
        [_summarize(signals), _summarize_regions(top_regions, "previous_anchored_lightgbm_region")],
        ignore_index=True,
    )

    report_dir = project_path("data/reports")
    chart_dir = report_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)
    signals.to_csv(
        report_dir / "manual_weak_previous_turning_signals_no_risk_gate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        report_dir / "manual_weak_previous_turning_summary_no_risk_gate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    top_regions.to_csv(
        report_dir / "manual_weak_previous_top_regions_no_risk_gate.csv",
        index=False,
        encoding="utf-8-sig",
    )
    market = _load_market()
    _make_chart(
        market,
        signals,
        "bottom",
        "000852 Previous Bottom Buy Signals, 2022-present",
        chart_dir / "previous_bottom_signals_2022_present_no_risk_gate.svg",
    )
    _make_chart(
        market,
        signals,
        "top",
        "000852 Previous Top Sell Signals, 2022-present",
        chart_dir / "previous_top_signals_2022_present_no_risk_gate.svg",
    )
    return signals, summary


def main() -> None:
    _, summary = build_previous_turning_signals_no_risk_gate()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
