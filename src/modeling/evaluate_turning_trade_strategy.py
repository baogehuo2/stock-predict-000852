from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.config import load_yaml, project_path
from src.common.db import read_sql


PREDICTION_TEMPLATE = "data/reports/manual_weak_turning_walk_forward_predictions_{model_kind}_wf.csv"
SUMMARY_OUTPUT = "data/reports/manual_weak_turning_trade_strategy_summary.csv"
YEARLY_OUTPUT = "data/reports/manual_weak_turning_trade_strategy_yearly.csv"
TRADE_OUTPUT = "data/reports/manual_weak_turning_trade_strategy_trades.csv"
EQUITY_OUTPUT = "data/reports/manual_weak_turning_trade_strategy_equity.csv"

DEFAULT_MODELS = ["lightgbm", "logistic", "xgboost", "bp"]
DEFAULT_STRATEGIES = ["scale_in", "risk_exit", "compressed_regions"]

TRADE_COLUMNS = [
    "model_kind",
    "strategy",
    "entry_date",
    "exit_date",
    "entry_price",
    "exit_price",
    "holding_days",
    "trade_return",
    "win",
    "entry_count",
    "invested_cash",
    "entry_bottom_proba",
    "entry_top_proba",
    "exit_bottom_proba",
    "exit_top_proba",
    "entry_is_manual_buy_window",
    "entry_manual_buy_window_rate",
    "exit_is_manual_sell_window",
    "exit_reason",
    "status",
]


@dataclass(frozen=True)
class StrategyParams:
    scale_in_max_entries: int = 3
    stop_loss: float = -0.08
    take_profit: float = 0.15
    max_holding_days: int = 60
    region_gap_days: int = 10


@dataclass(frozen=True)
class Lot:
    entry_date: pd.Timestamp
    entry_price: float
    cash_value: float
    shares: float
    bottom_proba: float
    top_proba: float
    is_manual_buy_window: int


def _normalize_model_kind(model_kind: str) -> str:
    normalized = model_kind.lower().strip()
    if normalized == "xboost":
        return "xgboost"
    if normalized in {"bpnn", "bp_neural_network", "neural_network", "mlp"}:
        return "bp"
    return normalized


def _normalize_strategy(strategy: str) -> str:
    normalized = strategy.lower().strip().replace("-", "_")
    aliases = {
        "base": "baseline",
        "single": "baseline",
        "scale": "scale_in",
        "add": "scale_in",
        "risk": "risk_exit",
        "stop": "risk_exit",
        "compress": "compressed_regions",
        "compressed": "compressed_regions",
    }
    return aliases.get(normalized, normalized)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int).astype(bool)


def _load_target_index() -> str:
    cfg = load_yaml(project_path("config", "bottom_model.yaml"))
    return str(cfg.get("model", {}).get("target_index", "000852"))


def _load_market_close(target_index: str) -> pd.DataFrame:
    market = read_sql(
        """
        SELECT trade_date, index_code, close
        FROM market_index_daily
        WHERE index_code = :target_index
        ORDER BY trade_date
        """,
        {"target_index": target_index},
    )
    if market.empty:
        raise RuntimeError(f"No market_index_daily rows found for {target_index}.")
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    market["close"] = pd.to_numeric(market["close"], errors="coerce")
    return market.dropna(subset=["trade_date", "close"]).sort_values("trade_date").reset_index(drop=True)


def _load_predictions(model_kind: str, market: pd.DataFrame) -> pd.DataFrame:
    model_kind = _normalize_model_kind(model_kind)
    path = project_path(PREDICTION_TEMPLATE.format(model_kind=model_kind))
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")

    prediction = pd.read_csv(path, parse_dates=["trade_date"])
    required = {
        "trade_date",
        "manual_weak_bottom_proba",
        "manual_weak_top_proba",
        "weak_combined_bottom_signal",
        "weak_combined_top_signal",
        "is_manual_buy_window",
        "is_manual_sell_window",
    }
    missing = sorted(required - set(prediction.columns))
    if missing:
        raise RuntimeError(f"{path} is missing columns: {missing}")

    prediction = prediction.merge(market[["trade_date", "close"]], on="trade_date", how="left")
    prediction["close"] = pd.to_numeric(prediction["close"], errors="coerce")
    prediction = prediction.dropna(subset=["close"]).sort_values("trade_date").reset_index(drop=True)
    prediction["bottom_signal"] = _bool_series(prediction["weak_combined_bottom_signal"])
    prediction["top_signal"] = _bool_series(prediction["weak_combined_top_signal"])
    prediction["is_manual_buy_window"] = pd.to_numeric(
        prediction["is_manual_buy_window"], errors="coerce"
    ).fillna(0).astype(int)
    prediction["is_manual_sell_window"] = pd.to_numeric(
        prediction["is_manual_sell_window"], errors="coerce"
    ).fillna(0).astype(int)
    prediction["model_kind"] = model_kind
    return prediction


def _compress_signals(
    prediction: pd.DataFrame,
    signal_col: str,
    proba_col: str,
    max_gap_days: int,
) -> pd.Series:
    compressed = pd.Series(False, index=prediction.index)
    signal = prediction[prediction[signal_col]].sort_values("trade_date").copy()
    if signal.empty:
        return compressed

    gaps = signal["trade_date"].diff().dt.days.fillna(max_gap_days + 1)
    signal["region_id"] = (gaps > max_gap_days).cumsum()
    representative_indexes = signal.groupby("region_id", sort=True).head(1).index.tolist()
    compressed.loc[representative_indexes] = True
    return compressed


def _prepare_strategy_prediction(
    prediction: pd.DataFrame,
    strategy: str,
    params: StrategyParams,
) -> pd.DataFrame:
    result = prediction.copy()
    if strategy == "compressed_regions":
        result["bottom_signal"] = _compress_signals(
            result,
            "bottom_signal",
            "manual_weak_bottom_proba",
            params.region_gap_days,
        )
        result["top_signal"] = _compress_signals(
            result,
            "top_signal",
            "manual_weak_top_proba",
            params.region_gap_days,
        )
    return result


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _position_values(lots: list[Lot]) -> tuple[float, float, float]:
    invested_cash = float(sum(lot.cash_value for lot in lots))
    shares = float(sum(lot.shares for lot in lots))
    avg_entry_price = invested_cash / shares if shares else np.nan
    return invested_cash, shares, avg_entry_price


def _make_trade_row(
    model_kind: str,
    strategy: str,
    lots: list[Lot],
    exit_date: pd.Timestamp,
    exit_price: float,
    exit_bottom_proba: float,
    exit_top_proba: float,
    exit_is_manual_sell_window: int,
    exit_reason: str,
    status: str,
) -> dict:
    invested_cash, shares, avg_entry_price = _position_values(lots)
    entry_date = min(lot.entry_date for lot in lots)
    trade_return = exit_price / avg_entry_price - 1.0 if shares else np.nan
    manual_buy_values = [lot.is_manual_buy_window for lot in lots]
    first_lot = sorted(lots, key=lambda lot: lot.entry_date)[0]
    return {
        "model_kind": model_kind,
        "strategy": strategy,
        "entry_date": entry_date.date(),
        "exit_date": exit_date.date(),
        "entry_price": avg_entry_price,
        "exit_price": exit_price,
        "holding_days": int((exit_date - entry_date).days),
        "trade_return": trade_return,
        "win": int(trade_return > 0) if np.isfinite(trade_return) else 0,
        "entry_count": int(len(lots)),
        "invested_cash": invested_cash,
        "entry_bottom_proba": float(first_lot.bottom_proba),
        "entry_top_proba": float(first_lot.top_proba),
        "exit_bottom_proba": exit_bottom_proba,
        "exit_top_proba": exit_top_proba,
        "entry_is_manual_buy_window": int(max(manual_buy_values)) if manual_buy_values else 0,
        "entry_manual_buy_window_rate": float(np.mean(manual_buy_values)) if manual_buy_values else np.nan,
        "exit_is_manual_sell_window": exit_is_manual_sell_window,
        "exit_reason": exit_reason,
        "status": status,
    }


def _buy_value(
    strategy: str,
    cash: float,
    cycle_start_equity: float,
    lots: list[Lot],
    params: StrategyParams,
) -> float:
    if strategy == "scale_in":
        if len(lots) >= params.scale_in_max_entries:
            return 0.0
        return float(min(cash, cycle_start_equity / params.scale_in_max_entries))
    if lots:
        return 0.0
    return float(cash)


def _exit_reason(
    strategy: str,
    lots: list[Lot],
    trade_date: pd.Timestamp,
    close: float,
    top_signal: bool,
    params: StrategyParams,
) -> str:
    if not lots:
        return ""

    _, _, avg_entry_price = _position_values(lots)
    position_return = close / avg_entry_price - 1.0
    holding_days = int((trade_date - min(lot.entry_date for lot in lots)).days)

    if strategy == "risk_exit":
        if position_return <= params.stop_loss:
            return "stop_loss"
        if position_return >= params.take_profit:
            return "take_profit"
        if holding_days >= params.max_holding_days:
            return "max_holding_days"

    if top_signal:
        return "top_signal"
    return ""


def _simulate_strategy(
    prediction: pd.DataFrame,
    model_kind: str,
    strategy: str,
    params: StrategyParams,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    cash = 1.0
    lots: list[Lot] = []
    cycle_start_equity = 1.0
    trades: list[dict] = []
    equity_rows: list[dict] = []
    diagnostics = {
        "buy_executions": 0,
        "sell_executions": 0,
        "ignored_bottom_while_long": 0,
        "ignored_top_while_flat": 0,
        "risk_stop_loss_exits": 0,
        "risk_take_profit_exits": 0,
        "risk_max_holding_exits": 0,
    }

    for row in prediction.itertuples(index=False):
        trade_date = pd.Timestamp(row.trade_date)
        close = float(row.close)
        bottom_signal = bool(row.bottom_signal)
        top_signal = bool(row.top_signal)
        opened_today = False

        if bottom_signal:
            if not lots:
                cycle_start_equity = cash
            value = _buy_value(strategy, cash, cycle_start_equity, lots, params)
            if value > 1e-12:
                lots.append(
                    Lot(
                        entry_date=trade_date,
                        entry_price=close,
                        cash_value=value,
                        shares=value / close,
                        bottom_proba=float(row.manual_weak_bottom_proba),
                        top_proba=float(row.manual_weak_top_proba),
                        is_manual_buy_window=int(row.is_manual_buy_window),
                    )
                )
                cash -= value
                diagnostics["buy_executions"] += 1
                opened_today = True
            elif lots:
                diagnostics["ignored_bottom_while_long"] += 1
        elif top_signal and not lots:
            diagnostics["ignored_top_while_flat"] += 1

        reason = "" if opened_today else _exit_reason(strategy, lots, trade_date, close, top_signal, params)
        if reason:
            _, shares, _ = _position_values(lots)
            cash += shares * close
            trades.append(
                _make_trade_row(
                    model_kind=model_kind,
                    strategy=strategy,
                    lots=lots,
                    exit_date=trade_date,
                    exit_price=close,
                    exit_bottom_proba=float(row.manual_weak_bottom_proba),
                    exit_top_proba=float(row.manual_weak_top_proba),
                    exit_is_manual_sell_window=int(row.is_manual_sell_window),
                    exit_reason=reason,
                    status="closed",
                )
            )
            diagnostics["sell_executions"] += 1
            if reason == "stop_loss":
                diagnostics["risk_stop_loss_exits"] += 1
            elif reason == "take_profit":
                diagnostics["risk_take_profit_exits"] += 1
            elif reason == "max_holding_days":
                diagnostics["risk_max_holding_exits"] += 1
            lots = []
            cycle_start_equity = cash

        _, shares, _ = _position_values(lots)
        position_value = shares * close
        equity = cash + position_value
        equity_rows.append(
            {
                "model_kind": model_kind,
                "strategy": strategy,
                "trade_date": trade_date.date(),
                "close": close,
                "equity": equity,
                "cash": cash,
                "position_value": position_value,
                "position": int(position_value > 1e-12),
                "capital_exposure": position_value / equity if equity else 0.0,
                "bottom_signal": int(bottom_signal),
                "top_signal": int(top_signal),
            }
        )

    if lots:
        last = prediction.iloc[-1]
        trades.append(
            _make_trade_row(
                model_kind=model_kind,
                strategy=strategy,
                lots=lots,
                exit_date=pd.Timestamp(last["trade_date"]),
                exit_price=float(last["close"]),
                exit_bottom_proba=float(last["manual_weak_bottom_proba"]),
                exit_top_proba=float(last["manual_weak_top_proba"]),
                exit_is_manual_sell_window=int(last["is_manual_sell_window"]),
                exit_reason="mark_to_market",
                status="open_mark_to_market",
            )
        )

    trades_df = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    equity_df = pd.DataFrame(equity_rows)
    return trades_df, equity_df, diagnostics


def _summarize_model(
    model_kind: str,
    strategy: str,
    prediction: pd.DataFrame,
    trades: pd.DataFrame,
    diagnostics: dict[str, int],
    equity: pd.DataFrame,
) -> dict:
    closed = trades[trades["status"] == "closed"].copy()
    open_trades = trades[trades["status"] != "closed"].copy()
    equity_series = pd.to_numeric(equity["equity"], errors="coerce")
    total_return = float(equity_series.iloc[-1] - 1.0) if not equity_series.empty else np.nan
    buy_hold_return = float(prediction["close"].iloc[-1] / prediction["close"].iloc[0] - 1.0)

    summary = {
        "model_kind": model_kind,
        "strategy": strategy,
        "start_date": prediction["trade_date"].min().date(),
        "end_date": prediction["trade_date"].max().date(),
        "bottom_signal_count": int(prediction["bottom_signal"].sum()),
        "top_signal_count": int(prediction["top_signal"].sum()),
        "buy_executions": diagnostics["buy_executions"],
        "sell_executions": diagnostics["sell_executions"],
        "closed_trades": int(len(closed)),
        "open_trades": int(len(open_trades)),
        "win_rate": float(closed["win"].mean()) if not closed.empty else np.nan,
        "avg_trade_return": float(closed["trade_return"].mean()) if not closed.empty else np.nan,
        "median_trade_return": float(closed["trade_return"].median()) if not closed.empty else np.nan,
        "best_trade_return": float(closed["trade_return"].max()) if not closed.empty else np.nan,
        "worst_trade_return": float(closed["trade_return"].min()) if not closed.empty else np.nan,
        "avg_holding_days": float(closed["holding_days"].mean()) if not closed.empty else np.nan,
        "avg_entry_count": float(closed["entry_count"].mean()) if not closed.empty else np.nan,
        "strategy_total_return": total_return,
        "buy_hold_return": buy_hold_return,
        "excess_vs_buy_hold": total_return - buy_hold_return if np.isfinite(total_return) else np.nan,
        "max_drawdown": _max_drawdown(equity_series),
        "binary_exposure": float(equity["position"].mean()) if not equity.empty else np.nan,
        "capital_exposure": float(equity["capital_exposure"].mean()) if not equity.empty else np.nan,
        "entry_manual_buy_window_rate": float(closed["entry_manual_buy_window_rate"].mean())
        if not closed.empty
        else np.nan,
        "exit_manual_sell_window_rate": float(closed["exit_is_manual_sell_window"].mean())
        if not closed.empty
        else np.nan,
        "ignored_bottom_while_long": diagnostics["ignored_bottom_while_long"],
        "ignored_top_while_flat": diagnostics["ignored_top_while_flat"],
        "risk_stop_loss_exits": diagnostics["risk_stop_loss_exits"],
        "risk_take_profit_exits": diagnostics["risk_take_profit_exits"],
        "risk_max_holding_exits": diagnostics["risk_max_holding_exits"],
    }
    if not open_trades.empty:
        summary["open_entry_date"] = str(open_trades.iloc[-1]["entry_date"])
        summary["open_mark_to_market_return"] = float(open_trades.iloc[-1]["trade_return"])
    else:
        summary["open_entry_date"] = ""
        summary["open_mark_to_market_return"] = np.nan
    return summary


def _summarize_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    closed = trades[trades["status"] == "closed"].copy()
    if closed.empty:
        return pd.DataFrame(
            columns=[
                "model_kind",
                "strategy",
                "entry_year",
                "closed_trades",
                "win_rate",
                "avg_trade_return",
                "median_trade_return",
                "total_compound_return",
                "avg_holding_days",
                "avg_entry_count",
            ]
        )

    closed["entry_year"] = pd.to_datetime(closed["entry_date"]).dt.year
    rows = []
    for (model_kind, strategy, entry_year), part in closed.groupby(
        ["model_kind", "strategy", "entry_year"],
        sort=True,
    ):
        rows.append(
            {
                "model_kind": model_kind,
                "strategy": strategy,
                "entry_year": int(entry_year),
                "closed_trades": int(len(part)),
                "win_rate": float(part["win"].mean()),
                "avg_trade_return": float(part["trade_return"].mean()),
                "median_trade_return": float(part["trade_return"].median()),
                "total_compound_return": float((1.0 + part["trade_return"]).prod() - 1.0),
                "avg_holding_days": float(part["holding_days"].mean()),
                "avg_entry_count": float(part["entry_count"].mean()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_model(
    model_kind: str,
    market: pd.DataFrame,
    strategies: list[str],
    params: StrategyParams,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    model_kind = _normalize_model_kind(model_kind)
    base_prediction = _load_predictions(model_kind, market)
    trade_parts = []
    equity_parts = []
    summaries = []

    for strategy in strategies:
        strategy = _normalize_strategy(strategy)
        prediction = _prepare_strategy_prediction(base_prediction, strategy, params)
        trades, equity, diagnostics = _simulate_strategy(prediction, model_kind, strategy, params)
        trade_parts.append(trades)
        equity_parts.append(equity)
        summaries.append(_summarize_model(model_kind, strategy, prediction, trades, diagnostics, equity))

    trades_df = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame(columns=TRADE_COLUMNS)
    equity_df = pd.concat(equity_parts, ignore_index=True) if equity_parts else pd.DataFrame()
    return trades_df, equity_df, summaries


def evaluate(
    models: list[str],
    strategies: list[str],
    params: StrategyParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_index = _load_target_index()
    market = _load_market_close(target_index)
    normalized_strategies = [_normalize_strategy(strategy) for strategy in strategies]

    all_trades = []
    all_equity = []
    summaries = []
    for model_kind in models:
        trades, equity, model_summaries = evaluate_model(
            model_kind,
            market,
            normalized_strategies,
            params,
        )
        all_trades.append(trades)
        all_equity.append(equity)
        summaries.extend(model_summaries)

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(columns=TRADE_COLUMNS)
    equity_df = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    summary_df = pd.DataFrame(summaries)
    yearly_df = _summarize_yearly(trades_df)

    project_path(SUMMARY_OUTPUT).parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(project_path(SUMMARY_OUTPUT), index=False, encoding="utf-8-sig")
    yearly_df.to_csv(project_path(YEARLY_OUTPUT), index=False, encoding="utf-8-sig")
    trades_df.to_csv(project_path(TRADE_OUTPUT), index=False, encoding="utf-8-sig")
    equity_df.to_csv(project_path(EQUITY_OUTPUT), index=False, encoding="utf-8-sig")
    return summary_df, yearly_df, trades_df, equity_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    parser.add_argument("--scale-in-max-entries", type=int, default=3)
    parser.add_argument("--stop-loss", type=float, default=-0.08)
    parser.add_argument("--take-profit", type=float, default=0.15)
    parser.add_argument("--max-holding-days", type=int, default=60)
    parser.add_argument("--region-gap-days", type=int, default=10)
    args = parser.parse_args()

    params = StrategyParams(
        scale_in_max_entries=args.scale_in_max_entries,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_holding_days=args.max_holding_days,
        region_gap_days=args.region_gap_days,
    )
    summary, yearly, trades, _ = evaluate(args.models, args.strategies, params)
    pd.set_option("display.max_columns", None)
    print("[summary]")
    print(summary.to_string(index=False))
    print("\n[yearly]")
    print(yearly.to_string(index=False))
    print("\n[trades]")
    print(trades.to_string(index=False))


if __name__ == "__main__":
    main()
