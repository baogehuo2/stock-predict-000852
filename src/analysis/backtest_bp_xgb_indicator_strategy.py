from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.config import load_yaml, project_path
from src.common.db import read_sql
from src.features.technical_indicators import cci, kdj, rsi_cn


BP_PREDICTION_PATH = "data/reports/manual_weak_turning_walk_forward_predictions_bp_wf.csv"
XGBOOST_PREDICTION_PATH = "data/reports/manual_weak_turning_walk_forward_predictions_xgboost_wf.csv"
SUMMARY_OUTPUT = "data/reports/bp_xgb_indicator_strategy_summary.csv"
TRADES_OUTPUT = "data/reports/bp_xgb_indicator_strategy_trades.csv"
EQUITY_OUTPUT = "data/reports/bp_xgb_indicator_strategy_equity.csv"


@dataclass(frozen=True)
class StrategyParams:
    first_entry_pct: float = 0.50
    add_entry_pct: float = 0.25
    max_position_pct: float = 0.75
    add_window_days: int = 5
    cooldown_days: int = 3
    stop_loss: float = -0.06
    take_profit: float = 0.15
    max_holding_days: int = 45
    top_reduce_pct: float = 0.50
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    exclude_start: str | None = None
    exclude_end: str | None = None


@dataclass(frozen=True)
class Lot:
    entry_date: pd.Timestamp
    entry_pos: int
    entry_price: float
    cash_value: float
    shares: float
    reason: str


def _output_path(path_value: str, tag: str | None) -> Path:
    path = project_path(path_value)
    if not tag:
        return path
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


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
        raise RuntimeError(f"No market rows found for {target_index}.")
    market["trade_date"] = pd.to_datetime(market["trade_date"])
    for column in ["open", "high", "low", "close"]:
        market[column] = pd.to_numeric(market[column], errors="coerce")
    market = market.dropna(subset=["trade_date", "open", "high", "low", "close"]).sort_values("trade_date")
    return _add_indicator_features(market).reset_index(drop=True)


def _add_indicator_features(market: pd.DataFrame) -> pd.DataFrame:
    result = market.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]

    result["rsi6"] = rsi_cn(close, 6)
    result["rsi_turn_down"] = (result["rsi6"] < result["rsi6"].shift(1)).astype(int)
    result["rsi_overbought"] = (result["rsi6"] > 80).astype(int)

    kdj_frame = kdj(high, low, close)
    result["kdj_k"] = kdj_frame["kdj_k"]
    result["kdj_d"] = kdj_frame["kdj_d"]
    result["kdj_j"] = kdj_frame["kdj_j"]
    result["kdj_k_minus_d"] = kdj_frame["kdj_k_minus_d"]
    result["kdj_turn_down"] = (
        (result["kdj_j"] < result["kdj_j"].shift(1))
        & (result["kdj_k"] < result["kdj_k"].shift(1))
    ).astype(int)
    result["kdj_dead_cross"] = (
        (result["kdj_k_minus_d"] < 0) & (result["kdj_k_minus_d"].shift(1) >= 0)
    ).astype(int)

    result["cci14"] = cci(high, low, close, window=14)
    result["cci_turn_down_below_100"] = (
        (result["cci14"] < 100) & (result["cci14"] < result["cci14"].shift(1))
    ).astype(int)
    return result


def _load_prediction(path_value: str, prefix: str, start_date: str) -> pd.DataFrame:
    path = project_path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Prediction file not found: {path}")
    prediction = pd.read_csv(path, parse_dates=["trade_date"])
    prediction = prediction[prediction["trade_date"] >= pd.Timestamp(start_date)].copy()
    return prediction.rename(
        columns={
            "weak_combined_bottom_signal": f"{prefix}_bottom_signal",
            "weak_combined_top_signal": f"{prefix}_top_signal",
            "manual_weak_bottom_proba": f"{prefix}_bottom_proba",
            "manual_weak_top_proba": f"{prefix}_top_proba",
        }
    )[
        [
            "trade_date",
            f"{prefix}_bottom_signal",
            f"{prefix}_top_signal",
            f"{prefix}_bottom_proba",
            f"{prefix}_top_proba",
        ]
    ]


def _load_backtest_frame(start_date: str) -> pd.DataFrame:
    target_index = _load_target_index()
    market = _load_market(target_index, start_date)
    bp = _load_prediction(BP_PREDICTION_PATH, "bp", start_date)
    xgb = _load_prediction(XGBOOST_PREDICTION_PATH, "xgb", start_date)
    data = market.merge(bp, on="trade_date", how="left").merge(xgb, on="trade_date", how="left")
    for column in ["bp_bottom_signal", "bp_top_signal", "xgb_bottom_signal", "xgb_top_signal"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0).astype(int)
    for column in ["bp_bottom_proba", "bp_top_proba", "xgb_bottom_proba", "xgb_top_proba"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.sort_values("trade_date").reset_index(drop=True)


def _in_excluded_period(trade_date: pd.Timestamp, params: StrategyParams) -> bool:
    if not params.exclude_start or not params.exclude_end:
        return False
    return pd.Timestamp(params.exclude_start) <= trade_date <= pd.Timestamp(params.exclude_end)


def _indicator_stop_reason(row) -> str:
    reasons = []
    for column in [
        "cci_turn_down_below_100",
        "kdj_turn_down",
        "rsi_turn_down",
        "kdj_dead_cross",
        "rsi_overbought",
    ]:
        if int(getattr(row, column, 0)) == 1:
            reasons.append(column)
    return "indicator_stop:" + "|".join(reasons) if reasons else ""


def _position(lots: list[Lot]) -> tuple[float, float, float]:
    invested_cash = float(sum(lot.cash_value for lot in lots))
    shares = float(sum(lot.shares for lot in lots))
    avg_entry_price = invested_cash / shares if shares else np.nan
    return invested_cash, shares, avg_entry_price


def _trade_return(exit_price: float, avg_entry_price: float, fee_rate: float) -> float:
    gross = exit_price / avg_entry_price - 1.0
    return gross - 2 * fee_rate


def _sell_lots(
    lots: list[Lot],
    sell_fraction: float,
    exit_date: pd.Timestamp,
    exit_pos: int,
    exit_price: float,
    reason: str,
    params: StrategyParams,
) -> tuple[list[Lot], dict]:
    sell_fraction = min(max(sell_fraction, 0.0), 1.0)
    invested_cash, shares, avg_entry_price = _position(lots)
    if not lots or sell_fraction <= 0:
        raise RuntimeError("No position to sell.")

    sold_cash = invested_cash * sell_fraction
    sold_shares = shares * sell_fraction
    remaining_lots = [
        Lot(
            entry_date=lot.entry_date,
            entry_pos=lot.entry_pos,
            entry_price=lot.entry_price,
            cash_value=lot.cash_value * (1.0 - sell_fraction),
            shares=lot.shares * (1.0 - sell_fraction),
            reason=lot.reason,
        )
        for lot in lots
        if lot.cash_value * (1.0 - sell_fraction) > 1e-12
    ]
    entry_date = min(lot.entry_date for lot in lots)
    entry_pos = min(lot.entry_pos for lot in lots)
    realized_return = _trade_return(exit_price, avg_entry_price, params.fee_rate)
    trade = {
        "entry_date": entry_date.date().isoformat(),
        "exit_date": exit_date.date().isoformat(),
        "entry_price": avg_entry_price,
        "exit_price": exit_price,
        "holding_trade_days": int(exit_pos - entry_pos),
        "holding_calendar_days": int((exit_date - entry_date).days),
        "trade_return": realized_return,
        "win": int(realized_return > 0),
        "entry_count": int(len(lots)),
        "sell_fraction": sell_fraction,
        "sold_cash": sold_cash,
        "sold_shares": sold_shares,
        "exit_reason": reason,
        "status": "closed",
    }
    return remaining_lots, trade


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return np.nan
    return float((equity / equity.cummax() - 1.0).min())


def run_backtest(data: pd.DataFrame, params: StrategyParams) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cash = 1.0
    lots: list[Lot] = []
    last_entry_date: pd.Timestamp | None = None
    last_exit_pos: int | None = None
    previous_xgb_top = False
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for trade_pos, row in enumerate(data.itertuples(index=False)):
        trade_date = pd.Timestamp(row.trade_date)
        close = float(row.close)
        buy_price = close * (1.0 + params.slippage_rate)
        sell_price = close * (1.0 - params.slippage_rate)
        blocked = _in_excluded_period(trade_date, params)
        invested_cash, shares, avg_entry_price = _position(lots)
        equity_before = cash + shares * close
        position_ratio = invested_cash / equity_before if equity_before > 0 else 0.0

        opened_today = False
        if not blocked and int(row.bp_bottom_signal) == 1 and int(row.xgb_top_signal) == 0:
            cooldown_ok = last_exit_pos is None or trade_pos - last_exit_pos >= params.cooldown_days
            if not lots and cooldown_ok:
                buy_value = min(cash, equity_before * params.first_entry_pct)
                if buy_value > 1e-12:
                    lots.append(
                        Lot(
                            entry_date=trade_date,
                            entry_pos=trade_pos,
                            entry_price=buy_price,
                            cash_value=buy_value,
                            shares=buy_value / buy_price,
                            reason="bp_bottom_initial",
                        )
                    )
                    cash -= buy_value
                    last_entry_date = trade_date
                    opened_today = True
            elif lots and last_entry_date is not None:
                first_entry_pos = min(lot.entry_pos for lot in lots)
                in_add_window = (trade_pos - first_entry_pos) <= params.add_window_days
                if in_add_window and position_ratio < params.max_position_pct - 1e-9:
                    add_value = min(cash, equity_before * params.add_entry_pct)
                    room_value = max(0.0, equity_before * params.max_position_pct - invested_cash)
                    buy_value = min(add_value, room_value)
                    if buy_value > 1e-12:
                        lots.append(
                            Lot(
                                entry_date=trade_date,
                                entry_pos=trade_pos,
                                entry_price=buy_price,
                                cash_value=buy_value,
                                shares=buy_value / buy_price,
                                reason="bp_bottom_add",
                            )
                        )
                        cash -= buy_value
                        last_entry_date = trade_date
                        opened_today = True

        invested_cash, shares, avg_entry_price = _position(lots)
        exit_reason = ""
        sell_fraction = 0.0
        if lots and not opened_today:
            position_return = close / avg_entry_price - 1.0
            holding_days = int(trade_pos - min(lot.entry_pos for lot in lots))
            indicator_reason = _indicator_stop_reason(row)
            if position_return <= params.stop_loss:
                exit_reason = "hard_stop_loss"
                sell_fraction = 1.0
            elif position_return >= params.take_profit:
                exit_reason = "hard_take_profit_half"
                sell_fraction = min(0.5, 1.0)
            elif indicator_reason:
                exit_reason = indicator_reason
                sell_fraction = 1.0
            elif int(row.xgb_top_signal) == 1:
                if previous_xgb_top:
                    exit_reason = "xgb_top_confirmed_clear"
                    sell_fraction = 1.0
                else:
                    exit_reason = "xgb_top_reduce_half"
                    sell_fraction = params.top_reduce_pct
            elif holding_days >= params.max_holding_days:
                exit_reason = "max_holding_days"
                sell_fraction = 1.0

        if lots and sell_fraction > 0:
            before_cash = cash
            lots, trade = _sell_lots(
                lots,
                sell_fraction,
                trade_date,
                trade_pos,
                sell_price,
                exit_reason,
                params,
            )
            cash = before_cash + trade["sold_shares"] * sell_price * (1.0 - params.fee_rate)
            trades.append(trade)
            if not lots:
                last_exit_pos = trade_pos
                last_entry_date = None

        invested_cash, shares, avg_entry_price = _position(lots)
        position_value = shares * close
        equity = cash + position_value
        equity_rows.append(
            {
                "trade_date": trade_date.date().isoformat(),
                "close": close,
                "equity": equity,
                "cash": cash,
                "position_value": position_value,
                "position_ratio": position_value / equity if equity else 0.0,
                "bp_bottom_signal": int(row.bp_bottom_signal),
                "xgb_top_signal": int(row.xgb_top_signal),
                "blocked_excluded_period": int(blocked),
            }
        )
        previous_xgb_top = int(row.xgb_top_signal) == 1

    if lots:
        last = data.iloc[-1]
        lots, trade = _sell_lots(
            lots,
            1.0,
            pd.Timestamp(last["trade_date"]),
            int(len(data) - 1),
            float(last["close"]) * (1.0 - params.slippage_rate),
            "mark_to_market",
            params,
        )
        trade["status"] = "open_mark_to_market"
        trades.append(trade)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    closed = trades_df[trades_df["status"] == "closed"].copy() if not trades_df.empty else pd.DataFrame()
    equity_series = pd.to_numeric(equity_df["equity"], errors="coerce")
    summary = {
        "strategy": "bp_xgb_indicator_strategy",
        "start_date": data["trade_date"].min().date().isoformat(),
        "end_date": data["trade_date"].max().date().isoformat(),
        "total_return": float(equity_series.iloc[-1] - 1.0) if not equity_series.empty else np.nan,
        "max_drawdown": _max_drawdown(equity_series),
        "closed_trades": int(len(closed)),
        "open_trades": int((trades_df["status"] != "closed").sum()) if not trades_df.empty else 0,
        "win_rate": float(closed["win"].mean()) if not closed.empty else np.nan,
        "avg_trade_return": float(closed["trade_return"].mean()) if not closed.empty else np.nan,
        "median_trade_return": float(closed["trade_return"].median()) if not closed.empty else np.nan,
        "best_trade_return": float(closed["trade_return"].max()) if not closed.empty else np.nan,
        "worst_trade_return": float(closed["trade_return"].min()) if not closed.empty else np.nan,
        "avg_holding_trade_days": float(closed["holding_trade_days"].mean()) if not closed.empty else np.nan,
        "avg_holding_calendar_days": float(closed["holding_calendar_days"].mean()) if not closed.empty else np.nan,
        "avg_position_ratio": float(equity_df["position_ratio"].mean()) if not equity_df.empty else np.nan,
        "bp_bottom_signal_count": int(data["bp_bottom_signal"].sum()),
        "xgb_top_signal_count": int(data["xgb_top_signal"].sum()),
        "fee_rate": params.fee_rate,
        "slippage_rate": params.slippage_rate,
        "exclude_start": params.exclude_start or "",
        "exclude_end": params.exclude_end or "",
    }
    summary_df = pd.DataFrame([summary])
    return summary_df, trades_df, equity_df


def write_outputs(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    tag: str | None = None,
) -> tuple[Path, Path, Path]:
    summary_path = _output_path(SUMMARY_OUTPUT, tag)
    trades_path = _output_path(TRADES_OUTPUT, tag)
    equity_path = _output_path(EQUITY_OUTPUT, tag)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    trades.to_csv(trades_path, index=False, encoding="utf-8-sig")
    equity.to_csv(equity_path, index=False, encoding="utf-8-sig")
    return summary_path, trades_path, equity_path


def _run_one_case(data: pd.DataFrame, params: StrategyParams, tag: str) -> pd.DataFrame:
    summary, trades, equity = run_backtest(data, params)
    write_outputs(summary, trades, equity, tag)
    summary = summary.copy()
    summary["case"] = tag
    return summary


def run_default_cases(start_date: str = "2022-01-01") -> pd.DataFrame:
    data = _load_backtest_frame(start_date)
    cases = [
        ("all_no_cost", StrategyParams()),
        ("exclude_no_cost", StrategyParams(exclude_start="2023-12-01", exclude_end="2024-01-31")),
        ("all_cost", StrategyParams(fee_rate=0.0003, slippage_rate=0.0005)),
        (
            "exclude_cost",
            StrategyParams(
                fee_rate=0.0003,
                slippage_rate=0.0005,
                exclude_start="2023-12-01",
                exclude_end="2024-01-31",
            ),
        ),
    ]
    compare = pd.concat([_run_one_case(data, params, tag) for tag, params in cases], ignore_index=True)
    compare_path = project_path("data/reports/bp_xgb_indicator_strategy_summary_compare.csv")
    compare.to_csv(compare_path, index=False, encoding="utf-8-sig")
    return compare


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest BP bottom + XGBoost top + indicator stop strategy.")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--first-entry-pct", type=float, default=0.50)
    parser.add_argument("--add-entry-pct", type=float, default=0.25)
    parser.add_argument("--max-position-pct", type=float, default=0.75)
    parser.add_argument("--add-window-days", type=int, default=5)
    parser.add_argument("--cooldown-days", type=int, default=3)
    parser.add_argument("--stop-loss", type=float, default=-0.06)
    parser.add_argument("--take-profit", type=float, default=0.15)
    parser.add_argument("--max-holding-days", type=int, default=45)
    parser.add_argument("--top-reduce-pct", type=float, default=0.50)
    parser.add_argument("--fee-rate", type=float, default=0.0)
    parser.add_argument("--slippage-rate", type=float, default=0.0)
    parser.add_argument("--exclude-start")
    parser.add_argument("--exclude-end")
    parser.add_argument("--output-tag")
    parser.add_argument("--run-default-cases", action="store_true")
    args = parser.parse_args()

    if args.run_default_cases:
        compare = run_default_cases(args.start_date)
        pd.set_option("display.max_columns", None)
        print(compare.to_string(index=False))
        return

    params = StrategyParams(
        first_entry_pct=args.first_entry_pct,
        add_entry_pct=args.add_entry_pct,
        max_position_pct=args.max_position_pct,
        add_window_days=args.add_window_days,
        cooldown_days=args.cooldown_days,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_holding_days=args.max_holding_days,
        top_reduce_pct=args.top_reduce_pct,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        exclude_start=args.exclude_start,
        exclude_end=args.exclude_end,
    )
    data = _load_backtest_frame(args.start_date)
    summary, trades, equity = run_backtest(data, params)
    write_outputs(summary, trades, equity, args.output_tag)
    pd.set_option("display.max_columns", None)
    print("[summary]")
    print(summary.to_string(index=False))
    print("\n[trades]")
    print(trades.to_string(index=False))


if __name__ == "__main__":
    main()
