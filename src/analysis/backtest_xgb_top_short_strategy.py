from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.backtest_bp_xgb_indicator_strategy import _load_backtest_frame, _max_drawdown
from src.common.config import project_path


SUMMARY_OUTPUT = "data/reports/xgb_top_short_strategy_summary.csv"
TRADES_OUTPUT = "data/reports/xgb_top_short_strategy_trades.csv"
EQUITY_OUTPUT = "data/reports/xgb_top_short_strategy_equity.csv"


@dataclass(frozen=True)
class ShortStrategyParams:
    first_short_pct: float = 0.50
    add_short_pct: float = 0.25
    max_short_pct: float = 0.75
    add_window_days: int = 5
    cooldown_days: int = 3
    stop_loss: float = -0.06
    take_profit: float = 0.15
    max_holding_days: int = 45
    bottom_cover_pct: float = 0.50
    fee_rate: float = 0.0
    slippage_rate: float = 0.0
    exclude_start: str | None = None
    exclude_end: str | None = None


@dataclass(frozen=True)
class ShortLot:
    entry_date: pd.Timestamp
    entry_pos: int
    entry_price: float
    notional_value: float
    shares: float
    reason: str


def _output_path(path_value: str, tag: str | None) -> Path:
    path = project_path(path_value)
    if not tag:
        return path
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def _in_excluded_period(trade_date: pd.Timestamp, params: ShortStrategyParams) -> bool:
    if not params.exclude_start or not params.exclude_end:
        return False
    return pd.Timestamp(params.exclude_start) <= trade_date <= pd.Timestamp(params.exclude_end)


def _add_short_cover_indicators(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["rsi_turn_up"] = (result["rsi6"] > result["rsi6"].shift(1)).astype(int)
    result["rsi_oversold"] = (result["rsi6"] < 20).astype(int)
    result["kdj_turn_up"] = (
        (result["kdj_j"] > result["kdj_j"].shift(1))
        & (result["kdj_k"] > result["kdj_k"].shift(1))
    ).astype(int)
    result["kdj_golden_cross"] = (
        (result["kdj_k_minus_d"] > 0) & (result["kdj_k_minus_d"].shift(1) <= 0)
    ).astype(int)
    result["cci_turn_up_above_minus_100"] = (
        (result["cci14"] > -100) & (result["cci14"] > result["cci14"].shift(1))
    ).astype(int)
    return result


def _indicator_cover_reason(row) -> str:
    reasons = []
    for column in [
        "cci_turn_up_above_minus_100",
        "kdj_turn_up",
        "rsi_turn_up",
        "kdj_golden_cross",
        "rsi_oversold",
    ]:
        if int(getattr(row, column, 0)) == 1:
            reasons.append(column)
    return "indicator_cover:" + "|".join(reasons) if reasons else ""


def _position(lots: list[ShortLot], close: float | None = None) -> tuple[float, float, float, float]:
    entry_notional = float(sum(lot.notional_value for lot in lots))
    shares = float(sum(lot.shares for lot in lots))
    avg_entry_price = entry_notional / shares if shares else np.nan
    current_exposure = shares * float(close) if shares and close is not None else entry_notional
    return entry_notional, shares, avg_entry_price, current_exposure


def _unrealized_pnl(lots: list[ShortLot], close: float) -> float:
    return float(sum(lot.shares * (lot.entry_price - close) for lot in lots))


def _short_return(exit_price: float, avg_entry_price: float, fee_rate: float) -> float:
    gross = 1.0 - exit_price / avg_entry_price
    return gross - 2 * fee_rate


def _cover_lots(
    lots: list[ShortLot],
    cover_fraction: float,
    exit_date: pd.Timestamp,
    exit_pos: int,
    exit_price: float,
    reason: str,
    params: ShortStrategyParams,
) -> tuple[list[ShortLot], dict, float]:
    cover_fraction = min(max(cover_fraction, 0.0), 1.0)
    entry_notional, shares, avg_entry_price, _ = _position(lots)
    if not lots or cover_fraction <= 0:
        raise RuntimeError("No short position to cover.")

    covered_entry_notional = entry_notional * cover_fraction
    covered_shares = shares * cover_fraction
    cover_notional = covered_shares * exit_price
    realized_pnl = covered_shares * (avg_entry_price - exit_price)
    exit_fee = cover_notional * params.fee_rate
    remaining_lots = [
        ShortLot(
            entry_date=lot.entry_date,
            entry_pos=lot.entry_pos,
            entry_price=lot.entry_price,
            notional_value=lot.notional_value * (1.0 - cover_fraction),
            shares=lot.shares * (1.0 - cover_fraction),
            reason=lot.reason,
        )
        for lot in lots
        if lot.notional_value * (1.0 - cover_fraction) > 1e-12
    ]
    entry_date = min(lot.entry_date for lot in lots)
    entry_pos = min(lot.entry_pos for lot in lots)
    trade_return = _short_return(exit_price, avg_entry_price, params.fee_rate)
    trade = {
        "entry_date": entry_date.date().isoformat(),
        "exit_date": exit_date.date().isoformat(),
        "entry_price": avg_entry_price,
        "exit_price": exit_price,
        "holding_trade_days": int(exit_pos - entry_pos),
        "holding_calendar_days": int((exit_date - entry_date).days),
        "trade_return": trade_return,
        "win": int(trade_return > 0),
        "entry_count": int(len(lots)),
        "cover_fraction": cover_fraction,
        "covered_entry_notional": covered_entry_notional,
        "covered_shares": covered_shares,
        "realized_pnl": realized_pnl,
        "exit_fee": exit_fee,
        "exit_reason": reason,
        "status": "closed",
    }
    cash_delta = realized_pnl - exit_fee
    return remaining_lots, trade, cash_delta


def run_short_backtest(
    data: pd.DataFrame,
    params: ShortStrategyParams,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = _add_short_cover_indicators(data)
    cash = 1.0
    lots: list[ShortLot] = []
    last_entry_date: pd.Timestamp | None = None
    last_exit_pos: int | None = None
    previous_bp_bottom = False
    trades: list[dict] = []
    equity_rows: list[dict] = []

    for trade_pos, row in enumerate(data.itertuples(index=False)):
        trade_date = pd.Timestamp(row.trade_date)
        close = float(row.close)
        short_entry_price = close * (1.0 - params.slippage_rate)
        cover_price = close * (1.0 + params.slippage_rate)
        blocked = _in_excluded_period(trade_date, params)
        entry_notional, shares, avg_entry_price, current_exposure = _position(lots, close)
        equity_before = cash + _unrealized_pnl(lots, close)
        short_ratio = current_exposure / equity_before if equity_before > 0 else 0.0

        opened_today = False
        if not blocked and int(row.xgb_top_signal) == 1 and int(row.bp_bottom_signal) == 0:
            cooldown_ok = last_exit_pos is None or trade_pos - last_exit_pos >= params.cooldown_days
            if not lots and cooldown_ok:
                notional = max(0.0, equity_before * params.first_short_pct)
                entry_fee = notional * params.fee_rate
                if notional > 1e-12 and cash > entry_fee:
                    lots.append(
                        ShortLot(
                            entry_date=trade_date,
                            entry_pos=trade_pos,
                            entry_price=short_entry_price,
                            notional_value=notional,
                            shares=notional / short_entry_price,
                            reason="xgb_top_initial_short",
                        )
                    )
                    cash -= entry_fee
                    last_entry_date = trade_date
                    opened_today = True
            elif lots and last_entry_date is not None:
                first_entry_pos = min(lot.entry_pos for lot in lots)
                in_add_window = (trade_pos - first_entry_pos) <= params.add_window_days
                if in_add_window and short_ratio < params.max_short_pct - 1e-9:
                    add_notional = max(0.0, equity_before * params.add_short_pct)
                    room_notional = max(0.0, equity_before * params.max_short_pct - current_exposure)
                    notional = min(add_notional, room_notional)
                    entry_fee = notional * params.fee_rate
                    if notional > 1e-12 and cash > entry_fee:
                        lots.append(
                            ShortLot(
                                entry_date=trade_date,
                                entry_pos=trade_pos,
                                entry_price=short_entry_price,
                                notional_value=notional,
                                shares=notional / short_entry_price,
                                reason="xgb_top_add_short",
                            )
                        )
                        cash -= entry_fee
                        last_entry_date = trade_date
                        opened_today = True

        entry_notional, shares, avg_entry_price, current_exposure = _position(lots, close)
        exit_reason = ""
        cover_fraction = 0.0
        if lots and not opened_today:
            position_return = 1.0 - close / avg_entry_price
            holding_days = int(trade_pos - min(lot.entry_pos for lot in lots))
            indicator_reason = _indicator_cover_reason(row)
            if position_return <= params.stop_loss:
                exit_reason = "hard_short_stop_loss"
                cover_fraction = 1.0
            elif position_return >= params.take_profit:
                exit_reason = "hard_short_take_profit_half"
                cover_fraction = min(0.5, 1.0)
            elif indicator_reason:
                exit_reason = indicator_reason
                cover_fraction = 1.0
            elif int(row.bp_bottom_signal) == 1:
                if previous_bp_bottom:
                    exit_reason = "bp_bottom_confirmed_cover"
                    cover_fraction = 1.0
                else:
                    exit_reason = "bp_bottom_cover_half"
                    cover_fraction = params.bottom_cover_pct
            elif holding_days >= params.max_holding_days:
                exit_reason = "max_holding_days"
                cover_fraction = 1.0

        if lots and cover_fraction > 0:
            lots, trade, cash_delta = _cover_lots(
                lots,
                cover_fraction,
                trade_date,
                trade_pos,
                cover_price,
                exit_reason,
                params,
            )
            cash += cash_delta
            trades.append(trade)
            if not lots:
                last_exit_pos = trade_pos
                last_entry_date = None

        entry_notional, shares, avg_entry_price, current_exposure = _position(lots, close)
        unrealized_pnl = _unrealized_pnl(lots, close)
        equity = cash + unrealized_pnl
        equity_rows.append(
            {
                "trade_date": trade_date.date().isoformat(),
                "close": close,
                "equity": equity,
                "cash": cash,
                "short_entry_notional": entry_notional,
                "short_current_exposure": current_exposure,
                "short_ratio": current_exposure / equity if equity else 0.0,
                "unrealized_pnl": unrealized_pnl,
                "bp_bottom_signal": int(row.bp_bottom_signal),
                "xgb_top_signal": int(row.xgb_top_signal),
                "blocked_excluded_period": int(blocked),
            }
        )
        previous_bp_bottom = int(row.bp_bottom_signal) == 1

    if lots:
        last = data.iloc[-1]
        lots, trade, cash_delta = _cover_lots(
            lots,
            1.0,
            pd.Timestamp(last["trade_date"]),
            int(len(data) - 1),
            float(last["close"]) * (1.0 + params.slippage_rate),
            "mark_to_market",
            params,
        )
        cash += cash_delta
        trade["status"] = "open_mark_to_market"
        trades.append(trade)

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    closed = trades_df[trades_df["status"] == "closed"].copy() if not trades_df.empty else pd.DataFrame()
    equity_series = pd.to_numeric(equity_df["equity"], errors="coerce")
    summary = {
        "strategy": "xgb_top_short_only_strategy",
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
        "avg_short_ratio": float(equity_df["short_ratio"].mean()) if not equity_df.empty else np.nan,
        "bp_bottom_signal_count": int(data["bp_bottom_signal"].sum()),
        "xgb_top_signal_count": int(data["xgb_top_signal"].sum()),
        "fee_rate": params.fee_rate,
        "slippage_rate": params.slippage_rate,
        "exclude_start": params.exclude_start or "",
        "exclude_end": params.exclude_end or "",
    }
    return pd.DataFrame([summary]), trades_df, equity_df


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


def _run_one_case(
    data: pd.DataFrame,
    params: ShortStrategyParams,
    tag: str,
) -> pd.DataFrame:
    summary, trades, equity = run_short_backtest(data, params)
    write_outputs(summary, trades, equity, tag)
    summary = summary.copy()
    summary["case"] = tag
    return summary


def run_default_cases(start_date: str = "2022-01-01") -> pd.DataFrame:
    data = _load_backtest_frame(start_date)
    cases = [
        ("all_no_cost", ShortStrategyParams()),
        ("exclude_no_cost", ShortStrategyParams(exclude_start="2023-12-01", exclude_end="2024-01-31")),
        ("all_cost", ShortStrategyParams(fee_rate=0.0003, slippage_rate=0.0005)),
        (
            "exclude_cost",
            ShortStrategyParams(
                fee_rate=0.0003,
                slippage_rate=0.0005,
                exclude_start="2023-12-01",
                exclude_end="2024-01-31",
            ),
        ),
    ]
    rows = [_run_one_case(data, params, tag) for tag, params in cases]
    compare = pd.concat(rows, ignore_index=True)
    compare_path = project_path("data/reports/xgb_top_short_strategy_summary_compare.csv")
    compare.to_csv(compare_path, index=False, encoding="utf-8-sig")
    return compare


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest XGBoost top short-only strategy.")
    parser.add_argument("--start-date", default="2022-01-01")
    parser.add_argument("--first-short-pct", type=float, default=0.50)
    parser.add_argument("--add-short-pct", type=float, default=0.25)
    parser.add_argument("--max-short-pct", type=float, default=0.75)
    parser.add_argument("--add-window-days", type=int, default=5)
    parser.add_argument("--cooldown-days", type=int, default=3)
    parser.add_argument("--stop-loss", type=float, default=-0.06)
    parser.add_argument("--take-profit", type=float, default=0.15)
    parser.add_argument("--max-holding-days", type=int, default=45)
    parser.add_argument("--bottom-cover-pct", type=float, default=0.50)
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

    params = ShortStrategyParams(
        first_short_pct=args.first_short_pct,
        add_short_pct=args.add_short_pct,
        max_short_pct=args.max_short_pct,
        add_window_days=args.add_window_days,
        cooldown_days=args.cooldown_days,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        max_holding_days=args.max_holding_days,
        bottom_cover_pct=args.bottom_cover_pct,
        fee_rate=args.fee_rate,
        slippage_rate=args.slippage_rate,
        exclude_start=args.exclude_start,
        exclude_end=args.exclude_end,
    )
    data = _load_backtest_frame(args.start_date)
    summary, trades, equity = run_short_backtest(data, params)
    write_outputs(summary, trades, equity, args.output_tag)
    pd.set_option("display.max_columns", None)
    print("[summary]")
    print(summary.to_string(index=False))
    print("\n[trades]")
    print(trades.to_string(index=False))


if __name__ == "__main__":
    main()
