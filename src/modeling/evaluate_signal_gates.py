from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import project_path


DEFAULT_BUY_THRESHOLDS = {7: 0.60}


def _parse_threshold_map(value: str) -> dict[int, float]:
    result = {}
    for item in value.split(","):
        horizon, threshold = item.split(":", 1)
        result[int(horizon.strip())] = float(threshold.strip())
    return result


def _causal_regime(data: pd.DataFrame) -> pd.Series:
    regime = pd.Series("neutral", index=data.index, dtype="object")
    regime.loc[pd.to_numeric(data["f_is_bull_trend"], errors="coerce").fillna(0) == 1] = "bull"
    regime.loc[pd.to_numeric(data["f_is_bear_trend"], errors="coerce").fillna(0) == 1] = "bear"
    return regime


def _gate_mask(data: pd.DataFrame, gate: str, base_threshold: float) -> pd.Series:
    if gate == "no_gate":
        return data["buy_proba"] >= base_threshold
    if gate == "hard_gate":
        return (data["causal_regime"] == "bull") & (data["buy_proba"] >= base_threshold)
    if gate == "soft_gate":
        threshold = pd.Series(max(base_threshold, 0.75), index=data.index)
        threshold.loc[data["causal_regime"] == "bull"] = base_threshold
        threshold.loc[data["causal_regime"] == "bear"] = max(base_threshold, 0.85)
        return data["buy_proba"] >= threshold
    raise ValueError(f"Unknown gate: {gate}")


def _non_overlapping(signal: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if signal.empty:
        return signal
    if signal["test_year"].nunique() > 1:
        parts = [
            _non_overlapping(year_part, horizon)
            for _, year_part in signal.groupby("test_year", sort=True)
        ]
        return pd.concat(parts).sort_values("trade_date") if parts else signal.iloc[0:0]
    selected = []
    next_allowed_position = -1
    for idx, row in signal.sort_values("trade_date").iterrows():
        position = int(row["trade_pos"])
        if position >= next_allowed_position:
            selected.append(idx)
            next_allowed_position = position + horizon
    return signal.loc[selected]


def _metrics(
    universe: pd.DataFrame,
    signal: pd.DataFrame,
    gate: str,
    sample_mode: str,
    scope: str,
) -> dict:
    correct = signal["future_ret"] > 0
    return {
        "scope": scope,
        "test_year": int(universe["test_year"].iloc[0]) if scope == "year" else None,
        "horizon": int(universe["horizon"].iloc[0]),
        "gate": gate,
        "sample_mode": sample_mode,
        "base_buy_threshold": float(universe["base_buy_threshold"].iloc[0]),
        "universe_rows": int(len(universe)),
        "signal_count": int(len(signal)),
        "coverage": float(len(signal) / len(universe)) if len(universe) else 0.0,
        "precision": float(correct.mean()) if len(signal) else None,
        "avg_future_ret": float(signal["future_ret"].mean()) if len(signal) else None,
        "median_future_ret": float(signal["future_ret"].median()) if len(signal) else None,
        "avg_buy_proba": float(signal["buy_proba"].mean()) if len(signal) else None,
        "bull_signal_count": int((signal["causal_regime"] == "bull").sum()),
        "neutral_signal_count": int((signal["causal_regime"] == "neutral").sum()),
        "bear_signal_count": int((signal["causal_regime"] == "bear").sum()),
    }


def _evaluate_scope(data: pd.DataFrame, scope: str) -> list[dict]:
    rows = []
    for horizon, universe in data.groupby("horizon", sort=True):
        base_threshold = float(universe["base_buy_threshold"].iloc[0])
        for gate in ["no_gate", "hard_gate", "soft_gate"]:
            gate_mask = _gate_mask(universe, gate, base_threshold)
            signal = universe[gate_mask]
            rows.append(_metrics(universe, signal, gate, "all_signals", scope))
            rows.append(
                _metrics(universe, _non_overlapping(signal, int(horizon)), gate, "non_overlapping", scope)
            )
    return rows


def evaluate_signal_gates(
    detail_csv: str = "data/reports/walk_forward_buy_7d_details.csv",
    buy_thresholds: dict[int, float] | None = None,
    output_csv: str = "data/reports/signal_gate_quality_7d_buy.csv",
) -> list[dict]:
    thresholds = buy_thresholds or DEFAULT_BUY_THRESHOLDS
    details = pd.read_csv(project_path(detail_csv), parse_dates=["trade_date"])
    required = {
        "trade_date",
        "test_year",
        "horizon",
        "future_ret",
        "buy_proba",
        "f_is_bull_trend",
        "f_is_bear_trend",
    }
    missing = sorted(required - set(details.columns))
    if missing:
        raise RuntimeError(f"Walk-forward detail report is missing columns: {missing}. Rerun walk-forward evaluation.")
    details = details[details["horizon"].isin(thresholds)].copy()
    details["causal_regime"] = _causal_regime(details)
    details["base_buy_threshold"] = details["horizon"].map(thresholds)
    details["trade_pos"] = details.groupby(["test_year", "horizon"]).cumcount()

    rows = []
    for _, year_part in details.groupby("test_year", sort=True):
        rows.extend(_evaluate_scope(year_part, "year"))
    rows.extend(_evaluate_scope(details, "aggregate"))
    path = project_path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    print(f"saved signal gate report path={path} rows={len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare causal signal gates for the 7-day buy model.")
    parser.add_argument("--detail-csv", default="data/reports/walk_forward_buy_7d_details.csv")
    parser.add_argument("--buy-thresholds", default="7:0.60")
    parser.add_argument("--output-csv", default="data/reports/signal_gate_quality_7d_buy.csv")
    args = parser.parse_args()
    rows = evaluate_signal_gates(
        detail_csv=args.detail_csv,
        buy_thresholds=_parse_threshold_map(args.buy_thresholds),
        output_csv=args.output_csv,
    )
    report = pd.DataFrame(rows)
    focus = report[(report["scope"] == "aggregate") & (report["sample_mode"] == "non_overlapping")]
    cols = [
        "horizon",
        "gate",
        "signal_count",
        "precision",
        "avg_future_ret",
        "median_future_ret",
        "bull_signal_count",
        "neutral_signal_count",
        "bear_signal_count",
    ]
    print(focus[cols].to_string(index=False))


if __name__ == "__main__":
    main()
