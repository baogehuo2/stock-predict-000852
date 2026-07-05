from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import get_config, project_path
from src.modeling.bottom_data import load_bottom_dataset, load_bottom_features
from src.modeling.walk_forward_bottom_lgbm import (
    _fit_model,
    _non_overlapping,
    _positive_probability,
)


def _metrics(universe: pd.DataFrame, signal: pd.DataFrame, sample_mode: str) -> dict:
    count = len(signal)
    return {
        "test_year": int(universe["test_year"].iloc[0]),
        "year_complete": bool(universe["year_complete"].iloc[0]),
        "model": universe["model"].iloc[0],
        "opportunity_threshold": float(universe["opportunity_threshold"].iloc[0]),
        "risk_threshold": float(universe["risk_threshold"].iloc[0]),
        "sample_mode": sample_mode,
        "candidate_rows": len(universe),
        "signal_count": count,
        "coverage": count / len(universe),
        "terminal_precision": float(signal["terminal_rebound_label"].mean()) if count else None,
        "quality_precision": float(signal["quality_bottom_label"].mean()) if count else None,
        "natural_terminal_precision": float(universe["terminal_rebound_label"].mean()),
        "natural_quality_precision": float(universe["quality_bottom_label"].mean()),
        "actual_up_rate": float((signal["future_ret_15d"] > 0).mean()) if count else None,
        "avg_future_ret_15d": float(signal["future_ret_15d"].mean()) if count else None,
        "avg_future_mfe_15d": float(signal["future_mfe_15d"].mean()) if count else None,
        "avg_future_mae_15d": float(signal["future_mae_15d"].mean()) if count else None,
        "continuation_risk_rate": float(signal["continuation_risk_label"].mean()) if count else None,
        "natural_avg_future_ret_15d": float(universe["future_ret_15d"].mean()),
        "natural_continuation_risk_rate": float(universe["continuation_risk_label"].mean()),
    }


def evaluate_bottom_risk_gate(
    start_year: int | None = None,
    end_year: int | None = None,
    models: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = get_config("bottom_model.yaml")
    model_cfg = cfg["model"]
    data = load_bottom_dataset()
    if data.empty:
        raise RuntimeError("Bottom dataset is empty. Run build_bottom_dataset first.")
    features = load_bottom_features(data)
    required = ["terminal_rebound_label", "quality_bottom_label", "continuation_risk_label", "future_ret_15d"]
    complete = data.dropna(subset=required).copy()
    horizon = int(model_cfg["horizon"])
    start_year = start_year or int(model_cfg["evaluation_start_year"])
    end_year = end_year or int(complete["trade_date"].dt.year.max())
    models = models or ["logistic", "lightgbm"]
    opportunity_thresholds = [float(value) for value in model_cfg["probability_thresholds"]]
    risk_thresholds = [float(value) for value in model_cfg["risk_thresholds"]]
    train_start = pd.Timestamp(str(model_cfg["train_start"]))
    min_train_rows = int(model_cfg["min_train_rows"])
    random_state = int(model_cfg["random_state"])
    latest_complete_date = complete["trade_date"].max()
    detail_parts = []
    summary_rows = []

    for year in range(start_year, end_year + 1):
        test_start = pd.Timestamp(f"{year}-01-01")
        test_end = pd.Timestamp(f"{year}-12-31")
        year_rows = complete[
            (complete["trade_date"] >= test_start) & (complete["trade_date"] <= test_end)
        ]
        if year_rows.empty:
            continue
        first_position = int(year_rows["trade_pos"].min())
        train = complete[
            (complete["trade_date"] >= train_start)
            & (complete["trade_pos"] <= first_position - horizon - 1)
            & (complete["is_candidate"] == 1)
        ].copy()
        test = year_rows[year_rows["is_candidate"] == 1].copy()
        if len(train) < min_train_rows or test.empty:
            continue
        for model_name in models:
            opportunity_model, opportunity_features = _fit_model(
                model_name, train, features, "terminal_rebound_label", random_state
            )
            risk_model, risk_features = _fit_model(
                model_name, train, features, "continuation_risk_label", random_state
            )
            base = test[
                [
                    "trade_date", "trade_pos", "terminal_rebound_label", "quality_bottom_label",
                    "continuation_risk_label", "future_ret_15d", "future_mfe_15d", "future_mae_15d",
                ]
            ].copy()
            base["opportunity_proba"] = _positive_probability(
                opportunity_model, test, opportunity_features
            )
            base["risk_proba"] = _positive_probability(risk_model, test, risk_features)
            base["test_year"] = year
            base["year_complete"] = latest_complete_date >= test_end
            base["model"] = model_name
            detail_parts.append(base)
            for opportunity_threshold in opportunity_thresholds:
                for risk_threshold in risk_thresholds:
                    universe = base.copy()
                    universe["opportunity_threshold"] = opportunity_threshold
                    universe["risk_threshold"] = risk_threshold
                    signal = universe[
                        (universe["opportunity_proba"] >= opportunity_threshold)
                        & (universe["risk_proba"] <= risk_threshold)
                    ]
                    summary_rows.append(_metrics(universe, signal, "all_signals"))
                    summary_rows.append(
                        _metrics(universe, _non_overlapping(signal, horizon), "non_overlapping")
                    )

    if not detail_parts:
        raise RuntimeError("No bottom risk-gate folds were produced.")
    details = pd.concat(detail_parts, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    for frame, key in [(details, "risk_gate_detail"), (summary, "risk_gate_summary")]:
        path = project_path(str(cfg["outputs"][key]))
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    return details, summary


def aggregate_risk_gate(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = ["model", "opportunity_threshold", "risk_threshold", "sample_mode"]
    for keys, part in summary.groupby(groups, sort=True):
        signal_count = int(part["signal_count"].sum())
        candidate_rows = int(part["candidate_rows"].sum())

        def weighted(column: str, weight_column: str) -> float | None:
            valid = part[column].notna() & (part[weight_column] > 0)
            if not valid.any():
                return None
            return float(
                (part.loc[valid, column] * part.loc[valid, weight_column]).sum()
                / part.loc[valid, weight_column].sum()
            )

        row = {
            **dict(zip(groups, keys)),
            "evaluated_years": int(part["test_year"].nunique()),
            "complete_years": int(part.loc[part["year_complete"], "test_year"].nunique()),
            "candidate_rows": candidate_rows,
            "signal_count": signal_count,
            "coverage": signal_count / candidate_rows,
        }
        for column in [
            "terminal_precision", "quality_precision", "actual_up_rate", "avg_future_ret_15d",
            "avg_future_mfe_15d", "avg_future_mae_15d", "continuation_risk_rate",
        ]:
            row[column] = weighted(column, "signal_count")
        for column in [
            "natural_terminal_precision", "natural_quality_precision",
            "natural_avg_future_ret_15d", "natural_continuation_risk_rate",
        ]:
            row[column] = weighted(column, "candidate_rows")
        rows.append(row)
    result = pd.DataFrame(rows)
    result["quality_precision_lift"] = result["quality_precision"] - result["natural_quality_precision"]
    result["avg_return_lift"] = result["avg_future_ret_15d"] - result["natural_avg_future_ret_15d"]
    result["risk_rate_reduction"] = result["natural_continuation_risk_rate"] - result["continuation_risk_rate"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate opportunity model plus continuation-risk veto.")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--models", default="logistic,lightgbm")
    args = parser.parse_args()
    _, summary = evaluate_bottom_risk_gate(
        start_year=args.start_year,
        end_year=args.end_year,
        models=[item.strip() for item in args.models.split(",") if item.strip()],
    )
    aggregate = aggregate_risk_gate(summary)
    focus = aggregate[aggregate["sample_mode"] == "non_overlapping"].sort_values(
        ["avg_return_lift", "quality_precision_lift"], ascending=False
    )
    print(focus.to_string(index=False))


if __name__ == "__main__":
    main()
