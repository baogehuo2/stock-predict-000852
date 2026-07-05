from __future__ import annotations

import pandas as pd

from src.common.config import get_config, project_path
from src.modeling.bottom_data import load_bottom_dataset


RULES = {
    "week_close_below_boll_lower": lambda data: data["f_week_close_below_boll_lower"] == 1,
    "week_low_below_boll_lower": lambda data: data["f_week_low_below_boll_lower"] == 1,
    "week_boll_lower_reclaim": lambda data: data["f_week_boll_lower_reclaim"] == 1,
    "week_uptrend_pullback_below_mid": lambda data: data["f_week_uptrend_pullback_below_mid"] == 1,
    "week_kdj_golden_cross": lambda data: data["f_week_kdj_golden_cross"] == 1,
    "week_oversold_kdj_golden_cross": lambda data: data["f_week_oversold_kdj_golden_cross"] == 1,
    "week_down_volume_expand": lambda data: data["f_week_down_volume_expand"] == 1,
    "week_lower_break_or_reclaim": lambda data: (
        (data["f_week_close_below_boll_lower"] == 1)
        | (data["f_week_boll_lower_reclaim"] == 1)
    ),
    "week_lower_zone_and_kdj_cross": lambda data: (
        (data["f_week_low_below_boll_lower"] == 1)
        & (data["f_week_kdj_golden_cross"] == 1)
    ),
    "week_confirmation_score_ge_2": lambda data: data["f_week_bottom_confirmation_score"] >= 2,
}


def evaluate_bottom_weekly_rules() -> pd.DataFrame:
    data = load_bottom_dataset()
    required = ["quality_bottom_label", "future_ret_15d", "continuation_risk_label"]
    candidates = data[(data["is_candidate"] == 1)].dropna(subset=required).copy()
    candidates["year"] = candidates["trade_date"].dt.year
    rows = []
    for rule_name, rule in RULES.items():
        mask = rule(candidates).fillna(False)
        for year_name, part_mask in [
            ("all", mask),
            *[(str(year), mask & (candidates["year"] == year)) for year in sorted(candidates["year"].unique())],
        ]:
            selected = candidates[part_mask]
            if selected.empty:
                continue
            rows.append(
                {
                    "rule": rule_name,
                    "year": year_name,
                    "signal_count": len(selected),
                    "quality_precision": float(selected["quality_bottom_label"].mean()),
                    "actual_up_rate": float((selected["future_ret_15d"] > 0).mean()),
                    "avg_future_ret_15d": float(selected["future_ret_15d"].mean()),
                    "avg_future_mfe_15d": float(selected["future_mfe_15d"].mean()),
                    "avg_future_mae_15d": float(selected["future_mae_15d"].mean()),
                    "continuation_risk_rate": float(selected["continuation_risk_label"].mean()),
                    "candidate_quality_baseline": float(candidates["quality_bottom_label"].mean()),
                    "candidate_return_baseline": float(candidates["future_ret_15d"].mean()),
                }
            )
    report = pd.DataFrame(rows)
    cfg = get_config("bottom_model.yaml")
    path = project_path("data/reports/bottom_weekly_rule_diagnostics_v0_2.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False, encoding="utf-8-sig")
    return report


def main() -> None:
    report = evaluate_bottom_weekly_rules()
    print(
        report[report["year"] == "all"]
        .sort_values(["avg_future_ret_15d", "quality_precision"], ascending=False)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
