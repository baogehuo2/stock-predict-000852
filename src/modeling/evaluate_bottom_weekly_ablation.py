from __future__ import annotations

import pandas as pd

from src.common.config import get_config, project_path
from src.modeling.walk_forward_bottom_lgbm import walk_forward_bottom_evaluation


VARIANTS = {
    "daily_v0_1": "config/bottom_features_v0_1.json",
    "daily_weekly_v0_2": "config/bottom_features_v0_2.json",
}


def evaluate_bottom_weekly_ablation() -> pd.DataFrame:
    reports = []
    for variant, manifest in VARIANTS.items():
        _, summary, aggregate = walk_forward_bottom_evaluation(
            label_col="quality_bottom_label",
            feature_manifest=manifest,
            write_reports=False,
        )
        aggregate.insert(0, "variant", variant)
        reports.append(aggregate)
    report = pd.concat(reports, ignore_index=True)
    cfg = get_config("bottom_model.yaml")
    path = project_path(str(cfg["outputs"]["weekly_ablation"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False, encoding="utf-8-sig")
    return report


def main() -> None:
    report = evaluate_bottom_weekly_ablation()
    focus = report[report["sample_mode"] == "non_overlapping"].sort_values(
        ["variant", "model", "threshold"]
    )
    print(
        focus[
            [
                "variant", "model", "threshold", "signal_count", "precision",
                "natural_precision", "precision_lift", "avg_future_ret_15d",
                "avg_return_lift", "continuation_risk_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
