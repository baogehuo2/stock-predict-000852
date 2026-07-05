from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import project_path
from src.modeling.train_manual_turning_models import load_manual_model_frame


LABEL_COLUMNS = [
    "is_manual_buy_window",
    "is_manual_sell_window",
    "quality_bottom_label",
    "continuation_risk_label",
    "terminal_rebound_label",
    "path_rebound_label",
]


def _label_summary(data: pd.DataFrame, group_col: str, label_col: str) -> pd.DataFrame:
    rows = []
    for group_value, part in data.groupby(group_col, sort=True):
        if label_col not in part:
            continue
        label = pd.to_numeric(part[label_col], errors="coerce")
        valid = label.dropna()
        positives = int(valid.sum()) if len(valid) else 0
        rows.append(
            {
                group_col: group_value,
                "label": label_col,
                "rows": int(len(valid)),
                "positive_count": positives,
                "negative_count": int(len(valid) - positives),
                "positive_rate": float(positives / len(valid)) if len(valid) else None,
            }
        )
    return pd.DataFrame(rows)


def audit_manual_turning_samples(
    target_index: str = "000852",
    output: str = "data/reports/manual_turning_sample_distribution_000852.csv",
) -> Path:
    data = load_manual_model_frame(target_index)
    data["year"] = pd.to_datetime(data["trade_date"]).dt.year.astype(int)
    frames = []
    for label_col in LABEL_COLUMNS:
        if label_col not in data:
            continue
        frames.append(_label_summary(data, "year", label_col))
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    output_path = project_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit sparse positive labels for manual turning models.")
    parser.add_argument("--target-index", default="000852")
    parser.add_argument("--output", default="data/reports/manual_turning_sample_distribution_000852.csv")
    args = parser.parse_args()
    print(audit_manual_turning_samples(args.target_index, args.output))


if __name__ == "__main__":
    main()
