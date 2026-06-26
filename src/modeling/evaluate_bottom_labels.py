from __future__ import annotations

import argparse

import pandas as pd

from src.common.config import get_config, project_path
from src.modeling.bottom_data import load_bottom_dataset


LABELS = ["terminal_rebound_label", "path_rebound_label", "quality_bottom_label"]


def evaluate_bottom_labels(output_csv: str | None = None) -> pd.DataFrame:
    cfg = get_config("bottom_model.yaml")
    data = load_bottom_dataset()
    if data.empty:
        raise RuntimeError("Bottom dataset is empty. Run build_bottom_dataset first.")
    complete = data.dropna(subset=["future_ret_15d", "future_mfe_15d", "future_mae_15d"]).copy()
    complete["year"] = complete["trade_date"].dt.year
    rows = []
    for scope, scoped in [("all_days", complete), ("candidates", complete[complete["is_candidate"] == 1])]:
        for year_name, part in [("all", scoped), *[(str(year), group) for year, group in scoped.groupby("year")]]:
            if part.empty:
                continue
            for label in LABELS:
                positive = part[label].astype(int) == 1
                rows.append(
                    {
                        "scope": scope,
                        "year": year_name,
                        "label": label,
                        "rows": len(part),
                        "positive_count": int(positive.sum()),
                        "positive_rate": float(positive.mean()),
                        "avg_future_ret_15d": float(part["future_ret_15d"].mean()),
                        "avg_future_mfe_15d": float(part["future_mfe_15d"].mean()),
                        "avg_future_mae_15d": float(part["future_mae_15d"].mean()),
                        "continuation_risk_rate": float(part["continuation_risk_label"].mean()),
                    }
                )
    report = pd.DataFrame(rows)
    output = output_csv or str(cfg["outputs"]["label_report"])
    path = project_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False, encoding="utf-8-sig")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare bottom-fishing label baselines.")
    parser.add_argument("--output-csv")
    args = parser.parse_args()
    report = evaluate_bottom_labels(args.output_csv)
    print(report[(report["scope"] == "candidates") & (report["year"] == "all")].to_string(index=False))


if __name__ == "__main__":
    main()
