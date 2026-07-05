from __future__ import annotations

import json

import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import get_database_name, read_sql


def assert_bottom_weekly_database() -> None:
    cfg = get_config("bottom_weekly_model.yaml")
    expected = str(cfg["database"]["expected_name"])
    actual = get_database_name()
    if actual != expected:
        raise RuntimeError(
            f"Weekly bottom workflow requires database {expected}, but config/db.yaml points to {actual}."
        )


def load_bottom_weekly_dataset() -> pd.DataFrame:
    assert_bottom_weekly_database()
    cfg = get_config("bottom_weekly_model.yaml")
    table = str(cfg["outputs"]["dataset_table"])
    data = read_sql(f"SELECT * FROM `{table}` ORDER BY week_end_date")
    if data.empty:
        return data
    for column in ["week_start_date", "week_end_date"]:
        data[column] = pd.to_datetime(data[column])
    feature_dicts = data["feature_json"].map(
        lambda value: json.loads(value) if isinstance(value, str) else (value or {})
    )
    features = pd.json_normalize(feature_dicts)
    return pd.concat([data.drop(columns=["feature_json"]), features], axis=1)


def load_bottom_weekly_features(data: pd.DataFrame, manifest_path: str | None = None) -> list[str]:
    cfg = get_config("bottom_weekly_model.yaml")
    path = project_path(manifest_path or str(cfg["features"]["manifest"]))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not all(isinstance(item, str) for item in manifest):
        raise ValueError(f"Weekly bottom feature manifest must be a string list: {path}")
    missing = [feature for feature in manifest if feature not in data.columns]
    if missing:
        raise RuntimeError(f"Weekly bottom dataset is missing manifest features: {missing}")
    augmented = [
        column
        for column in data.columns
        if column.startswith("f_event_week_") and column not in manifest and data[column].notna().any()
    ]
    return [*manifest, *augmented]
