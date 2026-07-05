from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import get_database_name, read_sql

EVENT_V2_WINDOWS = [3, 5, 10, 20]
EVENT_V2_GROUPS = {
    "policy": ("event_type", "政策"),
    "macro": ("event_type", "宏观"),
    "liquidity": ("event_type", "流动性"),
    "regulation": ("event_type", "监管"),
    "industry": ("event_type", "产业"),
    "diplomacy": ("event_type", "外交"),
    "small_cap": ("affected_style", "小盘"),
    "growth": ("affected_style", "成长"),
}


def assert_bottom_database() -> None:
    cfg = get_config("bottom_model.yaml")
    expected = str(cfg["database"]["expected_name"])
    actual = get_database_name()
    if actual != expected:
        raise RuntimeError(
            f"Bottom workflow requires database {expected}, but config/db.yaml points to {actual}."
        )


def load_bottom_dataset() -> pd.DataFrame:
    assert_bottom_database()
    cfg = get_config("bottom_model.yaml")
    table = str(cfg["outputs"]["dataset_table"])
    data = read_sql(f"SELECT * FROM `{table}` ORDER BY trade_date")
    if data.empty:
        return data
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    feature_dicts = data["feature_json"].map(
        lambda value: json.loads(value) if isinstance(value, str) else (value or {})
    )
    features = pd.json_normalize(feature_dicts)
    return pd.concat([data.drop(columns=["feature_json"]), features], axis=1)


def _json_feature_frame(table: str, target_index: str) -> pd.DataFrame:
    data = read_sql(
        f"SELECT trade_date,index_code,feature_json FROM `{table}` "
        "WHERE index_code=:target_index ORDER BY trade_date",
        {"target_index": target_index},
    )
    if data.empty:
        return data
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    feature_dicts = data["feature_json"].map(
        lambda value: json.loads(value) if isinstance(value, str) else (value or {})
    )
    features = pd.json_normalize(feature_dicts).add_prefix("f_")
    return pd.concat([data.drop(columns=["feature_json"]), features], axis=1)


def _buy_v1_manifest() -> list[str]:
    cfg = get_config("bottom_model.yaml")
    path = project_path(
        str(cfg.get("feature_augments", {}).get("buy_v1_manifest", "config/buy_features_v1.json"))
    )
    if not path.exists():
        return []
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not all(isinstance(item, str) for item in manifest):
        raise ValueError(f"Buy feature manifest must be a string list: {path}")
    return manifest


def _augment_with_buy_v1_features(data: pd.DataFrame) -> pd.DataFrame:
    cfg = get_config("bottom_model.yaml")
    if not bool(cfg.get("feature_augments", {}).get("use_buy_v1_features", True)):
        return data

    index_codes = data["index_code"].dropna().astype(str).unique().tolist() if "index_code" in data else []
    target_index = index_codes[0] if len(index_codes) == 1 else str(cfg["model"]["target_index"])
    buy = _json_feature_frame("model_dataset_daily", target_index)
    if buy.empty:
        return data

    manifest = _buy_v1_manifest()
    selected = [
        feature
        for feature in manifest
        if feature in buy.columns and feature not in data.columns and buy[feature].notna().any()
    ]
    if not selected:
        return data
    return data.merge(
        buy[["trade_date", "index_code", *selected]],
        on=["trade_date", "index_code"],
        how="left",
        validate="one_to_one",
    )


def _contains(value: object, keyword: str) -> bool:
    if value is None or pd.isna(value):
        return False
    return keyword in str(value)


def _direction_score(events: pd.DataFrame) -> pd.Series:
    direction = events["impact_direction"].fillna("").astype(str)
    strength = pd.to_numeric(events["impact_strength"], errors="coerce").fillna(1.0).clip(lower=0)
    score = pd.Series(0.0, index=events.index)
    score.loc[direction.str.contains("利多|利好|多头", regex=True)] = strength
    score.loc[direction.str.contains("利空", regex=True)] = -strength
    return score


def _aggregate_event_part(part: pd.DataFrame, prefix: str) -> dict[str, float]:
    score = pd.to_numeric(part["event_score"], errors="coerce").fillna(0.0)
    strength = pd.to_numeric(part["impact_strength"], errors="coerce").fillna(0.0)
    direction = pd.to_numeric(part["direction_score"], errors="coerce").fillna(0.0)
    return {
        f"{prefix}_count": float(len(part)),
        f"{prefix}_score_sum": float(score.sum()),
        f"{prefix}_score_mean": float(score.mean()) if len(part) else 0.0,
        f"{prefix}_positive_count": float((direction > 0).sum()),
        f"{prefix}_negative_count": float((direction < 0).sum()),
        f"{prefix}_direction_sum": float(direction.sum()),
        f"{prefix}_max_strength": float(strength.max()) if len(part) else 0.0,
    }


def _event_v2_daily_features(target_dates: pd.Series) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(target_dates).sort_values().unique())
    result = pd.DataFrame({"trade_date": dates})
    events = read_sql(
        "SELECT trade_date,event_type,event_stage,expectation_level,surprise_level,"
        "affected_style,impact_direction,impact_strength,event_score "
        "FROM event_daily ORDER BY trade_date"
    )
    if events.empty:
        return result

    events["trade_date"] = pd.to_datetime(events["trade_date"])
    events["event_score"] = pd.to_numeric(events["event_score"], errors="coerce").fillna(0.0)
    events["impact_strength"] = pd.to_numeric(events["impact_strength"], errors="coerce").fillna(0.0)
    events["direction_score"] = _direction_score(events)

    rows = []
    for trade_date, part in events.groupby("trade_date", sort=True):
        record = {"trade_date": pd.Timestamp(trade_date).normalize()}
        record.update(_aggregate_event_part(part, "f_m2_event_all"))
        for group, (column, keyword) in EVENT_V2_GROUPS.items():
            group_part = part[part[column].map(lambda value, key=keyword: _contains(value, key))]
            record.update(_aggregate_event_part(group_part, f"f_m2_event_{group}"))
        surprise_up = part[part["surprise_level"].map(lambda value: _contains(value, "超预期"))]
        surprise_down = part[part["surprise_level"].map(lambda value: _contains(value, "不及预期"))]
        expectation_hot = part[part["expectation_level"].map(lambda value: _contains(value, "预期充分"))]
        record.update(_aggregate_event_part(surprise_up, "f_m2_event_surprise_up"))
        record.update(_aggregate_event_part(surprise_down, "f_m2_event_surprise_down"))
        record.update(_aggregate_event_part(expectation_hot, "f_m2_event_expectation_hot"))
        rows.append(record)

    daily = pd.DataFrame(rows)
    result = result.merge(daily, on="trade_date", how="left").fillna(0.0)
    feature_cols = [column for column in result.columns if column.startswith("f_m2_event_")]
    result = result.sort_values("trade_date").reset_index(drop=True)
    rolling_features: dict[str, pd.Series] = {}
    for window in EVENT_V2_WINDOWS:
        for column in feature_cols:
            values = pd.to_numeric(result[column], errors="coerce").fillna(0.0)
            if column.endswith("_max_strength"):
                rolling_features[f"{column}_roll_{window}d"] = values.rolling(window, min_periods=1).max()
            else:
                rolling_features[f"{column}_roll_{window}d"] = values.rolling(window, min_periods=1).sum()
    if rolling_features:
        result = pd.concat([result, pd.DataFrame(rolling_features, index=result.index)], axis=1)
    result = result.replace([np.inf, -np.inf], np.nan).copy()
    return result


def _augment_with_event_v2_features(data: pd.DataFrame) -> pd.DataFrame:
    cfg = get_config("bottom_model.yaml")
    if not bool(cfg.get("feature_augments", {}).get("use_event_v2_features", True)):
        return data

    event_features = _event_v2_daily_features(data["trade_date"])
    feature_cols = [
        column
        for column in event_features.columns
        if column.startswith("f_m2_event_") and column not in data.columns and event_features[column].notna().any()
    ]
    if not feature_cols:
        return data
    return data.merge(
        event_features[["trade_date", *feature_cols]],
        on="trade_date",
        how="left",
        validate="many_to_one",
    )


def augment_bottom_features(data: pd.DataFrame) -> pd.DataFrame:
    data = _augment_with_buy_v1_features(data)
    data = _augment_with_event_v2_features(data)
    return data


def load_bottom_features(data: pd.DataFrame, manifest_path: str | None = None) -> list[str]:
    cfg = get_config("bottom_model.yaml")
    path = project_path(manifest_path or str(cfg["features"]["manifest"]))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not all(isinstance(item, str) for item in manifest):
        raise ValueError(f"Bottom feature manifest must be a string list: {path}")
    missing = [feature for feature in manifest if feature not in data.columns]
    if missing:
        raise RuntimeError(f"Bottom dataset is missing manifest features: {missing}")
    augmented = []
    buy_manifest = _buy_v1_manifest()
    for feature in buy_manifest:
        if feature in data.columns and feature not in manifest and data[feature].notna().any():
            augmented.append(feature)
    augmented.extend(
        column
        for column in data.columns
        if column.startswith("f_m2_event_") and column not in manifest and column not in augmented and data[column].notna().any()
    )
    return [*manifest, *augmented]
