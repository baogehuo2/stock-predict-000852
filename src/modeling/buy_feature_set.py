from __future__ import annotations

import json

import pandas as pd

from src.common.config import project_path


DEFAULT_BUY_FEATURE_VERSION = "buy-features-v1.0"
DEFAULT_BUY_FEATURE_MANIFEST = "config/buy_features_v1.json"


def load_buy_feature_manifest(path: str = DEFAULT_BUY_FEATURE_MANIFEST) -> list[str]:
    manifest_path = project_path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Buy feature manifest not found: {manifest_path}")
    features = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(features, list) or not features or not all(isinstance(item, str) for item in features):
        raise ValueError(f"Buy feature manifest must be a non-empty string list: {manifest_path}")
    if len(features) != len(set(features)):
        raise ValueError(f"Buy feature manifest contains duplicate columns: {manifest_path}")
    return features


def select_buy_features(data: pd.DataFrame, buy_config: dict | None = None) -> list[str]:
    buy_config = buy_config or {}
    manifest = str(buy_config.get("feature_manifest", DEFAULT_BUY_FEATURE_MANIFEST))
    features = load_buy_feature_manifest(manifest)
    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise RuntimeError(f"Buy feature manifest columns are missing from dataset: {missing}")
    return features
