from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import get_database_name, read_sql
from src.common.logger import get_logger
from src.modeling.walk_forward_bottom_weekly_lgbm import (
    SIDE_CONFIG,
    _fit_model,
    _tune_model,
)


logger = get_logger(__name__)


def _load_weekly_dataset(cfg: dict) -> pd.DataFrame:
    expected = str(cfg["database"]["expected_name"])
    actual = get_database_name()
    if actual != expected:
        raise RuntimeError(
            f"Weekly bottom final training requires database {expected}, but config/db.yaml points to {actual}."
        )
    table = str(cfg["outputs"]["dataset_table"])
    data = read_sql(f"SELECT * FROM `{table}` ORDER BY week_end_date")
    if data.empty:
        raise RuntimeError(f"{table} is empty. Run build_bottom_weekly_dataset first.")
    for column in ["week_start_date", "week_end_date"]:
        data[column] = pd.to_datetime(data[column])
    feature_dicts = data["feature_json"].map(
        lambda value: json.loads(value) if isinstance(value, str) else (value or {})
    )
    features = pd.json_normalize(feature_dicts)
    return pd.concat([data.drop(columns=["feature_json"]), features], axis=1)


def _load_features(cfg: dict, data: pd.DataFrame) -> list[str]:
    path = project_path(str(cfg["features"]["manifest"]))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not all(isinstance(item, str) for item in manifest):
        raise ValueError(f"Weekly bottom feature manifest must be a string list: {path}")
    missing = [feature for feature in manifest if feature not in data.columns]
    if missing:
        raise RuntimeError(f"Weekly bottom dataset is missing manifest features: {missing}")
    event_features = [
        column
        for column in data.columns
        if column.startswith("f_event_week_") and column not in manifest and data[column].notna().any()
    ]
    return [*manifest, *event_features]


def _model_file_name(model_tag: str, label_mode: str, side: str, model_name: str) -> str:
    safe_label = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in label_mode)
    return f"{model_tag}_{safe_label}_{side}_{model_name}.joblib"


def train_bottom_weekly_final_models(
    config_file: str = "bottom_weekly_model.yaml",
    models: list[str] | None = None,
    sides: list[str] | None = None,
    use_candidates_only: bool = True,
) -> pd.DataFrame:
    cfg = get_config(config_file)
    model_cfg = cfg["model"]
    data = _load_weekly_dataset(cfg)
    features = _load_features(cfg, data)
    complete = data.dropna(subset=["future_ret_3w"]).copy()
    if complete.empty:
        raise RuntimeError("Weekly bottom dataset has no complete future_ret_3w rows.")

    train_start = pd.Timestamp(str(model_cfg["train_start"]))
    train = complete[complete["week_end_date"] >= train_start].sort_values("week_pos").copy()
    if train.empty:
        raise RuntimeError(f"No weekly rows on or after train_start={train_start.date()}.")

    model_names = models or [str(value) for value in model_cfg.get("model_kinds", ["logistic", "lightgbm"])]
    side_names = sides or ["bottom", "top"]
    invalid_sides = sorted(set(side_names) - set(SIDE_CONFIG))
    if invalid_sides:
        raise ValueError(f"Unsupported sides: {invalid_sides}")

    thresholds = [float(value) for value in model_cfg["probability_thresholds"]]
    horizon_weeks = int(model_cfg["horizon_weeks"])
    random_state = int(model_cfg["random_state"])
    min_train_rows = int(model_cfg["min_train_rows"])
    enable_tuning = bool(model_cfg.get("enable_tuning", False))
    validation_fraction = float(model_cfg.get("tuning_validation_fraction", 0.20))
    min_validation_rows = int(model_cfg.get("tuning_min_validation_rows", 20))
    min_signal_count = int(model_cfg.get("tuning_min_signal_count", 2))

    output_dir = project_path(str(cfg["outputs"]["final_model_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    model_tag = str(model_cfg["model_tag"])
    label_mode = str(cfg["manual_labels"]["label_mode"])
    manifest_rows = []

    for side in side_names:
        label_col = SIDE_CONFIG[side]["label"]
        candidate_col = SIDE_CONFIG[side]["candidate"]
        side_train = train.copy()
        if use_candidates_only:
            side_train = side_train[side_train[candidate_col] == 1].copy()
        if len(side_train) < min_train_rows:
            raise RuntimeError(
                f"Not enough rows for final weekly {side} model: {len(side_train)} < {min_train_rows}"
            )
        if side_train[label_col].astype(int).nunique() < 2:
            raise RuntimeError(f"{label_col} has only one class in final training data.")

        for model_name in model_names:
            selected_params: dict = {}
            tuning_status = "disabled"
            if enable_tuning:
                selected_params, tuning_info = _tune_model(
                    model_name,
                    side_train,
                    features,
                    label_col,
                    random_state,
                    thresholds,
                    horizon_weeks,
                    validation_fraction,
                    min_validation_rows,
                    min_signal_count,
                )
                tuning_status = str(tuning_info.get("tuning_status", "unknown"))
            fitted, usable = _fit_model(
                model_name,
                side_train,
                features,
                label_col,
                random_state,
                selected_params,
            )
            path = output_dir / _model_file_name(model_tag, label_mode, side, model_name)
            bundle = {
                "classifier": fitted,
                "features": usable,
                "model": model_name,
                "side": side,
                "label_col": label_col,
                "candidate_col": candidate_col,
                "model_version": str(model_cfg["model_version"]),
                "feature_version": str(model_cfg["feature_version"]),
                "model_tag": model_tag,
                "label_mode": label_mode,
                "horizon_weeks": horizon_weeks,
                "use_candidates_only": bool(use_candidates_only),
                "tuned_params": selected_params,
                "tuning_status": tuning_status,
                "trained_from": side_train["week_end_date"].min().strftime("%Y-%m-%d"),
                "trained_through": side_train["week_end_date"].max().strftime("%Y-%m-%d"),
                "train_rows": int(len(side_train)),
                "positive_rows": int(side_train[label_col].astype(int).sum()),
            }
            joblib.dump(bundle, path)
            manifest_rows.append(
                {
                    "config_file": config_file,
                    "model_file": str(path.relative_to(project_path("."))),
                    "model": model_name,
                    "side": side,
                    "label_col": label_col,
                    "candidate_col": candidate_col,
                    "model_version": bundle["model_version"],
                    "feature_version": bundle["feature_version"],
                    "model_tag": model_tag,
                    "label_mode": label_mode,
                    "horizon_weeks": horizon_weeks,
                    "use_candidates_only": bool(use_candidates_only),
                    "tuning_status": tuning_status,
                    "tuned_params_json": json.dumps(selected_params, ensure_ascii=False, sort_keys=True),
                    "trained_from": bundle["trained_from"],
                    "trained_through": bundle["trained_through"],
                    "train_rows": bundle["train_rows"],
                    "positive_rows": bundle["positive_rows"],
                    "usable_features": len(usable),
                }
            )
            logger.info(
                "trained final weekly model path=%s side=%s model=%s rows=%s positives=%s features=%s",
                path,
                side,
                model_name,
                bundle["train_rows"],
                bundle["positive_rows"],
                len(usable),
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = project_path(str(cfg["outputs"]["final_model_manifest"]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train final full-history weekly top/bottom models.")
    parser.add_argument("--config-file", default="bottom_weekly_model.yaml")
    parser.add_argument("--models")
    parser.add_argument("--sides", default="bottom,top")
    parser.add_argument("--all-weeks", action="store_true")
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()] if args.models else None
    sides = [item.strip() for item in args.sides.split(",") if item.strip()]
    manifest = train_bottom_weekly_final_models(
        config_file=args.config_file,
        models=models,
        sides=sides,
        use_candidates_only=not args.all_weeks,
    )
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
