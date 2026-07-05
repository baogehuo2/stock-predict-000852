from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common.config import get_config, project_path
from src.common.logger import get_logger
from src.modeling.walk_forward_bottom_weekly_lgbm import _non_overlapping


logger = get_logger(__name__)


@dataclass(frozen=True)
class Policy:
    side: str
    enabled_models: tuple[str, ...]
    thresholds: dict[str, float]
    min_votes: int


def _read_details() -> pd.DataFrame:
    cfg = get_config("bottom_weekly_model.yaml")
    path = project_path(str(cfg["outputs"]["walk_forward_detail"]))
    if not path.exists():
        raise FileNotFoundError(f"weekly detail report not found: {path}")
    data = pd.read_csv(path, parse_dates=["week_start_date", "week_end_date"])
    data["turning_proba"] = pd.to_numeric(data["turning_proba"], errors="coerce")
    data["future_ret_3w"] = pd.to_numeric(data["future_ret_3w"], errors="coerce")
    data["week_pos"] = pd.to_numeric(data["week_pos"], errors="coerce").astype(int)
    return data


def _side_frame(details: pd.DataFrame, side: str) -> pd.DataFrame:
    label_col = "manual_weak_bottom_label" if side == "bottom" else "manual_weak_top_label"
    part = details[details["side"] == side].copy()
    base_cols = [
        "week_start_date",
        "week_end_date",
        "week_pos",
        "test_year",
        "future_ret_3w",
        label_col,
    ]
    base = part[base_cols].drop_duplicates("week_end_date").copy()
    proba = part.pivot_table(
        index="week_end_date",
        columns="model",
        values="turning_proba",
        aggfunc="max",
    ).reset_index()
    merged = base.merge(proba, on="week_end_date", how="left")
    return merged.sort_values("week_pos").reset_index(drop=True)


def _evaluate_policy(frame: pd.DataFrame, policy: Policy, horizon_weeks: int) -> dict:
    label_col = "manual_weak_bottom_label" if policy.side == "bottom" else "manual_weak_top_label"
    votes = np.zeros(len(frame), dtype=int)
    score_parts = []
    for model in policy.enabled_models:
        proba = pd.to_numeric(frame[model], errors="coerce").fillna(0.0)
        passed = proba >= float(policy.thresholds[model])
        votes += passed.astype(int).to_numpy()
        score_parts.append(proba.to_numpy())
    scored = frame[["week_start_date", "week_end_date", "week_pos", "future_ret_3w", label_col]].copy()
    scored["vote_count"] = votes
    scored["turning_proba"] = np.mean(score_parts, axis=0) if score_parts else 0.0
    signal = _non_overlapping(scored[scored["vote_count"] >= policy.min_votes], horizon_weeks)
    signal_count = int(len(signal))
    natural_precision = float(scored[label_col].mean())
    natural_ret = float(scored["future_ret_3w"].mean())
    precision = float(signal[label_col].mean()) if signal_count else np.nan
    avg_ret = float(signal["future_ret_3w"].mean()) if signal_count else np.nan
    return {
        "side": policy.side,
        "enabled_models": ",".join(policy.enabled_models),
        "thresholds_json": json.dumps(policy.thresholds, sort_keys=True),
        "min_votes": int(policy.min_votes),
        "candidate_rows": int(len(scored)),
        "signal_count": signal_count,
        "precision": precision,
        "natural_precision": natural_precision,
        "precision_lift": precision - natural_precision if signal_count else np.nan,
        "avg_future_ret_3w": avg_ret,
        "natural_avg_future_ret_3w": natural_ret,
        "avg_return_lift": avg_ret - natural_ret if signal_count else np.nan,
    }


def _score_row(row: dict, side: str) -> float:
    signal_count = int(row["signal_count"])
    if signal_count <= 0 or pd.isna(row["precision"]):
        return -1e9
    precision_lift = float(row["precision_lift"])
    return_lift = float(row["avg_return_lift"])
    signed_return_lift = return_lift if side == "bottom" else -return_lift
    min_signal_count = 8
    min_precision_lift = 0.00
    min_signed_return_lift = -0.005
    penalty = 0.0
    if signal_count < min_signal_count:
        penalty += (min_signal_count - signal_count) * 12.0
    if precision_lift < min_precision_lift:
        penalty += (min_precision_lift - precision_lift) * 70.0
    if signed_return_lift < min_signed_return_lift:
        penalty += (min_signed_return_lift - signed_return_lift) * 100.0
    return (
        signal_count * 6.0
        + max(precision_lift, -0.5) * 35.0
        + signed_return_lift * 250.0
        - penalty
    )


def _random_policy(
    side: str,
    models: list[str],
    threshold_grid: list[float],
    rng: np.random.Generator,
) -> Policy:
    enabled = tuple(model for model in models if rng.random() < 0.65)
    if not enabled:
        enabled = (str(rng.choice(models)),)
    thresholds = {model: float(rng.choice(threshold_grid)) for model in enabled}
    min_votes = int(rng.integers(1, len(enabled) + 1))
    return Policy(side=side, enabled_models=enabled, thresholds=thresholds, min_votes=min_votes)


def _mutate_policy(
    policy: Policy,
    models: list[str],
    threshold_grid: list[float],
    rng: np.random.Generator,
) -> Policy:
    enabled = set(policy.enabled_models)
    thresholds = dict(policy.thresholds)
    action = rng.choice(["threshold", "toggle", "votes"])
    if action == "threshold" and enabled:
        model = str(rng.choice(sorted(enabled)))
        thresholds[model] = float(rng.choice(threshold_grid))
    elif action == "toggle":
        model = str(rng.choice(models))
        if model in enabled and len(enabled) > 1:
            enabled.remove(model)
            thresholds.pop(model, None)
        else:
            enabled.add(model)
            thresholds.setdefault(model, float(rng.choice(threshold_grid)))
    else:
        pass
    enabled_tuple = tuple(model for model in models if model in enabled)
    min_votes = int(policy.min_votes)
    if action == "votes":
        min_votes = int(rng.integers(1, len(enabled_tuple) + 1))
    min_votes = min(max(1, min_votes), len(enabled_tuple))
    thresholds = {model: float(thresholds[model]) for model in enabled_tuple}
    return Policy(policy.side, enabled_tuple, thresholds, min_votes)


def _optimize_side(
    frame: pd.DataFrame,
    side: str,
    horizon_weeks: int,
    models: list[str],
    threshold_grid: list[float],
    random_state: int,
    generations: int,
    population_size: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state + (0 if side == "bottom" else 10_000))
    population = [
        _random_policy(side, models, threshold_grid, rng)
        for _ in range(population_size)
    ]
    rows = []
    for generation in range(generations):
        evaluated = []
        for policy in population:
            row = _evaluate_policy(frame, policy, horizon_weeks)
            row["generation"] = generation
            row["objective_score"] = _score_row(row, side)
            evaluated.append((row["objective_score"], policy, row))
        evaluated.sort(key=lambda item: item[0], reverse=True)
        rows.extend(item[2] for item in evaluated[: max(5, population_size // 5)])
        elites = [item[1] for item in evaluated[: max(3, population_size // 6)]]
        next_population = elites.copy()
        while len(next_population) < population_size:
            parent = elites[int(rng.integers(0, len(elites)))]
            if rng.random() < 0.25:
                next_population.append(_random_policy(side, models, threshold_grid, rng))
            else:
                next_population.append(_mutate_policy(parent, models, threshold_grid, rng))
        population = next_population
    result = pd.DataFrame(rows).drop_duplicates(
        ["side", "enabled_models", "thresholds_json", "min_votes"]
    )
    return result.sort_values(
        ["objective_score", "signal_count", "precision_lift"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def optimize_bottom_weekly_signal_policy() -> pd.DataFrame:
    cfg = get_config("bottom_weekly_model.yaml")
    model_cfg = cfg["model"]
    details = _read_details()
    horizon_weeks = int(model_cfg["horizon_weeks"])
    models = [str(model) for model in model_cfg.get("model_kinds", [])]
    threshold_grid = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    random_state = int(model_cfg.get("random_state", 42))
    parts = []
    for side in ["bottom", "top"]:
        frame = _side_frame(details, side)
        parts.append(
            _optimize_side(
                frame=frame,
                side=side,
                horizon_weeks=horizon_weeks,
                models=models,
                threshold_grid=threshold_grid,
                random_state=random_state,
                generations=35,
                population_size=48,
            )
        )
    result = pd.concat(parts, ignore_index=True)
    path = project_path(str(cfg["outputs"]["optimized_policy_report"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("saved weekly optimized signal policy report path=%s rows=%s", path, len(result))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize weekly top/bottom signal policy.")
    parser.parse_args()
    report = optimize_bottom_weekly_signal_policy()
    print(report.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
