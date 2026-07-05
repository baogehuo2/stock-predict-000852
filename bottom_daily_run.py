from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.analysis.plot_bp_xgboost_rf_prediction_signals import build_html
from src.common.config import ensure_project_dirs, get_config
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.features.build_bottom_dataset import build_bottom_dataset
from src.features.build_manual_turning_labels import build_manual_turning_labels
from src.modeling.blend_manual_turning_predictions import blend_predictions
from src.modeling.build_optimized_turning_signals import build_optimized_turning_signals
from src.modeling.train_manual_turning_models import (
    train_manual_turning_models,
    walk_forward_manual_turning_models,
)
from src.modeling.walk_forward_bottom_lgbm import walk_forward_bottom_evaluation


logger = get_logger(__name__)


DEFAULT_WALK_FORWARD_MODELS = [
    "l1_logistic_kbest",
    "xgboost_small",
    "svm",
]


@dataclass(frozen=True)
class BottomStep:
    name: str
    description: str
    action: Callable[[], object]


def _target_index() -> str:
    return str(get_config("bottom_model.yaml")["model"]["target_index"])


def _build_manual_labels() -> object:
    return build_manual_turning_labels(target_index=_target_index())


def _build_dataset() -> object:
    return build_bottom_dataset(target_index=_target_index())


def _walk_forward_baseline() -> object:
    cfg = get_config("bottom_model.yaml")
    thresholds = [float(value) for value in cfg["model"]["probability_thresholds"]]
    return walk_forward_bottom_evaluation(thresholds=thresholds)


def _train_sparse_positive_baseline() -> object:
    return train_manual_turning_models("l1_logistic_kbest", tune=True, target_index=_target_index())


def _walk_forward_sparse_positive_models() -> object:
    results = {}
    for model_kind in DEFAULT_WALK_FORWARD_MODELS:
        logger.info("running manual turning walk-forward model=%s", model_kind)
        results[model_kind] = walk_forward_manual_turning_models(
            model_kind,
            start_year=2022,
            tune=True,
            target_index=_target_index(),
        )
    return results


def _blend_equal_predictions() -> Path:
    return blend_predictions(
        models=DEFAULT_WALK_FORWARD_MODELS,
        output="data/reports/manual_weak_turning_walk_forward_predictions_blend_equal_wf.csv",
    )


def _blend_equal_rescue_predictions() -> Path:
    return blend_predictions(
        models=DEFAULT_WALK_FORWARD_MODELS,
        output="data/reports/manual_weak_turning_walk_forward_predictions_blend_equal_low_recall_wf.csv",
        bottom_rescue_model="l1_logistic_kbest",
        bottom_rescue_threshold=0.50,
    )


def _build_optimized_signals() -> object:
    return build_optimized_turning_signals(start_year=2022)


def _build_daily_signal_report() -> Path:
    return build_html(
        models=["blend_equal"],
        start_date="2022-01-01",
        output="data/reports/zz1000_blend_equal_daily_pgt070.html",
        bottom_threshold=0.70,
        top_threshold=0.70,
        min_daily_proba=0.70,
        use_file_signals=True,
    )


def _build_rescue_signal_report() -> Path:
    return build_html(
        models=["blend_equal_low_recall"],
        start_date="2022-01-01",
        output="data/reports/zz1000_blend_equal_rescue_low_recall_strategy_signals.html",
        bottom_threshold=0.50,
        top_threshold=0.55,
        use_file_signals=True,
    )


STEPS: dict[str, BottomStep] = {
    "manual_labels": BottomStep(
        "manual_labels",
        "Validate manual bottom/top regions and expand them to daily labels.",
        _build_manual_labels,
    ),
    "build_dataset": BottomStep(
        "build_dataset",
        "Build the isolated bottom-fishing feature dataset.",
        _build_dataset,
    ),
    "walk_forward_baseline": BottomStep(
        "walk_forward_baseline",
        "Run the original LightGBM bottom-fishing walk-forward evaluation.",
        _walk_forward_baseline,
    ),
    "train_sparse_baseline": BottomStep(
        "train_sparse_baseline",
        "Train the L1/KBest sparse-positive baseline on fixed splits.",
        _train_sparse_positive_baseline,
    ),
    "walk_forward_sparse_models": BottomStep(
        "walk_forward_sparse_models",
        "Run rolling manual turning models: L1 logistic, XGBoost-small, and SVM.",
        _walk_forward_sparse_positive_models,
    ),
    "blend_equal": BottomStep(
        "blend_equal",
        "Build equal-weight blended walk-forward probabilities.",
        _blend_equal_predictions,
    ),
    "blend_equal_rescue": BottomStep(
        "blend_equal_rescue",
        "Build equal-weight blended probabilities with low-recall bottom rescue.",
        _blend_equal_rescue_predictions,
    ),
    "optimized_signals": BottomStep(
        "optimized_signals",
        "Build optimized bottom/top turning signals and summary charts.",
        _build_optimized_signals,
    ),
    "daily_signal_report": BottomStep(
        "daily_signal_report",
        "Render daily blended signal K-line HTML report.",
        _build_daily_signal_report,
    ),
    "rescue_signal_report": BottomStep(
        "rescue_signal_report",
        "Render low-recall rescue strategy K-line HTML report.",
        _build_rescue_signal_report,
    ),
}


DEFAULT_FLOW = [
    "manual_labels",
    "build_dataset",
    "walk_forward_sparse_models",
    "blend_equal",
    "blend_equal_rescue",
    "daily_signal_report",
    "rescue_signal_report",
]


def list_steps() -> None:
    print("available bottom workflow steps:")
    for step in STEPS.values():
        print(f"- {step.name}: {step.description}")
    print("\ndefault flow:")
    for name in DEFAULT_FLOW:
        print(f"- {name}")


def run_steps(step_names: Sequence[str]) -> None:
    ensure_project_dirs()
    disable_env_proxies()
    for name in step_names:
        if name not in STEPS:
            available = ", ".join(STEPS)
            raise KeyError(f"Unknown bottom step: {name}. Available steps: {available}")
        step = STEPS[name]
        logger.info("start bottom step=%s", name)
        result = step.action()
        logger.info("finished bottom step=%s result=%s", name, result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the independent bottom-fishing research workflow."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=sorted(STEPS),
        help="Steps to run. Defaults to the stable bottom workflow.",
    )
    parser.add_argument(
        "--list-steps",
        action="store_true",
        help="Print available bottom workflow steps and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_steps:
        list_steps()
        return
    run_steps(args.steps or DEFAULT_FLOW)


if __name__ == "__main__":
    main()
