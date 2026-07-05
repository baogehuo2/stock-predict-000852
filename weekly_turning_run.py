from __future__ import annotations

import argparse

from src.analysis.build_bottom_weekly_evaluation_html import build_bottom_weekly_evaluation_html
from src.analysis.build_bottom_weekly_signal_html import build_bottom_weekly_signal_html
from src.common.network import disable_env_proxies
from src.common.pipeline_entry import run_named_steps
from src.features.build_bottom_weekly_dataset import build_bottom_weekly_dataset
from src.modeling.optimize_bottom_weekly_signal_policy import optimize_bottom_weekly_signal_policy
from src.modeling.train_bottom_weekly_final_models import train_bottom_weekly_final_models
from src.modeling.walk_forward_bottom_weekly_lgbm import walk_forward_bottom_weekly_evaluation


STEPS = {
    "build_bottom_weekly_dataset": build_bottom_weekly_dataset,
    "evaluate_bottom_weekly": walk_forward_bottom_weekly_evaluation,
    "train_bottom_weekly_final": train_bottom_weekly_final_models,
    "optimize_bottom_weekly_policy": optimize_bottom_weekly_signal_policy,
    "build_bottom_weekly_html": build_bottom_weekly_evaluation_html,
    "build_bottom_weekly_signal_html": build_bottom_weekly_signal_html,
}

DEFAULT_FLOW = [
    "build_bottom_weekly_dataset",
    "evaluate_bottom_weekly",
    "train_bottom_weekly_final",
    "optimize_bottom_weekly_policy",
    "build_bottom_weekly_html",
    "build_bottom_weekly_signal_html",
]


def run_steps(steps: list[str], continue_on_error: bool = True) -> None:
    run_named_steps(steps, STEPS, continue_on_error=continue_on_error)


def main() -> None:
    disable_env_proxies()
    parser = argparse.ArgumentParser(description="Run weekly turning pipeline.")
    parser.add_argument("--step", choices=sorted(STEPS), action="append", help="Run one or more specific steps.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    run_steps(args.step or DEFAULT_FLOW, continue_on_error=not args.stop_on_error)


if __name__ == "__main__":
    main()
