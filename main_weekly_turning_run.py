from __future__ import annotations

import argparse

from src.analysis.build_bottom_weekly_evaluation_html import build_bottom_weekly_evaluation_html
from src.analysis.build_bottom_weekly_signal_html import build_bottom_weekly_signal_html
from src.common.config import ensure_project_dirs
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.features.build_bottom_weekly_dataset import build_bottom_weekly_dataset
from src.modeling.optimize_bottom_weekly_signal_policy import optimize_bottom_weekly_signal_policy
from src.modeling.train_bottom_weekly_final_models import train_bottom_weekly_final_models
from src.modeling.walk_forward_bottom_weekly_lgbm import walk_forward_bottom_weekly_evaluation


logger = get_logger(__name__)


STEPS = {
    "build_dataset": build_bottom_weekly_dataset,
    "evaluate": walk_forward_bottom_weekly_evaluation,
    "train_final": train_bottom_weekly_final_models,
    "optimize_policy": optimize_bottom_weekly_signal_policy,
    "build_evaluation_html": build_bottom_weekly_evaluation_html,
    "build_signal_html": build_bottom_weekly_signal_html,
}

DEFAULT_FLOW = [
    "build_dataset",
    "evaluate",
    "train_final",
    "optimize_policy",
    "build_evaluation_html",
    "build_signal_html",
]


def run_steps(steps: list[str], continue_on_error: bool = True) -> None:
    ensure_project_dirs()
    for step in steps:
        logger.info("start weekly_turning step=%s", step)
        try:
            result = STEPS[step]()
            logger.info("finish weekly_turning step=%s result=%s", step, result)
        except Exception:
            logger.exception("failed weekly_turning step=%s", step)
            if not continue_on_error:
                raise


def main() -> None:
    disable_env_proxies()
    parser = argparse.ArgumentParser(description="Run weekly top/bottom turning workflow.")
    parser.add_argument("--step", choices=sorted(STEPS), action="append", help="Run one or more specific steps.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    run_steps(args.step or DEFAULT_FLOW, continue_on_error=not args.stop_on_error)


if __name__ == "__main__":
    main()
