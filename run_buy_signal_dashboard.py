from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from src.common.config import ensure_project_dirs
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.report.generate_buy_signal_report import generate_buy_signal_report
from src.report.generate_walk_forward_buy_report import generate_walk_forward_buy_report


logger = get_logger(__name__)

StepFunc = Callable[[], Path]

STEPS: dict[str, StepFunc] = {
    "report_buy_signal": generate_buy_signal_report,
    "report_walk_forward_buy": generate_walk_forward_buy_report,
}

DEFAULT_FLOW = ["report_buy_signal"]


def run_steps(steps: list[str], continue_on_error: bool = True) -> None:
    ensure_project_dirs()
    for step in steps:
        logger.info("start step=%s", step)
        try:
            result = STEPS[step]()
            logger.info("finish step=%s result=%s", step, result)
        except Exception:
            logger.exception("failed step=%s", step)
            if not continue_on_error:
                raise


def main() -> None:
    disable_env_proxies()
    parser = argparse.ArgumentParser(description="Run Buy signal dashboard reports.")
    parser.add_argument("--step", choices=sorted(STEPS), action="append", help="Run one or more report steps.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    run_steps(args.step or DEFAULT_FLOW, continue_on_error=not args.stop_on_error)


if __name__ == "__main__":
    main()
