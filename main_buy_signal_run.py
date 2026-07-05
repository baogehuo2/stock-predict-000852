from __future__ import annotations

import argparse

from src.common.config import ensure_project_dirs
from src.common.config import get_config
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.features.build_market_features import build_market_features
from src.features.build_model_dataset import build_model_dataset
from src.features.build_technical_indicator_self_check import build_technical_indicator_self_check
from src.modeling.generate_buy_signal import generate_buy_signals
from src.modeling.train_final_buy_model import train_final_buy_model


logger = get_logger(__name__)

STEPS = {
    "build_market_features": build_market_features,
    "technical_indicator_self_check": build_technical_indicator_self_check,
    "build_dataset": build_model_dataset,
    "train_buy_final": train_final_buy_model,
    "generate_buy_signal": generate_buy_signals,
}

DEFAULT_FLOW = [
    "build_market_features",
    "technical_indicator_self_check",
    "build_dataset",
    "train_buy_final",
    "generate_buy_signal",
]


def run_steps(steps: list[str], continue_on_error: bool = True) -> None:
    ensure_project_dirs()
    get_config()
    for step in steps:
        logger.info("start buy step=%s", step)
        try:
            result = STEPS[step]()
            logger.info("finish buy step=%s result=%s", step, result)
        except Exception:
            logger.exception("failed buy step=%s", step)
            if not continue_on_error:
                raise


def main() -> None:
    disable_env_proxies()
    parser = argparse.ArgumentParser(description="Run buy signal research pipeline.")
    parser.add_argument("--step", choices=sorted(STEPS), action="append", help="Run one or more buy-specific steps.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    run_steps(args.step or DEFAULT_FLOW, continue_on_error=not args.stop_on_error)


if __name__ == "__main__":
    main()
