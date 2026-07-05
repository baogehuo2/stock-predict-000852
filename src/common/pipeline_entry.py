from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from src.common.config import ensure_project_dirs
from src.common.logger import get_logger


logger = get_logger(__name__)

StepFunc = Callable[[], Any]
SkipFunc = Callable[[str], bool]


def run_named_steps(
    steps: Iterable[str],
    registry: dict[str, StepFunc],
    *,
    continue_on_error: bool = True,
    skip_step: SkipFunc | None = None,
    stop_on_error_steps: set[str] | None = None,
) -> None:
    ensure_project_dirs()
    stop_on_error_steps = stop_on_error_steps or set()
    for step in steps:
        if step not in registry:
            raise KeyError(f"unknown step: {step}")
        if skip_step is not None and skip_step(step):
            logger.info("skip step=%s", step)
            continue
        logger.info("start step=%s", step)
        try:
            result = registry[step]()
            logger.info("finish step=%s result=%s", step, result)
        except Exception:
            logger.exception("failed step=%s", step)
            if not continue_on_error or step in stop_on_error_steps:
                raise
