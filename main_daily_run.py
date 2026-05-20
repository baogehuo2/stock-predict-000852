from __future__ import annotations

import argparse

from src.collectors.collect_etf_akshare import collect_etfs
from src.collectors.collect_futures_akshare import collect_futures
from src.collectors.collect_guba_eastmoney import collect_guba
from src.collectors.collect_index_akshare import collect_indices
from src.collectors.collect_news import collect_news
from src.common.config import ensure_project_dirs
from src.common.db import init_database
from src.common.logger import get_logger
from src.common.network import disable_env_proxies
from src.features.build_market_features import build_market_features
from src.features.build_model_dataset import build_model_dataset
from src.features.build_sentiment_features import build_sentiment_features
from src.llm.extract_event import extract_events_for_recent_news
from src.modeling.predict import predict
from src.modeling.train_lgbm import train_models
from src.report.generate_html_report import generate_html_report


logger = get_logger(__name__)

STEPS = {
    "init_db": init_database,
    "collect_index": collect_indices,
    "collect_etf": collect_etfs,
    "collect_futures": collect_futures,
    "collect_guba": collect_guba,
    "collect_news": collect_news,
    "build_market_features": build_market_features,
    "build_sentiment_features": build_sentiment_features,
    "extract_events": extract_events_for_recent_news,
    "build_dataset": build_model_dataset,
    "train": train_models,
    "predict": predict,
    "report": generate_html_report,
}

DEFAULT_FLOW = [
    "init_db",
    "collect_index",
    "collect_etf",
    "collect_futures",
    "collect_guba",
    "collect_news",
    "build_market_features",
    "build_sentiment_features",
    "extract_events",
    "build_dataset",
    "train",
    "predict",
    "report",
]


def run_steps(steps: list[str], continue_on_error: bool = True) -> None:
    ensure_project_dirs()
    for step in steps:
        logger.info("start step=%s", step)
        try:
            result = STEPS[step]()
            logger.info("finish step=%s result=%s", step, result)
        except Exception:
            logger.exception("failed step=%s", step)
            if not continue_on_error or step == "extract_events":
                raise


def main() -> None:
    disable_env_proxies()
    parser = argparse.ArgumentParser(description="Run zz1000 daily pipeline.")
    parser.add_argument("--step", choices=sorted(STEPS), action="append", help="Run one or more specific steps.")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()
    run_steps(args.step or DEFAULT_FLOW, continue_on_error=not args.stop_on_error)


if __name__ == "__main__":
    main()
