from __future__ import annotations

import argparse

from src.llm.extract_event import extract_events_for_recent_news


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()
    print(extract_events_for_recent_news(limit=args.limit))


if __name__ == "__main__":
    main()

