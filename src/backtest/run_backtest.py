from __future__ import annotations

import argparse
import json

from src.modeling.evaluate import evaluate_models


def main() -> None:
    argparse.ArgumentParser().parse_args()
    print(json.dumps(evaluate_models(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

