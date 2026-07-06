from __future__ import annotations

import argparse
from dataclasses import dataclass

from src.label_tool.app import run


@dataclass(frozen=True)
class LabelToolProfile:
    index_code: str
    index_name: str
    label_path: str
    port: int
    default_freq: str = "daily"
    allowed_freqs: str = "daily,weekly,monthly,intraday"
    enable_candidates: bool = True


PROFILES = {
    "zz1000": LabelToolProfile(
        index_code="000852",
        index_name="\u4e2d\u8bc11000",
        label_path="data/manual_labels/market_turning_regions.csv",
        port=8765,
    ),
    "hs300": LabelToolProfile(
        index_code="000300",
        index_name="\u6caa\u6df1300",
        label_path="data/manual_labels/market_turning_regions_hs300.csv",
        port=8766,
    ),
    "zz1000-weekly": LabelToolProfile(
        index_code="000852",
        index_name="\u4e2d\u8bc11000",
        label_path="data/manual_labels/market_turning_regions_weekly_zz1000.csv",
        port=8775,
        default_freq="weekly",
        allowed_freqs="weekly",
        enable_candidates=False,
    ),
    "hs300-weekly": LabelToolProfile(
        index_code="000300",
        index_name="\u6caa\u6df1300",
        label_path="data/manual_labels/market_turning_regions_weekly_hs300.csv",
        port=8776,
        default_freq="weekly",
        allowed_freqs="weekly",
        enable_candidates=False,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the manual turning label web tool.")
    parser.add_argument("--target", choices=sorted(PROFILES), default="zz1000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="Override the target default port.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved settings without starting the server.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = PROFILES[args.target]
    port = args.port or profile.port
    if args.dry_run:
        print(f"target={args.target}")
        print(f"url=http://{args.host}:{port}/")
        print(f"index={profile.index_name} {profile.index_code}")
        print(f"label_path={profile.label_path}")
        print(f"default_freq={profile.default_freq}")
        print(f"allowed_freqs={profile.allowed_freqs}")
        print(f"enable_candidates={profile.enable_candidates}")
        return

    run(
        host=args.host,
        port=port,
        label_path=profile.label_path,
        index_code=profile.index_code,
        index_name=profile.index_name,
        default_freq=profile.default_freq,
        allowed_freqs=profile.allowed_freqs,
        enable_candidates=profile.enable_candidates,
    )


if __name__ == "__main__":
    main()
