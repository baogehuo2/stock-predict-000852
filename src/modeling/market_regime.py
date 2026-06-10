from __future__ import annotations

import pandas as pd

from src.common.config import get_config


def assign_evaluation_regime(trade_dates: pd.Series) -> pd.Series:
    cfg = get_config().get("market_regime", {})
    dates = pd.to_datetime(trade_dates)
    evaluation_start = pd.Timestamp(cfg.get("evaluation_start", "2021-01-01"))
    regime = pd.Series("outside", index=trade_dates.index, dtype="object")
    regime.loc[dates >= evaluation_start] = "bull"
    for period in cfg.get("bear_periods", []):
        start = pd.Timestamp(period["start"])
        end = pd.Timestamp(period["end"])
        regime.loc[(dates >= start) & (dates <= end)] = "bear"
    return regime
