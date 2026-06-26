from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.common.config import get_config
from src.common.db import execute_sql, get_database_name, read_sql, upsert_dataframe
from src.common.logger import get_logger
from src.modeling.bottom_data import assert_bottom_database


logger = get_logger(__name__)


def _consecutive_days(mask: pd.Series) -> pd.Series:
    values = mask.fillna(False).astype(bool)
    groups = values.ne(values.shift(fill_value=False)).cumsum()
    days = values.groupby(groups).cumcount() + 1
    return days.where(values, 0).astype(float)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _weekly_kdj(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple[float, float, float, float]:
    k_value = 50.0
    d_value = 50.0
    previous_diff = 0.0
    for idx in range(len(close)):
        start = max(0, idx - 8)
        lowest = float(np.min(low[start : idx + 1]))
        highest = float(np.max(high[start : idx + 1]))
        rsv = 50.0 if highest == lowest else (float(close[idx]) - lowest) / (highest - lowest) * 100
        k_value = 2 / 3 * k_value + 1 / 3 * rsv
        d_value = 2 / 3 * d_value + 1 / 3 * k_value
        if idx == len(close) - 2:
            previous_diff = k_value - d_value
    j_value = 3 * k_value - 2 * d_value
    return k_value, d_value, j_value, previous_diff


def build_causal_weekly_features(target: pd.DataFrame) -> pd.DataFrame:
    data = target.sort_values("trade_date").reset_index(drop=True).copy()
    week_keys = data["trade_date"].dt.to_period("W-FRI")
    completed: list[dict[str, float]] = []
    rows: list[dict[str, float]] = []
    current_key = None
    current: dict[str, float] = {}

    for idx, row in data.iterrows():
        week_key = week_keys.iloc[idx]
        if current_key is None or week_key != current_key:
            if current_key is not None:
                completed.append(current.copy())
            current_key = week_key
            current = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "days": 1.0,
            }
        else:
            current["high"] = max(current["high"], float(row["high"]))
            current["low"] = min(current["low"], float(row["low"]))
            current["close"] = float(row["close"])
            current["volume"] += float(row["volume"])
            current["days"] += 1.0

        bars = [*completed, current]
        opens = np.array([bar["open"] for bar in bars], dtype=float)
        highs = np.array([bar["high"] for bar in bars], dtype=float)
        lows = np.array([bar["low"] for bar in bars], dtype=float)
        closes = np.array([bar["close"] for bar in bars], dtype=float)
        volumes = np.array([bar["volume"] for bar in bars], dtype=float)
        days = np.array([bar["days"] for bar in bars], dtype=float)

        def gap(window: int) -> float:
            return float(closes[-1] / closes[-window:].mean() - 1) if len(closes) >= window else np.nan

        def return_n(window: int) -> float:
            return float(closes[-1] / closes[-window - 1] - 1) if len(closes) > window else np.nan

        ma20 = float(closes[-20:].mean()) if len(closes) >= 20 else np.nan
        std20 = float(closes[-20:].std(ddof=1)) if len(closes) >= 20 else np.nan
        boll_lower = ma20 - 2 * std20 if np.isfinite(ma20) and np.isfinite(std20) else np.nan
        boll_upper = ma20 + 2 * std20 if np.isfinite(ma20) and np.isfinite(std20) else np.nan
        boll_range = boll_upper - boll_lower if np.isfinite(boll_lower) else np.nan
        k_value, d_value, j_value, previous_kdj_diff = _weekly_kdj(highs, lows, closes)
        kdj_diff = k_value - d_value

        prior_uptrend = 0.0
        if len(completed) >= 24:
            prior_closes = np.array([bar["close"] for bar in completed], dtype=float)
            prior_ma20 = float(prior_closes[-20:].mean())
            prior_ma20_four_weeks_ago = float(prior_closes[-24:-4].mean())
            prior_uptrend = float(
                prior_closes[-1] > prior_ma20 and prior_ma20 > prior_ma20_four_weeks_ago
            )

        previous_completed = completed[-5:]
        previous_weekly_volume = np.array([bar["volume"] for bar in previous_completed], dtype=float)
        previous_daily_volume = np.array(
            [bar["volume"] / bar["days"] for bar in previous_completed], dtype=float
        )
        volume_ratio = (
            float(current["volume"] / previous_weekly_volume.mean())
            if len(previous_weekly_volume) == 5 and previous_weekly_volume.mean() != 0
            else np.nan
        )
        volume_pace_ratio = (
            float((current["volume"] / current["days"]) / previous_daily_volume.mean())
            if len(previous_daily_volume) == 5 and previous_daily_volume.mean() != 0
            else np.nan
        )

        week_range = current["high"] - current["low"]
        lower_shadow = min(current["open"], current["close"]) - current["low"]
        close_below_lower = float(np.isfinite(boll_lower) and current["close"] < boll_lower)
        low_below_lower = float(np.isfinite(boll_lower) and current["low"] < boll_lower)
        lower_reclaim = float(low_below_lower == 1 and current["close"] >= boll_lower)
        close_below_mid = float(np.isfinite(ma20) and current["close"] < ma20)
        uptrend_pullback = float(prior_uptrend == 1 and close_below_mid == 1)
        golden_cross = float(kdj_diff > 0 and previous_kdj_diff <= 0)
        kdj_oversold = float(j_value < 0 or k_value < 25)
        oversold_golden_cross = float(golden_cross == 1 and kdj_oversold == 1)
        down_volume_expand = float(
            np.isfinite(return_n(1)) and return_n(1) < 0
            and np.isfinite(volume_pace_ratio) and volume_pace_ratio >= 1.2
        )
        long_lower_shadow = float(week_range > 0 and lower_shadow / week_range >= 0.45)

        rows.append(
            {
                "trade_date": row["trade_date"],
                "week_progress": current["days"] / 5.0,
                "week_ret_1w": return_n(1),
                "week_ret_2w": return_n(2),
                "week_ret_4w": return_n(4),
                "week_range": week_range / current["close"],
                "week_body_return": current["close"] / current["open"] - 1,
                "week_lower_shadow_ratio": lower_shadow / week_range if week_range > 0 else 0.0,
                "week_drawdown_13w": closes[-1] / highs[-13:].max() - 1 if len(highs) >= 13 else np.nan,
                "week_drawdown_26w": closes[-1] / highs[-26:].max() - 1 if len(highs) >= 26 else np.nan,
                "week_ma5_gap": gap(5),
                "week_ma10_gap": gap(10),
                "week_ma20_gap": gap(20),
                "week_ma20_slope_4w": (
                    ma20 / closes[-24:-4].mean() - 1 if len(closes) >= 24 else np.nan
                ),
                "week_boll_width": boll_range / ma20 if np.isfinite(boll_range) else np.nan,
                "week_boll_position": (
                    (current["close"] - boll_lower) / boll_range
                    if np.isfinite(boll_range) and boll_range != 0
                    else np.nan
                ),
                "week_close_below_boll_lower": close_below_lower,
                "week_low_below_boll_lower": low_below_lower,
                "week_boll_lower_reclaim": lower_reclaim,
                "week_close_below_boll_mid": close_below_mid,
                "week_prior_uptrend": prior_uptrend,
                "week_uptrend_pullback_below_mid": uptrend_pullback,
                "week_kdj_k": k_value,
                "week_kdj_d": d_value,
                "week_kdj_j": j_value,
                "week_kdj_k_minus_d": kdj_diff,
                "week_kdj_golden_cross": golden_cross,
                "week_kdj_oversold": kdj_oversold,
                "week_oversold_kdj_golden_cross": oversold_golden_cross,
                "week_volume_pace_ratio_5w": volume_pace_ratio,
                "week_volume_ratio_5w": volume_ratio,
                "week_down_volume_expand": down_volume_expand,
                "week_bottom_confirmation_score": float(
                    close_below_lower
                    + lower_reclaim
                    + uptrend_pullback
                    + golden_cross
                    + oversold_golden_cross
                    + long_lower_shadow
                ),
            }
        )
    return pd.DataFrame(rows)


def _load_bottom_config() -> dict:
    return get_config("bottom_model.yaml")


def _market_frame(target_index: str) -> pd.DataFrame:
    data = read_sql(
        "SELECT trade_date, index_code, open, high, low, close, volume "
        "FROM market_index_daily ORDER BY trade_date, index_code"
    )
    if data.empty:
        raise RuntimeError("market_index_daily is empty.")
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    for col in ["open", "high", "low", "close", "volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    target = data[data["index_code"] == target_index].copy()
    if target.empty:
        raise RuntimeError(f"No market rows found for target index {target_index}.")
    return target.sort_values("trade_date").reset_index(drop=True), data


def build_bottom_feature_frame(target: pd.DataFrame, all_indices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = target[["trade_date", "index_code"]].copy()
    open_ = target["open"]
    high = target["high"]
    low = target["low"]
    close = target["close"]
    volume = target["volume"]
    ret_1d = close.pct_change()

    for window in [1, 3, 5, 10]:
        out[f"ret_{window}d"] = close.pct_change(window)
    out["downside_ret_3d"] = ret_1d.clip(upper=0).rolling(3).sum()
    out["downside_ret_5d"] = ret_1d.clip(upper=0).rolling(5).sum()
    for window in [20, 60, 120]:
        out[f"drawdown_{window}d"] = close / close.rolling(window).max() - 1
    for window in [5, 10, 20]:
        out[f"distance_low_{window}d"] = close / low.rolling(window).min() - 1

    moving_averages = {window: close.rolling(window).mean() for window in [5, 10, 20, 60]}
    for window, average in moving_averages.items():
        out[f"ma{window}_gap"] = close / average - 1
    out["ma20_slope_5d"] = moving_averages[20] / moving_averages[20].shift(5) - 1
    out["ma60_slope_10d"] = moving_averages[60] / moving_averages[60].shift(10) - 1
    out["below_ma20_days"] = _consecutive_days(close < moving_averages[20])
    out["below_ma60_days"] = _consecutive_days(close < moving_averages[60])

    out["rsi6"] = _rsi(close, 6)
    out["rsi14"] = _rsi(close, 14)
    high_14 = high.rolling(14).max()
    low_14 = low.rolling(14).min()
    out["wr14"] = (high_14 - close) / (high_14 - low_14).replace(0, np.nan) * -100
    typical_price = (high + low + close) / 3
    typical_mean = typical_price.rolling(14).mean()
    mean_deviation = (typical_price - typical_mean).abs().rolling(14).mean()
    out["cci14"] = (typical_price - typical_mean) / (0.015 * mean_deviation.replace(0, np.nan))

    boll_mid = moving_averages[20]
    boll_std = close.rolling(20).std()
    boll_upper = boll_mid + 2 * boll_std
    boll_lower = boll_mid - 2 * boll_std
    out["boll_width"] = (boll_upper - boll_lower) / boll_mid
    out["boll_position"] = (close - boll_lower) / (boll_upper - boll_lower).replace(0, np.nan)
    out["boll_lower_break"] = (close < boll_lower).astype(float)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["macd_hist_delta_1d"] = out["macd_hist"].diff()
    out["macd_hist_delta_3d"] = out["macd_hist"].diff(3)

    low_9 = low.rolling(9, min_periods=1).min()
    high_9 = high.rolling(9, min_periods=1).max()
    rsv = (close - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
    out["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    out["kdj_d"] = out["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    out["kdj_j"] = 3 * out["kdj_k"] - 2 * out["kdj_d"]
    out["kdj_k_minus_d"] = out["kdj_k"] - out["kdj_d"]
    out["kdj_golden_cross"] = (
        (out["kdj_k_minus_d"] > 0) & (out["kdj_k_minus_d"].shift(1) <= 0)
    ).astype(float)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1
    ).max(axis=1)
    out["atr14"] = true_range.rolling(14).mean() / close
    for window in [5, 10, 20]:
        out[f"volatility_{window}d"] = ret_1d.rolling(window).std()
    out["volatility_expand"] = out["volatility_5d"] / out["volatility_20d"].replace(0, np.nan)

    day_range = (high - low).replace(0, np.nan)
    lower_shadow = pd.concat([open_, close], axis=1).min(axis=1) - low
    out["intraday_range"] = day_range / previous_close
    out["body_return"] = close / open_ - 1
    out["lower_shadow_ratio"] = lower_shadow / day_range
    out["long_lower_shadow"] = (out["lower_shadow_ratio"] >= 0.5).astype(float)
    out["gap_down"] = (open_ < previous_close * 0.995).astype(float)
    out["down_streak"] = _consecutive_days(ret_1d < 0)
    out["up_streak"] = _consecutive_days(ret_1d > 0)
    out["volume_zscore_20d"] = (volume - volume.rolling(20).mean()) / volume.rolling(20).std()
    out["volume_ratio_5d_20d"] = volume.rolling(5).mean() / volume.rolling(20).mean().replace(0, np.nan)

    close_pivot = all_indices.pivot(index="trade_date", columns="index_code", values="close").sort_index()
    relative = pd.DataFrame(index=close_pivot.index)
    for code, name in [("000300", "hs300"), ("000905", "zz500"), ("399006", "cyb")]:
        relative[f"relative_{name}_1d"] = close_pivot[target["index_code"].iloc[0]].pct_change() - close_pivot[code].pct_change()
        relative[f"relative_{name}_5d"] = close_pivot[target["index_code"].iloc[0]].pct_change(5) - close_pivot[code].pct_change(5)
    out = out.merge(relative.reset_index(), on="trade_date", how="left")

    trend_votes = pd.concat(
        [
            (close > moving_averages[20]).map({True: 1, False: -1}),
            (close > moving_averages[60]).map({True: 1, False: -1}),
            (out["ma20_slope_5d"] > 0).map({True: 1, False: -1}),
            (out["ma60_slope_10d"] > 0).map({True: 1, False: -1}),
        ],
        axis=1,
    )
    unavailable = moving_averages[60].isna() | out["ma60_slope_10d"].isna()
    out["regime_trend_score"] = trend_votes.sum(axis=1).where(~unavailable)
    regime = pd.Series(0, index=out.index)
    regime.loc[out["regime_trend_score"] >= 2] = 1
    regime.loc[out["regime_trend_score"] <= -2] = -1
    groups = regime.ne(regime.shift(fill_value=0)).cumsum()
    out["regime_state_days"] = (regime.groupby(groups).cumcount() + 1).where(regime != 0, 0)

    candidate_cfg = cfg["candidate"]
    out["candidate_depth"] = (
        (out["drawdown_20d"] <= float(candidate_cfg["drawdown_20d"]))
        | (out["drawdown_60d"] <= float(candidate_cfg["drawdown_60d"]))
    ).astype(float)
    out["candidate_oversold"] = (
        (out["rsi6"] <= float(candidate_cfg["rsi6"]))
        | (out["boll_position"] <= float(candidate_cfg["boll_position"]))
        | (out["wr14"] <= float(candidate_cfg["wr14"]))
    ).astype(float)
    out["candidate_trend"] = (
        (out["below_ma20_days"] >= int(candidate_cfg["below_ma20_days"]))
        | (out["ma20_gap"] <= float(candidate_cfg["ma20_gap"]))
    ).astype(float)
    out["candidate_stress"] = (
        (out["ret_3d"] <= float(candidate_cfg["ret_3d"]))
        | (out["volatility_expand"] >= float(candidate_cfg["volatility_expand"]))
        | (out["long_lower_shadow"] == 1)
    ).astype(float)
    condition_cols = ["candidate_depth", "candidate_oversold", "candidate_trend", "candidate_stress"]
    out["candidate_score"] = out[condition_cols].sum(axis=1)
    out["is_candidate"] = (
        out["candidate_score"] >= int(candidate_cfg["min_conditions"])
    ).astype(int)
    weekly = build_causal_weekly_features(target)
    out = out.merge(weekly, on="trade_date", how="left")
    return out


def add_bottom_labels(frame: pd.DataFrame, target: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    result = frame.copy()
    horizon = int(cfg["model"]["horizon"])
    label_cfg = cfg["labels"]
    close = target["close"].reset_index(drop=True)
    high = target["high"].reset_index(drop=True)
    low = target["low"].reset_index(drop=True)
    high_paths = pd.concat([high.shift(-step) / close - 1 for step in range(1, horizon + 1)], axis=1)
    low_paths = pd.concat([low.shift(-step) / close - 1 for step in range(1, horizon + 1)], axis=1)
    high_paths.columns = range(1, horizon + 1)
    low_paths.columns = range(1, horizon + 1)
    result["future_ret_15d"] = close.shift(-horizon) / close - 1
    result["future_mfe_15d"] = high_paths.max(axis=1)
    result["future_mae_15d"] = low_paths.min(axis=1)

    rebound_target = float(label_cfg["rebound_target"])
    hit_mask = high_paths >= rebound_target
    result["target_hit_day"] = hit_mask.apply(
        lambda row: next((int(day) for day, hit in row.items() if bool(hit)), np.nan), axis=1
    )
    pre_target_mae = []
    for idx, hit_day in result["target_hit_day"].items():
        end_day = horizon if pd.isna(hit_day) else int(hit_day)
        pre_target_mae.append(low_paths.loc[idx, list(range(1, end_day + 1))].min())
    result["pre_target_mae"] = pre_target_mae

    complete = result["future_ret_15d"].notna()
    result["terminal_rebound_label"] = np.where(
        complete, (result["future_ret_15d"] > float(label_cfg["terminal_return"])).astype(int), np.nan
    )
    result["path_rebound_label"] = np.where(
        complete,
        (
            (result["future_mfe_15d"] >= rebound_target)
            & (result["future_mae_15d"] >= float(label_cfg["path_max_adverse_excursion"]))
        ).astype(int),
        np.nan,
    )
    result["quality_bottom_label"] = np.where(
        complete,
        (
            (result["future_mfe_15d"] >= rebound_target)
            & (result["future_ret_15d"] > float(label_cfg["quality_terminal_return"]))
            & (result["pre_target_mae"] >= float(label_cfg["quality_max_adverse_excursion"]))
        ).astype(int),
        np.nan,
    )
    result["continuation_risk_label"] = np.where(
        complete,
        (result["future_mae_15d"] <= float(label_cfg["continuation_risk"])).astype(int),
        np.nan,
    )
    result["trade_pos"] = np.arange(len(result))
    return result


def _ensure_table(table: str) -> None:
    execute_sql(
        f"""
        CREATE TABLE IF NOT EXISTS `{table}` (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            trade_date DATE NOT NULL,
            index_code VARCHAR(20) NOT NULL,
            trade_pos INT NOT NULL,
            is_candidate TINYINT NOT NULL,
            candidate_score INT NOT NULL,
            future_ret_15d DECIMAL(12,8) NULL,
            future_mfe_15d DECIMAL(12,8) NULL,
            future_mae_15d DECIMAL(12,8) NULL,
            target_hit_day INT NULL,
            pre_target_mae DECIMAL(12,8) NULL,
            terminal_rebound_label TINYINT NULL,
            path_rebound_label TINYINT NULL,
            quality_bottom_label TINYINT NULL,
            continuation_risk_label TINYINT NULL,
            feature_json JSON NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uk_bottom_dataset (trade_date, index_code),
            KEY idx_bottom_candidate (is_candidate, trade_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )
    database = get_database_name()
    existing = set(
        read_sql(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=:database AND TABLE_NAME=:table",
            {"database": database, "table": table},
        )["COLUMN_NAME"].tolist()
    )
    for column in ["future_ret_15d", "future_mfe_15d", "future_mae_15d"]:
        if column not in existing:
            execute_sql(f"ALTER TABLE `{table}` ADD COLUMN `{column}` DECIMAL(12,8) NULL")


def build_bottom_dataset() -> int:
    assert_bottom_database()
    cfg = _load_bottom_config()
    target_index = str(cfg["model"]["target_index"])
    target, all_indices = _market_frame(target_index)
    feature_frame = build_bottom_feature_frame(target, all_indices, cfg)
    data = add_bottom_labels(feature_frame, target, cfg)
    feature_cols = [col for col in data.columns if col not in {"trade_date", "index_code", "is_candidate"}]
    excluded = {
        "trade_pos", "future_ret_15d", "future_mfe_15d", "future_mae_15d",
        "target_hit_day", "pre_target_mae", "terminal_rebound_label", "path_rebound_label",
        "quality_bottom_label", "continuation_risk_label",
    }
    feature_cols = [col for col in feature_cols if col not in excluded]
    records = []
    for _, row in data.iterrows():
        features = {
            f"f_{col}": None if pd.isna(row[col]) else float(row[col])
            for col in feature_cols
        }
        records.append(
            {
                "trade_date": row["trade_date"].date(),
                "index_code": row["index_code"],
                "trade_pos": int(row["trade_pos"]),
                "is_candidate": int(row["is_candidate"]),
                "candidate_score": int(row["candidate_score"]),
                "future_ret_15d": row["future_ret_15d"],
                "future_mfe_15d": row["future_mfe_15d"],
                "future_mae_15d": row["future_mae_15d"],
                "target_hit_day": row["target_hit_day"],
                "pre_target_mae": row["pre_target_mae"],
                "terminal_rebound_label": row["terminal_rebound_label"],
                "path_rebound_label": row["path_rebound_label"],
                "quality_bottom_label": row["quality_bottom_label"],
                "continuation_risk_label": row["continuation_risk_label"],
                "feature_json": json.dumps(features, ensure_ascii=False),
            }
        )
    table = str(cfg["outputs"]["dataset_table"])
    _ensure_table(table)
    count = upsert_dataframe(pd.DataFrame(records), table, ["trade_date", "index_code"])
    logger.info("built bottom dataset rows=%s candidates=%s table=%s", count, int(data["is_candidate"].sum()), table)
    return count


def main() -> None:
    argparse.ArgumentParser(description="Build the isolated bottom-fishing research dataset.").parse_args()
    print(build_bottom_dataset())


if __name__ == "__main__":
    main()
