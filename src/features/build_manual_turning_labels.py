from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import execute_sql, get_database_name, read_sql, upsert_dataframe
from src.common.logger import get_logger
from src.modeling.bottom_data import assert_bottom_database


logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "region_id",
    "start_date",
    "end_date",
    "region_type",
    "cycle",
    "level",
    "label_freq",
    "confidence",
    "usable_for_signal",
    "entry_start",
    "entry_end",
    "exit_start",
    "exit_end",
    "reason",
    "notes",
]
LEVEL_SCORE = {"minor": 1, "swing": 2, "major": 3}
CYCLE_SCORE = {"short": 1, "medium": 2, "long": 3}
LABEL_FREQ_SCORE = {"daily": 1, "weekly": 2, "monthly": 3}


def load_manual_regions(path: str | Path | None = None) -> pd.DataFrame:
    cfg = get_config("bottom_model.yaml")
    label_path = project_path(path or str(cfg["outputs"]["manual_region_file"]))
    if not label_path.exists():
        raise FileNotFoundError(f"Manual turning region file not found: {label_path}")
    regions = pd.read_csv(label_path, dtype=str).fillna("")
    missing = [col for col in REQUIRED_COLUMNS if col not in regions.columns]
    if missing:
        raise RuntimeError(f"Manual turning region file is missing columns: {missing}")
    regions = regions[REQUIRED_COLUMNS].copy()
    for col in ["start_date", "end_date", "entry_start", "entry_end", "exit_start", "exit_end"]:
        regions[col] = pd.to_datetime(regions[col], errors="coerce")
    regions["confidence"] = pd.to_numeric(regions["confidence"], errors="raise").astype(int)
    regions["usable_for_signal"] = pd.to_numeric(
        regions["usable_for_signal"], errors="raise"
    ).astype(int)
    _validate_regions(regions)
    return regions


def _validate_regions(regions: pd.DataFrame) -> None:
    duplicated = regions["region_id"][regions["region_id"].duplicated()].tolist()
    if duplicated:
        raise RuntimeError(f"Duplicate manual region_id values: {duplicated}")
    invalid_types = sorted(set(regions["region_type"]) - {"bottom", "top"})
    if invalid_types:
        raise RuntimeError(f"Invalid region_type values: {invalid_types}")
    invalid_levels = sorted(set(regions["level"]) - set(LEVEL_SCORE))
    if invalid_levels:
        raise RuntimeError(f"Invalid level values: {invalid_levels}")
    invalid_cycles = sorted(set(regions["cycle"]) - set(CYCLE_SCORE))
    if invalid_cycles:
        raise RuntimeError(f"Invalid cycle values: {invalid_cycles}")
    invalid_label_freqs = sorted(set(regions["label_freq"]) - set(LABEL_FREQ_SCORE))
    if invalid_label_freqs:
        raise RuntimeError(f"Invalid label_freq values: {invalid_label_freqs}")
    bad_confidence = regions[~regions["confidence"].isin([1, 2, 3])]
    if not bad_confidence.empty:
        raise RuntimeError(f"Invalid confidence rows: {bad_confidence['region_id'].tolist()}")
    bad_usable = regions[~regions["usable_for_signal"].isin([0, 1])]
    if not bad_usable.empty:
        raise RuntimeError(f"Invalid usable_for_signal rows: {bad_usable['region_id'].tolist()}")
    bad_dates = regions[regions["start_date"].isna() | regions["end_date"].isna()]
    if not bad_dates.empty:
        raise RuntimeError(f"Invalid region date rows: {bad_dates['region_id'].tolist()}")
    reversed_dates = regions[regions["start_date"] > regions["end_date"]]
    if not reversed_dates.empty:
        raise RuntimeError(f"Reversed date rows: {reversed_dates['region_id'].tolist()}")
    for row in regions.itertuples(index=False):
        if row.region_type == "bottom":
            _validate_optional_window(row.region_id, row.entry_start, row.entry_end, row.start_date, row.end_date)
        if row.region_type == "top":
            _validate_optional_window(row.region_id, row.exit_start, row.exit_end, row.start_date, row.end_date)


def _validate_optional_window(
    region_id: str,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    region_start: pd.Timestamp,
    region_end: pd.Timestamp,
) -> None:
    if pd.isna(window_start) and pd.isna(window_end):
        return
    if pd.isna(window_start) or pd.isna(window_end):
        raise RuntimeError(f"Region {region_id} has an incomplete signal window.")
    if window_start > window_end:
        raise RuntimeError(f"Region {region_id} has a reversed signal window.")
    if window_start < region_start or window_end > region_end:
        raise RuntimeError(f"Region {region_id} signal window is outside its region range.")


def _trade_dates(target_index: str) -> pd.DataFrame:
    data = read_sql(
        "SELECT trade_date, index_code FROM market_index_daily "
        "WHERE index_code=:target_index ORDER BY trade_date",
        {"target_index": target_index},
    )
    if data.empty:
        raise RuntimeError(f"No market_index_daily rows found for {target_index}.")
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    return data


def _join_ids(values: list[str]) -> str:
    return "|".join(sorted(set(value for value in values if value)))


def _join_ordered(values: list[str]) -> str:
    seen = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return "|".join(seen)


def build_manual_turning_daily(path: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert_bottom_database()
    cfg = get_config("bottom_model.yaml")
    target_index = str(cfg["model"]["target_index"])
    regions = load_manual_regions(path)
    daily = _trade_dates(target_index)
    daily["is_manual_bottom_region"] = 0
    daily["is_manual_top_region"] = 0
    daily["is_manual_buy_window"] = 0
    daily["is_manual_sell_window"] = 0
    daily["bottom_level_score"] = 0.0
    daily["top_level_score"] = 0.0
    daily["bottom_confidence"] = 0
    daily["top_confidence"] = 0
    daily["bottom_cycle_score"] = 0
    daily["top_cycle_score"] = 0
    daily["bottom_label_freq_score"] = 0
    daily["top_label_freq_score"] = 0
    daily["label_mode"] = str(get_config("bottom_model.yaml")["manual_models"]["label_mode"])
    bottom_ids: list[list[str]] = [[] for _ in range(len(daily))]
    top_ids: list[list[str]] = [[] for _ in range(len(daily))]
    bottom_cycles: list[list[str]] = [[] for _ in range(len(daily))]
    top_cycles: list[list[str]] = [[] for _ in range(len(daily))]
    bottom_label_freqs: list[list[str]] = [[] for _ in range(len(daily))]
    top_label_freqs: list[list[str]] = [[] for _ in range(len(daily))]

    for region in regions.itertuples(index=False):
        region_mask = (daily["trade_date"] >= region.start_date) & (daily["trade_date"] <= region.end_date)
        if not region_mask.any():
            logger.warning("manual region has no matching trade dates region_id=%s", region.region_id)
            continue
        idxs = daily.index[region_mask].tolist()
        level_score = float(LEVEL_SCORE[region.level] * region.confidence)
        cycle_score = int(CYCLE_SCORE[region.cycle])
        label_freq_score = int(LABEL_FREQ_SCORE[region.label_freq])
        if region.region_type == "bottom":
            daily.loc[region_mask, "is_manual_bottom_region"] = 1
            daily.loc[region_mask, "bottom_level_score"] = daily.loc[
                region_mask, "bottom_level_score"
            ].clip(lower=level_score)
            daily.loc[region_mask, "bottom_confidence"] = daily.loc[
                region_mask, "bottom_confidence"
            ].clip(lower=int(region.confidence))
            daily.loc[region_mask, "bottom_cycle_score"] = daily.loc[
                region_mask, "bottom_cycle_score"
            ].clip(lower=cycle_score)
            daily.loc[region_mask, "bottom_label_freq_score"] = daily.loc[
                region_mask, "bottom_label_freq_score"
            ].clip(lower=label_freq_score)
            for idx in idxs:
                bottom_ids[idx].append(region.region_id)
                bottom_cycles[idx].append(region.cycle)
                bottom_label_freqs[idx].append(region.label_freq)
            if region.usable_for_signal == 1 and pd.notna(region.entry_start):
                entry_mask = (
                    (daily["trade_date"] >= region.entry_start)
                    & (daily["trade_date"] <= region.entry_end)
                )
                daily.loc[entry_mask, "is_manual_buy_window"] = 1
        else:
            daily.loc[region_mask, "is_manual_top_region"] = 1
            daily.loc[region_mask, "top_level_score"] = daily.loc[
                region_mask, "top_level_score"
            ].clip(lower=level_score)
            daily.loc[region_mask, "top_confidence"] = daily.loc[
                region_mask, "top_confidence"
            ].clip(lower=int(region.confidence))
            daily.loc[region_mask, "top_cycle_score"] = daily.loc[
                region_mask, "top_cycle_score"
            ].clip(lower=cycle_score)
            daily.loc[region_mask, "top_label_freq_score"] = daily.loc[
                region_mask, "top_label_freq_score"
            ].clip(lower=label_freq_score)
            for idx in idxs:
                top_ids[idx].append(region.region_id)
                top_cycles[idx].append(region.cycle)
                top_label_freqs[idx].append(region.label_freq)
            if region.usable_for_signal == 1 and pd.notna(region.exit_start):
                exit_mask = (
                    (daily["trade_date"] >= region.exit_start)
                    & (daily["trade_date"] <= region.exit_end)
                )
                daily.loc[exit_mask, "is_manual_sell_window"] = 1

    daily["manual_bottom_region_ids"] = [_join_ids(values) for values in bottom_ids]
    daily["manual_top_region_ids"] = [_join_ids(values) for values in top_ids]
    daily["manual_bottom_cycles"] = [_join_ordered(values) for values in bottom_cycles]
    daily["manual_top_cycles"] = [_join_ordered(values) for values in top_cycles]
    daily["manual_bottom_label_freqs"] = [_join_ordered(values) for values in bottom_label_freqs]
    daily["manual_top_label_freqs"] = [_join_ordered(values) for values in top_label_freqs]
    daily["manual_state"] = "neutral"
    daily.loc[daily["is_manual_bottom_region"] == 1, "manual_state"] = "bottom"
    daily.loc[daily["is_manual_top_region"] == 1, "manual_state"] = "top"
    both = (daily["is_manual_bottom_region"] == 1) & (daily["is_manual_top_region"] == 1)
    daily.loc[both, "manual_state"] = "overlap"
    daily["trade_date"] = daily["trade_date"].dt.date
    return regions, daily


def _ensure_table(table: str) -> None:
    execute_sql(
        f"""
        CREATE TABLE IF NOT EXISTS `{table}` (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            trade_date DATE NOT NULL,
            index_code VARCHAR(20) NOT NULL,
            manual_state VARCHAR(16) NOT NULL,
            is_manual_bottom_region TINYINT NOT NULL,
            is_manual_top_region TINYINT NOT NULL,
            is_manual_buy_window TINYINT NOT NULL,
            is_manual_sell_window TINYINT NOT NULL,
            label_mode VARCHAR(32) NOT NULL,
            bottom_level_score DECIMAL(12,6) NOT NULL,
            top_level_score DECIMAL(12,6) NOT NULL,
            bottom_confidence TINYINT NOT NULL,
            top_confidence TINYINT NOT NULL,
            bottom_cycle_score TINYINT NOT NULL,
            top_cycle_score TINYINT NOT NULL,
            bottom_label_freq_score TINYINT NOT NULL,
            top_label_freq_score TINYINT NOT NULL,
            manual_bottom_region_ids VARCHAR(255) NOT NULL,
            manual_top_region_ids VARCHAR(255) NOT NULL,
            manual_bottom_cycles VARCHAR(64) NOT NULL,
            manual_top_cycles VARCHAR(64) NOT NULL,
            manual_bottom_label_freqs VARCHAR(64) NOT NULL,
            manual_top_label_freqs VARCHAR(64) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uk_manual_turning_daily (trade_date, index_code),
            KEY idx_manual_state (manual_state, trade_date),
            KEY idx_manual_buy_window (is_manual_buy_window, trade_date),
            KEY idx_manual_sell_window (is_manual_sell_window, trade_date)
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
    if "label_mode" not in existing:
        execute_sql(
            f"ALTER TABLE `{table}` ADD COLUMN `label_mode` VARCHAR(32) NOT NULL "
            "DEFAULT 'post_hoc_weak' AFTER `is_manual_sell_window`"
        )
    additions = [
        ("bottom_cycle_score", "TINYINT NOT NULL DEFAULT 0 AFTER `top_confidence`"),
        ("top_cycle_score", "TINYINT NOT NULL DEFAULT 0 AFTER `bottom_cycle_score`"),
        ("bottom_label_freq_score", "TINYINT NOT NULL DEFAULT 0 AFTER `top_cycle_score`"),
        ("top_label_freq_score", "TINYINT NOT NULL DEFAULT 0 AFTER `bottom_label_freq_score`"),
        ("manual_bottom_cycles", "VARCHAR(64) NOT NULL DEFAULT '' AFTER `manual_top_region_ids`"),
        ("manual_top_cycles", "VARCHAR(64) NOT NULL DEFAULT '' AFTER `manual_bottom_cycles`"),
        ("manual_bottom_label_freqs", "VARCHAR(64) NOT NULL DEFAULT '' AFTER `manual_top_cycles`"),
        ("manual_top_label_freqs", "VARCHAR(64) NOT NULL DEFAULT '' AFTER `manual_bottom_label_freqs`"),
    ]
    for column, definition in additions:
        if column not in existing:
            execute_sql(f"ALTER TABLE `{table}` ADD COLUMN `{column}` {definition}")


def _summary(regions: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    region_data = regions.copy()
    region_data["split"] = "train_manual_2016_2021"
    region_data.loc[region_data["start_date"] < pd.Timestamp("2016-01-01"), "split"] = "pretrain_reference"
    region_data.loc[region_data["start_date"] >= pd.Timestamp("2022-01-01"), "split"] = "oot_diagnostic_2022_plus"
    for (split, region_type), part in region_data.groupby(["split", "region_type"], sort=True):
        rows.append(
            {
                "scope": "regions",
                "split": split,
                "type": region_type,
                "rows": int(len(part)),
                "usable_rows": int(part["usable_for_signal"].sum()),
                "major_rows": int((part["level"] == "major").sum()),
                "confidence_avg": float(part["confidence"].mean()),
            }
        )
    daily_data = daily.copy()
    daily_data["trade_date"] = pd.to_datetime(daily_data["trade_date"])
    daily_data["split"] = "train_manual_2016_2021"
    daily_data.loc[daily_data["trade_date"] < pd.Timestamp("2016-01-01"), "split"] = "pretrain_reference"
    daily_data.loc[daily_data["trade_date"] >= pd.Timestamp("2022-01-01"), "split"] = "oot_diagnostic_2022_plus"
    for split, part in daily_data.groupby("split", sort=True):
        rows.append(
            {
                "scope": "daily",
                "split": split,
                "type": "bottom",
                "rows": int(part["is_manual_bottom_region"].sum()),
                "usable_rows": int(part["is_manual_buy_window"].sum()),
                "major_rows": int(((part["bottom_level_score"] >= 9) & (part["is_manual_bottom_region"] == 1)).sum()),
                "confidence_avg": float(
                    part.loc[part["is_manual_bottom_region"] == 1, "bottom_confidence"].mean()
                )
                if int(part["is_manual_bottom_region"].sum())
                else 0.0,
            }
        )
        rows.append(
            {
                "scope": "daily",
                "split": split,
                "type": "top",
                "rows": int(part["is_manual_top_region"].sum()),
                "usable_rows": int(part["is_manual_sell_window"].sum()),
                "major_rows": int(((part["top_level_score"] >= 9) & (part["is_manual_top_region"] == 1)).sum()),
                "confidence_avg": float(
                    part.loc[part["is_manual_top_region"] == 1, "top_confidence"].mean()
                )
                if int(part["is_manual_top_region"].sum())
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def build_manual_turning_labels(path: str | Path | None = None, write_database: bool = True) -> pd.DataFrame:
    cfg = get_config("bottom_model.yaml")
    regions, daily = build_manual_turning_daily(path)
    table = str(cfg["outputs"]["manual_daily_table"])
    if write_database:
        _ensure_table(table)
        upsert_dataframe(daily, table, ["trade_date", "index_code"])
    report = _summary(regions, daily)
    report_path = project_path(str(cfg["outputs"]["manual_region_report"]))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False, encoding="utf-8-sig")
    logger.info("built manual turning labels rows=%s table=%s", len(daily), table)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and expand manual bottom/top regions to daily labels.")
    parser.add_argument("--path")
    parser.add_argument("--no-database", action="store_true")
    args = parser.parse_args()
    report = build_manual_turning_labels(args.path, write_database=not args.no_database)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
