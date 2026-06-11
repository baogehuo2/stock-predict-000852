from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.common.config import get_config, project_path
from src.common.db import execute_sql, upsert_dataframe
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.logger import get_logger
from src.modeling.data import load_dataset


disable_broken_dask_autoload()

logger = get_logger(__name__)

SIGNAL_COLUMNS = [
    "trade_date",
    "model_version",
    "feature_version",
    "model_tag",
    "horizon",
    "causal_regime",
    "buy_proba",
    "buy_threshold",
    "direction",
    "signal_strength",
    "pred_ret",
    "future_ret_7d",
]


def _ensure_signal_table() -> None:
    execute_sql(
        """
        CREATE TABLE IF NOT EXISTS buy_signal_daily (
            id BIGINT PRIMARY KEY AUTO_INCREMENT,
            trade_date DATE NOT NULL,
            model_version VARCHAR(64) NOT NULL,
            feature_version VARCHAR(64) NOT NULL,
            model_tag VARCHAR(64) NOT NULL,
            horizon INT NOT NULL,
            causal_regime VARCHAR(16) NOT NULL,
            buy_proba DECIMAL(12,8) NOT NULL,
            buy_threshold DECIMAL(12,8) NOT NULL,
            direction VARCHAR(16) NOT NULL,
            signal_strength VARCHAR(16) NOT NULL,
            pred_ret DECIMAL(12,8) NULL,
            future_ret_7d DECIMAL(12,8) NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY uk_buy_signal (trade_date, model_version, horizon),
            KEY idx_buy_signal_date (trade_date),
            KEY idx_buy_signal_direction (direction)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
        """
    )


def _refresh_realized_returns(model_version: str) -> None:
    execute_sql(
        """
        UPDATE buy_signal_daily AS s
        JOIN model_dataset_daily AS d
          ON d.trade_date = s.trade_date
         AND d.index_code = :index_code
        SET s.future_ret_7d = d.future_ret_7d
        WHERE s.model_version = :model_version
          AND s.future_ret_7d IS NULL
          AND d.future_ret_7d IS NOT NULL
        """,
        {
            "index_code": get_config()["project"]["target_index"],
            "model_version": model_version,
        },
    )


def _select_rows(
    data: pd.DataFrame,
    trade_date: str | None,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if trade_date and (start_date or end_date):
        raise ValueError("trade_date cannot be combined with start_date or end_date.")
    selected = data.sort_values("trade_date").copy()
    if trade_date:
        selected = selected[selected["trade_date"] == pd.Timestamp(trade_date)]
    elif start_date or end_date:
        if start_date:
            selected = selected[selected["trade_date"] >= pd.Timestamp(start_date)]
        if end_date:
            selected = selected[selected["trade_date"] <= pd.Timestamp(end_date)]
    else:
        selected = selected.tail(1)
    if selected.empty:
        raise RuntimeError("No dataset rows matched the requested signal dates.")
    return selected


def _causal_regime(data: pd.DataFrame) -> pd.Series:
    regime = pd.Series("neutral", index=data.index, dtype="object")
    regime.loc[pd.to_numeric(data["f_is_bull_trend"], errors="coerce").fillna(0) == 1] = "bull"
    regime.loc[pd.to_numeric(data["f_is_bear_trend"], errors="coerce").fillna(0) == 1] = "bear"
    return regime


def _load_bundle(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Buy model not found: {path}")
    bundle = joblib.load(path)
    required = {"classifier", "features", "horizon", "feature_version"}
    missing = sorted(required - set(bundle))
    if missing:
        raise RuntimeError(f"Buy model bundle is missing metadata: {missing}")
    return bundle


def build_buy_signal_frame(
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    cfg = get_config()
    buy_cfg = cfg.get("binary_buy_model", {})
    model_version = str(buy_cfg.get("model_version", "buy-signal-v1.0"))
    model_tag = str(buy_cfg.get("model_tag", "buy_v1_7d"))
    model_file = str(
        buy_cfg.get("final_model_file", "models/buy_lgbm_buy_signal_v1_7d.joblib")
    )
    buy_threshold = float(buy_cfg.get("signal_threshold", 0.60))
    bundle = _load_bundle(project_path(model_file))
    horizon = int(bundle["horizon"])
    if horizon != 7:
        raise RuntimeError(f"Model 1.0 signal generator requires horizon=7, got {horizon}.")
    if bundle.get("model_version") != model_version:
        raise RuntimeError(
            f"Model version mismatch: config={model_version}, bundle={bundle.get('model_version')}."
        )

    data = load_dataset()
    if data.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    selected = _select_rows(data, trade_date, start_date, end_date)
    features = list(bundle["features"])
    missing = [feature for feature in features if feature not in selected.columns]
    if missing:
        raise RuntimeError(f"Signal dataset is missing model features: {missing}")

    classifier = bundle["classifier"]
    classes = list(classifier.named_steps["model"].classes_)
    buy_class_index = classes.index(1)
    buy_proba = classifier.predict_proba(selected[features])[:, buy_class_index]
    output = pd.DataFrame(
        {
            "trade_date": selected["trade_date"].dt.date,
            "model_version": model_version,
            "feature_version": str(bundle["feature_version"]),
            "model_tag": model_tag,
            "horizon": horizon,
            "causal_regime": _causal_regime(selected),
            "buy_proba": buy_proba,
            "buy_threshold": buy_threshold,
            "pred_ret": None,
            "future_ret_7d": pd.to_numeric(selected.get("future_ret_7d"), errors="coerce"),
        },
        index=selected.index,
    )
    output["direction"] = "neutral"
    long_mask = (output["causal_regime"] == "bull") & (output["buy_proba"] >= buy_threshold)
    output.loc[long_mask, "direction"] = "long"
    output["signal_strength"] = "normal"
    output = output.reset_index(drop=True)[SIGNAL_COLUMNS]
    if output.duplicated(["trade_date", "model_version", "horizon"]).any():
        raise RuntimeError("Signal output contains duplicate date/model/horizon rows.")
    return output


def generate_buy_signals(
    trade_date: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    output_csv: str = "data/reports/buy_signal_v1.csv",
    write_database: bool = True,
) -> pd.DataFrame:
    signals = build_buy_signal_frame(trade_date, start_date, end_date)
    output_path = project_path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output_path, index=False, encoding="utf-8-sig")
    if write_database:
        _ensure_signal_table()
        _refresh_realized_returns(str(signals["model_version"].iloc[0]))
        upsert_dataframe(
            signals,
            "buy_signal_daily",
            ["trade_date", "model_version", "horizon"],
        )
    logger.info(
        "generated buy signals rows=%s long=%s neutral=%s path=%s",
        len(signals),
        int((signals["direction"] == "long").sum()),
        int((signals["direction"] == "neutral").sum()),
        output_path,
    )
    return signals


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate unified 7-day Buy model 1.0 signals.")
    parser.add_argument("--trade-date")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--output-csv", default="data/reports/buy_signal_v1.csv")
    parser.add_argument("--no-database", action="store_true")
    args = parser.parse_args()
    signals = generate_buy_signals(
        trade_date=args.trade_date,
        start_date=args.start_date,
        end_date=args.end_date,
        output_csv=args.output_csv,
        write_database=not args.no_database,
    )
    print(signals.to_string(index=False))


if __name__ == "__main__":
    main()
