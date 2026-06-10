from __future__ import annotations

import argparse

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.common.config import get_config, project_path
from src.common.lightgbm_compat import disable_broken_dask_autoload
from src.common.logger import get_logger
from src.modeling.data import feature_columns, load_dataset
from src.modeling.train_buy_lgbm import GUBA_SENTIMENT_FEATURES, _parse_csv_floats, _signal_metrics


disable_broken_dask_autoload()
from lightgbm import LGBMClassifier


logger = get_logger(__name__)


def _year_bounds(year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    return pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")


def _fit_buy_model(
    train: pd.DataFrame,
    features: list[str],
    label_col: str,
    random_state: int,
) -> tuple[Pipeline, list[str]]:
    usable_features = [col for col in features if train[col].notna().any()]
    dropped_features = [col for col in features if col not in usable_features]
    if dropped_features:
        logger.warning("drop all-missing training features label=%s features=%s", label_col, dropped_features)

    y_train = train[label_col].astype(int)
    if y_train.nunique() < 2:
        raise RuntimeError(f"{label_col} training label has only one class.")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.03,
                    num_leaves=31,
                    random_state=random_state,
                    class_weight="balanced",
                    verbose=-1,
                ),
            ),
        ]
    )
    model.fit(train[usable_features], y_train)
    return model, usable_features


def _evaluate_year(
    test: pd.DataFrame,
    model: Pipeline,
    features: list[str],
    horizon: int,
    thresholds: list[float],
    train_start: str,
    train_end: str,
    test_year: int,
) -> list[dict]:
    ret_col = f"future_ret_{horizon}d"
    label_col = f"buy_label_{horizon}d"
    part = test.dropna(subset=[ret_col, label_col]).copy()
    if part.empty:
        return []

    part["pred_buy"] = model.predict(part[features]).astype(int)
    proba = model.predict_proba(part[features])
    buy_class_index = list(model.named_steps["model"].classes_).index(1)
    part["buy_proba"] = proba[:, buy_class_index]

    rows = []
    rules: list[tuple[str, pd.DataFrame, float | None]] = [
        ("baseline_all_rows", part, None),
        ("predicted_buy_label", part[part["pred_buy"] == 1], None),
    ]
    for threshold in thresholds:
        rules.append((f"buy_probability", part[part["buy_proba"] >= threshold], threshold))

    for rule, signal, threshold in rules:
        metrics = _signal_metrics(part, signal, ret_col, label_col, rule, threshold)
        metrics.update(
            {
                "test_year": test_year,
                "horizon": horizon,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": f"{test_year}-01-01",
                "test_end": f"{test_year}-12-31",
                "test_rows": int(len(part)),
            }
        )
        rows.append(metrics)
    return rows


def walk_forward_buy_evaluation(
    start_year: int = 2021,
    end_year: int = 2026,
    train_start: str = "2016-01-01",
    horizons: list[int] | None = None,
    thresholds: list[float] | None = None,
    min_train_rows: int = 200,
    output_csv: str | None = None,
) -> list[dict]:
    cfg = get_config()
    buy_cfg = cfg.get("binary_buy_model", {})
    horizons = horizons or [int(h) for h in buy_cfg.get("horizons", [5, 7])]
    thresholds = thresholds or [float(x) for x in buy_cfg.get("probability_thresholds", [0.35, 0.4, 0.45, 0.5, 0.55, 0.6])]

    df = load_dataset()
    if df.empty:
        raise RuntimeError("model_dataset_daily is empty. Build dataset first.")
    features = feature_columns(df)
    if not cfg.get("features", {}).get("use_guba_sentiment", False):
        features = [col for col in features if col not in GUBA_SENTIMENT_FEATURES]
    if not features:
        raise RuntimeError("No feature columns found in dataset.")

    rows = []
    for test_year in range(start_year, end_year + 1):
        test_start, test_end = _year_bounds(test_year)
        train_end = pd.Timestamp(f"{test_year - 1}-12-31")
        train_window_start = pd.Timestamp(train_start)
        train_base = df[(df["trade_date"] >= train_window_start) & (df["trade_date"] <= train_end)].copy()
        test_base = df[(df["trade_date"] >= test_start) & (df["trade_date"] <= test_end)].copy()
        if test_base.empty:
            logger.warning("skip test_year=%s because no rows", test_year)
            continue

        for horizon in horizons:
            ret_col = f"future_ret_{horizon}d"
            label_col = f"buy_label_{horizon}d"
            if label_col not in df.columns:
                raise RuntimeError(f"{label_col} is missing. Run build_dataset first.")
            train_h = train_base.dropna(subset=[ret_col, label_col]).copy()
            if len(train_h) < min_train_rows:
                logger.warning(
                    "skip test_year=%s horizon=%sd because train rows=%s < %s",
                    test_year,
                    horizon,
                    len(train_h),
                    min_train_rows,
                )
                continue
            model, usable_features = _fit_buy_model(
                train_h,
                features,
                label_col,
                random_state=cfg["model"].get("random_state", 42),
            )
            rows.extend(
                _evaluate_year(
                    test_base,
                    model,
                    usable_features,
                    horizon,
                    thresholds,
                    train_start=train_start,
                    train_end=train_end.strftime("%Y-%m-%d"),
                    test_year=test_year,
                )
            )
            logger.info(
                "walk-forward evaluated test_year=%s horizon=%sd train_rows=%s test_rows=%s",
                test_year,
                horizon,
                len(train_h),
                len(test_base),
            )

    if output_csv:
        out_path = project_path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"saved walk-forward buy report path={out_path} rows={len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward yearly evaluation for binary buy models.")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--train-start", default="2016-01-01")
    parser.add_argument("--horizons", default="5,7", help="Comma-separated horizons, e.g. 5,7.")
    parser.add_argument("--thresholds", default=None, help="Comma-separated buy probability thresholds.")
    parser.add_argument("--min-train-rows", type=int, default=200)
    parser.add_argument("--output-csv", help="Optional CSV output path.")
    args = parser.parse_args()

    rows = walk_forward_buy_evaluation(
        start_year=args.start_year,
        end_year=args.end_year,
        train_start=args.train_start,
        horizons=[int(x) for x in args.horizons.split(",") if x.strip()],
        thresholds=_parse_csv_floats(args.thresholds) if args.thresholds else None,
        min_train_rows=args.min_train_rows,
        output_csv=args.output_csv,
    )
    df = pd.DataFrame(rows)
    if df.empty:
        print("No walk-forward rows.")
        return
    cols = [
        "test_year",
        "horizon",
        "rule",
        "threshold",
        "signal_count",
        "coverage",
        "buy_precision",
        "buy_recall",
        "avg_future_ret",
        "median_future_ret",
        "avg_buy_proba",
    ]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
