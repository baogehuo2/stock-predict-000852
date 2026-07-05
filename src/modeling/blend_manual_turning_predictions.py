from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import project_path


PREDICTION_TEMPLATE = "data/reports/manual_weak_turning_walk_forward_predictions_{model_kind}_wf.csv"
DEFAULT_MODELS = ["l1_logistic_kbest", "xgboost_small", "svm"]


def _normalize_weights(models: list[str], weights: list[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / len(models)] * len(models)
    if len(weights) != len(models):
        raise ValueError("weights length must match models length.")
    total = sum(float(value) for value in weights)
    if total <= 0:
        raise ValueError("weights sum must be positive.")
    return [float(value) / total for value in weights]


def _load_prediction(model_kind: str) -> pd.DataFrame:
    path = project_path(PREDICTION_TEMPLATE.format(model_kind=model_kind))
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path, dtype={"index_code": str}, parse_dates=["trade_date"])
    required = ["trade_date", "index_code", "manual_weak_bottom_proba", "manual_weak_top_proba"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")
    return data


def blend_predictions(
    models: list[str] | None = None,
    weights: list[float] | None = None,
    output: str = "data/reports/manual_weak_turning_walk_forward_predictions_blend_equal_wf.csv",
    bottom_rescue_model: str | None = None,
    bottom_rescue_threshold: float | None = None,
    top_rescue_model: str | None = None,
    top_rescue_threshold: float | None = None,
) -> Path:
    model_list = models or DEFAULT_MODELS
    weight_list = _normalize_weights(model_list, weights)
    frames = []
    for model_kind, weight in zip(model_list, weight_list):
        frame = _load_prediction(model_kind).copy()
        frame["blend_model"] = model_kind
        frame["blend_weight"] = weight
        frames.append(frame)

    base_cols = [
        "trade_date",
        "index_code",
        "is_manual_buy_window",
        "is_manual_sell_window",
        "manual_state",
        "label_mode",
        "future_ret_15d",
        "future_mfe_15d",
        "future_mae_15d",
        "quality_bottom_label",
        "continuation_risk_label",
        "test_year",
        "train_start",
        "train_end",
        "train_rows",
    ]
    base = frames[0][[column for column in base_cols if column in frames[0].columns]].copy()
    key_cols = ["trade_date", "index_code"]
    for frame, model_kind, weight in zip(frames, model_list, weight_list):
        proba = frame[
            [
                "trade_date",
                "index_code",
                "manual_weak_bottom_proba",
                "manual_weak_top_proba",
            ]
        ].copy()
        proba[f"{model_kind}_bottom_proba"] = pd.to_numeric(
            proba["manual_weak_bottom_proba"], errors="coerce"
        )
        proba[f"{model_kind}_top_proba"] = pd.to_numeric(
            proba["manual_weak_top_proba"], errors="coerce"
        )
        proba = proba.drop(columns=["manual_weak_bottom_proba", "manual_weak_top_proba"])
        base = base.merge(proba, on=key_cols, how="inner", validate="one_to_one")
        base[f"{model_kind}_weight"] = weight

    bottom = pd.Series(0.0, index=base.index)
    top = pd.Series(0.0, index=base.index)
    for model_kind, weight in zip(model_list, weight_list):
        bottom += pd.to_numeric(base[f"{model_kind}_bottom_proba"], errors="coerce").fillna(0.0) * weight
        top += pd.to_numeric(base[f"{model_kind}_top_proba"], errors="coerce").fillna(0.0) * weight
    base["blend_bottom_proba"] = bottom
    base["blend_top_proba"] = top
    base["manual_weak_bottom_proba"] = bottom
    base["manual_weak_top_proba"] = top
    base_bottom_signal = (base["blend_bottom_proba"] >= 0.70) & (base["blend_top_proba"] < 0.50)
    base_top_signal = (base["blend_top_proba"] >= 0.70) & (base["blend_bottom_proba"] < 0.50)
    bottom_rescue_signal = pd.Series(False, index=base.index)
    top_rescue_signal = pd.Series(False, index=base.index)
    if bottom_rescue_model and bottom_rescue_threshold is not None:
        bottom_col = f"{bottom_rescue_model}_bottom_proba"
        top_col = f"{bottom_rescue_model}_top_proba"
        if bottom_col not in base or top_col not in base:
            raise RuntimeError(f"Missing rescue columns for {bottom_rescue_model}.")
        bottom_rescue_signal = (
            (pd.to_numeric(base[bottom_col], errors="coerce") >= bottom_rescue_threshold)
            & (pd.to_numeric(base[top_col], errors="coerce") < 0.50)
        )
    if top_rescue_model and top_rescue_threshold is not None:
        top_col = f"{top_rescue_model}_top_proba"
        bottom_col = f"{top_rescue_model}_bottom_proba"
        if top_col not in base or bottom_col not in base:
            raise RuntimeError(f"Missing rescue columns for {top_rescue_model}.")
        top_rescue_signal = (
            (pd.to_numeric(base[top_col], errors="coerce") >= top_rescue_threshold)
            & (pd.to_numeric(base[bottom_col], errors="coerce") < 0.50)
        )
    base["weak_combined_bottom_signal"] = (base_bottom_signal | bottom_rescue_signal).astype(int)
    base["weak_combined_top_signal"] = (base_top_signal | top_rescue_signal).astype(int)
    if bottom_rescue_model:
        rescue_bottom_col = f"{bottom_rescue_model}_bottom_proba"
        base.loc[bottom_rescue_signal, "manual_weak_bottom_proba"] = pd.to_numeric(
            base.loc[bottom_rescue_signal, rescue_bottom_col], errors="coerce"
        )
    if top_rescue_model:
        rescue_top_col = f"{top_rescue_model}_top_proba"
        base.loc[top_rescue_signal, "manual_weak_top_proba"] = pd.to_numeric(
            base.loc[top_rescue_signal, rescue_top_col], errors="coerce"
        )
    base["bottom_signal_source"] = ""
    base.loc[base_bottom_signal, "bottom_signal_source"] = "blend"
    base.loc[bottom_rescue_signal, "bottom_signal_source"] = bottom_rescue_model or ""
    base.loc[base_bottom_signal & bottom_rescue_signal, "bottom_signal_source"] = "blend+rescue"
    base["top_signal_source"] = ""
    base.loc[base_top_signal, "top_signal_source"] = "blend"
    base.loc[top_rescue_signal, "top_signal_source"] = top_rescue_model or ""
    base.loc[base_top_signal & top_rescue_signal, "top_signal_source"] = "blend+rescue"
    base["blend_members"] = ",".join(model_list)
    base["blend_weights"] = ",".join(f"{weight:.6f}" for weight in weight_list)
    if bottom_rescue_model:
        base["bottom_rescue_rule"] = f"{bottom_rescue_model}>={bottom_rescue_threshold}"
    if top_rescue_model:
        base["top_rescue_rule"] = f"{top_rescue_model}>={top_rescue_threshold}"
    output_path = project_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Blend manual turning walk-forward prediction probabilities.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--weights", nargs="+", type=float)
    parser.add_argument("--output", default="data/reports/manual_weak_turning_walk_forward_predictions_blend_equal_wf.csv")
    parser.add_argument("--bottom-rescue-model")
    parser.add_argument("--bottom-rescue-threshold", type=float)
    parser.add_argument("--top-rescue-model")
    parser.add_argument("--top-rescue-threshold", type=float)
    args = parser.parse_args()
    print(
        blend_predictions(
            args.models,
            args.weights,
            args.output,
            args.bottom_rescue_model,
            args.bottom_rescue_threshold,
            args.top_rescue_model,
            args.top_rescue_threshold,
        )
    )


if __name__ == "__main__":
    main()
