from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features.build_bottom_dataset import add_bottom_labels, build_causal_weekly_features
from src.features.build_bottom_weekly_dataset import (
    add_manual_weekly_labels,
    apply_auto_future_return_labels,
    build_monthly_state_features,
    build_weekly_event_features,
    build_weekly_feature_frame,
    build_weekly_market_bars,
)
from src.features.build_manual_turning_labels import build_manual_turning_daily
from src.modeling.build_optimized_turning_signals import (
    select_optimized_bottom_signals,
    select_optimized_top_signals,
)
from src.modeling.turning_signal_postprocess import (
    assign_bottom_signal_layers,
    compress_top_signal_regions,
)
from src.modeling.walk_forward_bottom_lgbm import _non_overlapping
from src.modeling.walk_forward_bottom_weekly_lgbm import (
    _balanced_resample,
    _fit_model,
    _negative_bagging_fraction,
    _positive_probability,
    _positive_weight,
)


class BottomLabelTests(unittest.TestCase):
    def test_quality_label_uses_target_path_and_terminal_return(self) -> None:
        dates = pd.date_range("2024-01-01", periods=17, freq="D")
        close = pd.Series(
            [100.0, 100.0, 100.5, 100.4, 100.6, 100.7, 100.8, 101.0, 101.1,
             101.2, 101.3, 101.4, 101.5, 101.6, 101.7, 101.8, 101.9]
        )
        target = pd.DataFrame(
            {
                "trade_date": dates,
                "index_code": "000852",
                "close": close,
                "high": [100.2, 100.5, 103.0, 100.8, 101.0, 101.1, 101.2, 101.3, 101.4,
                         101.5, 101.6, 101.7, 101.8, 101.9, 102.0, 102.1, 102.2],
                "low": [99.8, 99.2, 99.5, 100.0, 100.1, 100.2, 100.3, 100.5, 100.6,
                        100.7, 100.8, 100.9, 101.0, 101.1, 101.2, 101.3, 101.4],
            }
        )
        frame = target[["trade_date", "index_code"]].copy()
        config = {
            "model": {"horizon": 15},
            "labels": {
                "terminal_return": 0.01,
                "rebound_target": 0.025,
                "quality_terminal_return": 0.005,
                "quality_max_adverse_excursion": -0.015,
                "path_max_adverse_excursion": -0.02,
                "continuation_risk": -0.025,
            },
        }

        result = add_bottom_labels(frame, target, config)

        self.assertEqual(result.loc[0, "target_hit_day"], 2)
        self.assertAlmostEqual(result.loc[0, "pre_target_mae"], -0.008)
        self.assertEqual(result.loc[0, "quality_bottom_label"], 1)
        self.assertEqual(result.loc[0, "continuation_risk_label"], 0)
        self.assertTrue(np.isnan(result.loc[2, "quality_bottom_label"]))

    def test_non_overlapping_uses_market_positions(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-20"]),
                "trade_pos": [1, 3, 16],
            }
        )

        selected = _non_overlapping(signals, horizon=15)

        self.assertEqual(selected["trade_pos"].tolist(), [1, 16])

    def test_weekly_features_do_not_use_later_days_in_same_week(self) -> None:
        dates = pd.to_datetime(
            [
                "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
            ]
        )
        market = pd.DataFrame(
            {
                "trade_date": dates,
                "open": np.arange(100, 110, dtype=float),
                "high": np.arange(101, 111, dtype=float),
                "low": np.arange(99, 109, dtype=float),
                "close": np.arange(100.5, 110.5, dtype=float),
                "volume": np.arange(1000, 11000, 1000, dtype=float),
            }
        )

        prefix = build_causal_weekly_features(market.iloc[:8].copy()).iloc[-1]
        full = build_causal_weekly_features(market).iloc[7]

        for column in build_causal_weekly_features(market).columns.drop("trade_date"):
            left = prefix[column]
            right = full[column]
            if pd.isna(left) and pd.isna(right):
                continue
            self.assertAlmostEqual(float(left), float(right), places=12, msg=column)

    def test_weekly_volume_features_use_average_daily_volume(self) -> None:
        dates = pd.to_datetime(
            [
                "2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
                "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12",
                "2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19",
                "2024-01-22", "2024-01-23", "2024-01-24", "2024-01-25", "2024-01-26",
                "2024-01-29", "2024-01-30", "2024-01-31", "2024-02-01", "2024-02-02",
                "2024-02-05", "2024-02-06", "2024-02-07", "2024-02-08",
            ]
        )
        rows = []
        close = 100.0
        for date in dates:
            close += 0.5
            rows.append(
                {
                    "trade_date": date,
                    "index_code": "000852",
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0,
                }
            )
        daily = pd.DataFrame(rows)
        weekly = build_weekly_market_bars(daily)
        cfg = {
            "candidate": {
                "volume_expand_ratio": 1.2,
                "bottom_min_conditions": 2,
                "top_min_conditions": 2,
                "bottom_drawdown_13w": -0.08,
                "bottom_boll_position": 0.20,
                "bottom_rsi6": 35.0,
                "bottom_kdj_j": 20.0,
                "top_runup_13w": 0.12,
                "top_boll_position": 0.80,
                "top_rsi6": 70.0,
                "top_kdj_j": 85.0,
            }
        }

        features = build_weekly_feature_frame(weekly, weekly, cfg)

        self.assertEqual(weekly["trade_days"].tolist(), [5, 5, 5, 5, 5, 4])
        self.assertEqual(weekly["volume"].tolist()[-2:], [5000.0, 4000.0])
        self.assertEqual(weekly["avg_daily_volume"].tolist()[-2:], [1000.0, 1000.0])
        self.assertAlmostEqual(features.loc[5, "week_avg_daily_volume_ratio_5w"], 1.0)
        self.assertAlmostEqual(features.loc[5, "week_close_position"], (114.5 - 112.5) / (115.0 - 112.5))
        self.assertAlmostEqual(features.loc[5, "week_open_position"], (112.8 - 112.5) / (115.0 - 112.5))
        self.assertAlmostEqual(features.loc[5, "week_close_to_low"], 114.5 / 112.5 - 1)
        self.assertAlmostEqual(features.loc[5, "week_high_to_close"], 115.0 / 114.5 - 1)

    def test_manual_weekly_labels_overlap_week_ranges(self) -> None:
        weeks = pd.DataFrame(
            {
                "week_start_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
                "week_end_date": pd.to_datetime(["2024-01-05", "2024-01-12"]),
            }
        )
        regions = pd.DataFrame(
            {
                "region_id": ["B202401"],
                "start_date": pd.to_datetime(["2024-01-03"]),
                "end_date": pd.to_datetime(["2024-01-10"]),
                "region_type": ["bottom"],
                "cycle": ["medium"],
                "level": ["swing"],
                "label_freq": ["weekly"],
                "confidence": [2],
                "usable_for_signal": [1],
                "entry_start": pd.to_datetime(["2024-01-05"]),
                "entry_end": pd.to_datetime(["2024-01-08"]),
                "exit_start": [pd.NaT],
                "exit_end": [pd.NaT],
                "reason": [""],
                "notes": [""],
            }
        )
        cfg = {"manual_labels": {"label_mode": "post_hoc_weekly_weak"}}

        labeled = add_manual_weekly_labels(weeks, regions, cfg)

        self.assertEqual(labeled["is_manual_bottom_region"].tolist(), [1, 1])
        self.assertEqual(labeled["manual_weak_bottom_label"].tolist(), [1, 1])
        self.assertEqual(labeled["bottom_level_score"].tolist(), [4.0, 4.0])

    def test_auto_future_return_weekly_labels_use_three_week_thresholds(self) -> None:
        frame = pd.DataFrame(
            {
                "future_ret_3w": [0.051, 0.050, -0.051, -0.050, np.nan],
                "manual_weak_bottom_label": [0, 0, 0, 0, 1],
                "manual_weak_top_label": [0, 0, 0, 0, 1],
                "label_mode": ["manual"] * 5,
                "manual_state": ["neutral"] * 5,
            }
        )
        cfg = {
            "model": {"horizon_weeks": 3},
            "manual_labels": {
                "label_source": "auto_future_return",
                "label_mode": "auto_future_return_3w_5pct",
                "auto_bottom_return_threshold": 0.05,
                "auto_top_return_threshold": -0.05,
            },
        }

        labeled = apply_auto_future_return_labels(frame, cfg)

        self.assertEqual(labeled["manual_weak_bottom_label"].tolist(), [1, 0, 0, 0, 0])
        self.assertEqual(labeled["manual_weak_top_label"].tolist(), [0, 0, 1, 0, 0])
        self.assertEqual(
            labeled["manual_state"].tolist(),
            ["bottom", "neutral", "top", "neutral", "neutral"],
        )
        self.assertEqual(set(labeled["label_mode"]), {"auto_future_return_3w_5pct"})

    def test_monthly_state_features_only_use_days_available_by_week_end(self) -> None:
        dates = pd.bdate_range("2023-01-02", "2024-04-30")
        close = 100.0
        rows = []
        for date in dates:
            close += 0.2
            if date > pd.Timestamp("2024-04-12"):
                close += 10.0
            rows.append(
                {
                    "trade_date": date,
                    "index_code": "000852",
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000.0,
                }
            )
        daily = pd.DataFrame(rows)
        weekly = pd.DataFrame({"week_end_date": pd.to_datetime(["2024-04-12", "2024-04-26"])})

        features = build_monthly_state_features(daily, weekly)

        first = features.iloc[0]
        second = features.iloc[1]
        available = daily[daily["trade_date"] <= pd.Timestamp("2024-04-12")]
        april = available[available["trade_date"].dt.to_period("M") == pd.Period("2024-04")]
        expected_position = (
            april["close"].iloc[-1] - april["low"].min()
        ) / (april["high"].max() - april["low"].min())

        self.assertAlmostEqual(first["month_close_position"], expected_position)
        self.assertLess(first["month_ret_1m"], second["month_ret_1m"])

    def test_weekly_event_features_only_use_current_week_events(self) -> None:
        weeks = pd.DataFrame(
            {
                "week_start_date": pd.to_datetime(["2024-01-01", "2024-01-08"]),
                "week_end_date": pd.to_datetime(["2024-01-05", "2024-01-12"]),
            }
        )
        events = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(["2024-01-05", "2024-01-08"]),
                "event_type": ["政策", "宏观"],
                "event_stage": ["确认", "确认"],
                "expectation_level": ["中性", "中性"],
                "surprise_level": ["符合预期", "符合预期"],
                "affected_style": ["小盘", "成长"],
                "impact_direction": ["利多", "利空"],
                "impact_strength": [2, 3],
                "event_score": [2.0, -3.0],
            }
        )

        features = build_weekly_event_features(weeks, events)

        self.assertEqual(features["f_event_week_all_count"].tolist(), [1.0, 1.0])
        self.assertEqual(features["f_event_week_all_score_sum"].tolist(), [2.0, -3.0])
        self.assertEqual(features["f_event_week_policy_count"].tolist(), [1.0, 0.0])
        self.assertEqual(features["f_event_week_macro_count"].tolist(), [0.0, 1.0])

    def test_weekly_imbalance_helpers_cap_weight_and_resample_negatives(self) -> None:
        frame = pd.DataFrame(
            {
                "week_pos": range(20),
                "label": [1, 1, *([0] * 18)],
                "feature": np.arange(20, dtype=float),
            }
        )
        imbalance = {
            "enabled": True,
            "max_scale_pos_weight": 5.0,
            "negative_sample_ratio": 3.0,
        }

        weight = _positive_weight(frame["label"], imbalance)
        bagging_fraction = _negative_bagging_fraction(frame["label"], imbalance, {})
        sampled = _balanced_resample(frame, "label", negative_ratio=3.0, random_state=42)

        self.assertEqual(weight, 5.0)
        self.assertAlmostEqual(bagging_fraction, 6 / 18)
        self.assertEqual(int((sampled["label"] == 1).sum()), 2)
        self.assertEqual(int((sampled["label"] == 0).sum()), 6)

    def test_weekly_lda_qda_models_fit_and_predict_probability(self) -> None:
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(
            {
                "f_a": rng.normal(size=60),
                "f_b": rng.normal(size=60),
                "f_c": rng.normal(size=60),
            }
        )
        frame["label"] = ((frame["f_a"] + frame["f_b"] * 0.5) > 0).astype(int)
        features = ["f_a", "f_b", "f_c"]

        for model_name, params in [
            ("lda", {"solver": "lsqr", "shrinkage": "auto"}),
            ("qda", {"reg_param": 0.50}),
        ]:
            model, usable = _fit_model(model_name, frame, features, "label", 42, params)
            proba = _positive_probability(model, frame, usable)
            self.assertEqual(len(proba), len(frame))
            self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

    def test_manual_labels_are_marked_as_weak(self) -> None:
        regions = pd.DataFrame(
            [
                {
                    "region_id": "B1",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-03",
                    "region_type": "bottom",
                    "cycle": "short",
                    "level": "minor",
                    "label_freq": "daily",
                    "confidence": 1,
                    "usable_for_signal": 1,
                    "entry_start": "2024-01-01",
                    "entry_end": "2024-01-02",
                    "exit_start": "",
                    "exit_end": "",
                    "reason": "test",
                    "notes": "",
                },
                {
                    "region_id": "T1",
                    "start_date": "2024-01-04",
                    "end_date": "2024-01-05",
                    "region_type": "top",
                    "cycle": "medium",
                    "level": "swing",
                    "label_freq": "weekly",
                    "confidence": 2,
                    "usable_for_signal": 1,
                    "entry_start": "",
                    "entry_end": "",
                    "exit_start": "2024-01-04",
                    "exit_end": "2024-01-05",
                    "reason": "test",
                    "notes": "",
                },
            ]
        )
        trade_dates = pd.DataFrame(
            {
                "trade_date": pd.date_range("2024-01-01", periods=5, freq="D"),
                "index_code": "000852",
            }
        )
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "regions.csv"
            regions.to_csv(path, index=False)
            with patch("src.features.build_manual_turning_labels.assert_bottom_database"), patch(
                "src.features.build_manual_turning_labels._trade_dates",
                return_value=trade_dates,
            ):
                _, daily = build_manual_turning_daily(path=path)

        self.assertIn("label_mode", daily.columns)
        self.assertEqual(set(daily["label_mode"]), {"post_hoc_weak"})
        for column in [
            "bottom_cycle_score",
            "top_cycle_score",
            "bottom_label_freq_score",
            "top_label_freq_score",
            "manual_bottom_cycles",
            "manual_top_cycles",
            "manual_bottom_label_freqs",
            "manual_top_label_freqs",
        ]:
            self.assertIn(column, daily.columns)

    def test_optimized_signal_thresholds_apply_side_veto(self) -> None:
        prediction = pd.DataFrame(
            {
                "manual_weak_bottom_proba": [0.71, 0.72, 0.40],
                "manual_weak_top_proba": [0.49, 0.51, 0.61],
                "manual_weak_risk_proba": [0.44, 0.44, 0.10],
                "f_week_kdj_k_minus_d": [-9.0, -9.0, 0.0],
            }
        )

        bottom = select_optimized_bottom_signals(prediction)
        top = select_optimized_top_signals(prediction)

        self.assertEqual(bottom.tolist(), [True, False, False])
        self.assertEqual(top.tolist(), [False, False, False])

    def test_top_signals_are_compressed_to_representative_regions(self) -> None:
        signals = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-20"]
                ),
                "manual_weak_top_proba": [0.7, 0.9, 0.8],
                "future_ret_15d": [-0.03, -0.04, 0.01],
                "continuation_risk_label": [1, 1, 0],
            }
        )

        representatives, regions = compress_top_signal_regions(signals, max_gap_days=10)

        self.assertEqual(regions["signal_count"].tolist(), [2, 1])
        self.assertEqual(
            representatives["trade_date"].dt.strftime("%Y-%m-%d").tolist(),
            ["2024-01-02", "2024-01-20"],
        )

    def test_bottom_signals_are_layered_by_risk(self) -> None:
        prediction = pd.DataFrame(
            {
                "manual_weak_bottom_proba": [0.55, 0.72, 0.73],
                "manual_weak_top_proba": [0.60, 0.40, 0.40],
                "manual_weak_risk_proba": [0.20, 0.60, 0.30],
                "f_week_kdj_k_minus_d": [0.0, 0.0, 0.0],
            }
        )

        layered = assign_bottom_signal_layers(
            prediction,
            watch_threshold=0.50,
            action_threshold=0.70,
            top_veto_threshold=0.50,
            risk_threshold=0.45,
            week_kdj_min=-10,
        )

        self.assertEqual(
            layered["signal_layer"].tolist(),
            ["bottom_watch", "bottom_candidate", "bottom_action"],
        )


if __name__ == "__main__":
    unittest.main()
