from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features.build_bottom_dataset import add_bottom_labels, build_causal_weekly_features
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
