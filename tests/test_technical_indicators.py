from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.technical_indicators import kdj, rsi_cn, sma_cn, wr


class TechnicalIndicatorTests(unittest.TestCase):
    def test_sma_cn_uses_recursive_previous_value(self) -> None:
        values = pd.Series([1.0, 2.0, 3.0])

        result = sma_cn(values, window=3, weight=1)

        self.assertAlmostEqual(result.iloc[0], 1.0)
        self.assertAlmostEqual(result.iloc[1], (2.0 + 2.0 * 1.0) / 3.0)
        self.assertAlmostEqual(result.iloc[2], (3.0 + 2.0 * result.iloc[1]) / 3.0)

    def test_rsi_cn_differs_from_simple_rolling_rsi(self) -> None:
        close = pd.Series([10.0, 11.0, 10.5, 10.0, 10.2, 10.1, 10.8, 10.4, 10.3])

        result = rsi_cn(close, window=6)
        delta = close.diff()
        rolling_gain = delta.clip(lower=0).rolling(6).mean()
        rolling_loss = (-delta.clip(upper=0)).rolling(6).mean()
        rolling_rsi = 100 - 100 / (1 + rolling_gain / rolling_loss.replace(0, np.nan))

        self.assertFalse(np.isclose(result.iloc[-1], rolling_rsi.iloc[-1]))
        self.assertTrue(0 <= result.dropna().iloc[-1] <= 100)

    def test_kdj_initializes_k_and_d_from_50(self) -> None:
        high = pd.Series([10.0, 11.0])
        low = pd.Series([9.0, 9.5])
        close = pd.Series([9.5, 10.5])

        result = kdj(high, low, close)

        self.assertAlmostEqual(result.loc[0, "kdj_k"], 50.0)
        self.assertAlmostEqual(result.loc[0, "kdj_d"], 50.0)
        self.assertAlmostEqual(result.loc[1, "kdj_k"], (75.0 + 2 * 50.0) / 3, places=8)

    def test_wr_negative_keeps_oversold_threshold_direction(self) -> None:
        high = pd.Series([10.0] * 14)
        low = pd.Series([0.0] * 14)
        close = pd.Series([1.0] * 14)

        result = wr(high, low, close, window=14, sign="negative")

        self.assertAlmostEqual(result.iloc[-1], -90.0)
        self.assertLessEqual(result.iloc[-1], -80.0)


if __name__ == "__main__":
    unittest.main()
