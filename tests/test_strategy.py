"""Tests for the strategy interface contract."""

import pandas as pd
import numpy as np
import pytest
from strategy import Strategy, Signal


def _make_bars(n: int = 50, base_price: float = 100.0) -> pd.DataFrame:
    """Generate synthetic OHLCV bars for testing."""
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": prices + np.random.randn(n) * 0.1,
        "high": prices + abs(np.random.randn(n) * 0.5),
        "low": prices - abs(np.random.randn(n) * 0.5),
        "close": prices,
        "volume": np.random.randint(1000, 100000, n).astype(float),
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="5min")
    return df


class TestStrategyInterface:
    def test_has_version(self):
        s = Strategy({"rsi_period": 14, "ema_period": 20, "universe": ["SPY"]})
        assert hasattr(s, "VERSION")
        assert isinstance(s.VERSION, str)

    def test_compute_indicators_adds_columns(self):
        s = Strategy({"rsi_period": 14, "ema_period": 20, "universe": ["SPY"]})
        df = _make_bars()
        result = s.compute_indicators(df)
        assert "rsi" in result.columns
        assert "ema" in result.columns
        assert "atr" in result.columns
        assert "volume_ratio" in result.columns

    def test_generate_signals_returns_list(self):
        s = Strategy({"rsi_period": 14, "ema_period": 20, "volume_multiplier": 0.1,
                       "volume_lookback": 5, "atr_period": 14, "universe": ["SPY"]})
        bars = {"SPY": _make_bars()}
        signals = s.generate_signals(bars)
        assert isinstance(signals, list)

    def test_signal_has_required_fields(self):
        sig = Signal(
            symbol="SPY", side="long", signal_type="test",
            strength=0.8, rsi=35, ema=100, atr=2, volume_ratio=2.0,
            price=101, context={},
        )
        assert sig.symbol == "SPY"
        assert sig.side in ("long", "short")
        assert 0 <= sig.strength <= 1
        assert sig.atr > 0

    def test_should_exit_returns_tuple(self):
        s = Strategy({"rsi_period": 14, "ema_period": 20, "universe": ["SPY"]})
        trade = {"entry_price": 100, "side": "long", "atr_at_entry": 2}
        result = s.should_exit(trade, 105, 2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)

    def test_reload_params(self):
        s = Strategy({"rsi_period": 14, "ema_period": 20, "universe": ["SPY"]})
        # Should not raise
        s.reload_params()

    def test_ignores_symbols_not_in_universe(self):
        s = Strategy({"rsi_period": 14, "ema_period": 20, "volume_multiplier": 0.1,
                       "volume_lookback": 5, "atr_period": 14, "universe": ["SPY"]})
        bars = {"AAPL": _make_bars()}
        signals = s.generate_signals(bars)
        assert len(signals) == 0


class TestShouldExit:
    def test_no_exit_below_trailing_threshold(self):
        s = Strategy({"rsi_period": 14, "ema_period": 20, "universe": ["SPY"]})
        trade = {"entry_price": 100, "side": "long", "atr_at_entry": 2}
        should, reason = s.should_exit(trade, 102, 2)  # only 1 ATR profit
        assert should is False

    def test_trailing_stop_above_threshold_no_exit(self):
        s = Strategy({"rsi_period": 14, "ema_period": 20, "universe": ["SPY"]})
        trade = {"entry_price": 100, "side": "long", "atr_at_entry": 2}
        # Price $104 = 2 ATR profit (>= 1.5 threshold)
        # trailing_stop = 104-5 = 99, but price 104 > 99 → no exit
        should, reason = s.should_exit(trade, 104, 5)
        assert should is False

    def test_short_no_exit_below_threshold(self):
        s = Strategy({"rsi_period": 14, "ema_period": 20, "universe": ["SPY"]})
        trade = {"entry_price": 100, "side": "short", "atr_at_entry": 2}
        # profit_atr = (100-99)/2 = 0.5 → below 1.5 threshold
        should, _ = s.should_exit(trade, 99, 2)
        assert should is False
