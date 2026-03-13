"""Tests for risk management module - validates locked constraints."""

import pytest
from risk import (
    position_size, stop_loss, take_profit, trailing_activation,
    compute_position_limits, circuit_breaker, can_open_position,
    POSITION_SIZE_PCT, STOP_LOSS_ATR_MULT, MAX_POSITIONS, MAX_DAILY_LOSS_PCT,
)


class TestPositionSize:
    def test_basic_calculation(self):
        # 2% of $100k at $500/share = $2000 / $500 = 4 shares
        assert position_size(100_000, 500) == 4

    def test_rounds_down(self):
        # 2% of $100k at $300/share = $2000 / $300 = 6.67 -> 6 shares
        assert position_size(100_000, 300) == 6

    def test_zero_price(self):
        assert position_size(100_000, 0) == 0

    def test_zero_portfolio(self):
        assert position_size(0, 100) == 0

    def test_negative_values(self):
        assert position_size(-100_000, 100) == 0
        assert position_size(100_000, -100) == 0

    def test_constraint_is_two_percent(self):
        assert POSITION_SIZE_PCT == 0.02


class TestStopLoss:
    def test_long_stop(self):
        # Entry $100, ATR $2, stop = $100 - 1.5*$2 = $97
        assert stop_loss(100, 2, "long") == 97.0

    def test_short_stop(self):
        # Entry $100, ATR $2, stop = $100 + 1.5*$2 = $103
        assert stop_loss(100, 2, "short") == 103.0

    def test_constraint_is_1_5x_atr(self):
        assert STOP_LOSS_ATR_MULT == 1.5


class TestTakeProfit:
    def test_long_target(self):
        # Entry $100, ATR $2, target = $100 + 3*$2 = $106
        assert take_profit(100, 2, "long") == 106.0

    def test_short_target(self):
        assert take_profit(100, 2, "short") == 94.0


class TestTrailingActivation:
    def test_long_activation(self):
        # Entry $100, ATR $2, activation = $100 + 1.5*$2 = $103
        assert trailing_activation(100, 2, "long") == 103.0


class TestCircuitBreaker:
    def test_not_triggered(self):
        assert circuit_breaker(-100, 100_000) is False  # -0.1%

    def test_triggered_at_threshold(self):
        assert circuit_breaker(-3000, 100_000) is True  # -3%

    def test_triggered_beyond_threshold(self):
        assert circuit_breaker(-5000, 100_000) is True  # -5%

    def test_zero_portfolio(self):
        assert circuit_breaker(-1, 0) is True

    def test_constraint_is_3_percent(self):
        assert MAX_DAILY_LOSS_PCT == -0.03


class TestCanOpenPosition:
    def test_below_max(self):
        assert can_open_position(3) is True

    def test_at_max(self):
        assert can_open_position(5) is False

    def test_above_max(self):
        assert can_open_position(6) is False

    def test_max_is_five(self):
        assert MAX_POSITIONS == 5


class TestComputePositionLimits:
    def test_returns_all_limits(self):
        limits = compute_position_limits(100_000, 100, 2, "long")
        assert limits.max_shares == 20  # 2% of $100k / $100
        assert limits.stop_price == 97.0
        assert limits.take_profit_price == 106.0
        assert limits.trailing_activation_price == 103.0
