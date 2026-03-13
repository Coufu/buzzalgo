"""
Risk Management Module - LOCKED
================================
This file contains position sizing, stop-loss, and circuit breaker logic.
Claude Code is NEVER allowed to modify this file.
Any changes require explicit operator approval.
"""

import logging
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

# Hard constraints - do not modify
POSITION_SIZE_PCT = 0.02        # 2% of portfolio per trade
STOP_LOSS_ATR_MULT = 1.5        # 1.5x ATR(14) from entry
TAKE_PROFIT_ATR_MULT = 3.0      # 3x ATR(14) from entry
TRAILING_ACTIVATION_MULT = 1.5  # Start trailing after 1.5x ATR profit
MAX_POSITIONS = 5               # Maximum concurrent open positions
MAX_DAILY_LOSS_PCT = -0.03      # -3% daily P&L circuit breaker


@dataclass
class PositionLimits:
    max_shares: int
    stop_price: float
    take_profit_price: float
    trailing_activation_price: float


def position_size(portfolio_value: float, entry_price: float) -> int:
    """Calculate the number of shares for a trade based on fixed 2% risk.

    Args:
        portfolio_value: Current total portfolio value.
        entry_price: Expected entry price per share.

    Returns:
        Number of shares to buy (integer, floored).
    """
    if entry_price <= 0 or portfolio_value <= 0:
        return 0
    dollar_amount = portfolio_value * POSITION_SIZE_PCT
    shares = int(dollar_amount // entry_price)
    return max(shares, 0)


def stop_loss(entry_price: float, atr: float, side: str = "long") -> float:
    """Calculate stop-loss price using 1.5x ATR.

    Args:
        entry_price: The price at which the position was entered.
        atr: Current ATR(14) value.
        side: "long" or "short".

    Returns:
        Stop-loss price.
    """
    offset = STOP_LOSS_ATR_MULT * atr
    if side == "long":
        return round(entry_price - offset, 2)
    else:
        return round(entry_price + offset, 2)


def take_profit(entry_price: float, atr: float, side: str = "long") -> float:
    """Calculate take-profit price using 3x ATR.

    Args:
        entry_price: The price at which the position was entered.
        atr: Current ATR(14) value.
        side: "long" or "short".

    Returns:
        Take-profit price.
    """
    offset = TAKE_PROFIT_ATR_MULT * atr
    if side == "long":
        return round(entry_price + offset, 2)
    else:
        return round(entry_price - offset, 2)


def trailing_activation(entry_price: float, atr: float, side: str = "long") -> float:
    """Price level at which trailing stop activates.

    Args:
        entry_price: The price at which the position was entered.
        atr: Current ATR(14) value.
        side: "long" or "short".

    Returns:
        Trailing activation price.
    """
    offset = TRAILING_ACTIVATION_MULT * atr
    if side == "long":
        return round(entry_price + offset, 2)
    else:
        return round(entry_price - offset, 2)


def compute_position_limits(
    portfolio_value: float,
    entry_price: float,
    atr: float,
    side: str = "long",
) -> PositionLimits:
    """Compute all position limits in one call.

    Args:
        portfolio_value: Current total portfolio value.
        entry_price: Expected entry price per share.
        atr: Current ATR(14) value.
        side: "long" or "short".

    Returns:
        PositionLimits with max_shares, stop_price, take_profit_price,
        and trailing_activation_price.
    """
    return PositionLimits(
        max_shares=position_size(portfolio_value, entry_price),
        stop_price=stop_loss(entry_price, atr, side),
        take_profit_price=take_profit(entry_price, atr, side),
        trailing_activation_price=trailing_activation(entry_price, atr, side),
    )


def circuit_breaker(daily_pnl: float, portfolio_value: float) -> bool:
    """Check if the daily loss circuit breaker has been triggered.

    Args:
        daily_pnl: Today's realized + unrealized P&L in dollars.
        portfolio_value: Portfolio value at start of day.

    Returns:
        True if trading should be HALTED (circuit breaker triggered).
    """
    if portfolio_value <= 0:
        return True
    daily_return = daily_pnl / portfolio_value
    if daily_return <= MAX_DAILY_LOSS_PCT:
        logger.warning(
            "CIRCUIT BREAKER TRIGGERED: daily return %.2f%% <= %.2f%% threshold",
            daily_return * 100,
            MAX_DAILY_LOSS_PCT * 100,
        )
        return True
    return False


def can_open_position(current_open_positions: int) -> bool:
    """Check if we can open a new position.

    Args:
        current_open_positions: Number of currently open positions.

    Returns:
        True if a new position can be opened.
    """
    return current_open_positions < MAX_POSITIONS
