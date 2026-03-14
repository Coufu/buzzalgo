"""
Backtest Validation Harness
============================
Walk-forward validation for strategy changes.
Claude Code runs this to validate modifications before deploying.

Usage:
    python backtest.py                    # Run with defaults (60-day window)
    python backtest.py --days 90          # Custom window
    python backtest.py --train-split 0.67 # Custom train/test split
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

import risk
from strategy import Strategy

logger = logging.getLogger("backtest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

ET = ZoneInfo("America/New_York")
STRATEGY_JSON = Path(__file__).parent / "strategy.json"
SLIPPAGE_BPS = 5       # 5 basis points slippage per trade
COMMISSION_PER_SHARE = 0.0  # Alpaca is commission-free


@dataclass
class BacktestTrade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    entry_time: str
    exit_time: str
    exit_reason: str


@dataclass
class BacktestResult:
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    total_pnl: float
    daily_returns: list[float] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "total_trades": self.total_trades,
            "total_pnl": round(self.total_pnl, 2),
            "train_sharpe": round(self.train_sharpe, 4),
            "test_sharpe": round(self.test_sharpe, 4),
        }

    def __str__(self) -> str:
        return (
            f"Backtest Result:\n"
            f"  Sharpe Ratio:  {self.sharpe_ratio:.4f}\n"
            f"  Max Drawdown:  {self.max_drawdown:.2%}\n"
            f"  Win Rate:      {self.win_rate:.2%}\n"
            f"  Profit Factor: {self.profit_factor:.4f}\n"
            f"  Total Trades:  {self.total_trades}\n"
            f"  Total P&L:     ${self.total_pnl:.2f}\n"
            f"  Train Sharpe:  {self.train_sharpe:.4f}\n"
            f"  Test Sharpe:   {self.test_sharpe:.4f}"
        )


def fetch_historical_data(symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
    """Fetch historical bar data from Alpaca (stocks and/or crypto)."""
    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    end = datetime.now(ET)
    start = end - timedelta(days=days + 10)  # extra buffer for weekends

    stock_symbols = [s for s in symbols if "/" not in s]
    crypto_symbols = [s for s in symbols if "/" in s]

    result = {}

    if stock_symbols:
        client = StockHistoricalDataClient(api_key, secret_key)
        request = StockBarsRequest(
            symbol_or_symbols=stock_symbols,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        barset = client.get_stock_bars(request)
        result.update(_parse_barset(barset, stock_symbols))

    if crypto_symbols:
        crypto_client = CryptoHistoricalDataClient(api_key, secret_key)
        request = CryptoBarsRequest(
            symbol_or_symbols=crypto_symbols,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        barset = crypto_client.get_crypto_bars(request)
        result.update(_parse_barset(barset, crypto_symbols))

    return result


def _parse_barset(barset, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Parse an Alpaca barset response into 5-min DataFrames."""
    result = {}
    for symbol in symbols:
        if symbol not in barset.data:
            continue
        rows = []
        for bar in barset.data[symbol]:
            rows.append({
                "timestamp": bar.timestamp,
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            })
        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            df_5min = df.resample("5min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
            result[symbol] = df_5min
    return result


def apply_slippage(price: float, side: str, entry: bool) -> float:
    """Apply realistic slippage to a price.

    Slippage always makes the fill worse for the trader:
    - Long entry: price goes UP (pay more)
    - Long exit: price goes DOWN (receive less)
    - Short entry: price goes DOWN (sell for less)
    - Short exit: price goes UP (buy back for more)
    """
    slip = price * SLIPPAGE_BPS / 10000
    if (side == "long" and entry) or (side == "short" and not entry):
        return price + slip  # worse fill: pay more to buy
    return price - slip  # worse fill: receive less to sell


def simulate(strategy: Strategy, bars: dict[str, pd.DataFrame], portfolio_value: float = 100000) -> BacktestResult:
    """Run a simulated backtest over historical bars.

    Processes bars chronologically, generating signals and simulating trades
    with realistic slippage and risk management.
    """
    equity = portfolio_value
    peak_equity = equity
    max_drawdown = 0.0
    open_positions: list[dict] = []
    closed_trades: list[BacktestTrade] = []
    daily_equity: dict[str, float] = {}

    # Get all unique timestamps across symbols and sort
    all_timestamps = set()
    for df in bars.values():
        all_timestamps.update(df.index.tolist())
    all_timestamps = sorted(all_timestamps)

    if not all_timestamps:
        return BacktestResult(0, 0, 0, 0, 0, 0)

    lookback = max(strategy.rsi_period, strategy.ema_period, strategy.volume_lookback) + 10
    daily_pnl = 0.0
    current_day = None

    for i, ts in enumerate(all_timestamps):
        ts_date = ts.date() if hasattr(ts, 'date') else pd.Timestamp(ts).date()
        day_str = str(ts_date)

        if current_day != day_str:
            if current_day is not None:
                # Mark-to-market: include unrealized P&L in daily equity
                mtm_equity = equity
                for pos in open_positions:
                    symbol = pos["symbol"]
                    if symbol in bars and not bars[symbol].empty:
                        # Get last known price for this day
                        day_bars = bars[symbol][bars[symbol].index <= ts]
                        if not day_bars.empty:
                            last_price = day_bars.iloc[-1]["close"]
                            if pos["side"] == "long":
                                mtm_equity += (last_price - pos["entry_price"]) * pos["qty"]
                            else:
                                mtm_equity += (pos["entry_price"] - last_price) * pos["qty"]
                daily_equity[current_day] = mtm_equity
            current_day = day_str
            daily_pnl = 0.0

        # Build bar windows up to current timestamp
        bar_windows = {}
        for symbol, df in bars.items():
            mask = df.index <= ts
            window = df[mask].tail(lookback)
            if len(window) >= lookback:
                bar_windows[symbol] = window

        if not bar_windows:
            continue

        # Check open positions for exits
        positions_to_close = []
        for pos in open_positions:
            symbol = pos["symbol"]
            if symbol not in bar_windows:
                continue
            current_price = bar_windows[symbol].iloc[-1]["close"]
            current_atr = None
            indicators = strategy.compute_indicators(bar_windows[symbol])
            if not pd.isna(indicators.iloc[-1]["atr"]):
                current_atr = indicators.iloc[-1]["atr"]
            else:
                current_atr = pos["atr"]

            side = pos["side"]
            exit_price = None
            exit_reason = None

            # Stop-loss check: fill at market price (which triggered the stop), not the stop price
            if side == "long" and current_price <= pos["stop_price"]:
                exit_price = apply_slippage(current_price, side, False)
                exit_reason = "stop_loss"
            elif side == "short" and current_price >= pos["stop_price"]:
                exit_price = apply_slippage(current_price, side, False)
                exit_reason = "stop_loss"
            # Take-profit check: fill at market price that triggered TP
            elif side == "long" and current_price >= pos["take_profit_price"]:
                exit_price = apply_slippage(current_price, side, False)
                exit_reason = "take_profit"
            elif side == "short" and current_price <= pos["take_profit_price"]:
                exit_price = apply_slippage(current_price, side, False)
                exit_reason = "take_profit"
            else:
                # Strategy exit
                if current_atr and current_atr > 0:
                    should_exit, reason = strategy.should_exit(pos, current_price, current_atr)
                    if should_exit:
                        exit_price = apply_slippage(current_price, side, False)
                        exit_reason = reason

            if exit_price is not None:
                if side == "long":
                    pnl = (exit_price - pos["entry_price"]) * pos["qty"]
                else:
                    pnl = (pos["entry_price"] - exit_price) * pos["qty"]

                equity += pnl
                daily_pnl += pnl
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0
                max_drawdown = max(max_drawdown, dd)

                closed_trades.append(BacktestTrade(
                    symbol=symbol, side=side, entry_price=pos["entry_price"],
                    exit_price=exit_price, qty=pos["qty"], pnl=pnl,
                    entry_time=pos["entry_time"], exit_time=str(ts),
                    exit_reason=exit_reason,
                ))
                positions_to_close.append(pos)

        for pos in positions_to_close:
            open_positions.remove(pos)

        # Circuit breaker check (includes unrealized P&L)
        unrealized_pnl = 0.0
        for pos in open_positions:
            symbol = pos["symbol"]
            if symbol in bar_windows:
                cp = bar_windows[symbol].iloc[-1]["close"]
                if pos["side"] == "long":
                    unrealized_pnl += (cp - pos["entry_price"]) * pos["qty"]
                else:
                    unrealized_pnl += (pos["entry_price"] - cp) * pos["qty"]

        total_daily_pnl = daily_pnl + unrealized_pnl
        if equity > 0 and total_daily_pnl / equity <= risk.MAX_DAILY_LOSS_PCT:
            continue  # circuit breaker: skip new signals

        # Generate signals
        signals = strategy.generate_signals(bar_windows)
        for sig in sorted(signals, key=lambda s: s.strength, reverse=True):
            if not risk.can_open_position(len(open_positions)):
                break
            # Skip if already have position in symbol
            if any(p["symbol"] == sig.symbol for p in open_positions):
                continue

            entry_price = apply_slippage(sig.price, sig.side, True)
            limits = risk.compute_position_limits(equity, sig.price, sig.atr, sig.side)
            if limits.max_shares <= 0:
                continue

            open_positions.append({
                "symbol": sig.symbol,
                "side": sig.side,
                "entry_price": entry_price,
                "qty": limits.max_shares,
                "stop_price": limits.stop_price,
                "take_profit_price": limits.take_profit_price,
                "atr": sig.atr,
                "atr_at_entry": sig.atr,
                "entry_time": str(ts),
            })

    # Record final day with mark-to-market
    if current_day:
        mtm_equity = equity
        for pos in open_positions:
            symbol = pos["symbol"]
            if symbol in bars and not bars[symbol].empty:
                last_price = bars[symbol].iloc[-1]["close"]
                if pos["side"] == "long":
                    mtm_equity += (last_price - pos["entry_price"]) * pos["qty"]
                else:
                    mtm_equity += (pos["entry_price"] - last_price) * pos["qty"]
        daily_equity[current_day] = mtm_equity

    # Compute metrics
    wins = [t for t in closed_trades if t.pnl > 0]
    losses = [t for t in closed_trades if t.pnl <= 0]
    total_trades = len(closed_trades)
    win_rate = len(wins) / total_trades if total_trades else 0
    gross_profit = sum(t.pnl for t in wins) if wins else 0
    gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    total_pnl = sum(t.pnl for t in closed_trades)

    # Daily returns for Sharpe — use actual sample count for annualization
    equity_series = sorted(daily_equity.items())
    daily_returns = []
    for j in range(1, len(equity_series)):
        prev_eq = equity_series[j - 1][1]
        curr_eq = equity_series[j][1]
        if prev_eq > 0:
            daily_returns.append((curr_eq - prev_eq) / prev_eq)

    if daily_returns and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    return BacktestResult(
        sharpe_ratio=sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        total_pnl=total_pnl,
        daily_returns=daily_returns,
        trades=closed_trades,
    )


def walk_forward_validation(
    strategy: Strategy,
    bars: dict[str, pd.DataFrame],
    train_split: float = 0.67,
) -> BacktestResult:
    """Walk-forward validation: train on first portion, test on remainder.

    Args:
        strategy: Strategy instance to evaluate.
        bars: Historical bar data.
        train_split: Fraction of data to use for training (default 67%).

    Returns:
        BacktestResult with train and test Sharpe ratios.
    """
    # Split data
    train_bars = {}
    test_bars = {}
    for symbol, df in bars.items():
        split_idx = int(len(df) * train_split)
        train_bars[symbol] = df.iloc[:split_idx]
        test_bars[symbol] = df.iloc[split_idx:]

    logger.info("Walk-forward split: train=%d bars, test=%d bars (per symbol, avg)",
                np.mean([len(df) for df in train_bars.values()]),
                np.mean([len(df) for df in test_bars.values()]))

    train_result = simulate(strategy, train_bars)
    test_result = simulate(strategy, test_bars)

    # Use test result as primary, annotate with train
    test_result.train_sharpe = train_result.sharpe_ratio
    test_result.test_sharpe = test_result.sharpe_ratio

    logger.info("Train Sharpe: %.4f | Test Sharpe: %.4f", train_result.sharpe_ratio, test_result.sharpe_ratio)
    return test_result


def run_backtest(days: int = 60, train_split: float = 0.67, mode: str = "equity") -> BacktestResult:
    """Run a full backtest with walk-forward validation.

    Args:
        days: Number of days of historical data.
        train_split: Train/test split ratio.
        mode: "equity" or "crypto".

    Returns:
        BacktestResult from walk-forward validation.
    """
    with open(STRATEGY_JSON) as f:
        config = json.load(f)

    strategy = Strategy(mode=mode)
    if mode == "crypto" and "crypto" in config:
        symbols = config["crypto"].get("universe", strategy.universe)
    else:
        symbols = config.get("universe", strategy.universe)

    logger.info("Fetching %d days of historical data for %d symbols...", days, len(symbols))
    bars = fetch_historical_data(symbols, days)
    logger.info("Fetched data for %d symbols", len(bars))

    result = walk_forward_validation(strategy, bars, train_split)
    logger.info("\n%s", result)

    # Save result
    result_path = Path(__file__).parent / "backtest_result.json"
    with open(result_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info("Result saved to %s", result_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run strategy backtest")
    parser.add_argument("--days", type=int, default=60, help="Days of historical data")
    parser.add_argument("--train-split", type=float, default=0.67, help="Train/test split ratio")
    parser.add_argument("--mode", choices=["equity", "crypto"], default="equity", help="Asset class to backtest")
    args = parser.parse_args()

    result = run_backtest(args.days, args.train_split, args.mode)
    print(result)
    return 0 if result.sharpe_ratio > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
