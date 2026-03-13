"""
Trader — Main execution loop. Runs 24/7 via caffeinate -i.
=============================================================
This process handles:
  - Alpaca WebSocket streaming for real-time quotes
  - Signal computation using the active strategy
  - Order execution via Alpaca REST API
  - Trade logging to SQLite
  - Hot-reloading strategy.json between bars

Usage:
    caffeinate -i python trader.py
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from alpaca.data.live import StockDataStream
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

import db
import risk
from strategy import Strategy, Signal

logger = logging.getLogger("trader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trader.log"),
    ],
)

ET = ZoneInfo("America/New_York")
STRATEGY_JSON = Path(__file__).parent / "strategy.json"
BAR_HISTORY_WINDOW = 50  # bars to fetch for indicator computation


class Trader:
    def __init__(self):
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        self.trading_client = TradingClient(api_key, secret_key, paper=("paper" in base_url))
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.stream = StockDataStream(api_key, secret_key)

        self.strategy = Strategy()
        self.running = True
        self.circuit_breaker_triggered = False
        self.daily_pnl = 0.0
        self.starting_equity = 0.0
        self.today = None
        self.bar_buffers: dict[str, pd.DataFrame] = {}
        self.last_strategy_reload = 0

        db.init_db()
        logger.info("Trader initialized — paper trading mode")

    def _is_market_hours(self) -> bool:
        now = datetime.now(ET)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        weekday = now.weekday()
        return weekday < 5 and market_open <= now <= market_close

    def _reset_daily_state(self):
        today = datetime.now(ET).date().isoformat()
        if self.today != today:
            account = self.trading_client.get_account()
            self.starting_equity = float(account.equity)
            self.daily_pnl = 0.0
            self.circuit_breaker_triggered = False
            self.today = today
            logger.info("New trading day: %s | Equity: $%.2f", today, self.starting_equity)

    def _maybe_reload_strategy(self):
        """Hot-reload strategy params if strategy.json changed."""
        try:
            mtime = STRATEGY_JSON.stat().st_mtime
            if mtime > self.last_strategy_reload:
                self.strategy.reload_params()
                self.last_strategy_reload = mtime
                logger.info("Strategy reloaded from strategy.json")
        except FileNotFoundError:
            pass

    def _fetch_historical_bars(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch recent 5-min bars for indicator computation."""
        end = datetime.now(ET)
        start = end - timedelta(days=5)  # enough for ~50 5-min bars

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        barset = self.data_client.get_stock_bars(request)

        result = {}
        for symbol in symbols:
            if symbol in barset.data:
                bars = barset.data[symbol]
                rows = []
                for bar in bars:
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
                    # Resample to 5-min bars
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                    df_5min = df.resample("5min").agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }).dropna()
                    result[symbol] = df_5min.tail(BAR_HISTORY_WINDOW)
        return result

    def _execute_signal(self, sig: Signal):
        """Execute a trade based on a signal."""
        account = self.trading_client.get_account()
        portfolio_value = float(account.equity)

        with db.get_db() as conn:
            open_trades = db.get_open_trades(conn)

        if not risk.can_open_position(len(open_trades)):
            logger.info("Max positions reached (%d), skipping %s", risk.MAX_POSITIONS, sig.symbol)
            return

        # Check if we already have a position in this symbol
        for t in open_trades:
            if t["symbol"] == sig.symbol:
                logger.info("Already have open position in %s, skipping", sig.symbol)
                return

        limits = risk.compute_position_limits(portfolio_value, sig.price, sig.atr, sig.side)
        if limits.max_shares <= 0:
            logger.info("Position size is 0 for %s at $%.2f, skipping", sig.symbol, sig.price)
            return

        # Submit order
        order_side = OrderSide.BUY if sig.side == "long" else OrderSide.SELL
        try:
            order_request = MarketOrderRequest(
                symbol=sig.symbol,
                qty=limits.max_shares,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_request)
            logger.info(
                "ORDER SUBMITTED: %s %d %s @ ~$%.2f | stop=$%.2f target=$%.2f",
                sig.side.upper(), limits.max_shares, sig.symbol, sig.price,
                limits.stop_price, limits.take_profit_price,
            )

            # Log to database
            with db.get_db() as conn:
                db.log_trade(conn, {
                    "timestamp": datetime.now(ET).isoformat(),
                    "symbol": sig.symbol,
                    "side": sig.side,
                    "qty": limits.max_shares,
                    "entry_price": sig.price,
                    "stop_price": limits.stop_price,
                    "take_profit_price": limits.take_profit_price,
                    "strategy_version": self.strategy.VERSION,
                    "signal_type": sig.signal_type,
                    "signal_context": sig.context,
                    "status": "open",
                    "opened_at": datetime.now(ET).isoformat(),
                    "atr_at_entry": sig.atr,
                    "rsi_at_entry": sig.rsi,
                    "ema_at_entry": sig.ema,
                    "volume_ratio": sig.volume_ratio,
                })

        except Exception as e:
            logger.error("Order failed for %s: %s", sig.symbol, e)

    def _check_open_positions(self):
        """Check open trades for stop-loss, take-profit, and strategy exits."""
        with db.get_db() as conn:
            open_trades = db.get_open_trades(conn)

        if not open_trades:
            return

        try:
            positions = {p.symbol: p for p in self.trading_client.get_all_positions()}
        except Exception as e:
            logger.error("Failed to get positions: %s", e)
            return

        for trade in open_trades:
            symbol = trade["symbol"]
            if symbol not in positions:
                # Position was closed externally
                with db.get_db() as conn:
                    db.close_trade(conn, trade["id"], trade["entry_price"], 0, 0)
                continue

            pos = positions[symbol]
            current_price = float(pos.current_price)
            entry_price = trade["entry_price"]
            side = trade["side"]
            atr = trade.get("atr_at_entry", 1.0)

            # Check stop-loss
            if side == "long" and current_price <= trade["stop_price"]:
                self._close_position(trade, current_price, "stop_loss")
            elif side == "short" and current_price >= trade["stop_price"]:
                self._close_position(trade, current_price, "stop_loss")
            # Check take-profit
            elif side == "long" and current_price >= trade["take_profit_price"]:
                self._close_position(trade, current_price, "take_profit")
            elif side == "short" and current_price <= trade["take_profit_price"]:
                self._close_position(trade, current_price, "take_profit")
            else:
                # Check strategy-specific exits
                should_exit, reason = self.strategy.should_exit(trade, current_price, atr)
                if should_exit:
                    self._close_position(trade, current_price, reason)

    def _close_position(self, trade: dict, exit_price: float, reason: str):
        """Close a position and log the result."""
        symbol = trade["symbol"]
        side = trade["side"]
        qty = trade["qty"]
        entry_price = trade["entry_price"]

        if side == "long":
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty
        pnl_pct = pnl / (entry_price * qty) * 100

        # Submit closing order
        close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=close_side,
                time_in_force=TimeInForce.DAY,
            )
            self.trading_client.submit_order(order_request)
            logger.info(
                "POSITION CLOSED: %s %s | P&L: $%.2f (%.2f%%) | Reason: %s",
                side.upper(), symbol, pnl, pnl_pct, reason,
            )

            with db.get_db() as conn:
                db.close_trade(conn, trade["id"], exit_price, pnl, pnl_pct)

            self.daily_pnl += pnl

        except Exception as e:
            logger.error("Failed to close %s: %s", symbol, e)

    def _check_circuit_breaker(self):
        if self.circuit_breaker_triggered:
            return
        if risk.circuit_breaker(self.daily_pnl, self.starting_equity):
            self.circuit_breaker_triggered = True
            logger.warning("CIRCUIT BREAKER: Halting all trading for today")
            # Close all positions
            with db.get_db() as conn:
                open_trades = db.get_open_trades(conn)
            for trade in open_trades:
                try:
                    positions = {p.symbol: p for p in self.trading_client.get_all_positions()}
                    if trade["symbol"] in positions:
                        current_price = float(positions[trade["symbol"]].current_price)
                        self._close_position(trade, current_price, "circuit_breaker")
                except Exception as e:
                    logger.error("Failed to close position during circuit breaker: %s", e)

    def _log_daily_summary(self):
        """Log end-of-day performance summary."""
        with db.get_db() as conn:
            today = datetime.now(ET).date().isoformat()
            trades_today = db.get_trades_since(conn, today)
            closed = [t for t in trades_today if t["status"] == "closed"]
            wins = sum(1 for t in closed if (t.get("pnl") or 0) > 0)
            losses = sum(1 for t in closed if (t.get("pnl") or 0) <= 0)

            account = self.trading_client.get_account()
            ending_equity = float(account.equity)

            db.log_daily_performance(conn, {
                "date": today,
                "starting_equity": self.starting_equity,
                "ending_equity": ending_equity,
                "daily_pnl": self.daily_pnl,
                "daily_return_pct": (self.daily_pnl / self.starting_equity * 100) if self.starting_equity else 0,
                "trades_taken": len(closed),
                "wins": wins,
                "losses": losses,
                "circuit_breaker_triggered": 1 if self.circuit_breaker_triggered else 0,
                "strategy_version": self.strategy.VERSION,
            })

        logger.info(
            "Daily Summary: P&L=$%.2f | Trades=%d | W/L=%d/%d | Equity=$%.2f",
            self.daily_pnl, len(closed), wins, losses, ending_equity,
        )

    async def run(self):
        """Main trading loop."""
        logger.info("Starting trader main loop")

        def handle_shutdown(sig, frame):
            logger.info("Shutdown signal received")
            self.running = False

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        while self.running:
            try:
                self._reset_daily_state()
                self._maybe_reload_strategy()

                if not self._is_market_hours():
                    # Outside market hours — log summary if end of day
                    now = datetime.now(ET)
                    if now.hour == 16 and now.minute < 5 and self.today:
                        self._log_daily_summary()
                    await asyncio.sleep(30)
                    continue

                if self.circuit_breaker_triggered:
                    logger.debug("Circuit breaker active — waiting for next day")
                    await asyncio.sleep(60)
                    continue

                # Fetch bars and compute signals
                bars = self._fetch_historical_bars(self.strategy.universe)
                signals = self.strategy.generate_signals(bars)

                # Execute signals
                for sig in sorted(signals, key=lambda s: s.strength, reverse=True):
                    self._execute_signal(sig)

                # Check existing positions
                self._check_open_positions()

                # Check circuit breaker
                self._check_circuit_breaker()

                # Wait for next bar interval (5 minutes)
                await asyncio.sleep(300)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
                await asyncio.sleep(60)

        logger.info("Trader stopped")
        self._log_daily_summary()


def main():
    trader = Trader()
    asyncio.run(trader.run())


if __name__ == "__main__":
    main()
