"""
Trader - Main execution loop. Runs 24/7 via Docker.
=============================================================
This process handles:
  - Polling Alpaca for 5-min bar data
  - Signal computation using the active strategy
  - Order execution via Alpaca REST API
  - Trade logging to SQLite
  - Hot-reloading strategy.json between bars

Usage:
    python -u trader.py
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import urllib.request
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
BAR_TIMEFRAME = "15min"  # resample to 15-min bars for wider ATR/stops
MAX_TRADES_PER_DAY = 5   # cap daily trades to avoid overtrading
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def is_crypto_symbol(symbol: str) -> bool:
    """Check if a symbol is a crypto pair (e.g. BTC/USD)."""
    return "/" in symbol


def _notify_slack(text: str):
    """Send a notification to Slack. Fails silently."""
    if not SLACK_WEBHOOK_URL:
        return
    try:
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        logger.debug("Slack notification failed: %s", e)


class Trader:
    def __init__(self):
        api_key = os.environ["ALPACA_API_KEY"]
        secret_key = os.environ["ALPACA_SECRET_KEY"]
        base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

        self.trading_client = TradingClient(api_key, secret_key, paper=("paper" in base_url))
        self.data_client = StockHistoricalDataClient(api_key, secret_key)
        self.crypto_data_client = CryptoHistoricalDataClient(api_key, secret_key)

        self.strategy = Strategy(mode="equity")
        self.crypto_strategy = Strategy(mode="crypto")
        self.running = True
        self.circuit_breaker_triggered = False
        self.daily_pnl = 0.0
        self.starting_equity = 0.0
        self.today = None
        self.bar_buffers: dict[str, pd.DataFrame] = {}
        self.last_strategy_reload = 0
        self._daily_summary_logged = False
        self._pending_order_symbols: set[str] = set()
        self._trades_today = 0

        db.init_db()
        self._reconcile_positions()
        logger.info("Trader initialized - paper trading mode")

    def _reconcile_positions(self):
        """On startup, close any Alpaca positions not tracked in the DB."""
        try:
            alpaca_positions = {p.symbol: p for p in self.trading_client.get_all_positions()}
            with db.get_db() as conn:
                db_open = db.get_open_trades(conn)
            db_symbols = {t["symbol"] for t in db_open}

            orphans = set(alpaca_positions.keys()) - db_symbols
            if not orphans:
                logger.info("Position reconciliation: all %d positions tracked", len(alpaca_positions))
                return

            logger.warning("Found %d orphan positions on Alpaca, closing: %s", len(orphans), orphans)
            for sym in orphans:
                try:
                    self.trading_client.close_position(sym)
                    logger.info("Closed orphan position: %s", sym)
                except Exception as e:
                    logger.error("Failed to close orphan %s: %s", sym, e)

            _notify_slack(
                f":broom: *Startup reconciliation* — closed {len(orphans)} orphan position(s): "
                f"{', '.join(sorted(orphans)[:10])}"
            )
        except Exception as e:
            logger.error("Position reconciliation failed: %s", e)

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
            self._daily_summary_logged = False
            self._pending_order_symbols.clear()
            self._trades_today = 0
            self._reconcile_positions()
            self.today = today
            logger.info("New trading day: %s | Equity: $%.2f", today, self.starting_equity)

    def _maybe_reload_strategy(self):
        """Hot-reload strategy params if strategy.json changed."""
        try:
            mtime = STRATEGY_JSON.stat().st_mtime
            if mtime > self.last_strategy_reload:
                self.strategy.reload_params()
                self.crypto_strategy.reload_params()
                self.last_strategy_reload = mtime
                logger.info("Strategy reloaded from strategy.json")
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error("Invalid strategy.json, keeping current params: %s", e)

    def _fetch_historical_bars(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch recent 5-min bars for equity symbols."""
        end = datetime.now(ET)
        start = end - timedelta(days=10)  # enough for ~50 15-min bars

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            feed=DataFeed.IEX,
        )
        barset = self.data_client.get_stock_bars(request)

        result = self._parse_barset(barset, symbols)
        fetched = len(result)
        if fetched < len(symbols):
            logger.warning("Fetched stock bars for %d/%d symbols", fetched, len(symbols))
        return result

    def _fetch_crypto_bars(self, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch recent 5-min bars for crypto symbols."""
        end = datetime.now(ET)
        start = end - timedelta(days=5)

        request = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        barset = self.crypto_data_client.get_crypto_bars(request)

        result = self._parse_barset(barset, symbols)
        fetched = len(result)
        if fetched < len(symbols):
            logger.warning("Fetched crypto bars for %d/%d symbols", fetched, len(symbols))
        return result

    def _parse_barset(self, barset, symbols: list[str]) -> dict[str, pd.DataFrame]:
        """Parse an Alpaca barset response into DataFrames at BAR_TIMEFRAME."""
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
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df = df.set_index("timestamp")
                    resampled = df.resample(BAR_TIMEFRAME).agg({
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }).dropna()
                    result[symbol] = resampled.tail(BAR_HISTORY_WINDOW)
        return result

    def _get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L from Alpaca positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return sum(float(p.unrealized_pl) for p in positions)
        except Exception as e:
            logger.error("Failed to get unrealized P&L: %s", e)
            return 0.0

    def _execute_signal(self, sig: Signal):
        """Execute a trade based on a signal."""
        # Daily trade cap
        if self._trades_today >= MAX_TRADES_PER_DAY:
            logger.debug("Daily trade cap reached (%d), skipping %s", MAX_TRADES_PER_DAY, sig.symbol)
            return

        # Dedup: skip if we already submitted an order for this symbol this cycle
        if sig.symbol in self._pending_order_symbols:
            logger.info("Order already pending for %s, skipping", sig.symbol)
            return

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

        # For crypto, compute fractional quantity since risk.py floors to int
        if is_crypto_symbol(sig.symbol):
            dollar_amount = portfolio_value * risk.POSITION_SIZE_PCT
            qty = round(dollar_amount / sig.price, 6)  # 6 decimal places for crypto
            if qty <= 0:
                logger.info("Position size is 0 for %s at $%.2f, skipping", sig.symbol, sig.price)
                return
        else:
            qty = limits.max_shares
            if qty <= 0:
                logger.info("Position size is 0 for %s at $%.2f, skipping", sig.symbol, sig.price)
                return

        # Log to database FIRST so we track the intent even if the order fails
        trade_record = {
            "timestamp": datetime.now(ET).isoformat(),
            "symbol": sig.symbol,
            "side": sig.side,
            "qty": qty,
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
            "market_regime": sig.context.get("market_regime", ""),
        }

        with db.get_db() as conn:
            db.log_trade(conn, trade_record)

        # Submit order
        order_side = OrderSide.BUY if sig.side == "long" else OrderSide.SELL
        tif = TimeInForce.GTC if is_crypto_symbol(sig.symbol) else TimeInForce.DAY
        try:
            order_request = MarketOrderRequest(
                symbol=sig.symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )
            order = self.trading_client.submit_order(order_request)
            self._pending_order_symbols.add(sig.symbol)
            self._trades_today += 1
            logger.info(
                "ORDER SUBMITTED: %s %s %s @ ~$%.2f | stop=$%.2f target=$%.2f",
                sig.side.upper(), qty, sig.symbol, sig.price,
                limits.stop_price, limits.take_profit_price,
            )
            _notify_slack(
                f":chart_with_upwards_trend: *OPEN {sig.side.upper()}* {limits.max_shares} {sig.symbol} @ ${sig.price:.2f}\n"
                f"Stop: ${limits.stop_price:.2f} | Target: ${limits.take_profit_price:.2f} | Signal: {sig.signal_type}"
            )

        except Exception as e:
            logger.error("Order failed for %s: %s", sig.symbol, e)
            # Mark the DB record as failed since the order didn't go through
            with db.get_db() as conn:
                recent = conn.execute(
                    "SELECT id FROM trades WHERE symbol=? AND status='open' ORDER BY id DESC LIMIT 1",
                    (sig.symbol,),
                ).fetchone()
                if recent:
                    db.close_trade(conn, recent["id"], sig.price, 0, 0)

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
                # Position was closed externally — use Alpaca's last known price
                logger.warning("Position %s closed externally, reconciling", symbol)
                with db.get_db() as conn:
                    db.close_trade(conn, trade["id"], trade["entry_price"], 0, 0)
                continue

            pos = positions[symbol]
            current_price = float(pos.current_price)
            entry_price = trade["entry_price"]
            side = trade["side"]
            atr = trade.get("atr_at_entry") or 1.0

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
                # Check strategy-specific exits (guard against bad ATR)
                if atr > 0:
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

        if entry_price > 0 and qty > 0:
            pnl_pct = pnl / (entry_price * qty) * 100
        else:
            pnl_pct = 0.0

        # Submit closing order
        close_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        tif = TimeInForce.GTC if is_crypto_symbol(symbol) else TimeInForce.DAY
        try:
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=close_side,
                time_in_force=tif,
            )
            self.trading_client.submit_order(order_request)
            logger.info(
                "POSITION CLOSED: %s %s | P&L: $%.2f (%.2f%%) | Reason: %s",
                side.upper(), symbol, pnl, pnl_pct, reason,
            )

            with db.get_db() as conn:
                db.close_trade(conn, trade["id"], exit_price, pnl, pnl_pct)

            self.daily_pnl += pnl
            self._pending_order_symbols.discard(symbol)

            pnl_emoji = ":white_check_mark:" if pnl >= 0 else ":red_circle:"
            _notify_slack(
                f"{pnl_emoji} *CLOSE {side.upper()}* {symbol} @ ${exit_price:.2f}\n"
                f"P&L: ${pnl:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason}"
            )

        except Exception as e:
            logger.error("Failed to close %s: %s", symbol, e)

    def _check_circuit_breaker(self):
        if self.circuit_breaker_triggered:
            return

        # Include unrealized P&L for accurate circuit breaker
        unrealized = self._get_unrealized_pnl()
        total_pnl = self.daily_pnl + unrealized

        if risk.circuit_breaker(total_pnl, self.starting_equity):
            self.circuit_breaker_triggered = True
            logger.warning(
                "CIRCUIT BREAKER: Halting trading | Realized=$%.2f Unrealized=$%.2f Total=$%.2f",
                self.daily_pnl, unrealized, total_pnl,
            )
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
        if self._daily_summary_logged:
            return
        self._daily_summary_logged = True

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
                "daily_pnl": ending_equity - self.starting_equity,
                "daily_return_pct": ((ending_equity - self.starting_equity) / self.starting_equity * 100) if self.starting_equity else 0,
                "trades_taken": len(closed),
                "wins": wins,
                "losses": losses,
                "circuit_breaker_triggered": 1 if self.circuit_breaker_triggered else 0,
                "strategy_version": self.strategy.VERSION,
            })

        logger.info(
            "Daily Summary: P&L=$%.2f | Trades=%d | W/L=%d/%d | Equity=$%.2f",
            ending_equity - self.starting_equity, len(closed), wins, losses, ending_equity,
        )

    async def run(self):
        """Main trading loop."""
        logger.info("Starting trader main loop")

        shutdown_event = asyncio.Event()

        def handle_shutdown(sig, frame):
            logger.info("Shutdown signal received")
            self.running = False
            shutdown_event.set()

        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)

        while self.running:
            try:
                self._reset_daily_state()
                self._maybe_reload_strategy()

                market_open = self._is_market_hours()

                if not market_open:
                    # Log summary once after market close (4pm-5pm window)
                    now = datetime.now(ET)
                    if now.hour >= 16 and now.hour < 17 and self.today and not self._daily_summary_logged:
                        self._log_daily_summary()

                if self.circuit_breaker_triggered:
                    if not market_open:
                        # After hours with circuit breaker — still trade crypto
                        pass
                    else:
                        logger.debug("Circuit breaker active - waiting for next day")
                        try:
                            await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                            break
                        except asyncio.TimeoutError:
                            pass
                        continue

                # Clear pending orders at start of each cycle
                self._pending_order_symbols.clear()

                with db.get_db() as conn:
                    open_trades = db.get_open_trades(conn)
                open_syms = [t["symbol"] for t in open_trades]
                all_signals = []

                # Equity signals — only during market hours
                if market_open and not self.circuit_breaker_triggered:
                    bars = self._fetch_historical_bars(self.strategy.universe)
                    equity_signals = self.strategy.generate_signals(bars, open_symbols=open_syms)
                    all_signals.extend(equity_signals)

                # Crypto signals — 24/7
                if self.crypto_strategy.universe:
                    crypto_bars = self._fetch_crypto_bars(self.crypto_strategy.universe)
                    crypto_signals = self.crypto_strategy.generate_signals(crypto_bars, open_symbols=open_syms)
                    all_signals.extend(crypto_signals)

                if all_signals:
                    logger.info("Generated %d signals (%s)",
                                len(all_signals),
                                "equity+crypto" if market_open else "crypto only")

                # Execute signals
                for sig in sorted(all_signals, key=lambda s: s.strength, reverse=True):
                    self._execute_signal(sig)

                # Check existing positions (both equity and crypto)
                self._check_open_positions()

                # Check circuit breaker
                self._check_circuit_breaker()

                # Wait for next bar interval (5 minutes), but allow fast shutdown
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=900)  # 15 min
                    break  # shutdown requested
                except asyncio.TimeoutError:
                    pass

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error("Error in main loop: %s", e, exc_info=True)
                try:
                    await asyncio.wait_for(shutdown_event.wait(), timeout=60)
                    break
                except asyncio.TimeoutError:
                    pass

        logger.info("Trader stopped")
        self._log_daily_summary()


def main():
    trader = Trader()
    asyncio.run(trader.run())


if __name__ == "__main__":
    main()
