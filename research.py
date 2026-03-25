"""
Strategy Research Engine
========================
Backtests multiple strategy candidates against historical data to find
what actually works. Fetches data once, runs each strategy through the
same simulate() harness from backtest.py, and outputs a ranked comparison.

Usage:
    python3 research.py                        # run all strategies
    python3 research.py --strategy vwap        # run one strategy
    python3 research.py --days 90              # custom lookback
    python3 research.py --symbols 20           # number of symbols to test
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

import risk
from backtest import simulate, BacktestResult, apply_slippage
from strategy import Signal

logger = logging.getLogger("research")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

ET = ZoneInfo("America/New_York")

# Default universe: top 20 liquid names
DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN",
    "TSLA", "AMD", "JPM", "V", "AVGO", "NFLX", "CRM", "COST",
    "HD", "MA", "LLY", "UNH",
]


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_daily_data(symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
    """Fetch daily bars from Alpaca for swing strategy research."""
    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    end = datetime.now(ET)
    start = end - timedelta(days=int(days * 1.5))  # buffer for weekends

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=symbols, timeframe=TimeFrame.Day,
        start=start, end=end, feed=DataFeed.IEX,
    )

    logger.info("Fetching %d days of daily bars for %d symbols...", days, len(symbols))
    barset = client.get_stock_bars(request)

    result = {}
    for symbol in symbols:
        if symbol not in barset.data:
            continue
        rows = [{"timestamp": b.timestamp, "open": float(b.open), "high": float(b.high),
                 "low": float(b.low), "close": float(b.close), "volume": float(b.volume)}
                for b in barset.data[symbol]]
        if rows:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            result[symbol] = df
            logger.info("  %s: %d daily bars", symbol, len(df))
    return result


def fetch_research_data(symbols: list[str], days: int) -> dict[str, pd.DataFrame]:
    """Fetch 1-min bars from Alpaca and resample to 15-min."""
    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]

    end = datetime.now(ET)
    start = end - timedelta(days=days + 10)  # buffer for weekends

    client = StockHistoricalDataClient(api_key, secret_key)
    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Minute,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    logger.info("Fetching %d days of 1-min data for %d symbols from Alpaca...", days, len(symbols))
    t0 = time.time()
    barset = client.get_stock_bars(request)
    elapsed = time.time() - t0
    logger.info("Data fetch completed in %.1fs", elapsed)

    result = {}
    for symbol in symbols:
        if symbol not in barset.data:
            logger.warning("No data for %s", symbol)
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
            # Resample to 15-min bars
            df_15min = df.resample("15min").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }).dropna()
            result[symbol] = df_15min
            logger.info("  %s: %d bars (15-min)", symbol, len(df_15min))

    return result


# ---------------------------------------------------------------------------
# Base strategy interface
# ---------------------------------------------------------------------------

class ResearchStrategy:
    """Base class for research strategy candidates.

    Each subclass must implement compute_indicators() and generate_signals()
    with the same interface expected by backtest.simulate().
    """

    name: str = "base"

    # These defaults match backtest.py's lookback calculation
    rsi_period: int = 14
    ema_period: int = 10
    volume_lookback: int = 20
    atr_period: int = 14
    time_exit_minutes: int = 0  # disabled for research

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        raise NotImplementedError

    def should_exit(self, trade: dict, current_price: float, current_atr: float) -> tuple[bool, str]:
        """Default: no strategy exits (rely on stop/TP from risk.py)."""
        return False, ""


# ---------------------------------------------------------------------------
# Strategy 1: VWAP Mean Reversion
# ---------------------------------------------------------------------------

class VWAPMeanReversion(ResearchStrategy):
    """Price deviates >1.5 ATR from VWAP, enter when it starts reverting."""

    name = "vwap_mean_revert"
    rsi_period = 14
    ema_period = 10
    volume_lookback = 20
    atr_period = 14
    atr_deviation = 1.5  # ATR distance from VWAP to trigger

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.ema_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]

        # Intraday VWAP: reset cumulative sums each day
        df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
        df["tp_vol"] = df["typical_price"] * df["volume"]
        df["date"] = df.index.date
        df["cum_tp_vol"] = df.groupby("date")["tp_vol"].cumsum()
        df["cum_vol"] = df.groupby("date")["volume"].cumsum()
        df["vwap"] = df["cum_tp_vol"] / df["cum_vol"].replace(0, np.nan)

        # VWAP bands
        df["vwap_upper"] = df["vwap"] + self.atr_deviation * df["atr"]
        df["vwap_lower"] = df["vwap"] - self.atr_deviation * df["atr"]

        # Clean up temp columns
        df.drop(columns=["typical_price", "tp_vol", "date", "cum_tp_vol", "cum_vol"], inplace=True)
        return df

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for symbol, df in bars.items():
            min_bars = max(self.rsi_period, self.ema_period, self.volume_lookback) + 10
            if len(df) < min_bars:
                continue

            df = self.compute_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            if pd.isna(latest["atr"]) or pd.isna(latest["vwap"]) or latest["atr"] <= 0:
                continue

            price = latest["close"]
            vwap = latest["vwap"]
            atr = latest["atr"]
            rsi = latest["rsi"] if not pd.isna(latest["rsi"]) else 50
            ema = latest["ema"] if not pd.isna(latest["ema"]) else price
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 1.0

            prev_price = prev["close"]
            prev_vwap_lower = prev["vwap_lower"] if not pd.isna(prev["vwap_lower"]) else None
            prev_vwap_upper = prev["vwap_upper"] if not pd.isna(prev["vwap_upper"]) else None

            # Long: price was below lower VWAP band, now reverting up
            if prev_vwap_lower is not None and prev_price < prev_vwap_lower and price > latest["vwap_lower"]:
                deviation = abs(price - vwap) / atr
                strength = min(1.0, deviation * 0.3 + volume_ratio * 0.1)
                strength = max(0.01, strength)
                signals.append(Signal(
                    symbol=symbol, side="long", signal_type="vwap_revert_long",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"vwap": round(vwap, 4), "deviation_atr": round(deviation, 2)},
                ))

            # Short: price was above upper VWAP band, now reverting down
            elif prev_vwap_upper is not None and prev_price > prev_vwap_upper and price < latest["vwap_upper"]:
                deviation = abs(price - vwap) / atr
                strength = min(1.0, deviation * 0.3 + volume_ratio * 0.1)
                strength = max(0.01, strength)
                signals.append(Signal(
                    symbol=symbol, side="short", signal_type="vwap_revert_short",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"vwap": round(vwap, 4), "deviation_atr": round(deviation, 2)},
                ))

        return sorted(signals, key=lambda s: s.strength, reverse=True)[:5]


# ---------------------------------------------------------------------------
# Strategy 2: Opening Range Breakout
# ---------------------------------------------------------------------------

class OpeningRangeBreakout(ResearchStrategy):
    """First 30 min establishes range. Trade the breakout with volume confirm."""

    name = "opening_range"
    rsi_period = 14
    ema_period = 10
    volume_lookback = 20
    atr_period = 14
    range_minutes = 30
    volume_confirm = 1.5  # volume must be 1.5x average

    def __init__(self):
        # Track which symbol+date combos already fired
        self._fired: set[str] = set()

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.ema_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
        return df

    def _get_opening_range(self, df: pd.DataFrame) -> tuple[float | None, float | None]:
        """Get today's opening range (high/low of first 30 minutes)."""
        if df.empty:
            return None, None
        today = df.index[-1].date()
        today_bars = df[df.index.date == today]
        if today_bars.empty:
            return None, None

        market_open = today_bars.index[0]
        cutoff = market_open + timedelta(minutes=self.range_minutes)
        range_bars = today_bars[today_bars.index <= cutoff]

        if len(range_bars) < 2:
            return None, None
        return range_bars["high"].max(), range_bars["low"].min()

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for symbol, df in bars.items():
            min_bars = max(self.rsi_period, self.ema_period, self.volume_lookback) + 10
            if len(df) < min_bars:
                continue

            df = self.compute_indicators(df)
            latest = df.iloc[-1]

            if pd.isna(latest["atr"]) or latest["atr"] <= 0:
                continue

            today = df.index[-1].date()
            fire_key = f"{symbol}_{today}"
            if fire_key in self._fired:
                continue  # only one signal per symbol per day

            price = latest["close"]
            atr = latest["atr"]
            rsi = latest["rsi"] if not pd.isna(latest["rsi"]) else 50
            ema = latest["ema"] if not pd.isna(latest["ema"]) else price
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 1.0

            # Need to be past the opening range period
            today_bars = df[df.index.date == today]
            if len(today_bars) < 3:
                continue
            market_open = today_bars.index[0]
            current_time = df.index[-1]
            if (current_time - market_open).total_seconds() < self.range_minutes * 60:
                continue  # still in opening range

            range_high, range_low = self._get_opening_range(df)
            if range_high is None or range_low is None:
                continue

            # Volume confirmation
            if volume_ratio < self.volume_confirm:
                continue

            # Breakout above range
            if price > range_high:
                breakout_pct = (price - range_high) / atr
                strength = min(1.0, breakout_pct * 0.5 + volume_ratio * 0.15)
                strength = max(0.01, strength)
                self._fired.add(fire_key)
                signals.append(Signal(
                    symbol=symbol, side="long", signal_type="orb_long",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"range_high": round(range_high, 4), "range_low": round(range_low, 4)},
                ))

            # Breakdown below range
            elif price < range_low:
                breakdown_pct = (range_low - price) / atr
                strength = min(1.0, breakdown_pct * 0.5 + volume_ratio * 0.15)
                strength = max(0.01, strength)
                self._fired.add(fire_key)
                signals.append(Signal(
                    symbol=symbol, side="short", signal_type="orb_short",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"range_high": round(range_high, 4), "range_low": round(range_low, 4)},
                ))

        return sorted(signals, key=lambda s: s.strength, reverse=True)[:5]


# ---------------------------------------------------------------------------
# Strategy 3: RSI Divergence
# ---------------------------------------------------------------------------

class RSIDivergence(ResearchStrategy):
    """Bullish/bearish RSI divergence over 10-bar lookback."""

    name = "rsi_divergence"
    rsi_period = 14
    ema_period = 10
    volume_lookback = 20
    atr_period = 14
    divergence_lookback = 10

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.ema_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
        return df

    def _find_divergence(self, df: pd.DataFrame) -> str | None:
        """Check for RSI divergence in the last N bars.

        Returns 'bullish', 'bearish', or None.
        """
        if len(df) < self.divergence_lookback + 1:
            return None

        window = df.iloc[-(self.divergence_lookback + 1):]
        prices = window["close"].values
        rsis = window["rsi"].values

        if any(np.isnan(rsis)):
            return None

        # Find the bar with the lowest price in the lookback (excluding current)
        lookback_prices = prices[:-1]
        lookback_rsis = rsis[:-1]
        current_price = prices[-1]
        current_rsi = rsis[-1]

        # Bullish divergence: price makes new low but RSI makes higher low
        price_low_idx = np.argmin(lookback_prices)
        if current_price < lookback_prices[price_low_idx]:
            # Price made new low - check if RSI is higher than at previous low
            if current_rsi > lookback_rsis[price_low_idx]:
                return "bullish"

        # Bearish divergence: price makes new high but RSI makes lower high
        price_high_idx = np.argmax(lookback_prices)
        if current_price > lookback_prices[price_high_idx]:
            # Price made new high - check if RSI is lower than at previous high
            if current_rsi < lookback_rsis[price_high_idx]:
                return "bearish"

        return None

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for symbol, df in bars.items():
            min_bars = max(self.rsi_period, self.ema_period, self.volume_lookback, self.divergence_lookback) + 10
            if len(df) < min_bars:
                continue

            df = self.compute_indicators(df)
            latest = df.iloc[-1]

            if pd.isna(latest["atr"]) or latest["atr"] <= 0:
                continue

            price = latest["close"]
            atr = latest["atr"]
            rsi = latest["rsi"] if not pd.isna(latest["rsi"]) else 50
            ema = latest["ema"] if not pd.isna(latest["ema"]) else price
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 1.0

            divergence = self._find_divergence(df)
            if divergence is None:
                continue

            if divergence == "bullish" and rsi < 45:
                strength = min(1.0, (50 - rsi) / 30 + volume_ratio * 0.1)
                strength = max(0.01, strength)
                signals.append(Signal(
                    symbol=symbol, side="long", signal_type="rsi_div_bullish",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"divergence": "bullish", "rsi": round(rsi, 2)},
                ))
            elif divergence == "bearish" and rsi > 55:
                strength = min(1.0, (rsi - 50) / 30 + volume_ratio * 0.1)
                strength = max(0.01, strength)
                signals.append(Signal(
                    symbol=symbol, side="short", signal_type="rsi_div_bearish",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"divergence": "bearish", "rsi": round(rsi, 2)},
                ))

        return sorted(signals, key=lambda s: s.strength, reverse=True)[:5]


# ---------------------------------------------------------------------------
# Strategy 4: Volume Spike Reversal
# ---------------------------------------------------------------------------

class VolumeSpikeReversal(ResearchStrategy):
    """3x average volume bar that closes opposite to its initial move.

    Hammer/shooting star pattern indicating institutional activity.
    """

    name = "volume_spike"
    rsi_period = 14
    ema_period = 10
    volume_lookback = 20
    atr_period = 14
    volume_spike_mult = 3.0

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.ema_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
        return df

    def _is_hammer(self, row: pd.Series) -> bool:
        """Bullish hammer: opened down, closed up (or near open), long lower wick."""
        body = abs(row["close"] - row["open"])
        full_range = row["high"] - row["low"]
        if full_range <= 0:
            return False
        lower_wick = min(row["open"], row["close"]) - row["low"]
        # Lower wick > 2x body, close in upper half
        return lower_wick > 2 * body and row["close"] >= (row["high"] + row["low"]) / 2

    def _is_shooting_star(self, row: pd.Series) -> bool:
        """Bearish shooting star: opened up, closed down, long upper wick."""
        body = abs(row["close"] - row["open"])
        full_range = row["high"] - row["low"]
        if full_range <= 0:
            return False
        upper_wick = row["high"] - max(row["open"], row["close"])
        # Upper wick > 2x body, close in lower half
        return upper_wick > 2 * body and row["close"] <= (row["high"] + row["low"]) / 2

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for symbol, df in bars.items():
            min_bars = max(self.rsi_period, self.ema_period, self.volume_lookback) + 10
            if len(df) < min_bars:
                continue

            df = self.compute_indicators(df)
            latest = df.iloc[-1]

            if pd.isna(latest["atr"]) or latest["atr"] <= 0:
                continue

            price = latest["close"]
            atr = latest["atr"]
            rsi = latest["rsi"] if not pd.isna(latest["rsi"]) else 50
            ema = latest["ema"] if not pd.isna(latest["ema"]) else price
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 1.0

            # Must have volume spike
            if volume_ratio < self.volume_spike_mult:
                continue

            # Check for hammer (bullish reversal)
            if self._is_hammer(latest):
                strength = min(1.0, volume_ratio / 5.0)
                strength = max(0.01, strength)
                signals.append(Signal(
                    symbol=symbol, side="long", signal_type="vol_spike_hammer",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"pattern": "hammer", "vol_ratio": round(volume_ratio, 2)},
                ))

            # Check for shooting star (bearish reversal)
            elif self._is_shooting_star(latest):
                strength = min(1.0, volume_ratio / 5.0)
                strength = max(0.01, strength)
                signals.append(Signal(
                    symbol=symbol, side="short", signal_type="vol_spike_star",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"pattern": "shooting_star", "vol_ratio": round(volume_ratio, 2)},
                ))

        return sorted(signals, key=lambda s: s.strength, reverse=True)[:5]


# ---------------------------------------------------------------------------
# Strategy 5: Trend Following (Dual EMA + ADX)
# ---------------------------------------------------------------------------

class DualEMATrend(ResearchStrategy):
    """Fast EMA(8) crosses slow EMA(21) with ADX>25 confirming trend."""

    name = "dual_ema_trend"
    rsi_period = 14
    ema_period = 21  # use slow EMA for backtest lookback calc
    volume_lookback = 20
    atr_period = 14
    fast_ema = 8
    slow_ema = 21
    adx_period = 14
    adx_threshold = 25

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.slow_ema)  # for compatibility
        df["ema_fast"] = ta.ema(df["close"], length=self.fast_ema)
        df["ema_slow"] = ta.ema(df["close"], length=self.slow_ema)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]

        # ADX
        try:
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)
            adx_col = f"ADX_{self.adx_period}"
            if adx_df is not None and adx_col in adx_df.columns:
                df["adx"] = adx_df[adx_col]
            else:
                df["adx"] = np.nan
        except Exception:
            df["adx"] = np.nan

        return df

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for symbol, df in bars.items():
            min_bars = max(self.rsi_period, self.slow_ema, self.volume_lookback, self.adx_period) + 10
            if len(df) < min_bars:
                continue

            df = self.compute_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            if pd.isna(latest["atr"]) or latest["atr"] <= 0:
                continue
            if pd.isna(latest["ema_fast"]) or pd.isna(latest["ema_slow"]):
                continue
            if pd.isna(prev["ema_fast"]) or pd.isna(prev["ema_slow"]):
                continue

            price = latest["close"]
            atr = latest["atr"]
            rsi = latest["rsi"] if not pd.isna(latest["rsi"]) else 50
            ema = latest["ema"] if not pd.isna(latest["ema"]) else price
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 1.0
            adx = latest["adx"] if not pd.isna(latest.get("adx")) else 0

            # ADX must confirm trend strength
            if adx < self.adx_threshold:
                continue

            # Bullish crossover: fast EMA crosses above slow EMA
            if prev["ema_fast"] <= prev["ema_slow"] and latest["ema_fast"] > latest["ema_slow"]:
                separation = (latest["ema_fast"] - latest["ema_slow"]) / atr
                strength = min(1.0, separation * 0.5 + adx / 50.0 * 0.3)
                strength = max(0.01, strength)
                signals.append(Signal(
                    symbol=symbol, side="long", signal_type="dual_ema_long",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"adx": round(adx, 2), "ema_fast": round(latest["ema_fast"], 4),
                             "ema_slow": round(latest["ema_slow"], 4)},
                ))

            # Bearish crossover: fast EMA crosses below slow EMA
            elif prev["ema_fast"] >= prev["ema_slow"] and latest["ema_fast"] < latest["ema_slow"]:
                separation = (latest["ema_slow"] - latest["ema_fast"]) / atr
                strength = min(1.0, separation * 0.5 + adx / 50.0 * 0.3)
                strength = max(0.01, strength)
                signals.append(Signal(
                    symbol=symbol, side="short", signal_type="dual_ema_short",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"adx": round(adx, 2), "ema_fast": round(latest["ema_fast"], 4),
                             "ema_slow": round(latest["ema_slow"], 4)},
                ))

        return sorted(signals, key=lambda s: s.strength, reverse=True)[:5]


# ---------------------------------------------------------------------------
# Swing Strategies (daily bars)
# ---------------------------------------------------------------------------

class SwingWeeklyBreakout(ResearchStrategy):
    """Price closes above highest close of last 20 days with volume."""

    name = "swing_weekly_breakout"
    rsi_period = 14
    ema_period = 10
    volume_lookback = 20
    atr_period = 14
    breakout_period = 20
    _uses_daily_bars = True

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.ema_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
        df["highest_close"] = df["close"].rolling(window=self.breakout_period).max().shift(1)
        return df

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for symbol, df in bars.items():
            if len(df) < self.breakout_period + 5:
                continue
            df = self.compute_indicators(df)
            latest = df.iloc[-1]
            if pd.isna(latest["atr"]) or latest["atr"] <= 0:
                continue

            price = latest["close"]
            highest = latest["highest_close"]
            rsi = latest["rsi"] if not pd.isna(latest["rsi"]) else 50
            ema = latest["ema"] if not pd.isna(latest["ema"]) else price
            atr = latest["atr"]
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 1.0

            if pd.isna(highest):
                continue
            if price > highest and 50 <= rsi <= 65 and volume_ratio >= 1.2:
                strength = min(1.0, (price - highest) / atr * 0.5 + volume_ratio * 0.1)
                signals.append(Signal(
                    symbol=symbol, side="long", signal_type="swing_weekly_breakout",
                    strength=max(0.01, strength), rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"highest_close": round(highest, 2)},
                ))
        return sorted(signals, key=lambda s: s.strength, reverse=True)[:5]


class SwingSupportBounce(ResearchStrategy):
    """Price near 50-day low with RSI oversold and bounce confirmation."""

    name = "swing_support_bounce"
    rsi_period = 14
    ema_period = 10
    volume_lookback = 20
    atr_period = 14
    support_lookback = 50
    _uses_daily_bars = True

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.ema_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
        df["lowest_low"] = df["low"].rolling(window=self.support_lookback).min()
        return df

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for symbol, df in bars.items():
            if len(df) < self.support_lookback + 5:
                continue
            df = self.compute_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            if pd.isna(latest["atr"]) or latest["atr"] <= 0:
                continue

            price = latest["close"]
            atr = latest["atr"]
            rsi = latest["rsi"] if not pd.isna(latest["rsi"]) else 50
            ema = latest["ema"] if not pd.isna(latest["ema"]) else price
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 1.0
            lowest = latest["lowest_low"]

            if pd.isna(lowest):
                continue
            near_support = (price - lowest) < 0.5 * atr
            bounce = price > prev["close"]
            if near_support and bounce and rsi < 35:
                strength = min(1.0, (35 - rsi) / 20 + volume_ratio * 0.1)
                signals.append(Signal(
                    symbol=symbol, side="long", signal_type="swing_support_bounce",
                    strength=max(0.01, strength), rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"support": round(lowest, 2), "dist_atr": round((price - lowest) / atr, 2)},
                ))
        return sorted(signals, key=lambda s: s.strength, reverse=True)[:5]


class SwingEMACrossover(ResearchStrategy):
    """Daily EMA(10) crosses above EMA(50) with ADX confirmation."""

    name = "swing_ema_crossover"
    rsi_period = 14
    ema_period = 10  # not used directly, but needed for interface
    volume_lookback = 20
    atr_period = 14
    ema_fast = 10
    ema_slow = 50
    adx_period = 14
    _uses_daily_bars = True

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.ema_fast)
        df["ema_fast"] = ta.ema(df["close"], length=self.ema_fast)
        df["ema_slow"] = ta.ema(df["close"], length=self.ema_slow)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
        try:
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)
            adx_col = f"ADX_{self.adx_period}"
            df["adx"] = adx_df[adx_col] if adx_df is not None and adx_col in adx_df.columns else np.nan
        except Exception:
            df["adx"] = np.nan
        return df

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []
        for symbol, df in bars.items():
            if len(df) < self.ema_slow + 5:
                continue
            df = self.compute_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            if pd.isna(latest["atr"]) or latest["atr"] <= 0:
                continue

            price = latest["close"]
            atr = latest["atr"]
            rsi = latest["rsi"] if not pd.isna(latest["rsi"]) else 50
            ema = latest["ema"] if not pd.isna(latest["ema"]) else price
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 1.0
            adx = latest["adx"] if not pd.isna(latest.get("adx")) else 0

            ema_fast = latest.get("ema_fast")
            ema_slow = latest.get("ema_slow")
            prev_fast = prev.get("ema_fast")
            prev_slow = prev.get("ema_slow")

            if any(pd.isna(v) for v in [ema_fast, ema_slow, prev_fast, prev_slow]):
                continue

            crossed_up = prev_fast <= prev_slow and ema_fast > ema_slow
            if crossed_up and adx > 20 and volume_ratio >= 1.0:
                strength = min(1.0, adx / 40 + volume_ratio * 0.1)
                signals.append(Signal(
                    symbol=symbol, side="long", signal_type="swing_ema_crossover",
                    strength=max(0.01, strength), rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"adx": round(adx, 2), "ema_fast": round(ema_fast, 2), "ema_slow": round(ema_slow, 2)},
                ))
        return sorted(signals, key=lambda s: s.strength, reverse=True)[:5]


# ---------------------------------------------------------------------------
# Strategy 6: Pairs / Statistical Arbitrage
# ---------------------------------------------------------------------------

class PairsTrading(ResearchStrategy):
    """Market-neutral stat-arb: trade the spread between correlated stocks.

    Computes the price ratio of paired stocks, enters when the z-score
    exceeds a threshold, and exits when the spread reverts to the mean.
    Each pair produces two opposing signals (long underperformer, short
    outperformer).
    """

    name = "pairs_stat_arb"
    rsi_period = 14
    ema_period = 10
    volume_lookback = 20
    atr_period = 14

    # Pairs-specific parameters
    lookback_period = 20
    zscore_entry = 2.0
    zscore_exit = 0.5
    correlation_min = 0.7

    # Default pairs list
    default_pairs = [
        ("GOOGL", "META"),
        ("V", "MA"),
        ("JPM", "BAC"),
        ("XOM", "CVX"),
        ("AAPL", "MSFT"),
        ("HD", "LOW"),
        ("COST", "WMT"),
        ("AMD", "NVDA"),
        ("CRM", "ORCL"),
        ("UNH", "LLY"),
    ]

    def __init__(self):
        self._active_positions: dict[str, dict] = {}  # pair_tag -> {zscore_at_entry, side_a}

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.ema_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
        return df

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        signals = []

        for sym_a, sym_b in self.default_pairs:
            if sym_a not in bars or sym_b not in bars:
                continue

            df_a = bars[sym_a]
            df_b = bars[sym_b]

            # Align timestamps
            common_idx = df_a.index.intersection(df_b.index)
            if len(common_idx) < self.lookback_period + 5:
                continue

            close_a = df_a.loc[common_idx, "close"]
            close_b = df_b.loc[common_idx, "close"]

            # Check correlation
            corr = close_a.iloc[-self.lookback_period:].corr(close_b.iloc[-self.lookback_period:])
            if pd.isna(corr) or corr < self.correlation_min:
                continue

            # Compute ratio and z-score
            ratio = close_a / close_b
            ratio_mean = ratio.iloc[-self.lookback_period:].mean()
            ratio_std = ratio.iloc[-self.lookback_period:].std()

            if ratio_std <= 0 or pd.isna(ratio_std):
                continue

            current_ratio = ratio.iloc[-1]
            zscore = (current_ratio - ratio_mean) / ratio_std

            pair_tag = f"{sym_a}_{sym_b}"

            # Check for exit on active positions
            if pair_tag in self._active_positions:
                if abs(zscore) <= self.zscore_exit:
                    del self._active_positions[pair_tag]
                continue  # don't enter a new position while one is active

            # Check for entry
            if abs(zscore) < self.zscore_entry:
                continue

            # Compute indicators for each leg
            df_a_ind = self.compute_indicators(df_a)
            df_b_ind = self.compute_indicators(df_b)
            latest_a = df_a_ind.iloc[-1]
            latest_b = df_b_ind.iloc[-1]

            atr_a = latest_a["atr"] if not pd.isna(latest_a.get("atr")) else 0
            atr_b = latest_b["atr"] if not pd.isna(latest_b.get("atr")) else 0
            rsi_a = latest_a["rsi"] if not pd.isna(latest_a.get("rsi")) else 50
            rsi_b = latest_b["rsi"] if not pd.isna(latest_b.get("rsi")) else 50
            vol_ratio_a = latest_a["volume_ratio"] if not pd.isna(latest_a.get("volume_ratio")) else 1.0
            vol_ratio_b = latest_b["volume_ratio"] if not pd.isna(latest_b.get("volume_ratio")) else 1.0

            if atr_a <= 0 or atr_b <= 0:
                continue

            price_a = latest_a["close"]
            price_b = latest_b["close"]
            strength = min(1.0, abs(zscore) / 4.0)
            strength = max(0.1, strength)

            pair_context = {
                "pair": pair_tag,
                "zscore": round(float(zscore), 4),
                "correlation": round(float(corr), 4),
                "ratio": round(float(current_ratio), 6),
            }

            if zscore > self.zscore_entry:
                # Ratio too high: SHORT A, LONG B
                self._active_positions[pair_tag] = {"zscore_at_entry": zscore, "side_a": "short"}
                signals.append(Signal(
                    symbol=sym_b, side="long",
                    signal_type=f"pairs_long_{pair_tag}",
                    strength=strength, rsi=rsi_b, ema=price_b,
                    atr=atr_b, volume_ratio=vol_ratio_b, price=price_b,
                    context={"partner": sym_a, "leg": "long", **pair_context},
                ))
                signals.append(Signal(
                    symbol=sym_a, side="short",
                    signal_type=f"pairs_short_{pair_tag}",
                    strength=strength, rsi=rsi_a, ema=price_a,
                    atr=atr_a, volume_ratio=vol_ratio_a, price=price_a,
                    context={"partner": sym_b, "leg": "short", **pair_context},
                ))

            elif zscore < -self.zscore_entry:
                # Ratio too low: LONG A, SHORT B
                self._active_positions[pair_tag] = {"zscore_at_entry": zscore, "side_a": "long"}
                signals.append(Signal(
                    symbol=sym_a, side="long",
                    signal_type=f"pairs_long_{pair_tag}",
                    strength=strength, rsi=rsi_a, ema=price_a,
                    atr=atr_a, volume_ratio=vol_ratio_a, price=price_a,
                    context={"partner": sym_b, "leg": "long", **pair_context},
                ))
                signals.append(Signal(
                    symbol=sym_b, side="short",
                    signal_type=f"pairs_short_{pair_tag}",
                    strength=strength, rsi=rsi_b, ema=price_b,
                    atr=atr_b, volume_ratio=vol_ratio_b, price=price_b,
                    context={"partner": sym_a, "leg": "short", **pair_context},
                ))

        # Cap: 2 pairs max (4 signals)
        if len(signals) > 4:
            signals = signals[:4]

        return signals

    def should_exit(self, trade: dict, current_price: float, current_atr: float) -> tuple[bool, str]:
        """Pairs exit is handled inside generate_signals via active position tracking."""
        return False, ""


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES: dict[str, type[ResearchStrategy]] = {
    "vwap": VWAPMeanReversion,
    "orb": OpeningRangeBreakout,
    "rsi_div": RSIDivergence,
    "vol_spike": VolumeSpikeReversal,
    "dual_ema": DualEMATrend,
    "pairs": PairsTrading,
}

SWING_STRATEGIES: dict[str, type[ResearchStrategy]] = {
    "swing_breakout": SwingWeeklyBreakout,
    "swing_support": SwingSupportBounce,
    "swing_ema": SwingEMACrossover,
}


# ---------------------------------------------------------------------------
# Research runner
# ---------------------------------------------------------------------------

def run_research(
    strategy_names: list[str] | None = None,
    days: int = 90,
    num_symbols: int = 20,
) -> list[dict]:
    """Run backtests for selected strategies and return ranked results."""

    symbols = DEFAULT_UNIVERSE[:num_symbols]

    # Fetch data once, share across strategies
    bars = fetch_research_data(symbols, days)
    if not bars:
        logger.error("No data fetched. Check Alpaca credentials and market hours.")
        return []

    logger.info("Data ready: %d symbols, running strategies...\n", len(bars))

    if strategy_names is None:
        to_run = list(STRATEGIES.keys())
    else:
        to_run = strategy_names

    results = []

    for key in to_run:
        if key not in STRATEGIES:
            logger.warning("Unknown strategy: %s (available: %s)", key, ", ".join(STRATEGIES.keys()))
            continue

        strategy_cls = STRATEGIES[key]
        strategy = strategy_cls()
        logger.info("Running: %s ...", strategy.name)
        t0 = time.time()

        try:
            result = simulate(strategy, bars, portfolio_value=100_000)
            elapsed = time.time() - t0
            logger.info("  %s completed in %.1fs: Sharpe=%.4f, WinRate=%.1f%%, Trades=%d, P&L=$%.2f",
                        strategy.name, elapsed, result.sharpe_ratio,
                        result.win_rate * 100, result.total_trades, result.total_pnl)

            results.append({
                "strategy": strategy.name,
                "sharpe": round(result.sharpe_ratio, 4),
                "win_rate": round(result.win_rate * 100, 1),
                "max_dd": round(result.max_drawdown * 100, 2),
                "profit_factor": round(result.profit_factor, 4),
                "trades": result.total_trades,
                "pnl": round(result.total_pnl, 2),
                "daily_returns": result.daily_returns,
                "trade_details": [
                    {
                        "symbol": t.symbol, "side": t.side,
                        "entry_price": t.entry_price, "exit_price": t.exit_price,
                        "qty": t.qty, "pnl": round(t.pnl, 2),
                        "entry_time": t.entry_time, "exit_time": t.exit_time,
                        "exit_reason": t.exit_reason,
                    }
                    for t in result.trades
                ],
            })
        except Exception as e:
            logger.error("  %s FAILED: %s", strategy.name, e, exc_info=True)
            results.append({
                "strategy": strategy.name if hasattr(strategy, 'name') else key,
                "sharpe": 0, "win_rate": 0, "max_dd": 0,
                "profit_factor": 0, "trades": 0, "pnl": 0,
                "error": str(e),
            })

    # Sort by Sharpe descending
    results.sort(key=lambda r: r["sharpe"], reverse=True)
    return results


def print_results_table(results: list[dict]):
    """Print a formatted comparison table."""
    print()
    header = f"{'Strategy':<20} {'Sharpe':>8} {'WinRate':>8} {'MaxDD':>8} {'PF':>8} {'Trades':>8} {'P&L':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['strategy']:<20} "
            f"{r['sharpe']:>8.4f} "
            f"{r['win_rate']:>7.1f}% "
            f"{r['max_dd']:>7.2f}% "
            f"{r['profit_factor']:>8.4f} "
            f"{r['trades']:>8d} "
            f"${r['pnl']:>11,.2f}"
        )
    print()


def save_results(results: list[dict], path: Path):
    """Save detailed results to JSON (excluding large daily_returns arrays for readability)."""
    output = []
    for r in results:
        entry = {k: v for k, v in r.items() if k != "daily_returns"}
        output.append(entry)

    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info("Detailed results saved to %s", path)


def main():
    parser = argparse.ArgumentParser(description="Strategy Research Engine")
    all_names = list(STRATEGIES.keys()) + list(SWING_STRATEGIES.keys())
    parser.add_argument("--strategy", type=str, default=None,
                        help=f"Run a single strategy ({', '.join(all_names)})")
    parser.add_argument("--days", type=int, default=90, help="Days of historical data (default: 90)")
    parser.add_argument("--symbols", type=int, default=20, help="Number of symbols to test (default: 20)")
    parser.add_argument("--swing", action="store_true", help="Run swing strategies (daily bars) instead of intraday")
    args = parser.parse_args()

    if args.swing:
        strategy_names = [args.strategy] if args.strategy else list(SWING_STRATEGIES.keys())
        print(f"\n{'='*60}")
        print(f"  Swing Strategy Research Engine")
        print(f"  Days: {args.days} | Symbols: {args.symbols}")
        print(f"  Strategies: {', '.join(strategy_names)}")
        print(f"{'='*60}\n")

        symbols = DEFAULT_UNIVERSE[:args.symbols]
        bars = fetch_daily_data(symbols, args.days)
        if not bars:
            print("No data fetched.")
            return 1

        logger.info("Data ready: %d symbols, running swing strategies...\n", len(bars))
        results = []
        for key in strategy_names:
            if key not in SWING_STRATEGIES:
                logger.warning("Unknown swing strategy: %s", key)
                continue
            strategy = SWING_STRATEGIES[key]()
            logger.info("Running: %s ...", strategy.name)
            t0 = time.time()
            try:
                result = simulate(strategy, bars, portfolio_value=100_000)
                elapsed = time.time() - t0
                logger.info("  %s completed in %.1fs: Sharpe=%.4f, WinRate=%.1f%%, Trades=%d, P&L=$%.2f",
                            strategy.name, elapsed, result.sharpe_ratio,
                            result.win_rate * 100, result.total_trades, result.total_pnl)
                results.append({
                    "strategy": strategy.name,
                    "sharpe": round(result.sharpe_ratio, 4),
                    "win_rate": round(result.win_rate, 4),
                    "max_dd": round(result.max_drawdown * 100, 2),
                    "profit_factor": round(result.profit_factor, 4),
                    "trades": result.total_trades,
                    "pnl": round(result.total_pnl, 2),
                })
            except Exception as e:
                logger.error("  %s failed: %s", strategy.name, e, exc_info=True)
    else:
        strategy_names = [args.strategy] if args.strategy else None

        print(f"\n{'='*60}")
        print(f"  Strategy Research Engine")
        print(f"  Days: {args.days} | Symbols: {args.symbols}")
        if strategy_names:
            print(f"  Strategy: {args.strategy}")
        else:
            print(f"  Strategies: {', '.join(STRATEGIES.keys())}")
        print(f"{'='*60}\n")

        results = run_research(strategy_names, args.days, args.symbols)

    if not results:
        print("No results. Check logs above for errors.")
        return 1

    print_results_table(results)

    output_path = Path(__file__).parent / "research_results.json"
    save_results(results, output_path)

    # Print winner
    best = results[0]
    print(f"Best strategy: {best['strategy']} (Sharpe: {best['sharpe']:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
