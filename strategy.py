"""
Active Trading Strategy - Claude Code modifies this file nightly.
==================================================================
Version: 1.1.0
Name: Momentum + Mean Reversion Hybrid
Description: RSI(14) oversold/overbought signals with EMA(20)
             direction filter, volume confirmation, ADX regime
             detection, and time-of-day awareness on 5-min bars.

Claude Code may freely modify:
  - RSI periods & thresholds
  - Moving average types/periods
  - Volume filter thresholds
  - Universe (within equities)
  - Entry/exit conditions
  - Take profit multipliers
  - ADX thresholds for regime detection

Claude Code may NOT modify:
  - Position sizing (handled by risk.py)
  - Stop-loss formula (handled by risk.py)
  - Circuit breaker (handled by risk.py)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_ta as ta

logger = logging.getLogger(__name__)

STRATEGY_JSON = Path(__file__).parent / "strategy.json"


@dataclass
class Signal:
    symbol: str
    side: str          # "long" or "short"
    signal_type: str   # e.g., "rsi_oversold_bounce", "rsi_overbought_short"
    strength: float    # 0.0 to 1.0
    rsi: float
    ema: float
    atr: float
    volume_ratio: float
    price: float
    context: dict


class Strategy:
    """Baseline momentum + mean-reversion hybrid strategy."""

    VERSION = "1.1.0"

    def __init__(self, params: dict | None = None):
        if params is None:
            params = self._load_params()
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_oversold = params.get("rsi_oversold", 30)
        self.rsi_overbought = params.get("rsi_overbought", 70)
        self.ema_period = params.get("ema_period", 20)
        self.volume_multiplier = params.get("volume_multiplier", 1.5)
        self.volume_lookback = params.get("volume_lookback", 20)
        self.atr_period = params.get("atr_period", 14)
        self.adx_period = params.get("adx_period", 14)
        self.adx_trending = params.get("adx_trending_threshold", 25)
        self.adx_ranging = params.get("adx_ranging_threshold", 20)
        self.universe = params.get("universe", ["SPY", "QQQ", "IWM"])
        logger.info("Strategy v%s initialized: RSI(%d) EMA(%d) ADX(%d)",
                     self.VERSION, self.rsi_period, self.ema_period, self.adx_period)

    @staticmethod
    def _load_params() -> dict:
        if STRATEGY_JSON.exists():
            with open(STRATEGY_JSON) as f:
                data = json.load(f)
            return {**data.get("parameters", {}), "universe": data.get("universe", [])}
        return {}

    def reload_params(self):
        """Hot-reload parameters from strategy.json without restarting."""
        params = self._load_params()
        self.__init__(params)
        logger.info("Strategy parameters hot-reloaded")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI, EMA, ATR, ADX, and volume ratio columns to a DataFrame of bars.

        Expects columns: open, high, low, close, volume.
        """
        df = df.copy()
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.ema_period)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
        # Regime detection
        try:
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)
            adx_col = f"ADX_{self.adx_period}"
            if adx_df is not None and adx_col in adx_df.columns:
                df["adx"] = adx_df[adx_col]
            else:
                logger.warning("ADX calculation returned unexpected columns: %s",
                               list(adx_df.columns) if adx_df is not None else "None")
                df["adx"] = np.nan
        except Exception as e:
            logger.warning("ADX calculation failed: %s", e)
            df["adx"] = np.nan
        df["ema_slope"] = (df["ema"] - df["ema"].shift(5)) / df["ema"].shift(5) * 100
        return df

    def _classify_regime(self, adx: float, ema_slope: float) -> str:
        """Classify market regime from ADX and EMA slope."""
        if adx > self.adx_trending:
            return "trending_up" if ema_slope > 0 else "trending_down"
        elif adx < self.adx_ranging:
            return "ranging"
        return "transitional"

    @staticmethod
    def _classify_time_bucket(ts) -> str:
        """Classify a bar timestamp into a time-of-day bucket."""
        if hasattr(ts, "hour"):
            h, m = ts.hour, ts.minute
        else:
            return "unknown"
        if h == 9:
            return "open_30"
        elif h < 12:
            return "morning"
        elif h < 14:
            return "midday"
        elif h < 15 or (h == 15 and m < 30):
            return "afternoon"
        return "close_30"

    def generate_signals(self, bars: dict[str, pd.DataFrame]) -> list[Signal]:
        """Generate trading signals for all symbols.

        Args:
            bars: Dict mapping symbol -> DataFrame of OHLCV bars.
                  Each DataFrame must have at least `ema_period + rsi_period` rows.

        Returns:
            List of Signal objects for actionable setups.
        """
        signals = []
        for symbol, df in bars.items():
            if symbol not in self.universe:
                continue
            if len(df) < max(self.rsi_period, self.ema_period, self.volume_lookback) + 5:
                continue

            df = self.compute_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            if pd.isna(latest["rsi"]) or pd.isna(latest["ema"]) or pd.isna(latest["atr"]):
                continue

            rsi = latest["rsi"]
            ema = latest["ema"]
            atr = latest["atr"]
            price = latest["close"]
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 0
            adx = latest["adx"] if not pd.isna(latest.get("adx")) else 0
            ema_slope = latest["ema_slope"] if not pd.isna(latest.get("ema_slope")) else 0

            regime = self._classify_regime(adx, ema_slope)
            time_bucket = self._classify_time_bucket(df.index[-1])
            entry_hour = df.index[-1].hour if hasattr(df.index[-1], "hour") else None

            # Volume filter - skip low-volume bars
            if volume_ratio < self.volume_multiplier:
                continue

            signal = self._evaluate_entry(
                symbol, rsi, prev["rsi"], ema, atr, price, volume_ratio,
                regime=regime, adx=adx, ema_slope=ema_slope,
                time_bucket=time_bucket, entry_hour=entry_hour,
            )
            if signal is not None:
                signals.append(signal)

        return signals

    def _evaluate_entry(
        self, symbol: str, rsi: float, prev_rsi: float,
        ema: float, atr: float, price: float, volume_ratio: float,
        *, regime: str = "", adx: float = 0, ema_slope: float = 0,
        time_bucket: str = "", entry_hour: int | None = None,
    ) -> Signal | None:
        """Evaluate whether current conditions warrant an entry signal."""

        extra_context = {
            "market_regime": regime,
            "adx": round(adx, 2),
            "ema_slope": round(ema_slope, 4),
            "time_bucket": time_bucket,
            "entry_hour": entry_hour,
        }

        # Long signal: RSI crosses above oversold + price above EMA (momentum confirmation)
        if prev_rsi <= self.rsi_oversold and rsi > self.rsi_oversold and price > ema:
            # Skip longs in strong downtrend
            if regime == "trending_down" and adx > self.adx_trending:
                return None

            strength = min(1.0, (self.rsi_oversold - prev_rsi + 10) / 20 * volume_ratio / max(self.volume_multiplier, 0.01))
            return Signal(
                symbol=symbol,
                side="long",
                signal_type="rsi_oversold_bounce",
                strength=strength,
                rsi=rsi,
                ema=ema,
                atr=atr,
                volume_ratio=volume_ratio,
                price=price,
                context={
                    "prev_rsi": prev_rsi,
                    "price_above_ema": True,
                    "ema_distance_pct": round((price - ema) / ema * 100, 3),
                    **extra_context,
                },
            )

        # Short signal: RSI crosses below overbought + price below EMA
        if prev_rsi >= self.rsi_overbought and rsi < self.rsi_overbought and price < ema:
            # Skip shorts in strong uptrend
            if regime == "trending_up" and adx > self.adx_trending:
                return None

            strength = min(1.0, (prev_rsi - self.rsi_overbought + 10) / 20 * volume_ratio / max(self.volume_multiplier, 0.01))
            return Signal(
                symbol=symbol,
                side="short",
                signal_type="rsi_overbought_short",
                strength=strength,
                rsi=rsi,
                ema=ema,
                atr=atr,
                volume_ratio=volume_ratio,
                price=price,
                context={
                    "prev_rsi": prev_rsi,
                    "price_below_ema": True,
                    "ema_distance_pct": round((price - ema) / ema * 100, 3),
                    **extra_context,
                },
            )

        return None

    def should_exit(self, trade: dict, current_price: float, current_atr: float) -> tuple[bool, str]:
        """Check if an open trade should be exited based on strategy logic.

        Note: Stop-loss and take-profit exits are handled by risk.py.
        This method handles strategy-specific exit logic only.

        Args:
            trade: Trade dict from the database.
            current_price: Current market price.
            current_atr: Current ATR value.

        Returns:
            (should_exit, reason) tuple.
        """
        # Trailing stop logic: after price moves 1.5x ATR in our favor,
        # tighten stop to 1x ATR from current price
        entry_price = trade["entry_price"]
        side = trade["side"]
        atr_at_entry = trade.get("atr_at_entry", current_atr)

        if side == "long":
            profit_atr = (current_price - entry_price) / atr_at_entry
            if profit_atr >= 1.5:
                trailing_stop = current_price - current_atr
                if current_price <= trailing_stop:
                    return True, "trailing_stop"
        elif side == "short":
            profit_atr = (entry_price - current_price) / atr_at_entry
            if profit_atr >= 1.5:
                trailing_stop = current_price + current_atr
                if current_price >= trailing_stop:
                    return True, "trailing_stop"

        return False, ""
