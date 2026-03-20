"""
Active Trading Strategy - Claude Code modifies this file nightly.
==================================================================
Version: 3.0.0
Name: Keltner Breakout Long-Only
Description: Keltner channel breakout signals, long-only, afternoon
             session. v3.0.0: MAJOR PIVOT — kill criteria triggered on
             v2.0.0 (Sharpe -7.17, 5.8% win rate, 207 trades).
             MACD crossovers fired too frequently (163 trades in one day).
             Shorts were catastrophic (125 short trades, 1.6% WR).
             New approach: Keltner breakout (price > EMA + mult*ATR)
             captures confirmed volatility expansion in uptrends.
             Long-only eliminates the losing short side.
             Afternoon session (12-16 ET) targets the only profitable
             time windows (midday +$7.69, close_30 +$32.93).
             MACD crossover long retained as secondary signal.

Claude Code may freely modify:
  - Keltner channel multiplier
  - MACD parameters (fast, slow, signal periods)
  - RSI periods & thresholds (used as filter, not trigger)
  - Moving average types/periods
  - Volume filter thresholds
  - Universe (within equities)
  - Entry/exit conditions
  - Time-of-day filter windows
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

SECTOR_MAP = {
    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF", "XLF": "ETF",
    "XLE": "ETF", "XLK": "ETF", "XLV": "ETF", "XLI": "ETF", "XLP": "ETF",
    "GLD": "ETF", "SLV": "ETF", "EEM": "ETF", "TLT": "ETF", "HYG": "ETF",
    "ARKK": "ETF", "SOXL": "ETF", "TQQQ": "ETF",
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "GOOGL": "Tech", "META": "Tech",
    "AMD": "Tech", "AVGO": "Tech", "INTC": "Tech", "CSCO": "Tech", "ORCL": "Tech",
    "ADBE": "Tech", "CRM": "Tech", "SMCI": "Tech", "ARM": "Tech",
    # Consumer/Internet
    "AMZN": "Consumer", "TSLA": "Consumer", "NFLX": "Consumer", "DIS": "Consumer",
    "HD": "Consumer", "COST": "Consumer", "WMT": "Consumer", "PG": "Consumer",
    "KO": "Consumer", "PEP": "Consumer", "ABNB": "Consumer", "UBER": "Consumer",
    "RBLX": "Consumer", "SHOP": "Consumer",
    # Financial
    "JPM": "Financial", "V": "Financial", "MA": "Financial", "BAC": "Financial",
    "PYPL": "Financial", "SQ": "Financial", "SOFI": "Financial", "HOOD": "Financial",
    "COIN": "Financial", "NU": "Financial",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare",
    # Energy
    "XOM": "Energy", "ENPH": "Energy", "FSLR": "Energy",
    # Travel/Transport
    "CCL": "Travel", "AAL": "Travel", "DAL": "Travel", "UAL": "Travel", "F": "Travel", "GM": "Travel",
    # Crypto-adjacent
    "MARA": "Crypto", "RIOT": "Crypto", "MSTR": "Crypto",
    # China/LatAm
    "BABA": "International", "JD": "International", "PDD": "International",
    "BILI": "International", "GRAB": "International", "SE": "International", "MELI": "International",
    # Cloud/Cyber
    "CRWD": "Cloud", "PANW": "Cloud", "ZS": "Cloud", "NET": "Cloud",
    "DDOG": "Cloud", "MDB": "Cloud", "SNOW": "Cloud", "PATH": "Cloud", "U": "Cloud",
    # Other
    "RIVN": "EV", "LCID": "EV", "NIO": "EV",
    "DKNG": "Consumer", "SNAP": "Tech", "PLTR": "Tech", "ROKU": "Tech", "RKLB": "Tech",
    "T": "Telecom",
    # Crypto
    "BTC/USD": "Crypto", "ETH/USD": "Crypto", "SOL/USD": "Crypto", "AVAX/USD": "Crypto",
    "DOGE/USD": "Crypto", "LINK/USD": "Crypto", "LTC/USD": "Crypto", "UNI/USD": "Crypto",
    "XRP/USD": "Crypto", "BCH/USD": "Crypto", "DOT/USD": "Crypto",
    "AAVE/USD": "Crypto", "SHIB/USD": "Crypto", "GRT/USD": "Crypto",
    "SUSHI/USD": "Crypto", "BAT/USD": "Crypto", "CRV/USD": "Crypto",
}


@dataclass
class Signal:
    symbol: str
    side: str          # "long" or "short"
    signal_type: str   # e.g., "macd_crossover_long", "macd_crossover_short"
    strength: float    # 0.0 to 1.0
    rsi: float
    ema: float
    atr: float
    volume_ratio: float
    price: float
    context: dict


class Strategy:
    """Keltner channel breakout strategy, long-only, afternoon session."""

    VERSION = "3.0.0"

    def __init__(self, params: dict | None = None, mode: str = "equity"):
        self.mode = mode
        if params is None:
            params = self._load_params(mode)
        # Long-only mode: disable all short entries
        self.long_only = params.get("long_only", True)
        # Keltner channel breakout
        self.keltner_mult = params.get("keltner_mult", 2.0)
        # MACD parameters (primary signal)
        self.macd_fast = params.get("macd_fast", 5)
        self.macd_slow = params.get("macd_slow", 13)
        self.macd_signal = params.get("macd_signal", 5)
        # RSI as momentum filter (not trigger)
        self.rsi_period = params.get("rsi_period", 14)
        self.rsi_long_min = params.get("rsi_long_min", 50)
        self.rsi_long_max = params.get("rsi_long_max", 75)
        self.rsi_short_max = params.get("rsi_short_max", 50)
        # EMA trend filter
        self.ema_period = params.get("ema_period", 10)
        # Volume filter
        self.volume_multiplier = params.get("volume_multiplier", 1.5)
        self.volume_lookback = params.get("volume_lookback", 20)
        # ATR
        self.atr_period = params.get("atr_period", 14)
        # ADX regime detection
        self.adx_period = params.get("adx_period", 14)
        self.adx_trending = params.get("adx_trending_threshold", 25)
        self.adx_ranging = params.get("adx_ranging_threshold", 20)
        self.min_adx_entry = params.get("min_adx_entry", 20)
        # Minimum MACD histogram magnitude (as fraction of ATR) to filter weak crossovers
        self.min_macd_atr_ratio = params.get("min_macd_atr_ratio", 0.0)
        # Time-based exit
        self.time_exit_minutes = params.get("time_exit_minutes", 120)
        self.time_exit_max_profit_atr = params.get("time_exit_max_profit_atr", 0.5)
        # Time-of-day filter: only trade during these hours (ET)
        self.trade_start_hour = params.get("trade_start_hour", 12)
        self.trade_end_hour = params.get("trade_end_hour", 16)
        # Sentiment
        self.sentiment_enabled = params.get("sentiment_enabled", False)
        self.sentiment_weight = params.get("sentiment_weight", 0.3)
        self.sentiment_veto_threshold = params.get("sentiment_veto_threshold", 0.5)
        self.sentiment_lookback_hours = params.get("sentiment_lookback_hours", 4)
        # Multi-timeframe
        self.multi_timeframe_enabled = params.get("multi_timeframe_enabled", False)
        self.daily_trend_period = params.get("daily_trend_period", 5)
        # Sector filter
        self.max_positions_per_sector = params.get("max_positions_per_sector", 2)
        # Universe
        self.universe = params.get("universe", ["SPY", "QQQ", "IWM"])
        # Legacy params for compatibility (used by backtest/risk)
        self.rsi_oversold = params.get("rsi_oversold", 25)
        self.rsi_overbought = params.get("rsi_overbought", 75)
        logger.info("Strategy v%s initialized: Keltner(%.1f) MACD(%d,%d,%d) RSI(%d) EMA(%d) ADX(%d) hours=%d-%d long_only=%s",
                     self.VERSION, self.keltner_mult, self.macd_fast, self.macd_slow, self.macd_signal,
                     self.rsi_period, self.ema_period, self.adx_period,
                     self.trade_start_hour, self.trade_end_hour, self.long_only)

    @staticmethod
    def _load_params(mode: str = "equity") -> dict:
        if STRATEGY_JSON.exists():
            with open(STRATEGY_JSON) as f:
                data = json.load(f)
            if mode == "crypto" and "crypto" in data:
                crypto = data["crypto"]
                return {**crypto.get("parameters", {}), "universe": crypto.get("universe", [])}
            return {**data.get("parameters", {}), "universe": data.get("universe", [])}
        return {}

    def reload_params(self):
        """Hot-reload parameters from strategy.json without restarting."""
        params = self._load_params(self.mode)
        self.__init__(params, mode=self.mode)
        logger.info("Strategy parameters hot-reloaded")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD, RSI, EMA, ATR, ADX, and volume ratio columns."""
        df = df.copy()
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema"] = ta.ema(df["close"], length=self.ema_period)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
        # MACD
        macd_df = ta.macd(df["close"], fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        if macd_df is not None:
            macd_col = f"MACD_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
            signal_col = f"MACDs_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
            hist_col = f"MACDh_{self.macd_fast}_{self.macd_slow}_{self.macd_signal}"
            df["macd"] = macd_df[macd_col] if macd_col in macd_df.columns else np.nan
            df["macd_signal"] = macd_df[signal_col] if signal_col in macd_df.columns else np.nan
            df["macd_hist"] = macd_df[hist_col] if hist_col in macd_df.columns else np.nan
        else:
            df["macd"] = np.nan
            df["macd_signal"] = np.nan
            df["macd_hist"] = np.nan
        # Regime detection
        try:
            adx_df = ta.adx(df["high"], df["low"], df["close"], length=self.adx_period)
            adx_col = f"ADX_{self.adx_period}"
            if adx_df is not None and adx_col in adx_df.columns:
                df["adx"] = adx_df[adx_col]
            else:
                df["adx"] = np.nan
        except Exception as e:
            logger.warning("ADX calculation failed: %s", e)
            df["adx"] = np.nan
        df["ema_slope"] = (df["ema"] - df["ema"].shift(5)) / df["ema"].shift(5) * 100
        # Keltner channel upper band
        df["keltner_upper"] = df["ema"] + self.keltner_mult * df["atr"]
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

    def _daily_trend(self, df: pd.DataFrame) -> str:
        """Compute daily trend from 5-min bars by resampling to daily."""
        try:
            daily = df["close"].resample("1D").last().dropna()
            if len(daily) < self.daily_trend_period:
                return "neutral"
            ema = daily.ewm(span=self.daily_trend_period, adjust=False).mean()
            if len(ema) < 2:
                return "neutral"
            slope = (ema.iloc[-1] - ema.iloc[-2]) / ema.iloc[-2]
            if slope > 0.001:
                return "up"
            elif slope < -0.001:
                return "down"
        except Exception:
            pass
        return "neutral"

    def _is_trading_hour(self, ts) -> bool:
        """Check if timestamp falls within allowed trading hours (ET)."""
        if hasattr(ts, "hour"):
            return self.trade_start_hour <= ts.hour < self.trade_end_hour
        return False

    def generate_signals(self, bars: dict[str, pd.DataFrame], open_symbols: list[str] | None = None) -> list[Signal]:
        """Generate trading signals for all symbols."""
        # Load sentiment scores if enabled
        sentiment_map: dict[str, float] = {}
        if self.sentiment_enabled:
            try:
                import db as _db
                with _db.get_db() as conn:
                    sentiment_map = _db.get_aggregate_sentiment(conn, self.sentiment_lookback_hours)
            except Exception:
                pass

        # Compute daily trends for multi-timeframe filter
        daily_trends: dict[str, str] = {}
        if self.multi_timeframe_enabled:
            for symbol, df in bars.items():
                daily_trends[symbol] = self._daily_trend(df)

        # Sector counts for correlation filter
        sector_counts: dict[str, int] = {}
        if open_symbols:
            for sym in open_symbols:
                sector = SECTOR_MAP.get(sym, "Other")
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

        signals = []
        for symbol, df in bars.items():
            if symbol not in self.universe:
                continue
            min_bars = max(self.rsi_period, self.ema_period, self.volume_lookback) + 5
            if len(df) < min_bars:
                continue

            df = self.compute_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            if pd.isna(latest["rsi"]) or pd.isna(latest["ema"]) or pd.isna(latest["atr"]):
                continue

            # Time-of-day filter: only trade during afternoon session
            if not self._is_trading_hour(df.index[-1]):
                continue

            rsi = latest["rsi"]
            ema = latest["ema"]
            atr = latest["atr"]
            price = latest["close"]
            keltner_upper = latest["keltner_upper"] if not pd.isna(latest.get("keltner_upper")) else None
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 0
            adx = latest["adx"] if not pd.isna(latest.get("adx")) else 0
            ema_slope = latest["ema_slope"] if not pd.isna(latest.get("ema_slope")) else 0
            macd = latest["macd"] if not pd.isna(latest.get("macd")) else None
            macd_sig = latest["macd_signal"] if not pd.isna(latest.get("macd_signal")) else None
            macd_hist = latest["macd_hist"] if not pd.isna(latest.get("macd_hist")) else 0
            prev_macd = prev["macd"] if not pd.isna(prev.get("macd")) else None
            prev_macd_sig = prev["macd_signal"] if not pd.isna(prev.get("macd_signal")) else None

            regime = self._classify_regime(adx, ema_slope)
            time_bucket = self._classify_time_bucket(df.index[-1])
            entry_hour = df.index[-1].hour if hasattr(df.index[-1], "hour") else None

            # Volume filter
            if volume_ratio < self.volume_multiplier:
                continue

            # ADX minimum filter
            if self.min_adx_entry > 0 and adx < self.min_adx_entry:
                continue

            # Sector correlation filter
            sector = SECTOR_MAP.get(symbol, "Other")
            if sector != "ETF" and sector_counts.get(sector, 0) >= self.max_positions_per_sector:
                continue

            daily_trend = daily_trends.get(symbol, "neutral")

            extra_context = {
                "market_regime": regime,
                "adx": round(adx, 2),
                "ema_slope": round(ema_slope, 4),
                "time_bucket": time_bucket,
                "entry_hour": entry_hour,
                "sentiment_score": round(sentiment_map.get(symbol, 0.0), 3),
                "daily_trend": daily_trend,
            }

            # Primary signal: MACD crossover (long only when long_only=True)
            signal = None
            if macd is not None and macd_sig is not None and prev_macd is not None and prev_macd_sig is not None:
                signal = self._evaluate_entry(
                    symbol, rsi, price, ema, atr, volume_ratio,
                    macd, macd_sig, prev_macd, prev_macd_sig, macd_hist,
                    regime=regime, adx=adx, ema_slope=ema_slope,
                    time_bucket=time_bucket, entry_hour=entry_hour,
                    sentiment_score=sentiment_map.get(symbol, 0.0),
                    daily_trend=daily_trend,
                )

            # Standalone Keltner breakout signal (long only, very selective)
            if signal is None:
                signal = self._evaluate_keltner_breakout(
                    symbol, rsi, price, ema, atr, volume_ratio,
                    keltner_upper, regime=regime, adx=adx, ema_slope=ema_slope,
                    sentiment_score=sentiment_map.get(symbol, 0.0),
                    daily_trend=daily_trend, extra_context=extra_context,
                )

            if signal is not None:
                signals.append(signal)

        return signals

    def _evaluate_keltner_breakout(
        self, symbol: str, rsi: float, price: float,
        ema: float, atr: float, volume_ratio: float,
        keltner_upper: float | None,
        *, regime: str = "", adx: float = 0, ema_slope: float = 0,
        sentiment_score: float = 0.0, daily_trend: str = "neutral",
        extra_context: dict | None = None,
    ) -> Signal | None:
        """Evaluate Keltner channel breakout entry (long only).

        Fires only when price breaks ABOVE upper Keltner band with
        momentum + uptrend confirmation. Very selective.
        """
        if keltner_upper is None or atr <= 0:
            return None
        # Price must close above Keltner upper band
        if price <= keltner_upper:
            return None
        # RSI momentum confirmation: 50-75 (momentum but not overbought)
        if rsi < self.rsi_long_min or rsi > self.rsi_long_max:
            return None
        # Require uptrend: EMA slope > 0
        if ema_slope <= 0:
            return None
        # Skip if regime is downtrend
        if regime == "trending_down":
            return None
        # Multi-timeframe: skip longs if daily trend is down
        if self.multi_timeframe_enabled and daily_trend == "down":
            return None
        # Strength based on breakout magnitude + volume
        breakout_strength = min(1.0, (price - keltner_upper) / atr)
        strength = min(1.0, breakout_strength * volume_ratio / max(self.volume_multiplier, 0.01))
        strength = max(0.01, strength)
        if self.sentiment_enabled and sentiment_score != 0:
            strength *= (1.0 + sentiment_score * self.sentiment_weight)
            strength = max(0.01, min(1.0, strength))
        return Signal(
            symbol=symbol,
            side="long",
            signal_type="keltner_breakout_long",
            strength=strength,
            rsi=rsi,
            ema=ema,
            atr=atr,
            volume_ratio=volume_ratio,
            price=price,
            context={
                "keltner_upper": round(keltner_upper, 4),
                "breakout_pct": round((price - keltner_upper) / keltner_upper * 100, 4),
                **(extra_context or {}),
            },
        )

    def _evaluate_entry(
        self, symbol: str, rsi: float, price: float,
        ema: float, atr: float, volume_ratio: float,
        macd: float, macd_sig: float,
        prev_macd: float, prev_macd_sig: float, macd_hist: float,
        *, regime: str = "", adx: float = 0, ema_slope: float = 0,
        time_bucket: str = "", entry_hour: int | None = None,
        sentiment_score: float = 0.0, daily_trend: str = "neutral",
    ) -> Signal | None:
        """Evaluate MACD crossover entry conditions (secondary signal)."""

        extra_context = {
            "market_regime": regime,
            "adx": round(adx, 2),
            "ema_slope": round(ema_slope, 4),
            "time_bucket": time_bucket,
            "entry_hour": entry_hour,
            "sentiment_score": round(sentiment_score, 3),
            "daily_trend": daily_trend,
        }

        # Minimum MACD histogram magnitude filter (avoid weak crossovers)
        if self.min_macd_atr_ratio > 0 and atr > 0:
            if abs(macd_hist) < atr * self.min_macd_atr_ratio:
                return None

        # Bullish MACD crossover: MACD crosses above signal line
        # Confirmed by: RSI > 50 (momentum), price > EMA (trend), EMA slope > 0 (uptrend)
        if prev_macd <= prev_macd_sig and macd > macd_sig and price > ema and rsi >= self.rsi_long_min and ema_slope > 0:
            # Skip longs in strong downtrend
            if regime == "trending_down" and adx > self.adx_trending:
                return None
            # Multi-timeframe: skip longs if daily trend is down
            if self.multi_timeframe_enabled and daily_trend == "down":
                return None
            # Strength based on MACD histogram magnitude + volume
            hist_strength = min(1.0, abs(macd_hist) / atr * 2) if atr > 0 else 0.5
            strength = min(1.0, hist_strength * volume_ratio / max(self.volume_multiplier, 0.01))
            strength = max(0.01, strength)
            if self.sentiment_enabled and sentiment_score != 0:
                strength *= (1.0 + sentiment_score * self.sentiment_weight)
                strength = max(0.01, min(1.0, strength))
            return Signal(
                symbol=symbol,
                side="long",
                signal_type="macd_crossover_long",
                strength=strength,
                rsi=rsi,
                ema=ema,
                atr=atr,
                volume_ratio=volume_ratio,
                price=price,
                context={
                    "macd": round(macd, 6),
                    "macd_signal": round(macd_sig, 6),
                    "macd_hist": round(macd_hist, 6),
                    "price_above_ema": True,
                    **extra_context,
                },
            )

        # Bearish MACD crossover: MACD crosses below signal line
        # Confirmed by: RSI < 50, price < EMA
        # Disabled when long_only=True
        if not self.long_only and prev_macd >= prev_macd_sig and macd < macd_sig and price < ema and rsi <= self.rsi_short_max:
            # Skip shorts in strong uptrend
            if regime == "trending_up" and adx > self.adx_trending:
                return None
            # Multi-timeframe: skip shorts if daily trend is up
            if self.multi_timeframe_enabled and daily_trend == "up":
                return None
            hist_strength = min(1.0, abs(macd_hist) / atr * 2) if atr > 0 else 0.5
            strength = min(1.0, hist_strength * volume_ratio / max(self.volume_multiplier, 0.01))
            strength = max(0.01, strength)
            if self.sentiment_enabled and sentiment_score != 0:
                strength *= (1.0 - sentiment_score * self.sentiment_weight)
                strength = max(0.01, min(1.0, strength))
            return Signal(
                symbol=symbol,
                side="short",
                signal_type="macd_crossover_short",
                strength=strength,
                rsi=rsi,
                ema=ema,
                atr=atr,
                volume_ratio=volume_ratio,
                price=price,
                context={
                    "macd": round(macd, 6),
                    "macd_signal": round(macd_sig, 6),
                    "macd_hist": round(macd_hist, 6),
                    "price_below_ema": True,
                    **extra_context,
                },
            )

        return None

    def should_exit(self, trade: dict, current_price: float, current_atr: float) -> tuple[bool, str]:
        """Check if an open trade should be exited based on strategy logic.

        Note: Stop-loss and take-profit exits are handled by risk.py.
        This method handles strategy-specific exit logic only.
        """
        entry_price = trade["entry_price"]
        side = trade["side"]
        atr_at_entry = trade.get("atr_at_entry", current_atr)

        if side == "long":
            profit_atr = (current_price - entry_price) / atr_at_entry
        else:
            profit_atr = (entry_price - current_price) / atr_at_entry

        # Time-based exit: close flat trades after N minutes (live only)
        entry_time_str = trade.get("opened_at") or trade.get("entry_time")
        if entry_time_str and self.time_exit_minutes > 0:
            try:
                from datetime import datetime
                from zoneinfo import ZoneInfo
                entry_time = datetime.fromisoformat(str(entry_time_str))
                elapsed = (datetime.now(ZoneInfo("America/New_York")) - entry_time).total_seconds() / 60
                if elapsed >= self.time_exit_minutes and abs(profit_atr) < self.time_exit_max_profit_atr:
                    return True, "time_exit_flat"
            except (ValueError, TypeError):
                pass

        return False, ""
