"""
Active Trading Strategy - Claude Code modifies this file nightly.
==================================================================
Version: 1.6.0
Name: Momentum + Mean Reversion Hybrid
Description: RSI(14) oversold/overbought signals with EMA(10)
             direction filter, volume confirmation, ADX regime
             detection, RSI momentum filter, and time-of-day
             awareness on 5-min bars.
             v1.6.0: Removed BB squeeze breakout (generated 540+
             false signals, Sharpe -21 → -0.88).

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

    VERSION = "1.6.0"

    def __init__(self, params: dict | None = None, mode: str = "equity"):
        self.mode = mode
        if params is None:
            params = self._load_params(mode)
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
        self.time_exit_minutes = params.get("time_exit_minutes", 120)
        self.time_exit_max_profit_atr = params.get("time_exit_max_profit_atr", 0.5)
        self.sentiment_enabled = params.get("sentiment_enabled", False)
        self.sentiment_weight = params.get("sentiment_weight", 0.3)
        self.sentiment_veto_threshold = params.get("sentiment_veto_threshold", 0.5)
        self.sentiment_lookback_hours = params.get("sentiment_lookback_hours", 4)
        self.multi_timeframe_enabled = params.get("multi_timeframe_enabled", False)
        self.daily_trend_period = params.get("daily_trend_period", 5)
        self.max_positions_per_sector = params.get("max_positions_per_sector", 2)
        self.min_ema_slope = params.get("min_ema_slope", 0.0)
        self.min_adx_entry = params.get("min_adx_entry", 0)
        self.min_rsi_delta = params.get("min_rsi_delta", 3.0)
        self.universe = params.get("universe", ["SPY", "QQQ", "IWM"])
        logger.info("Strategy v%s initialized: RSI(%d) EMA(%d) ADX(%d)",
                     self.VERSION, self.rsi_period, self.ema_period, self.adx_period)

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

    def generate_signals(self, bars: dict[str, pd.DataFrame], open_symbols: list[str] | None = None) -> list[Signal]:
        """Generate trading signals for all symbols.

        Args:
            bars: Dict mapping symbol -> DataFrame of OHLCV bars.
                  Each DataFrame must have at least `ema_period + rsi_period` rows.

        Returns:
            List of Signal objects for actionable setups.
        """
        # Load sentiment scores if enabled
        sentiment_map: dict[str, float] = {}
        if self.sentiment_enabled:
            try:
                import db as _db
                with _db.get_db() as conn:
                    sentiment_map = _db.get_aggregate_sentiment(conn, self.sentiment_lookback_hours)
            except Exception:
                pass  # No sentiment data is fine — defaults to neutral

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

            # ADX minimum filter - skip ranging markets with no directional conviction
            if self.min_adx_entry > 0 and adx < self.min_adx_entry:
                continue

            # Multi-timeframe filter: skip signals against daily trend
            daily_trend = daily_trends.get(symbol, "neutral")
            if self.multi_timeframe_enabled and daily_trend != "neutral":
                # We'll pass daily_trend to _evaluate_entry for filtering
                pass

            # Sector correlation filter
            sector = SECTOR_MAP.get(symbol, "Other")
            if sector != "ETF" and sector_counts.get(sector, 0) >= self.max_positions_per_sector:
                continue

            signal = self._evaluate_entry(
                symbol, rsi, prev["rsi"], ema, atr, price, volume_ratio,
                regime=regime, adx=adx, ema_slope=ema_slope,
                time_bucket=time_bucket, entry_hour=entry_hour,
                sentiment_score=sentiment_map.get(symbol, 0.0),
                daily_trend=daily_trend,
            )
            if signal is not None:
                signals.append(signal)

        return signals

    def _evaluate_entry(
        self, symbol: str, rsi: float, prev_rsi: float,
        ema: float, atr: float, price: float, volume_ratio: float,
        *, regime: str = "", adx: float = 0, ema_slope: float = 0,
        time_bucket: str = "", entry_hour: int | None = None,
        sentiment_score: float = 0.0,
        daily_trend: str = "neutral",
    ) -> Signal | None:
        """Evaluate whether current conditions warrant an entry signal."""

        extra_context = {
            "market_regime": regime,
            "adx": round(adx, 2),
            "ema_slope": round(ema_slope, 4),
            "time_bucket": time_bucket,
            "entry_hour": entry_hour,
            "sentiment_score": round(sentiment_score, 3),
            "daily_trend": daily_trend,
        }

        # Long signal: RSI crosses above oversold + price above EMA (momentum confirmation)
        if prev_rsi <= self.rsi_oversold and rsi > self.rsi_oversold and price > ema:
            # Skip longs in strong downtrend
            if regime == "trending_down" and adx > self.adx_trending:
                return None
            # Skip longs when EMA slope is below minimum threshold
            if ema_slope <= self.min_ema_slope:
                return None
            # Skip weak RSI bounces — require minimum RSI acceleration
            if (rsi - prev_rsi) < self.min_rsi_delta:
                return None
            # Multi-timeframe: skip longs if daily trend is down
            if self.multi_timeframe_enabled and daily_trend == "down":
                return None
            strength = min(1.0, (self.rsi_oversold - prev_rsi + 10) / 20 * volume_ratio / max(self.volume_multiplier, 0.01))
            # Sentiment boost/dampen signal strength
            if self.sentiment_enabled and sentiment_score != 0:
                strength *= (1.0 + sentiment_score * self.sentiment_weight)
                strength = max(0.01, min(1.0, strength))
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
            # Multi-timeframe: skip shorts if daily trend is up
            if self.multi_timeframe_enabled and daily_trend == "up":
                return None
            # Skip weak RSI drops — require minimum RSI deceleration
            if (prev_rsi - rsi) < self.min_rsi_delta:
                return None
            strength = min(1.0, (prev_rsi - self.rsi_overbought + 10) / 20 * volume_ratio / max(self.volume_multiplier, 0.01))
            # Sentiment boost/dampen (invert: bearish sentiment boosts short strength)
            if self.sentiment_enabled and sentiment_score != 0:
                strength *= (1.0 - sentiment_score * self.sentiment_weight)
                strength = max(0.01, min(1.0, strength))
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

        # Note: trailing stop logic is handled by the backtest simulate() and
        # trader via stop_price/take_profit_price from risk.py. The should_exit()
        # method cannot implement a true trailing stop because it has no memory
        # of the peak price between calls.

        return False, ""
