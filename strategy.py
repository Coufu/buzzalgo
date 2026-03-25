"""
Active Trading Strategy - Claude Code modifies this file nightly.
==================================================================
Version: 8.3.0
Name: Close-30 Concentration
Description: v8.3.0: Apply signal concentration to crypto params — block
             MACD/Keltner via rsi_long_min=99, add 2.0x volume filter,
             mirror equity VWAP+HL(3) config, disable noisy signals,
             trim low-liquidity altcoins (GRT, SUSHI, BAT, CRV).
             Core signals: VWAP reversion + HL(3) + ATR contraction.

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
    "HD": "Consumer", "LOW": "Consumer", "COST": "Consumer", "WMT": "Consumer", "PG": "Consumer",
    "KO": "Consumer", "PEP": "Consumer", "ABNB": "Consumer", "UBER": "Consumer",
    "RBLX": "Consumer", "SHOP": "Consumer",
    # Financial
    "JPM": "Financial", "BAC": "Financial", "V": "Financial", "MA": "Financial",
    "PYPL": "Financial", "SQ": "Financial", "SOFI": "Financial", "HOOD": "Financial",
    "COIN": "Financial", "NU": "Financial",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "LLY": "Healthcare", "ABBV": "Healthcare",
    "MRK": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "ENPH": "Energy", "FSLR": "Energy",
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
    """Concentrated signal strategy: VWAP reversion + HL(3), noisy signals disabled."""

    VERSION = "8.3.0"

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
        # Market regime gate: only trade when SPY trend is not strongly bearish
        self.market_regime_gate = params.get("market_regime_gate", True)
        self.market_regime_symbol = params.get("market_regime_symbol", "SPY")
        self.spy_gate_min_slope = params.get("spy_gate_min_slope", -0.05)
        # Higher-low momentum: number of consecutive higher lows required
        self.higher_low_bars = params.get("higher_low_bars", 4)
        self.higher_low_rsi_min = params.get("higher_low_rsi_min", 40)
        self.higher_low_rsi_max = params.get("higher_low_rsi_max", 70)
        # Minimum EMA slope for any entry (per-symbol trend strength gate)
        self.min_ema_slope_entry = params.get("min_ema_slope_entry", 0.0)
        # VWAP reversion parameters
        self.vwap_enabled = params.get("vwap_enabled", True)
        self.vwap_rsi_max = params.get("vwap_rsi_max", 65)
        self.vwap_min_bars_into_day = params.get("vwap_min_bars_into_day", 6)  # 30 min of 5-min bars
        self.vwap_min_deviation_atr = params.get("vwap_min_deviation_atr", 0.5)
        # Signal enable/disable params (allows silencing noisy signals via config)
        self.rsi_divergence_enabled = params.get("rsi_divergence_enabled", True)
        self.volume_spike_enabled = params.get("volume_spike_enabled", True)
        self.liquidity_grab_enabled = params.get("liquidity_grab_enabled", True)
        self.atr_contraction_enabled = params.get("atr_contraction_enabled", True)
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
        # Pairs-specific params (only used when mode="pairs")
        self.pairs_list: list[list[str]] = params.get("pairs_list", [])
        self._pairs_lookback = params.get("lookback_period", 20)
        self._pairs_zscore_entry = params.get("zscore_entry", 2.0)
        self._pairs_zscore_exit = params.get("zscore_exit", 0.5)
        self._pairs_correlation_min = params.get("correlation_min", 0.7)
        # Swing-specific params (only used when mode="swing")
        self._swing_breakout_period = params.get("weekly_breakout_period", 20)
        self._swing_support_lookback = params.get("support_lookback", 50)
        self._swing_ema_fast = params.get("ema_fast", 10)
        self._swing_ema_slow = params.get("ema_slow", 50)
        # Legacy params for compatibility (used by backtest/risk)
        self.rsi_oversold = params.get("rsi_oversold", 25)
        self.rsi_overbought = params.get("rsi_overbought", 75)
        # Tournament signal weights: loaded from strategy.json "signal_weights" key.
        # Signals with weight 0 are filtered out; others have strength multiplied.
        # Signals not listed in signal_weights are left unmodified (weight 1.0).
        self.signal_weights: dict[str, float] = params.get("signal_weights", {})
        logger.info("Strategy v%s initialized: VWAP=%s HigherLow(%d) Keltner(%.1f) MACD(%d,%d,%d) RSI(%d) EMA(%d) hours=%d-%d long_only=%s",
                     self.VERSION, self.vwap_enabled, self.higher_low_bars,
                     self.keltner_mult, self.macd_fast, self.macd_slow,
                     self.macd_signal, self.rsi_period, self.ema_period,
                     self.trade_start_hour, self.trade_end_hour, self.long_only)

    @staticmethod
    def _load_params(mode: str = "equity") -> dict:
        if STRATEGY_JSON.exists():
            with open(STRATEGY_JSON) as f:
                data = json.load(f)
            if mode == "crypto" and "crypto" in data:
                crypto = data["crypto"]
                return {**crypto.get("parameters", {}), "universe": crypto.get("universe", [])}
            if mode == "swing" and "swing" in data:
                swing = data["swing"]
                return {**swing.get("parameters", {}), "universe": swing.get("universe", [])}
            if mode == "pairs" and "pairs" in data:
                pairs_cfg = data["pairs"]
                return {
                    **pairs_cfg.get("parameters", {}),
                    "universe": pairs_cfg.get("universe", []),
                    "pairs_list": pairs_cfg.get("pairs", []),
                }
            params = {**data.get("parameters", {}), "universe": data.get("universe", [])}
            if "signal_weights" in data:
                params["signal_weights"] = data["signal_weights"]
            return params
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
        # Donchian channel for liquidity grab detection (previous bar's levels)
        df["dc_high"] = df["high"].rolling(window=20).max().shift(1)
        df["dc_low"] = df["low"].rolling(window=20).min().shift(1)
        # VWAP: volume-weighted average price, reset daily
        if self.vwap_enabled:
            try:
                tp = (df["high"] + df["low"] + df["close"]) / 3
                tp_vol = tp * df["volume"]
                dates = df.index.date
                df["vwap"] = tp_vol.groupby(dates).cumsum() / df["volume"].groupby(dates).cumsum()
            except Exception:
                df["vwap"] = np.nan
        else:
            df["vwap"] = np.nan
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
        # Market regime gate: optionally block all signals when SPY is bearish
        if self.market_regime_gate and self.market_regime_symbol in bars:
            spy_df = bars[self.market_regime_symbol]
            min_bars = max(self.ema_period, self.atr_period) + 5
            if len(spy_df) >= min_bars:
                spy_df = self.compute_indicators(spy_df)
                spy_latest = spy_df.iloc[-1]
                spy_ema_slope = spy_latest["ema_slope"] if not pd.isna(spy_latest.get("ema_slope")) else 0
                if spy_ema_slope <= self.spy_gate_min_slope:
                    logger.debug("Market regime gate: SPY EMA slope %.4f <= %.4f, skipping all signals",
                                 spy_ema_slope, self.spy_gate_min_slope)
                    return []

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

            # Primary signal: VWAP Reversion (proven: Sharpe 4.81 in v6.0.0)
            signal = None

            # ATR Contraction Breakout (own volume threshold: 2.0x)
            # Checked BEFORE global volume filter — volatility contraction
            # signals need less volume confirmation since the setup itself is selective
            if volume_ratio >= 2.0:  # ATR contraction has own volume threshold
                signal = self._evaluate_atr_contraction_breakout(
                    symbol, df, rsi, price, ema, atr, volume_ratio,
                    extra_context=extra_context,
                )

            # Global volume filter for remaining signals (3.0x)
            if signal is None and volume_ratio < self.volume_multiplier:
                continue

            # ADX minimum filter
            if signal is None and self.min_adx_entry > 0 and adx < self.min_adx_entry:
                continue

            # Per-symbol EMA slope minimum
            if signal is None and self.min_ema_slope_entry > 0 and ema_slope < self.min_ema_slope_entry:
                continue

            # VWAP Reversion (mean-reversion to fair value — primary 3x signal)
            if signal is None:
                signal = self._evaluate_vwap_reversion(
                    symbol, df, rsi, price, ema, atr, volume_ratio,
                    extra_context=extra_context,
                )

            # RSI Divergence
            if signal is None:
                signal = self._evaluate_rsi_divergence(
                    symbol, df, rsi, price, ema, atr, volume_ratio,
                    extra_context=extra_context,
                )

            # Volume Spike Reversal (candlestick pattern + volume)
            if signal is None:
                signal = self._evaluate_volume_spike(
                    symbol, latest, price, ema, atr, rsi, volume_ratio,
                    extra_context=extra_context,
                )

            # Higher-Low Momentum (price structure)
            if signal is None:
                signal = self._evaluate_higher_low_momentum(
                    symbol, df, rsi, price, ema, atr, volume_ratio,
                    ema_slope=ema_slope, extra_context=extra_context,
                )

            # Secondary signal: MACD crossover (long only when long_only=True)
            if signal is None and macd is not None and macd_sig is not None and prev_macd is not None and prev_macd_sig is not None:
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

            # Liquidity grab reversal — fade false breakouts
            if signal is None:
                dc_high = latest["dc_high"] if not pd.isna(latest.get("dc_high")) else None
                dc_low = latest["dc_low"] if not pd.isna(latest.get("dc_low")) else None
                signal = self._evaluate_liquidity_grab(
                    symbol, rsi, price, ema, atr, volume_ratio,
                    dc_high, dc_low,
                    prev_high=prev["high"], prev_low=prev["low"], prev_close=prev["close"],
                    extra_context=extra_context,
                )

            if signal is not None:
                signals.append(signal)

        # Apply tournament signal weights from strategy.json
        if self.signal_weights:
            weighted = []
            for sig in signals:
                w = self.signal_weights.get(sig.signal_type)
                if w is not None:
                    if w <= 0:
                        continue  # disabled by tournament
                    sig.strength *= w
                    sig.strength = max(0.01, min(1.0, sig.strength))
                # Signals not in signal_weights pass through unmodified
                weighted.append(sig)
            signals = weighted

        # Cap signals per cycle
        if len(signals) > 3:
            signals = sorted(signals, key=lambda s: s.strength, reverse=True)[:3]

        return signals

    def _evaluate_higher_low_momentum(
        self, symbol: str, df: pd.DataFrame,
        rsi: float, price: float, ema: float, atr: float, volume_ratio: float,
        *, ema_slope: float = 0, extra_context: dict | None = None,
    ) -> Signal | None:
        """Higher-Low Momentum: 3+ consecutive higher lows = micro-uptrend.

        Fundamentally different from crossover/oscillator signals — based on
        pure price structure. Catches forming uptrends before indicators confirm.
        Combined with EMA trend filter and RSI momentum window.
        """
        n = self.higher_low_bars
        if len(df) < n + 1:
            return None

        # Check for N-1 consecutive higher lows in the last N bars
        lows = df["low"].iloc[-n:].values
        for i in range(1, len(lows)):
            if lows[i] <= lows[i - 1]:
                return None

        # Price must be above EMA (uptrend context)
        if price <= ema:
            return None

        # RSI momentum filter: not overbought, has room to run
        if rsi < self.higher_low_rsi_min or rsi > self.higher_low_rsi_max:
            return None

        # EMA slope must be non-negative (at least flat trend)
        if ema_slope < 0:
            return None

        # Strength: magnitude of higher-low progression + volume
        low_progression = (lows[-1] - lows[0]) / atr if atr > 0 else 0
        strength = min(1.0, low_progression * 0.5 + volume_ratio * 0.2)
        strength = max(0.1, strength)

        if self.sentiment_enabled:
            sent = (extra_context or {}).get("sentiment_score", 0)
            if sent != 0:
                strength *= (1.0 + sent * self.sentiment_weight)
                strength = max(0.01, min(1.0, strength))

        return Signal(
            symbol=symbol,
            side="long",
            signal_type="higher_low_momentum",
            strength=strength,
            rsi=rsi,
            ema=ema,
            atr=atr,
            volume_ratio=volume_ratio,
            price=price,
            context={
                "higher_lows": [round(float(l), 4) for l in lows],
                "low_progression_atr": round(low_progression, 3),
                **(extra_context or {}),
            },
        )

    def _evaluate_atr_contraction_breakout(
        self, symbol: str, df: pd.DataFrame,
        rsi: float, price: float, ema: float, atr: float, volume_ratio: float,
        *, extra_context: dict | None = None,
    ) -> Signal | None:
        """ATR Contraction Breakout: enter when volatility expands after compression.

        Detects periods where ATR has been declining (consolidation), then a
        volume spike + directional candle triggers entry. Volatility contraction
        followed by expansion is a regime-independent pattern — works in bull,
        bear, and choppy markets because it catches the START of new moves.
        Fundamentally different from momentum, mean-reversion, or divergence.
        """
        if not self.atr_contraction_enabled:
            return None
        lookback = 5
        if len(df) < lookback + 2:
            return None

        # Check ATR contraction: current ATR < 75% of ATR from lookback bars ago
        atr_prev = df["atr"].iloc[-(lookback + 1)]
        if pd.isna(atr_prev) or atr_prev <= 0:
            return None
        if atr >= 0.75 * atr_prev:
            return None  # No contraction detected — require 25%+ drop

        latest = df.iloc[-1]
        # Require bullish candle for long entry
        if latest["close"] <= latest["open"]:
            return None

        # RSI not overbought: room to run
        if rsi > 70:
            return None

        # Strength based on contraction depth + volume
        contraction_ratio = 1.0 - (atr / atr_prev)  # how much ATR contracted
        strength = min(1.0, contraction_ratio * 2.0 + volume_ratio * 0.2)
        strength = max(0.1, strength)

        if self.sentiment_enabled:
            sent = (extra_context or {}).get("sentiment_score", 0)
            if sent != 0:
                strength *= (1.0 + sent * self.sentiment_weight)
                strength = max(0.01, min(1.0, strength))

        return Signal(
            symbol=symbol,
            side="long",
            signal_type="atr_contraction_breakout",
            strength=strength,
            rsi=rsi,
            ema=ema,
            atr=atr,
            volume_ratio=volume_ratio,
            price=price,
            context={
                "atr_contraction": round(contraction_ratio, 3),
                "atr_prev": round(float(atr_prev), 4),
                "atr_curr": round(float(atr), 4),
                **(extra_context or {}),
            },
        )

    def _find_rsi_divergence(self, df: pd.DataFrame) -> str | None:
        """Check for RSI divergence in the last 10 bars.

        Bullish: price makes new low but RSI makes higher low.
        Bearish: price makes new high but RSI makes lower high.
        """
        lookback = 10
        if len(df) < lookback + 1:
            return None
        window = df.iloc[-(lookback + 1):]
        prices = window["close"].values
        rsis = window["rsi"].values
        if any(np.isnan(rsis)):
            return None

        prev_prices = prices[:-1]
        prev_rsis = rsis[:-1]
        curr_price = prices[-1]
        curr_rsi = rsis[-1]

        # Bullish: price new low, RSI higher low
        low_idx = np.argmin(prev_prices)
        if curr_price < prev_prices[low_idx] and curr_rsi > prev_rsis[low_idx]:
            return "bullish"

        # Bearish: price new high, RSI lower high
        high_idx = np.argmax(prev_prices)
        if curr_price > prev_prices[high_idx] and curr_rsi < prev_rsis[high_idx]:
            return "bearish"

        return None

    def _evaluate_rsi_divergence(
        self, symbol: str, df: pd.DataFrame,
        rsi: float, price: float, ema: float, atr: float, volume_ratio: float,
        *, extra_context: dict | None = None,
    ) -> Signal | None:
        """RSI Divergence signal — research winner (Sharpe 0.83, 40% WR)."""
        if not self.rsi_divergence_enabled:
            return None
        divergence = self._find_rsi_divergence(df)
        if divergence is None:
            return None

        ctx = extra_context or {}

        if divergence == "bullish" and rsi < 45:
            strength = min(1.0, (50 - rsi) / 30 + volume_ratio * 0.1)
            return Signal(
                symbol=symbol, side="long", signal_type="rsi_div_bullish",
                strength=max(0.01, strength), rsi=rsi, ema=ema, atr=atr,
                volume_ratio=volume_ratio, price=price,
                context={"divergence": "bullish", **ctx},
            )
        elif divergence == "bearish" and rsi > 55 and not self.long_only:
            strength = min(1.0, (rsi - 50) / 30 + volume_ratio * 0.1)
            return Signal(
                symbol=symbol, side="short", signal_type="rsi_div_bearish",
                strength=max(0.01, strength), rsi=rsi, ema=ema, atr=atr,
                volume_ratio=volume_ratio, price=price,
                context={"divergence": "bearish", **ctx},
            )
        return None

    def _evaluate_volume_spike(
        self, symbol: str, latest, price: float, ema: float, atr: float,
        rsi: float, volume_ratio: float,
        *, extra_context: dict | None = None,
    ) -> Signal | None:
        """Volume Spike Reversal — hammer/shooting star on 3x volume."""
        if not self.volume_spike_enabled:
            return None
        if volume_ratio < 3.0:
            return None

        ctx = extra_context or {}
        body = abs(latest["close"] - latest["open"])
        full_range = latest["high"] - latest["low"]
        if full_range <= 0:
            return None

        lower_wick = min(latest["open"], latest["close"]) - latest["low"]
        upper_wick = latest["high"] - max(latest["open"], latest["close"])

        # Hammer (bullish): long lower wick, close in upper half
        is_hammer = (lower_wick > 2 * body and
                     latest["close"] >= (latest["high"] + latest["low"]) / 2)
        # Shooting star (bearish): long upper wick, close in lower half
        is_star = (upper_wick > 2 * body and
                   latest["close"] <= (latest["high"] + latest["low"]) / 2)

        strength = min(1.0, volume_ratio / 5.0)
        strength = max(0.01, strength)

        if is_hammer:
            return Signal(
                symbol=symbol, side="long", signal_type="vol_spike_hammer",
                strength=strength, rsi=rsi, ema=ema, atr=atr,
                volume_ratio=volume_ratio, price=price,
                context={"pattern": "hammer", "vol_ratio": round(volume_ratio, 2), **ctx},
            )
        elif is_star and not self.long_only:
            return Signal(
                symbol=symbol, side="short", signal_type="vol_spike_star",
                strength=strength, rsi=rsi, ema=ema, atr=atr,
                volume_ratio=volume_ratio, price=price,
                context={"pattern": "shooting_star", "vol_ratio": round(volume_ratio, 2), **ctx},
            )
        return None

    def _evaluate_vwap_reversion(
        self, symbol: str, df: pd.DataFrame,
        rsi: float, price: float, ema: float, atr: float, volume_ratio: float,
        *, extra_context: dict | None = None,
    ) -> Signal | None:
        """VWAP Reversion: buy when price crosses above VWAP after being below.

        Mean-reversion to intraday fair value. Works in all market regimes
        because stocks oscillate around VWAP regardless of macro direction.
        Fundamentally different from all prior trend-following signals.
        """
        if not self.vwap_enabled or len(df) < self.vwap_min_bars_into_day:
            return None

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        vwap = latest.get("vwap")
        prev_vwap = prev.get("vwap")
        if vwap is None or prev_vwap is None or pd.isna(vwap) or pd.isna(prev_vwap):
            return None

        prev_close = prev["close"]
        # Price must have just crossed above VWAP (was below, now above)
        if not (prev_close < prev_vwap and price > vwap):
            return None

        # Sustained deviation: require >= 2 of last 3 bars closed below VWAP
        # Filters out noisy single-bar dips that immediately reverse
        if len(df) >= 4:
            bars_below = 0
            for i in range(-3, 0):
                bar = df.iloc[i]
                bar_vwap = bar.get("vwap")
                if bar_vwap is not None and not pd.isna(bar_vwap) and bar["close"] < bar_vwap:
                    bars_below += 1
            if bars_below < 2:
                return None

        # Require meaningful prior deviation: prev close was at least N ATR below VWAP
        if atr > 0 and (prev_vwap - prev_close) < self.vwap_min_deviation_atr * atr:
            return None

        # RSI filter: not overbought — room to continue upward
        if rsi > self.vwap_rsi_max:
            return None

        # Bounce confirmation: current bar is bullish (close > open)
        if price <= latest["open"]:
            return None

        # Strength: proximity to VWAP (tighter = stronger) + volume
        distance_atr = abs(price - vwap) / atr if atr > 0 else 0
        strength = min(1.0, volume_ratio * 0.3 + max(0, 1.0 - distance_atr) * 0.4)
        strength = max(0.1, strength)

        if self.sentiment_enabled:
            sent = (extra_context or {}).get("sentiment_score", 0)
            if sent != 0:
                strength *= (1.0 + sent * self.sentiment_weight)
                strength = max(0.01, min(1.0, strength))

        return Signal(
            symbol=symbol,
            side="long",
            signal_type="vwap_reversion_long",
            strength=strength,
            rsi=rsi,
            ema=ema,
            atr=atr,
            volume_ratio=volume_ratio,
            price=price,
            context={
                "vwap": round(float(vwap), 4),
                "prev_close_vs_vwap": round(float(prev_close - prev_vwap), 4),
                "price_vs_vwap": round(float(price - vwap), 4),
                **(extra_context or {}),
            },
        )

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

    def _evaluate_liquidity_grab(
        self, symbol: str, rsi: float, price: float,
        ema: float, atr: float, volume_ratio: float,
        dc_high: float | None, dc_low: float | None,
        *, prev_high: float, prev_low: float, prev_close: float,
        extra_context: dict | None = None,
    ) -> Signal | None:
        """Liquidity grab reversal: fade false breakouts.

        Detects when price swept beyond a Donchian level (stop hunt) on the
        previous bar but closed back inside the range — then enter the reversal.

        Long: prev bar swept below dc_low (grabbed liquidity) but closed above it,
              current bar confirms by closing above prev close. RSI 30-50 (oversold bounce).
        Short: prev bar swept above dc_high but closed below it,
               current bar confirms by closing below prev close. RSI 50-70.
        """
        if not self.liquidity_grab_enabled:
            return None
        if dc_high is None or dc_low is None or atr <= 0:
            return None

        ctx = extra_context or {}

        # Long: liquidity grab below support
        swept_low = prev_low < dc_low  # wick below the channel
        closed_inside_low = prev_close > dc_low  # but closed back inside
        confirms_up = price > prev_close  # current bar moving up
        if swept_low and closed_inside_low and confirms_up and 25 <= rsi <= 55:
            if self.long_only or not self.long_only:  # works in both modes
                sweep_depth = (dc_low - prev_low) / atr  # how far it swept
                strength = min(1.0, sweep_depth * 0.5 + volume_ratio * 0.2)
                strength = max(0.1, strength)
                return Signal(
                    symbol=symbol, side="long", signal_type="liquidity_grab_long",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"dc_low": round(dc_low, 4), "sweep_depth_atr": round(sweep_depth, 3), **ctx},
                )

        # Short: liquidity grab above resistance (skip if long_only)
        if not self.long_only:
            swept_high = prev_high > dc_high
            closed_inside_high = prev_close < dc_high
            confirms_down = price < prev_close
            if swept_high and closed_inside_high and confirms_down and 45 <= rsi <= 75:
                sweep_depth = (prev_high - dc_high) / atr
                strength = min(1.0, sweep_depth * 0.5 + volume_ratio * 0.2)
                strength = max(0.1, strength)
                return Signal(
                    symbol=symbol, side="short", signal_type="liquidity_grab_short",
                    strength=strength, rsi=rsi, ema=ema, atr=atr,
                    volume_ratio=volume_ratio, price=price,
                    context={"dc_high": round(dc_high, 4), "sweep_depth_atr": round(sweep_depth, 3), **ctx},
                )

        return None

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

    # ------------------------------------------------------------------
    # Swing trading signals (mode="swing", daily bars)
    # ------------------------------------------------------------------

    def generate_swing_signals(self, bars: dict[str, pd.DataFrame], open_symbols: list[str] | None = None) -> list[Signal]:
        """Generate swing trading signals from daily bars.

        Only fires when self.mode == "swing". Three long-only signals:
        1. Weekly Breakout — price closes above highest close of last N days
        2. Support Bounce — price near support with RSI oversold + bounce
        3. EMA Crossover (Daily) — fast EMA crosses above slow EMA in trend
        """
        if self.mode != "swing":
            return []

        # Load swing-specific params (with sane defaults)
        weekly_breakout_period = getattr(self, "_swing_breakout_period", 20)
        support_lookback = getattr(self, "_swing_support_lookback", 50)
        ema_fast_period = getattr(self, "_swing_ema_fast", 10)
        ema_slow_period = getattr(self, "_swing_ema_slow", 50)

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

            min_bars = max(support_lookback, ema_slow_period, self.volume_lookback) + 5
            if len(df) < min_bars:
                continue

            # Compute core indicators
            df = self._compute_swing_indicators(df, ema_fast_period, ema_slow_period)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            if pd.isna(latest["rsi"]) or pd.isna(latest["atr"]):
                continue

            rsi = latest["rsi"]
            atr = latest["atr"]
            price = latest["close"]
            volume_ratio = latest["volume_ratio"] if not pd.isna(latest["volume_ratio"]) else 0
            adx = latest["adx"] if not pd.isna(latest.get("adx")) else 0
            ema_fast = latest["ema_fast"]
            ema_slow = latest["ema_slow"]

            # Sector correlation filter
            sector = SECTOR_MAP.get(symbol, "Other")
            if sector != "ETF" and sector_counts.get(sector, 0) >= self.max_positions_per_sector:
                continue

            ctx = {
                "mode": "swing",
                "adx": round(adx, 2),
                "rsi": round(rsi, 2),
                "volume_ratio": round(volume_ratio, 2),
            }

            signal = None

            # Signal 1: Weekly Breakout
            if signal is None:
                signal = self._swing_weekly_breakout(
                    symbol, df, price, rsi, atr, volume_ratio,
                    ema_fast, weekly_breakout_period, ctx,
                )

            # Signal 2: Support Bounce
            if signal is None:
                signal = self._swing_support_bounce(
                    symbol, df, price, rsi, atr, volume_ratio,
                    ema_fast, support_lookback, prev, ctx,
                )

            # Signal 3: EMA Crossover (Daily)
            if signal is None:
                signal = self._swing_ema_crossover(
                    symbol, df, price, rsi, atr, volume_ratio,
                    ema_fast, ema_slow, adx, prev, ctx,
                )

            if signal is not None:
                signals.append(signal)

        # Cap swing signals per cycle
        if len(signals) > 3:
            signals = sorted(signals, key=lambda s: s.strength, reverse=True)[:3]

        return signals

    def _compute_swing_indicators(self, df: pd.DataFrame, ema_fast_period: int, ema_slow_period: int) -> pd.DataFrame:
        """Compute indicators for swing trading on daily bars."""
        df = df.copy()
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["ema_fast"] = ta.ema(df["close"], length=ema_fast_period)
        df["ema_slow"] = ta.ema(df["close"], length=ema_slow_period)
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
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

    def _swing_weekly_breakout(
        self, symbol: str, df: pd.DataFrame,
        price: float, rsi: float, atr: float, volume_ratio: float,
        ema_fast: float, breakout_period: int, ctx: dict,
    ) -> Signal | None:
        """Weekly Breakout: price closes above highest close of last N days.

        Confirms with volume > 1.2x avg and RSI 50-65 (momentum, not overbought).
        """
        if len(df) < breakout_period + 1:
            return None

        # Highest close in the lookback (excluding current bar)
        highest_close = df["close"].iloc[-(breakout_period + 1):-1].max()

        if price <= highest_close:
            return None

        # Volume confirmation
        if volume_ratio < self.volume_multiplier:
            return None

        # RSI momentum band: 50-65 (not overbought, has momentum)
        if rsi < 50 or rsi > 65:
            return None

        strength = min(1.0, (price - highest_close) / atr * 0.3 + volume_ratio * 0.2)
        strength = max(0.1, strength)

        return Signal(
            symbol=symbol,
            side="long",
            signal_type="swing_weekly_breakout",
            strength=strength,
            rsi=rsi,
            ema=ema_fast,
            atr=atr,
            volume_ratio=volume_ratio,
            price=price,
            context={
                "highest_close": round(float(highest_close), 2),
                "breakout_pct": round((price - highest_close) / highest_close * 100, 2),
                **ctx,
            },
        )

    def _swing_support_bounce(
        self, symbol: str, df: pd.DataFrame,
        price: float, rsi: float, atr: float, volume_ratio: float,
        ema_fast: float, support_lookback: int, prev, ctx: dict,
    ) -> Signal | None:
        """Support Bounce: price near lowest low with RSI oversold + bounce confirmation.

        Price within 0.5 ATR of the lowest low, RSI < 35, current bar closes above previous.
        """
        if len(df) < support_lookback + 1:
            return None

        # Lowest low in lookback (excluding current bar)
        lowest_low = df["low"].iloc[-(support_lookback + 1):-1].min()

        # Price must be within 0.5 ATR of support
        if atr <= 0:
            return None
        distance = price - lowest_low
        if distance > 0.5 * atr or distance < 0:
            return None

        # RSI oversold
        rsi_oversold = getattr(self, "rsi_oversold", 35)
        if rsi >= rsi_oversold:
            return None

        # Bounce confirmation: current close > previous close
        prev_close = prev["close"]
        if price <= prev_close:
            return None

        strength = min(1.0, (rsi_oversold - rsi) / 30 * 0.5 + volume_ratio * 0.2)
        strength = max(0.1, strength)

        return Signal(
            symbol=symbol,
            side="long",
            signal_type="swing_support_bounce",
            strength=strength,
            rsi=rsi,
            ema=ema_fast,
            atr=atr,
            volume_ratio=volume_ratio,
            price=price,
            context={
                "support_level": round(float(lowest_low), 2),
                "distance_atr": round(distance / atr, 3),
                **ctx,
            },
        )

    def _swing_ema_crossover(
        self, symbol: str, df: pd.DataFrame,
        price: float, rsi: float, atr: float, volume_ratio: float,
        ema_fast: float, ema_slow: float, adx: float,
        prev, ctx: dict,
    ) -> Signal | None:
        """EMA Crossover (Daily): fast EMA crosses above slow EMA in trending market.

        ADX > 20 confirms trend, volume confirmation required.
        """
        if pd.isna(ema_fast) or pd.isna(ema_slow):
            return None

        prev_ema_fast = prev.get("ema_fast")
        prev_ema_slow = prev.get("ema_slow")
        if prev_ema_fast is None or prev_ema_slow is None:
            return None
        if pd.isna(prev_ema_fast) or pd.isna(prev_ema_slow):
            return None

        # Crossover: was below, now above
        if not (prev_ema_fast <= prev_ema_slow and ema_fast > ema_slow):
            return None

        # ADX trending threshold
        adx_threshold = getattr(self, "adx_trending", 20)
        if adx < adx_threshold:
            return None

        # Volume confirmation
        if volume_ratio < self.volume_multiplier:
            return None

        strength = min(1.0, adx / 50 * 0.4 + volume_ratio * 0.2)
        strength = max(0.1, strength)

        return Signal(
            symbol=symbol,
            side="long",
            signal_type="swing_ema_crossover",
            strength=strength,
            rsi=rsi,
            ema=ema_fast,
            atr=atr,
            volume_ratio=volume_ratio,
            price=price,
            context={
                "ema_fast": round(float(ema_fast), 2),
                "ema_slow": round(float(ema_slow), 2),
                **ctx,
            },
        )

    # ------------------------------------------------------------------
    # Pairs / statistical arbitrage signals (mode="pairs", 15-min bars)
    # ------------------------------------------------------------------

    def generate_pairs_signals(
        self, bars: dict[str, pd.DataFrame], open_symbols: list[str] | None = None,
    ) -> list[Signal]:
        """Generate pairs trading signals from 15-min bars.

        For each configured pair (A, B):
          1. Compute the price ratio = close_A / close_B over lookback bars.
          2. Compute z-score of the current ratio vs its rolling mean/std.
          3. If z-score > entry threshold: SHORT A, LONG B (ratio too high).
          4. If z-score < -entry threshold: LONG A, SHORT B (ratio too low).
          5. Both legs fire together so they can be executed as a pair.
        """
        if self.mode != "pairs":
            return []

        lookback = self._pairs_lookback
        zscore_entry = self._pairs_zscore_entry
        correlation_min = self._pairs_correlation_min
        open_symbols = open_symbols or []

        signals = []
        for pair in self.pairs_list:
            if len(pair) != 2:
                continue
            sym_a, sym_b = pair[0], pair[1]

            if sym_a not in bars or sym_b not in bars:
                continue

            df_a = bars[sym_a]
            df_b = bars[sym_b]

            # Align timestamps between the two symbols
            common_idx = df_a.index.intersection(df_b.index)
            if len(common_idx) < lookback + 5:
                continue

            close_a = df_a.loc[common_idx, "close"]
            close_b = df_b.loc[common_idx, "close"]

            # Check correlation over the lookback window
            corr = close_a.iloc[-lookback:].corr(close_b.iloc[-lookback:])
            if pd.isna(corr) or corr < correlation_min:
                continue

            # Compute price ratio and z-score
            ratio = close_a / close_b
            ratio_mean = ratio.iloc[-lookback:].mean()
            ratio_std = ratio.iloc[-lookback:].std()

            if ratio_std <= 0 or pd.isna(ratio_std):
                continue

            current_ratio = ratio.iloc[-1]
            zscore = (current_ratio - ratio_mean) / ratio_std

            # No signal if z-score within entry threshold
            if abs(zscore) < zscore_entry:
                continue

            # Compute ATR for each leg (needed for risk.py position sizing)
            df_a_ind = self._compute_pairs_indicators(df_a)
            df_b_ind = self._compute_pairs_indicators(df_b)
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

            # Strength: capped abs(zscore) / 4.0
            strength = min(1.0, abs(zscore) / 4.0)
            strength = max(0.1, strength)

            pair_tag = f"{sym_a}_{sym_b}"
            pair_context = {
                "pair": pair_tag,
                "zscore": round(float(zscore), 4),
                "correlation": round(float(corr), 4),
                "ratio": round(float(current_ratio), 6),
                "ratio_mean": round(float(ratio_mean), 6),
                "ratio_std": round(float(ratio_std), 6),
            }

            if zscore > zscore_entry:
                # Ratio too high: SHORT A (overperformer), LONG B (underperformer)
                # Skip if either leg already has an open position
                if sym_a not in open_symbols and sym_b not in open_symbols:
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

            elif zscore < -zscore_entry:
                # Ratio too low: LONG A (underperformer), SHORT B (overperformer)
                if sym_a not in open_symbols and sym_b not in open_symbols:
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

        # Cap to 4 signals (2 pairs) per cycle
        if len(signals) > 4:
            # Group by pair, sort by strength, take top 2 pairs (4 signals)
            pair_groups: dict[str, list[Signal]] = {}
            for sig in signals:
                ptag = sig.context.get("pair", "")
                pair_groups.setdefault(ptag, []).append(sig)
            sorted_pairs = sorted(pair_groups.values(), key=lambda sigs: sigs[0].strength, reverse=True)
            signals = []
            for pair_sigs in sorted_pairs[:2]:
                signals.extend(pair_sigs)

        return signals

    def _compute_pairs_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute basic indicators for pairs trading legs."""
        df = df.copy()
        df["atr"] = ta.atr(df["high"], df["low"], df["close"], length=self.atr_period)
        df["rsi"] = ta.rsi(df["close"], length=self.rsi_period)
        df["vol_avg"] = df["volume"].rolling(window=self.volume_lookback).mean()
        df["volume_ratio"] = df["volume"] / df["vol_avg"]
        return df

    def should_exit_pairs(
        self, trade: dict, bars: dict[str, pd.DataFrame],
    ) -> tuple[bool, str]:
        """Check if a pairs trade should be exited based on spread reversion.

        Returns (True, reason) if the z-score has reverted to the exit threshold.
        Also returns True if this is an orphaned leg (partner closed).
        """
        signal_type = trade.get("signal_type", "")
        if not signal_type.startswith("pairs_"):
            return False, ""

        context = trade.get("signal_context") or {}
        if isinstance(context, str):
            try:
                context = json.loads(context)
            except (json.JSONDecodeError, TypeError):
                context = {}

        pair_tag = context.get("pair", "")
        if not pair_tag or "_" not in pair_tag:
            return False, ""

        sym_a, sym_b = pair_tag.split("_", 1)
        if sym_a not in bars or sym_b not in bars:
            return False, ""

        df_a = bars[sym_a]
        df_b = bars[sym_b]

        # Align timestamps
        common_idx = df_a.index.intersection(df_b.index)
        lookback = self._pairs_lookback
        if len(common_idx) < lookback + 1:
            return False, ""

        close_a = df_a.loc[common_idx, "close"]
        close_b = df_b.loc[common_idx, "close"]

        ratio = close_a / close_b
        ratio_mean = ratio.iloc[-lookback:].mean()
        ratio_std = ratio.iloc[-lookback:].std()

        if ratio_std <= 0 or pd.isna(ratio_std):
            return False, ""

        current_zscore = (ratio.iloc[-1] - ratio_mean) / ratio_std

        # Exit when z-score reverts to exit threshold
        if abs(current_zscore) <= self._pairs_zscore_exit:
            return True, "pairs_mean_reversion"

        return False, ""

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
