"""
Triple Bottom Pattern Scanner
==============================
Scans the stock universe for triple bottom chart patterns using
daily bars from Alpaca. Sends Slack notifications for detected
setups. Notification-only — does not trigger trades.

Usage:
    python scanner.py          # Run continuously (daily at 4:45pm ET)
    python scanner.py --once   # Single scan
"""

import argparse
import json
import logging
import os
import signal
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

logger = logging.getLogger("scanner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

STRATEGY_JSON = Path(__file__).parent / "strategy.json"
DATA_DIR = Path(__file__).parent / "data"
DEDUP_FILE = DATA_DIR / "scanner_notified.json"
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")

_shutdown = False


def _handle_signal(signum, frame):
    global _shutdown
    _shutdown = True
    logger.info("Shutdown signal received")


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _notify_slack(text: str):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        logger.debug("Slack notification failed: %s", e)


def load_config() -> dict:
    """Load scanner config from strategy.json."""
    defaults = {
        "enabled": True,
        "lookback_days": 90,
        "swing_window": 3,
        "support_tolerance_pct": 1.5,
        "min_bounce_pct": 2.0,
        "min_touches": 3,
        "recency_days": 5,
        "proximity_pct": 5.0,
        "dedup_days": 5,
    }
    try:
        with open(STRATEGY_JSON) as f:
            config = json.load(f)
        scanner_cfg = config.get("scanner", {})
        defaults.update(scanner_cfg)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return defaults


def load_universe() -> list[str]:
    """Load stock universe from strategy.json."""
    try:
        with open(STRATEGY_JSON) as f:
            config = json.load(f)
        return config.get("universe", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return ["SPY", "QQQ", "IWM"]


def load_dedup() -> dict:
    """Load dedup state: {symbol: iso_timestamp_of_last_notification}."""
    if DEDUP_FILE.exists():
        try:
            with open(DEDUP_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_dedup(state: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DEDUP_FILE, "w") as f:
        json.dump(state, f, indent=2)


def fetch_daily_bars(symbols: list[str], lookback_days: int) -> dict:
    """Fetch daily bars from Alpaca for all symbols.

    Returns: {symbol: [(date, open, high, low, close, volume), ...]} sorted by date.
    """
    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    client = StockHistoricalDataClient(api_key, secret_key)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=lookback_days + 10)  # buffer for weekends/holidays

    request = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=DataFeed.IEX,
    )

    bars_data = client.get_stock_bars(request)
    result = {}
    for symbol in symbols:
        symbol_bars = bars_data.get(symbol, []) if hasattr(bars_data, 'get') else []
        if not symbol_bars:
            # Try dict-style access on the barset
            try:
                symbol_bars = bars_data[symbol]
            except (KeyError, TypeError):
                continue
        rows = []
        for bar in symbol_bars:
            rows.append((
                bar.timestamp,
                float(bar.open),
                float(bar.high),
                float(bar.low),
                float(bar.close),
                int(bar.volume),
            ))
        rows.sort(key=lambda r: r[0])
        # Trim to lookback_days most recent bars
        result[symbol] = rows[-lookback_days:]

    return result


def find_swing_lows(closes: list[float], window: int) -> list[int]:
    """Find indices of swing lows (local minima with `window` bars on each side)."""
    lows = []
    for i in range(window, len(closes) - window):
        is_low = True
        for j in range(1, window + 1):
            if closes[i] >= closes[i - j] or closes[i] >= closes[i + j]:
                is_low = False
                break
        if is_low:
            lows.append(i)
    return lows


def detect_triple_bottom(bars: list[tuple], cfg: dict) -> dict | None:
    """Detect a triple bottom pattern in daily bars.

    Returns detection dict or None.
    """
    if len(bars) < 20:
        return None

    closes = [b[4] for b in bars]
    lows = [b[3] for b in bars]
    dates = [b[0] for b in bars]

    swing_window = cfg["swing_window"]
    tolerance_pct = cfg["support_tolerance_pct"] / 100.0
    min_bounce_pct = cfg["min_bounce_pct"] / 100.0
    min_touches = cfg["min_touches"]
    recency_days = cfg["recency_days"]
    proximity_pct = cfg["proximity_pct"] / 100.0

    # Find swing lows using the low prices
    swing_indices = find_swing_lows(lows, swing_window)
    if len(swing_indices) < min_touches:
        return None

    # Cluster swing lows within tolerance of each other
    swing_prices = [(i, lows[i]) for i in swing_indices]

    # Try each swing low as a potential support level
    best_cluster = None
    for _, base_price in swing_prices:
        cluster = []
        for idx, price in swing_prices:
            if abs(price - base_price) / base_price <= tolerance_pct:
                cluster.append((idx, price))
        if len(cluster) >= min_touches:
            if best_cluster is None or len(cluster) > len(best_cluster):
                best_cluster = cluster

    if best_cluster is None:
        return None

    support_level = sum(p for _, p in best_cluster) / len(best_cluster)

    # Check bounce magnitude from each touch
    bounces = []
    for idx, touch_price in best_cluster:
        # Find max price in the 5 bars after the touch (or to end)
        end_idx = min(idx + 6, len(closes))
        if idx + 1 >= len(closes):
            continue
        max_after = max(closes[idx + 1:end_idx])
        bounce_pct = (max_after - touch_price) / touch_price
        if bounce_pct >= min_bounce_pct:
            bounces.append((idx, touch_price, bounce_pct))

    if len(bounces) < min_touches:
        return None

    # Check recency: most recent touch within recency_days trading days
    most_recent_idx = bounces[-1][0]
    bars_from_end = len(bars) - 1 - most_recent_idx
    if bars_from_end > recency_days:
        return None

    # Check proximity: current price within proximity_pct above support
    current_price = closes[-1]
    pct_above = (current_price - support_level) / support_level
    if pct_above < 0 or pct_above > proximity_pct:
        return None

    return {
        "support": round(support_level, 2),
        "touches": len(bounces),
        "bounces": [round(b[2] * 100, 1) for b in bounces],
        "current_price": round(current_price, 2),
        "pct_above_support": round(pct_above * 100, 1),
        "most_recent_touch_date": dates[most_recent_idx].strftime("%Y-%m-%d")
            if hasattr(dates[most_recent_idx], "strftime")
            else str(dates[most_recent_idx]),
    }


def run_scan():
    """Run one scan cycle across the full universe."""
    cfg = load_config()
    if not cfg.get("enabled", True):
        logger.info("Scanner disabled in config — skipping")
        return

    universe = load_universe()
    if not universe:
        logger.warning("Empty universe — skipping scan")
        return

    logger.info("Scanning %d symbols for triple bottom patterns...", len(universe))

    # Fetch bars for all symbols
    try:
        all_bars = fetch_daily_bars(universe, cfg["lookback_days"])
    except Exception as e:
        logger.error("Failed to fetch daily bars: %s", e)
        return

    # Load dedup state
    dedup = load_dedup()
    now = datetime.now(timezone.utc)
    dedup_cutoff = now - timedelta(days=cfg["dedup_days"])

    # Clean expired dedup entries
    dedup = {
        sym: ts for sym, ts in dedup.items()
        if datetime.fromisoformat(ts) > dedup_cutoff
    }

    detections = 0
    for symbol in universe:
        if _shutdown:
            break

        bars = all_bars.get(symbol)
        if not bars:
            continue

        result = detect_triple_bottom(bars, cfg)
        if result is None:
            continue

        # Check dedup
        if symbol in dedup:
            logger.debug("Skipping %s — notified recently", symbol)
            continue

        detections += 1
        bounce_str = ", ".join(f"+{b}%" for b in result["bounces"])
        msg = (
            f"\U0001f4c8 *Triple Bottom Detected: {symbol}*\n"
            f"Support: ${result['support']:.2f} "
            f"(touched {result['touches']}x, last on {result['most_recent_touch_date']})\n"
            f"Bounces: {bounce_str}\n"
            f"Current: ${result['current_price']:.2f} "
            f"({result['pct_above_support']}% above support)"
        )
        logger.info("Triple bottom: %s @ $%.2f support", symbol, result["support"])
        _notify_slack(msg)

        dedup[symbol] = now.isoformat()

    save_dedup(dedup)
    logger.info("Scan complete: %d triple bottom patterns detected", detections)


def main():
    parser = argparse.ArgumentParser(description="Triple bottom pattern scanner")
    parser.add_argument("--once", action="store_true", help="Run a single scan and exit")
    args = parser.parse_args()

    logger.info("Triple bottom scanner starting")

    if args.once:
        run_scan()
        return

    # Continuous mode — scan once per day
    while not _shutdown:
        run_scan()
        # Sleep 24h in small increments
        for _ in range(86400 // 5):
            if _shutdown:
                break
            time.sleep(5)

    logger.info("Triple bottom scanner stopped")


if __name__ == "__main__":
    main()
