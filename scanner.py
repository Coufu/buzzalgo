"""
Triple Bottom Pattern Scanner
==============================
Scans the stock universe for triple bottom chart patterns using
daily bars from Alpaca. Sends Slack notifications for detected
setups. Notification-only — does not trigger trades.

Usage:
    python scanner.py          # Run continuously (daily at 4:45pm ET)
    python scanner.py --once   # Single scan
    python scanner.py --evaluate  # Score entry rules from past detections
"""

import argparse
import json
import logging
import os
import signal
import sqlite3
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
SCANNER_DB = DATA_DIR / "scanner.db"
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


def init_scanner_db():
    """Create the scanner detections table."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(SCANNER_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            detected_at TEXT NOT NULL,
            support_level REAL NOT NULL,
            entry_price REAL NOT NULL,
            pct_above_support REAL,
            num_touches INTEGER,
            avg_bounce_pct REAL,
            max_bounce_pct REAL,
            min_bounce_pct REAL,
            volume_ratio REAL,
            day_of_week INTEGER,
            bars_since_last_touch INTEGER,
            return_5d REAL,
            return_10d REAL,
            return_20d REAL,
            max_drawdown_20d REAL,
            broke_support INTEGER,
            outcome_filled INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_det_symbol_date
        ON detections (symbol, detected_at)
    """)
    conn.commit()
    conn.close()


def log_detection(detection: dict):
    """Log a pattern detection to the scanner database."""
    conn = sqlite3.connect(str(SCANNER_DB))
    conn.execute(
        """INSERT INTO detections
           (symbol, detected_at, support_level, entry_price, pct_above_support,
            num_touches, avg_bounce_pct, max_bounce_pct, min_bounce_pct,
            volume_ratio, day_of_week, bars_since_last_touch)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            detection["symbol"],
            detection["detected_at"],
            detection["support_level"],
            detection["entry_price"],
            detection["pct_above_support"],
            detection["num_touches"],
            detection["avg_bounce_pct"],
            detection["max_bounce_pct"],
            detection["min_bounce_pct"],
            detection.get("volume_ratio"),
            detection.get("day_of_week"),
            detection.get("bars_since_last_touch"),
        ),
    )
    conn.commit()
    conn.close()


def fill_outcomes():
    """Fetch forward returns for past detections and fill outcome columns."""
    conn = sqlite3.connect(str(SCANNER_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, symbol, detected_at, entry_price, support_level
           FROM detections WHERE outcome_filled = 0"""
    ).fetchall()

    if not rows:
        logger.info("No unfilled detections to evaluate")
        conn.close()
        return

    # Only fill outcomes for detections at least 20 trading days old (~30 calendar days)
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)
    eligible = [r for r in rows if datetime.fromisoformat(r["detected_at"]) < cutoff]

    if not eligible:
        logger.info("No detections old enough for outcome evaluation (need 30+ days)")
        conn.close()
        return

    symbols = list({r["symbol"] for r in eligible})
    logger.info("Filling outcomes for %d detections across %d symbols", len(eligible), len(symbols))

    # Fetch enough bars to cover outcomes
    all_bars = fetch_daily_bars(symbols, 120)

    for row in eligible:
        symbol = row["symbol"]
        bars = all_bars.get(symbol)
        if not bars:
            continue

        entry_price = row["entry_price"]
        support = row["support_level"]
        det_date = datetime.fromisoformat(row["detected_at"])

        # Find the bar index on or after detection date
        start_idx = None
        for i, bar in enumerate(bars):
            bar_date = bar[0] if hasattr(bar[0], 'date') else bar[0]
            if hasattr(bar_date, 'date'):
                bar_date_cmp = bar_date
            else:
                bar_date_cmp = datetime.fromisoformat(str(bar_date))
            if bar_date_cmp.date() >= det_date.date():
                start_idx = i
                break

        if start_idx is None:
            continue

        closes_after = [b[4] for b in bars[start_idx:]]
        lows_after = [b[3] for b in bars[start_idx:]]

        # Forward returns
        return_5d = None
        return_10d = None
        return_20d = None
        max_dd_20d = None
        broke = None

        if len(closes_after) > 5:
            return_5d = (closes_after[5] - entry_price) / entry_price * 100
        if len(closes_after) > 10:
            return_10d = (closes_after[10] - entry_price) / entry_price * 100
        if len(closes_after) > 20:
            return_20d = (closes_after[20] - entry_price) / entry_price * 100

        # Max drawdown in first 20 bars
        window = min(21, len(lows_after))
        if window > 1:
            worst_low = min(lows_after[1:window])
            max_dd_20d = (worst_low - entry_price) / entry_price * 100

        # Did price close below support?
        window_closes = closes_after[1:min(21, len(closes_after))]
        broke = 1 if any(c < support * 0.99 for c in window_closes) else 0

        conn.execute(
            """UPDATE detections SET
               return_5d=?, return_10d=?, return_20d=?,
               max_drawdown_20d=?, broke_support=?, outcome_filled=1
               WHERE id=?""",
            (return_5d, return_10d, return_20d, max_dd_20d, broke, row["id"]),
        )

    conn.commit()
    conn.close()
    logger.info("Outcome fill complete")


def evaluate_entry_rules():
    """Score entry rules based on historical detections with outcomes."""
    fill_outcomes()

    conn = sqlite3.connect(str(SCANNER_DB))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM detections WHERE outcome_filled = 1"
    ).fetchall()
    conn.close()

    if not rows:
        print("No detections with outcomes yet. Need 30+ days of data.")
        return

    detections = [dict(r) for r in rows]
    total = len(detections)
    print(f"\n{'='*70}")
    print(f"TRIPLE BOTTOM ENTRY RULE EVALUATION — {total} detections with outcomes")
    print(f"{'='*70}\n")

    # Define entry rules as filters
    rules = {
        "Baseline (all detections)": lambda d: True,
        "Tight proximity (<2% above)": lambda d: d["pct_above_support"] is not None and d["pct_above_support"] < 2.0,
        "Wide proximity (2-5% above)": lambda d: d["pct_above_support"] is not None and d["pct_above_support"] >= 2.0,
        "Strong bounces (avg >3%)": lambda d: d["avg_bounce_pct"] is not None and d["avg_bounce_pct"] > 3.0,
        "Weak bounces (avg <=3%)": lambda d: d["avg_bounce_pct"] is not None and d["avg_bounce_pct"] <= 3.0,
        "Fresh touch (<=2 bars ago)": lambda d: d["bars_since_last_touch"] is not None and d["bars_since_last_touch"] <= 2,
        "Older touch (3-5 bars ago)": lambda d: d["bars_since_last_touch"] is not None and d["bars_since_last_touch"] > 2,
        "4+ touches": lambda d: d["num_touches"] is not None and d["num_touches"] >= 4,
        "Exactly 3 touches": lambda d: d["num_touches"] is not None and d["num_touches"] == 3,
        "High volume (ratio >1.5)": lambda d: d["volume_ratio"] is not None and d["volume_ratio"] > 1.5,
        "Low volume (ratio <=1.5)": lambda d: d["volume_ratio"] is not None and d["volume_ratio"] <= 1.5,
        "Monday": lambda d: d["day_of_week"] == 0,
        "Tuesday": lambda d: d["day_of_week"] == 1,
        "Wednesday": lambda d: d["day_of_week"] == 2,
        "Thursday": lambda d: d["day_of_week"] == 3,
        "Friday": lambda d: d["day_of_week"] == 4,
    }

    header = f"{'Rule':<32} {'N':>4} {'Win%5d':>7} {'Avg5d':>7} {'Win%10d':>8} {'Avg10d':>7} {'Avg20d':>7} {'MaxDD':>7} {'Broke%':>7}"
    print(header)
    print("-" * len(header))

    for name, filt in rules.items():
        subset = [d for d in detections if filt(d)]
        n = len(subset)
        if n == 0:
            continue

        r5 = [d["return_5d"] for d in subset if d["return_5d"] is not None]
        r10 = [d["return_10d"] for d in subset if d["return_10d"] is not None]
        r20 = [d["return_20d"] for d in subset if d["return_20d"] is not None]
        dd = [d["max_drawdown_20d"] for d in subset if d["max_drawdown_20d"] is not None]
        broke = [d["broke_support"] for d in subset if d["broke_support"] is not None]

        win5 = f"{sum(1 for x in r5 if x > 0)/len(r5)*100:.0f}%" if r5 else "—"
        avg5 = f"{sum(r5)/len(r5):+.1f}%" if r5 else "—"
        win10 = f"{sum(1 for x in r10 if x > 0)/len(r10)*100:.0f}%" if r10 else "—"
        avg10 = f"{sum(r10)/len(r10):+.1f}%" if r10 else "—"
        avg20 = f"{sum(r20)/len(r20):+.1f}%" if r20 else "—"
        avg_dd = f"{sum(dd)/len(dd):+.1f}%" if dd else "—"
        broke_pct = f"{sum(broke)/len(broke)*100:.0f}%" if broke else "—"

        print(f"{name:<32} {n:>4} {win5:>7} {avg5:>7} {win10:>8} {avg10:>7} {avg20:>7} {avg_dd:>7} {broke_pct:>7}")

    print()

    # Slack summary of best rule
    if SLACK_WEBHOOK_URL:
        best_rule = None
        best_avg = -999
        for name, filt in rules.items():
            if name == "Baseline (all detections)":
                continue
            subset = [d for d in detections if filt(d)]
            r10 = [d["return_10d"] for d in subset if d["return_10d"] is not None]
            if len(r10) >= 3:
                avg = sum(r10) / len(r10)
                if avg > best_avg:
                    best_avg = avg
                    best_rule = name

        if best_rule:
            _notify_slack(
                f"\U0001f4ca *Triple Bottom Entry Rule Evaluation*\n"
                f"Evaluated {total} past detections\n"
                f"Best entry rule: *{best_rule}* (avg 10d return: {best_avg:+.1f}%)"
            )


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

    # Volume ratio: detection day volume vs 20-day average
    volumes = [b[5] for b in bars]
    vol_avg_20 = sum(volumes[-21:-1]) / 20 if len(volumes) > 20 else sum(volumes) / len(volumes)
    volume_ratio = volumes[-1] / vol_avg_20 if vol_avg_20 > 0 else 1.0

    bounce_pcts = [b[2] * 100 for b in bounces]
    bars_since = len(bars) - 1 - most_recent_idx

    return {
        "support": round(support_level, 2),
        "touches": len(bounces),
        "bounces": [round(p, 1) for p in bounce_pcts],
        "current_price": round(current_price, 2),
        "pct_above_support": round(pct_above * 100, 1),
        "most_recent_touch_date": dates[most_recent_idx].strftime("%Y-%m-%d")
            if hasattr(dates[most_recent_idx], "strftime")
            else str(dates[most_recent_idx]),
        "avg_bounce_pct": round(sum(bounce_pcts) / len(bounce_pcts), 1),
        "max_bounce_pct": round(max(bounce_pcts), 1),
        "min_bounce_pct": round(min(bounce_pcts), 1),
        "volume_ratio": round(volume_ratio, 2),
        "bars_since_last_touch": bars_since,
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

    init_scanner_db()
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

        # Log detection for outcome tracking
        det_date = now.strftime("%Y-%m-%d")
        log_detection({
            "symbol": symbol,
            "detected_at": now.isoformat(),
            "support_level": result["support"],
            "entry_price": result["current_price"],
            "pct_above_support": result["pct_above_support"],
            "num_touches": result["touches"],
            "avg_bounce_pct": result["avg_bounce_pct"],
            "max_bounce_pct": result["max_bounce_pct"],
            "min_bounce_pct": result["min_bounce_pct"],
            "volume_ratio": result["volume_ratio"],
            "day_of_week": now.weekday(),
            "bars_since_last_touch": result["bars_since_last_touch"],
        })

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
    parser.add_argument("--evaluate", action="store_true", help="Score entry rules from past detections")
    args = parser.parse_args()

    if args.evaluate:
        init_scanner_db()
        evaluate_entry_rules()
        return

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
