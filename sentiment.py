"""
Sentiment Scanner
==================
Scans financial news headlines via RSS and classifies sentiment
using a local Ollama model. Writes scores to the database for
use by the trading strategy.

Usage:
    python sentiment.py          # Run continuously (every 30 min)
    python sentiment.py --once   # Single scan
"""

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import db

logger = logging.getLogger("sentiment")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

STRATEGY_JSON = Path(__file__).parent / "strategy.json"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "deepseek-r1:14b")
SCAN_INTERVAL = int(os.environ.get("SENTIMENT_INTERVAL", "1800"))  # 30 min
MAX_HEADLINES_PER_SYMBOL = 10
HEADLINE_MAX_AGE_HOURS = 12

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


def load_universe() -> list[str]:
    """Load stock universe from strategy.json."""
    try:
        with open(STRATEGY_JSON) as f:
            config = json.load(f)
        return config.get("universe", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return ["SPY", "QQQ", "IWM"]


def fetch_headlines(symbol: str) -> list[dict]:
    """Fetch recent headlines for a symbol from Yahoo Finance RSS."""
    url = f"https://finance.yahoo.com/rss/headline?s={symbol}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BuzzAlgo/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read()
    except Exception as e:
        logger.debug("Failed to fetch RSS for %s: %s", symbol, e)
        return []

    headlines = []
    try:
        root = ET.fromstring(data)
        for item in root.iter("item"):
            title_el = item.find("title")
            pub_date_el = item.find("pubDate")
            if title_el is None or title_el.text is None:
                continue
            headline = title_el.text.strip()
            pub_date = pub_date_el.text.strip() if pub_date_el is not None and pub_date_el.text else ""
            headline_hash = hashlib.sha256(headline.encode()).hexdigest()[:16]
            headlines.append({
                "headline": headline,
                "pub_date": pub_date,
                "source": "yahoo_finance",
                "headline_hash": headline_hash,
            })
    except ET.ParseError as e:
        logger.debug("Failed to parse RSS XML for %s: %s", symbol, e)

    return headlines[:MAX_HEADLINES_PER_SYMBOL]


def classify_sentiment(headline: str, symbol: str) -> dict:
    """Classify a headline's sentiment using local Ollama model.

    Returns:
        {"sentiment": "bullish"|"neutral"|"bearish", "score": float, "confidence": float}
    """
    prompt = (
        f"Classify the sentiment of this stock headline for {symbol}.\n"
        f'Headline: "{headline}"\n'
        'Respond with ONLY a JSON object: '
        '{"sentiment": "bullish"|"neutral"|"bearish", '
        '"score": <float from -1.0 bearish to 1.0 bullish>, '
        '"confidence": <float 0.0 to 1.0>}'
    )

    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }).encode()

    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())

        response_text = result.get("response", "")

        # Strip <think>...</think> blocks from deepseek-r1
        response_text = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

        # Extract JSON object from response
        json_match = re.search(r"\{[^}]+\}", response_text)
        if json_match:
            parsed = json.loads(json_match.group())
            sentiment = parsed.get("sentiment", "neutral")
            if sentiment not in ("bullish", "neutral", "bearish"):
                sentiment = "neutral"
            score = max(-1.0, min(1.0, float(parsed.get("score", 0.0))))
            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
            return {"sentiment": sentiment, "score": score, "confidence": confidence}

    except Exception as e:
        logger.debug("Ollama classification failed for '%s': %s", headline[:50], e)

    return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}


def check_ollama_health() -> bool:
    """Check if Ollama is reachable."""
    try:
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/tags")
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception:
        return False


def run_scan():
    """Run one scan cycle: fetch headlines, classify, write to DB."""
    universe = load_universe()
    if not universe:
        logger.warning("Empty universe — skipping scan")
        return

    if not check_ollama_health():
        logger.warning("Ollama not reachable at %s — skipping scan", OLLAMA_HOST)
        return

    db.init_db()
    total_new = 0
    total_scanned = 0
    sentiment_totals = {"bullish": 0, "neutral": 0, "bearish": 0}

    for symbol in universe:
        if _shutdown:
            break

        headlines = fetch_headlines(symbol)
        if not headlines:
            continue

        # Filter out headlines already in DB
        with db.get_db() as conn:
            existing = set()
            for h in headlines:
                row = conn.execute(
                    "SELECT 1 FROM sentiment_scores WHERE headline_hash = ?",
                    (h["headline_hash"],),
                ).fetchone()
                if row:
                    existing.add(h["headline_hash"])

        new_headlines = [h for h in headlines if h["headline_hash"] not in existing]
        if not new_headlines:
            continue

        for h in new_headlines:
            if _shutdown:
                break

            result = classify_sentiment(h["headline"], symbol)
            total_scanned += 1
            sentiment_totals[result["sentiment"]] += 1

            record = {
                "symbol": symbol,
                "scanned_at": datetime.now(timezone.utc).isoformat(),
                "headline": h["headline"],
                "source": h["source"],
                "sentiment": result["sentiment"],
                "score": result["score"],
                "confidence": result["confidence"],
                "model": OLLAMA_MODEL,
                "headline_hash": h["headline_hash"],
            }

            with db.get_db() as conn:
                db.insert_sentiment(conn, record)
            total_new += 1

    logger.info(
        "Scan complete: %d new headlines classified for %d symbols "
        "(bullish=%d, neutral=%d, bearish=%d)",
        total_new, len(universe),
        sentiment_totals["bullish"], sentiment_totals["neutral"], sentiment_totals["bearish"],
    )

    if total_new > 0:
        # Compute overall market sentiment
        with db.get_db() as conn:
            agg = db.get_aggregate_sentiment(conn, hours=4)
        if agg:
            avg_score = sum(agg.values()) / len(agg)
            sentiment_label = "bullish" if avg_score > 0.2 else "bearish" if avg_score < -0.2 else "neutral"
            _notify_slack(
                f":newspaper: *Sentiment Scan Complete*\n"
                f"{total_new} new headlines | Overall: {sentiment_label} ({avg_score:+.2f})\n"
                f"Bullish: {sentiment_totals['bullish']} | "
                f"Neutral: {sentiment_totals['neutral']} | "
                f"Bearish: {sentiment_totals['bearish']}"
            )


def main():
    parser = argparse.ArgumentParser(description="Sentiment scanner")
    parser.add_argument("--once", action="store_true", help="Run a single scan and exit")
    args = parser.parse_args()

    logger.info("Sentiment scanner starting (model=%s, interval=%ds)", OLLAMA_MODEL, SCAN_INTERVAL)

    if args.once:
        run_scan()
        return

    while not _shutdown:
        run_scan()
        # Sleep in small increments to respond to shutdown quickly
        for _ in range(SCAN_INTERVAL // 5):
            if _shutdown:
                break
            time.sleep(5)

    logger.info("Sentiment scanner stopped")


if __name__ == "__main__":
    main()
