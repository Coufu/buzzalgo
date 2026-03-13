"""
Replay Mode
=============
Backtest a specific day's data against current and proposed strategy params.
Answers: "Would this change have helped today?"

Usage:
    python replay.py                          # Replay today
    python replay.py --date 2026-03-12        # Replay specific date
    python replay.py --compare params.json    # Compare current vs proposed params
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from backtest import fetch_historical_data, simulate
from strategy import Strategy

logger = logging.getLogger("replay")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

ET = ZoneInfo("America/New_York")
STRATEGY_JSON = Path(__file__).parent / "strategy.json"
REPLAY_REPORT = Path(__file__).parent / "replay_report.json"


def run_replay(target_date: str | None = None, compare_file: str | None = None):
    """Replay a specific day and optionally compare against proposed params."""
    with open(STRATEGY_JSON) as f:
        config = json.load(f)

    base_params = {**config.get("parameters", {}), "universe": config.get("universe", [])}
    universe = config.get("universe", ["SPY", "QQQ", "IWM"])

    if target_date is None:
        target_date = datetime.now(ET).strftime("%Y-%m-%d")

    # Fetch enough data for indicator warm-up
    logger.info("Fetching data for replay (target: %s)...", target_date)
    bars = fetch_historical_data(universe, days=10)
    logger.info("Fetched data for %d symbols", len(bars))

    if not bars:
        logger.error("No data available")
        return None

    # Run current strategy
    current_strategy = Strategy(base_params)
    current_result = simulate(current_strategy, bars)

    # Filter trades to target date
    current_day_trades = [t for t in current_result.trades if t.entry_time.startswith(target_date)]
    current_day_pnl = sum(t.pnl for t in current_day_trades)

    report = {
        "date": target_date,
        "current": {
            "total_trades": len(current_day_trades),
            "day_pnl": round(current_day_pnl, 2),
            "wins": len([t for t in current_day_trades if t.pnl > 0]),
            "losses": len([t for t in current_day_trades if t.pnl <= 0]),
            "sharpe": round(current_result.sharpe_ratio, 4),
            "trades": [
                {"symbol": t.symbol, "side": t.side, "pnl": round(t.pnl, 2), "exit_reason": t.exit_reason}
                for t in current_day_trades
            ],
        },
    }

    logger.info("Current strategy on %s: %d trades, P&L=$%.2f",
                target_date, len(current_day_trades), current_day_pnl)

    # Compare against proposed params if provided
    if compare_file:
        try:
            with open(compare_file) as f:
                proposed_params = json.load(f)
            # Merge: proposed overrides base
            merged = {**base_params, **proposed_params}
            proposed_strategy = Strategy(merged)
            proposed_result = simulate(proposed_strategy, bars)

            proposed_day_trades = [t for t in proposed_result.trades if t.entry_time.startswith(target_date)]
            proposed_day_pnl = sum(t.pnl for t in proposed_day_trades)

            report["proposed"] = {
                "params": proposed_params,
                "total_trades": len(proposed_day_trades),
                "day_pnl": round(proposed_day_pnl, 2),
                "wins": len([t for t in proposed_day_trades if t.pnl > 0]),
                "losses": len([t for t in proposed_day_trades if t.pnl <= 0]),
                "sharpe": round(proposed_result.sharpe_ratio, 4),
                "trades": [
                    {"symbol": t.symbol, "side": t.side, "pnl": round(t.pnl, 2), "exit_reason": t.exit_reason}
                    for t in proposed_day_trades
                ],
            }

            delta_pnl = proposed_day_pnl - current_day_pnl
            report["delta"] = {
                "pnl": round(delta_pnl, 2),
                "trades": len(proposed_day_trades) - len(current_day_trades),
                "verdict": "proposed is better" if delta_pnl > 0 else "current is better",
            }

            logger.info("Proposed strategy on %s: %d trades, P&L=$%.2f (delta: %+.2f)",
                        target_date, len(proposed_day_trades), proposed_day_pnl, delta_pnl)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error("Failed to load proposed params: %s", e)

    # Save report
    with open(REPLAY_REPORT, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Replay report saved to %s", REPLAY_REPORT)

    return report


def main():
    parser = argparse.ArgumentParser(description="Replay a trading day")
    parser.add_argument("--date", type=str, default=None, help="Date to replay (YYYY-MM-DD)")
    parser.add_argument("--compare", type=str, default=None, help="JSON file with proposed params")
    args = parser.parse_args()

    report = run_replay(args.date, args.compare)
    if report:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
