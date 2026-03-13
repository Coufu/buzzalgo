"""
Parameter Sweep
================
Brute-force grid search over strategy parameters using the backtest harness.
Runs overnight to find optimal parameter combinations.

Usage:
    python sweep.py                # Run with defaults
    python sweep.py --days 90      # Custom backtest window
    python sweep.py --top 10       # Show top 10 results (default 5)
"""

import argparse
import itertools
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from backtest import fetch_historical_data, simulate
from strategy import Strategy

logger = logging.getLogger("sweep")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

ET = ZoneInfo("America/New_York")
STRATEGY_JSON = Path(__file__).parent / "strategy.json"
SWEEP_RESULTS = Path(__file__).parent / "sweep_results.json"

# Parameter grid — each key maps to a list of values to try
PARAM_GRID = {
    "rsi_period": [10, 14, 20],
    "rsi_oversold": [30, 35, 40],
    "rsi_overbought": [60, 65, 70],
    "ema_period": [10, 20, 50],
    "volume_multiplier": [0.3, 0.5, 1.0],
}


def run_sweep(days: int = 60, top_n: int = 5):
    """Run parameter grid search."""
    with open(STRATEGY_JSON) as f:
        config = json.load(f)

    base_params = config.get("parameters", {})
    universe = config.get("universe", ["SPY", "QQQ", "IWM"])

    logger.info("Fetching %d days of historical data...", days)
    bars = fetch_historical_data(universe, days)
    logger.info("Fetched data for %d symbols", len(bars))

    # Generate all parameter combinations
    keys = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(itertools.product(*values))
    total = len(combos)
    logger.info("Testing %d parameter combinations...", total)

    results = []
    for i, combo in enumerate(combos):
        params = {**base_params, "universe": universe}
        for k, v in zip(keys, combo):
            params[k] = v

        strategy = Strategy(params)
        result = simulate(strategy, bars)

        combo_dict = dict(zip(keys, combo))
        results.append({
            "params": combo_dict,
            "sharpe": round(result.sharpe_ratio, 4),
            "win_rate": round(result.win_rate, 4),
            "max_drawdown": round(result.max_drawdown, 4),
            "profit_factor": round(result.profit_factor, 4),
            "total_trades": result.total_trades,
            "total_pnl": round(result.total_pnl, 2),
        })

        if (i + 1) % 50 == 0 or (i + 1) == total:
            logger.info("Progress: %d/%d (%.0f%%)", i + 1, total, (i + 1) / total * 100)

    # Sort by Sharpe ratio
    results.sort(key=lambda r: r["sharpe"], reverse=True)

    # Save all results
    output = {
        "sweep_date": datetime.now(ET).isoformat(),
        "days": days,
        "total_combinations": total,
        "results": results,
    }
    with open(SWEEP_RESULTS, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Results saved to %s", SWEEP_RESULTS)

    # Print top results
    print(f"\nTop {top_n} parameter combinations by Sharpe ratio:\n")
    print(f"{'Rank':<5} {'Sharpe':<8} {'WinRate':<8} {'MaxDD':<8} {'PF':<8} {'Trades':<7} {'P&L':<10} {'Params'}")
    print("-" * 100)
    for i, r in enumerate(results[:top_n]):
        print(
            f"{i+1:<5} {r['sharpe']:<8.4f} {r['win_rate']:<8.2%} {r['max_drawdown']:<8.2%} "
            f"{r['profit_factor']:<8.4f} {r['total_trades']:<7} ${r['total_pnl']:<9.2f} {r['params']}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Parameter sweep")
    parser.add_argument("--days", type=int, default=60, help="Days of historical data")
    parser.add_argument("--top", type=int, default=5, help="Number of top results to display")
    args = parser.parse_args()

    results = run_sweep(args.days, args.top)
    if results and results[0]["sharpe"] > 0:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
