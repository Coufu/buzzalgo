"""
Improvement Loop Orchestrator
===============================
Runs nightly at 5:00pm ET after market close.
Triggers Claude Code to analyze performance, modify strategy, and validate.

Usage:
    python improve.py          # Run one improvement cycle
    python improve.py --cron   # Start cron scheduler (runs at 5pm ET daily)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import schedule
import time

import db
from backtest import run_backtest

logger = logging.getLogger("improve")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("improve.log"),
    ],
)

ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).parent
STRATEGY_JSON = PROJECT_ROOT / "strategy.json"
IMPROVEMENT_LOG = PROJECT_ROOT / "improvement_log.md"
PERFORMANCE_REPORT = PROJECT_ROOT / "performance_report.md"


def generate_performance_report(days: int = 30) -> str:
    """Analyze recent trades and generate a performance report.

    Returns:
        Path to the generated performance_report.md
    """
    since = (datetime.now(ET) - timedelta(days=days)).isoformat()

    with db.get_db() as conn:
        trades = db.get_trades_since(conn, since)
        daily_perfs = conn.execute(
            "SELECT * FROM daily_performance WHERE date >= ? ORDER BY date",
            (since[:10],),
        ).fetchall()

    if not trades:
        logger.info("No trades in the last %d days - skipping report", days)
        return ""

    closed = [t for t in trades if t["status"] == "closed"]
    total = len(closed)

    if total == 0:
        logger.info("No closed trades - skipping report")
        return ""

    wins = [t for t in closed if (t.get("pnl") or 0) > 0]
    losses = [t for t in closed if (t.get("pnl") or 0) <= 0]
    win_rate = len(wins) / total if total else 0
    total_pnl = sum(t.get("pnl", 0) for t in closed)
    gross_profit = sum(t["pnl"] for t in wins) if wins else 0
    gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 1
    profit_factor = gross_profit / gross_loss if gross_loss else 0

    # Daily returns for Sharpe
    daily_returns = []
    for dp in daily_perfs:
        if dp["starting_equity"] and dp["starting_equity"] > 0 and dp["daily_pnl"] is not None:
            daily_returns.append(dp["daily_pnl"] / dp["starting_equity"])

    if daily_returns and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Max drawdown from daily equity
    equities = [dp["ending_equity"] for dp in daily_perfs if dp["ending_equity"]]
    max_dd = 0.0
    if equities:
        peak = equities[0]
        for eq in equities:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

    # Analyze by signal type
    signal_breakdown = {}
    for t in closed:
        sig = t.get("signal_type", "unknown")
        if sig not in signal_breakdown:
            signal_breakdown[sig] = {"count": 0, "wins": 0, "pnl": 0}
        signal_breakdown[sig]["count"] += 1
        signal_breakdown[sig]["pnl"] += t.get("pnl", 0)
        if (t.get("pnl") or 0) > 0:
            signal_breakdown[sig]["wins"] += 1

    # Analyze stop-loss vs take-profit exits
    exit_reasons = {}
    for t in closed:
        ctx = t.get("signal_context")
        if isinstance(ctx, str):
            try:
                ctx = json.loads(ctx)
            except json.JSONDecodeError:
                ctx = {}

    # Build report
    report = f"""# Performance Report
Generated: {datetime.now(ET).isoformat()}
Period: Last {days} days

## Summary Metrics
| Metric | Value |
|--------|-------|
| Sharpe Ratio | {sharpe:.4f} |
| Max Drawdown | {max_dd:.2%} |
| Win Rate | {win_rate:.2%} ({len(wins)}/{total}) |
| Profit Factor | {profit_factor:.4f} |
| Total P&L | ${total_pnl:.2f} |
| Total Trades | {total} |

## Signal Breakdown
| Signal Type | Count | Win Rate | Total P&L |
|------------|-------|----------|-----------|
"""
    for sig, data in signal_breakdown.items():
        wr = data["wins"] / data["count"] if data["count"] else 0
        report += f"| {sig} | {data['count']} | {wr:.1%} | ${data['pnl']:.2f} |\n"

    report += f"""
## Losing Trades Analysis
"""
    for t in sorted(losses, key=lambda x: x.get("pnl", 0))[:10]:
        report += f"- {t['symbol']} {t['side']}: ${t.get('pnl', 0):.2f} | RSI={t.get('rsi_at_entry', 'N/A')} | Signal={t.get('signal_type', 'N/A')}\n"

    report += f"""
## Daily Performance
| Date | P&L | Return | Trades | W/L |
|------|-----|--------|--------|-----|
"""
    for dp in daily_perfs[-20:]:
        report += f"| {dp['date']} | ${dp.get('daily_pnl', 0):.2f} | {dp.get('daily_return_pct', 0):.2f}% | {dp.get('trades_taken', 0)} | {dp.get('wins', 0)}/{dp.get('losses', 0)} |\n"

    report += f"""
## Current Strategy
Version: {json.load(open(STRATEGY_JSON)).get('version', 'unknown')}
"""

    with open(PERFORMANCE_REPORT, "w") as f:
        f.write(report)

    logger.info("Performance report generated: Sharpe=%.4f, Win Rate=%.1f%%, P&L=$%.2f",
                sharpe, win_rate * 100, total_pnl)
    return str(PERFORMANCE_REPORT)


def run_claude_code_improvement():
    """Trigger Claude Code to analyze and improve the strategy.

    This invokes Claude Code CLI with the improvement prompt,
    which reads the performance report and modifies strategy.py.
    """
    report_path = generate_performance_report()
    if not report_path:
        logger.info("No report generated - skipping improvement cycle")
        return

    prompt = f"""You are the strategy improvement engine for AlgoTrader Pro.

Read the following files to understand the current state:
1. CLAUDE.md - your operating context and constraints
2. performance_report.md - latest performance analysis
3. strategy.py - the current strategy implementation
4. improvement_log.md - history of past improvements
5. strategy.json - current parameters

Your task:
1. ANALYZE: Identify the biggest weakness in recent performance
2. HYPOTHESIZE: Form 1-3 specific, testable hypotheses for improvement
3. IMPLEMENT: Make minimal, surgical changes to strategy.py
4. VALIDATE: Run `python backtest.py` and check the result
5. DEPLOY OR REVERT:
   - If backtest Sharpe improves by >= 0.05 AND max drawdown doesn't increase by > 2%: commit changes
   - Otherwise: revert changes and log the failed hypothesis

CONSTRAINTS:
- You may ONLY modify strategy.py and strategy.json
- You may NOT modify risk.py, trader.py, or any other files
- Changes must be minimal and well-commented
- Log your hypothesis and outcome to improvement_log.md

After running the backtest, output the result clearly."""

    logger.info("Triggering Claude Code improvement cycle...")

    try:
        result = subprocess.run(
            ["claude", "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(PROJECT_ROOT),
        )
        logger.info("Claude Code output:\n%s", result.stdout[-2000:] if result.stdout else "No output")
        if result.returncode != 0:
            logger.error("Claude Code error:\n%s", result.stderr[-1000:] if result.stderr else "Unknown error")
    except subprocess.TimeoutExpired:
        logger.error("Claude Code improvement cycle timed out after 10 minutes")
    except FileNotFoundError:
        logger.error("Claude Code CLI not found - make sure 'claude' is in PATH")


def log_improvement(hypothesis: str, changes: str, backtest_sharpe: float,
                    previous_sharpe: float, deployed: bool, revert_reason: str = ""):
    """Append an entry to improvement_log.md."""
    entry = f"""
## {datetime.now(ET).strftime('%Y-%m-%d %H:%M')} ET

**Hypothesis:** {hypothesis}

**Changes:** {changes}

**Backtest Sharpe:** {backtest_sharpe:.4f} (previous: {previous_sharpe:.4f})

**Result:** {'DEPLOYED' if deployed else 'REVERTED'}
"""
    if revert_reason:
        entry += f"\n**Revert Reason:** {revert_reason}\n"
    entry += "---\n"

    with open(IMPROVEMENT_LOG, "a") as f:
        f.write(entry)

    # Also log to database
    with db.get_db() as conn:
        conn.execute(
            """INSERT INTO improvement_log
               (timestamp, hypothesis, changes_made, backtest_sharpe, previous_sharpe, deployed, revert_reason)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(ET).isoformat(), hypothesis, changes,
             backtest_sharpe, previous_sharpe, 1 if deployed else 0, revert_reason),
        )


def run_improvement_cycle():
    """Run one complete improvement cycle."""
    logger.info("=" * 60)
    logger.info("Starting improvement cycle at %s", datetime.now(ET).isoformat())
    logger.info("=" * 60)

    run_claude_code_improvement()

    logger.info("Improvement cycle complete")


def run_cron():
    """Run the improvement loop on a schedule (5pm ET daily)."""
    logger.info("Starting improvement cron scheduler - runs at 5:00pm ET")

    schedule.every().monday.at("17:00").do(run_improvement_cycle)
    schedule.every().tuesday.at("17:00").do(run_improvement_cycle)
    schedule.every().wednesday.at("17:00").do(run_improvement_cycle)
    schedule.every().thursday.at("17:00").do(run_improvement_cycle)
    schedule.every().friday.at("17:00").do(run_improvement_cycle)

    while True:
        schedule.run_pending()
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description="Strategy improvement loop")
    parser.add_argument("--cron", action="store_true", help="Run as cron scheduler")
    parser.add_argument("--report-only", action="store_true", help="Only generate performance report")
    args = parser.parse_args()

    if args.report_only:
        generate_performance_report()
    elif args.cron:
        run_cron()
    else:
        run_improvement_cycle()


if __name__ == "__main__":
    main()
