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
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import schedule
import time
from dotenv import load_dotenv

load_dotenv()

import db
from backtest import run_backtest

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")


def _notify_slack(text: str):
    """Send a Slack notification via webhook."""
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


def _build_improvement_slack_message(output: str) -> None:
    """Parse Claude Code output and send a detailed Slack summary."""
    if "SUMMARY_START" in output and "SUMMARY_END" in output:
        block = output.split("SUMMARY_START")[1].split("SUMMARY_END")[0].strip()
        lines = {}
        for line in block.splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                lines[key.strip().lower()] = val.strip()

        deployed = "deployed" in lines.get("outcome", "").lower()
        emoji = ":rocket:" if deployed else ":rewind:"
        msg = (
            f"{emoji} *Improvement Cycle Complete*\n"
            f"*Analysis:* {lines.get('analysis', 'N/A')}\n"
            f"*Hypothesis:* {lines.get('hypothesis', 'N/A')}\n"
            f"*Changes:* {lines.get('changes', 'N/A')}\n"
            f"*Backtest:* {lines.get('backtest', 'N/A')}\n"
            f"*Outcome:* {lines.get('outcome', 'N/A')}"
        )
        _notify_slack(msg)
    else:
        summary = output[-500:]
        _notify_slack(f":white_check_mark: *Improvement cycle complete*\n```{summary}```")


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


def _parse_signal_context(trade: dict) -> dict:
    """Parse signal_context JSON from a trade record."""
    ctx = trade.get("signal_context")
    if isinstance(ctx, str):
        try:
            return json.loads(ctx)
        except (json.JSONDecodeError, TypeError):
            return {}
    return ctx if isinstance(ctx, dict) else {}


def generate_performance_report(days: int = 30) -> str:
    """Analyze recent trades and generate a performance report.

    Returns:
        Path to the generated performance_report.md
    """
    since = (datetime.now(ET) - timedelta(days=days)).isoformat()

    with db.get_db() as conn:
        trades = db.get_trades_since(conn, since)
        daily_perfs = [
            dict(r) for r in conn.execute(
                "SELECT * FROM daily_performance WHERE date >= ? ORDER BY date",
                (since[:10],),
            ).fetchall()
        ]

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

    # Regime breakdown
    regime_breakdown = {}
    for t in closed:
        ctx = _parse_signal_context(t)
        regime = ctx.get("market_regime", "unknown")
        if regime not in regime_breakdown:
            regime_breakdown[regime] = {"count": 0, "wins": 0, "pnl": 0}
        regime_breakdown[regime]["count"] += 1
        regime_breakdown[regime]["pnl"] += t.get("pnl", 0)
        if (t.get("pnl") or 0) > 0:
            regime_breakdown[regime]["wins"] += 1

    # Time-of-day breakdown
    time_breakdown = {}
    for t in closed:
        ctx = _parse_signal_context(t)
        bucket = ctx.get("time_bucket", "unknown")
        if bucket not in time_breakdown:
            time_breakdown[bucket] = {"count": 0, "wins": 0, "pnl": 0}
        time_breakdown[bucket]["count"] += 1
        time_breakdown[bucket]["pnl"] += t.get("pnl", 0)
        if (t.get("pnl") or 0) > 0:
            time_breakdown[bucket]["wins"] += 1

    # Holding period analysis
    holding_periods = []
    for t in closed:
        opened = t.get("opened_at")
        closed_at = t.get("closed_at")
        if opened and closed_at:
            try:
                dt_open = datetime.fromisoformat(opened)
                dt_close = datetime.fromisoformat(closed_at)
                minutes = (dt_close - dt_open).total_seconds() / 60
                holding_periods.append({
                    "minutes": minutes,
                    "pnl": t.get("pnl", 0),
                    "won": (t.get("pnl") or 0) > 0,
                })
            except (ValueError, TypeError):
                pass

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

    report += """
## Market Regime Breakdown
| Regime | Count | Win Rate | Total P&L |
|--------|-------|----------|-----------|
"""
    for regime, data in regime_breakdown.items():
        wr = data["wins"] / data["count"] if data["count"] else 0
        report += f"| {regime} | {data['count']} | {wr:.1%} | ${data['pnl']:.2f} |\n"

    report += """
## Time-of-Day Breakdown
| Time Bucket | Count | Win Rate | Total P&L |
|-------------|-------|----------|-----------|
"""
    for bucket, data in sorted(time_breakdown.items()):
        wr = data["wins"] / data["count"] if data["count"] else 0
        report += f"| {bucket} | {data['count']} | {wr:.1%} | ${data['pnl']:.2f} |\n"

    if holding_periods:
        short_holds = [h for h in holding_periods if h["minutes"] < 30]
        medium_holds = [h for h in holding_periods if 30 <= h["minutes"] < 120]
        long_holds = [h for h in holding_periods if h["minutes"] >= 120]

        report += """
## Holding Period Analysis
| Duration | Count | Win Rate | Avg P&L |
|----------|-------|----------|---------|
"""
        for label, holds in [("< 30 min", short_holds), ("30-120 min", medium_holds), ("> 120 min", long_holds)]:
            if holds:
                wr = sum(1 for h in holds if h["won"]) / len(holds)
                avg_pnl = sum(h["pnl"] for h in holds) / len(holds)
                report += f"| {label} | {len(holds)} | {wr:.1%} | ${avg_pnl:.2f} |\n"

    report += f"""
## Losing Trades Analysis
"""
    for t in sorted(losses, key=lambda x: x.get("pnl", 0))[:10]:
        ctx = _parse_signal_context(t)
        report += (
            f"- {t['symbol']} {t['side']}: ${t.get('pnl', 0):.2f} | "
            f"RSI={t.get('rsi_at_entry', 'N/A')} | Signal={t.get('signal_type', 'N/A')} | "
            f"Regime={ctx.get('market_regime', 'N/A')} | Time={ctx.get('time_bucket', 'N/A')}\n"
        )

    report += f"""
## Daily Performance
| Date | P&L | Return | Trades | W/L |
|------|-----|--------|--------|-----|
"""
    for dp in daily_perfs[-20:]:
        report += f"| {dp['date']} | ${dp.get('daily_pnl', 0):.2f} | {dp.get('daily_return_pct', 0):.2f}% | {dp.get('trades_taken', 0)} | {dp.get('wins', 0)}/{dp.get('losses', 0)} |\n"

    try:
        with open(STRATEGY_JSON) as _f:
            _strategy_version = json.load(_f).get("version", "unknown")
    except (FileNotFoundError, json.JSONDecodeError):
        _strategy_version = "unknown"

    report += f"""
## Current Strategy
Version: {_strategy_version}
"""

    with open(PERFORMANCE_REPORT, "w") as f:
        f.write(report)

    logger.info("Performance report generated: Sharpe=%.4f, Win Rate=%.1f%%, P&L=$%.2f",
                sharpe, win_rate * 100, total_pnl)
    _notify_slack(
        f":bar_chart: *Performance Report*\n"
        f"Sharpe: {sharpe:.4f} | Win Rate: {win_rate:.1%} | P&L: ${total_pnl:.2f}\n"
        f"Max DD: {max_dd:.2%} | Trades: {total} | PF: {profit_factor:.2f}"
    )
    return str(PERFORMANCE_REPORT)


def check_kill_criteria() -> tuple[bool, str]:
    """Check if the strategy has hit kill criteria and needs a full pivot.

    Returns:
        (kill_switch_active, reason)
    """
    # Load kill criteria from strategy.json
    try:
        with open(STRATEGY_JSON) as f:
            config = json.load(f)
        criteria = config.get("kill_criteria", {})
    except (FileNotFoundError, json.JSONDecodeError):
        criteria = {}

    min_trades = criteria.get("min_trades", 50)
    min_sharpe = criteria.get("min_sharpe", 0.3)
    min_win_rate = criteria.get("min_win_rate", 0.35)

    with db.get_db() as conn:
        closed_trades = conn.execute(
            "SELECT pnl FROM trades WHERE status='closed'"
        ).fetchall()
        daily_perfs = [
            dict(r) for r in conn.execute(
                "SELECT * FROM daily_performance ORDER BY date DESC LIMIT 30"
            ).fetchall()
        ]

    total_closed = len(closed_trades)
    if total_closed < min_trades:
        return False, f"Only {total_closed}/{min_trades} trades — too early to evaluate"

    wins = sum(1 for t in closed_trades if (t["pnl"] or 0) > 0)
    win_rate = wins / total_closed

    daily_returns = [
        dp["daily_pnl"] / dp["starting_equity"]
        for dp in daily_perfs
        if dp["starting_equity"] and dp["starting_equity"] > 0 and dp["daily_pnl"] is not None
    ]
    if daily_returns and np.std(daily_returns) > 0:
        recent_sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        recent_sharpe = 0.0

    reasons = []
    if recent_sharpe < min_sharpe:
        reasons.append(f"Sharpe={recent_sharpe:.2f} < {min_sharpe}")
    if win_rate < min_win_rate:
        reasons.append(f"WinRate={win_rate:.1%} < {min_win_rate:.0%}")

    if reasons:
        reason = f"KILL CRITERIA: {', '.join(reasons)} after {total_closed} trades"
        logger.warning(reason)
        return True, reason

    return False, ""


def run_claude_code_improvement():
    """Trigger Claude Code to analyze and improve the strategy.

    This invokes Claude Code CLI with the improvement prompt,
    which reads the performance report and modifies strategy.py.
    """
    report_path = generate_performance_report()
    if not report_path:
        logger.info("No report generated - skipping improvement cycle")
        return

    # Load sweep results if available
    sweep_section = ""
    sweep_path = PROJECT_ROOT / "sweep_results.json"
    if sweep_path.exists():
        try:
            with open(sweep_path) as f:
                sweep_data = json.load(f)
            top_results = sweep_data.get("results", [])[:10]
            if top_results:
                sweep_section = "\n## Parameter Sweep Results (Top 10)\n"
                sweep_section += "Do NOT blindly adopt these — they may be overfit. Use as a guide.\n```json\n"
                sweep_section += json.dumps(top_results, indent=2)
                sweep_section += "\n```\n"
        except (json.JSONDecodeError, KeyError):
            pass

    kill_active, kill_reason = check_kill_criteria()

    kill_directive = ""
    if kill_active:
        kill_directive = f"""
## CRITICAL: KILL CRITERIA TRIGGERED
{kill_reason}

Parameter tweaks are NOT sufficient. You must try a FUNDAMENTALLY different approach:
- Different signal type (e.g., breakout, momentum, volatility contraction)
- Different indicator combination entirely
- Different universe focus (sector rotation, fewer symbols)
- Consider time-of-day filtering based on the performance data
- Bump the version to X.0.0 to reflect a major strategy pivot
Do NOT just adjust RSI thresholds or EMA periods — that has been tried.
"""

    prompt = f"""You are the strategy improvement engine for AlgoTrader Pro.

Read the following files to understand the current state:
1. CLAUDE.md - your operating context and constraints
2. performance_report.md - latest performance analysis (INCLUDES regime and time-of-day breakdowns)
3. strategy.py - the current strategy implementation
4. improvement_log.md - history of past improvements (AVOID repeating failed hypotheses)
5. strategy.json - current parameters and kill criteria
6. sweep_results.json - parameter sweep results (if available)
{kill_directive}{sweep_section}
## Analysis Framework

Before making changes, perform this structured analysis:

### 1. Regime Analysis
- Which market regimes (trending_up, trending_down, ranging, transitional) are profitable?
- Which are losing money? Consider filtering out unprofitable regimes.

### 2. Time-of-Day Analysis
- Which time buckets (open_30, morning, midday, afternoon, close_30) perform best?
- Should we avoid trading during certain periods?

### 3. Holding Period Analysis
- Are short holds (<30min) or longer holds (>120min) more profitable?
- Does this suggest our stops are too tight or our take-profits too aggressive?

### 4. Signal Quality
- What is the win rate and avg P&L per signal type?
- Are we generating too many low-quality signals?

### 5. Stop-Loss Analysis
- What % of trades hit stop-loss vs take-profit vs trailing-stop?
- If most trades hit stop-loss, the entry timing or stop distance may be wrong.

## Your Task
1. ANALYZE: Use the framework above to identify the #1 actionable weakness
2. HYPOTHESIZE: Form 1-2 specific, testable hypotheses
3. IMPLEMENT: Make minimal, surgical changes to strategy.py and/or strategy.json
4. VALIDATE: Run `python backtest.py` and check the result
5. DEPLOY OR REVERT:
   - If backtest Sharpe improves by >= 0.05 AND max drawdown doesn't increase by > 2%: commit changes
   - Otherwise: revert changes and log the failed hypothesis

## Constraints
- You may ONLY modify strategy.py and strategy.json
- You may NOT modify risk.py, trader.py, or any other files
- Changes must be minimal and well-commented
- Log your hypothesis and outcome to improvement_log.md
- After running the backtest, output the result clearly
- At the very end, output a summary block in this exact format:
  SUMMARY_START
  Analysis: <1-2 sentence key finding>
  Hypothesis: <what you tested>
  Changes: <what you changed>
  Backtest: <Sharpe, win rate, max DD numbers>
  Outcome: <DEPLOYED or REVERTED + reason>
  SUMMARY_END"""

    logger.info("Triggering Claude Code improvement cycle...")
    if kill_active:
        logger.warning("Kill switch active: %s", kill_reason)
        _notify_slack(f":rotating_light: *KILL CRITERIA TRIGGERED*\n{kill_reason}\nClaude Code will attempt a major strategy pivot.")

    _notify_slack(":brain: *Improvement cycle started* — Claude Code is analyzing strategy performance...")

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
            _notify_slack(f":x: *Improvement cycle failed* — Claude Code returned error (exit code {result.returncode})")
        else:
            output = (result.stdout or "No output").strip()
            # Try to extract structured summary
            slack_msg = _build_improvement_slack_message(output)
    except subprocess.TimeoutExpired:
        logger.error("Claude Code improvement cycle timed out after 10 minutes")
        _notify_slack(":warning: *Improvement cycle timed out* after 10 minutes")
    except FileNotFoundError:
        logger.error("Claude Code CLI not found - make sure 'claude' is in PATH")
        _notify_slack(":x: *Improvement cycle failed* — Claude Code CLI not found in PATH")


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


def send_daily_journal():
    """Send an end-of-day trading journal to Slack."""
    today = datetime.now(ET).strftime("%Y-%m-%d")

    with db.get_db() as conn:
        # Today's closed trades
        closed = conn.execute(
            "SELECT * FROM trades WHERE date(closed_at) = ? AND status = 'closed' ORDER BY pnl DESC",
            (today,),
        ).fetchall()
        closed = [dict(r) for r in closed]

        # Today's open trades
        open_trades = conn.execute(
            "SELECT * FROM trades WHERE status = 'open'",
        ).fetchall()
        open_trades = [dict(r) for r in open_trades]

        # Daily performance
        dp = conn.execute(
            "SELECT * FROM daily_performance WHERE date = ?",
            (today,),
        ).fetchone()

        # Sentiment
        sentiment_rows = conn.execute(
            "SELECT symbol, AVG(score) as avg FROM sentiment_scores "
            "WHERE scanned_at >= datetime('now', '-24 hours') GROUP BY symbol",
        ).fetchall()

    if not closed and not open_trades and not dp:
        logger.info("No trading activity today — skipping journal")
        return

    # Build journal
    total_pnl = sum(t.get("pnl", 0) for t in closed)
    wins = [t for t in closed if (t.get("pnl") or 0) > 0]
    losses = [t for t in closed if (t.get("pnl") or 0) <= 0]

    equity = dp["ending_equity"] if dp and dp["ending_equity"] else "N/A"
    equity_str = f"${equity:,.2f}" if isinstance(equity, (int, float)) else equity

    pnl_emoji = ":chart_with_upwards_trend:" if total_pnl >= 0 else ":chart_with_downwards_trend:"
    date_str = datetime.now(ET).strftime("%b %d")

    msg = f"{pnl_emoji} *Daily Journal — {date_str}*\n"
    msg += f"P&L: ${total_pnl:+.2f} | Trades: {len(closed)} ({len(wins)}W/{len(losses)}L)\n"

    if closed:
        best = max(closed, key=lambda t: t.get("pnl", 0))
        worst = min(closed, key=lambda t: t.get("pnl", 0))
        msg += f"Best: {best['symbol']} {best['side']} ${best.get('pnl', 0):+.2f}\n"
        exit_reason = worst.get("signal_type", "")
        msg += f"Worst: {worst['symbol']} {worst['side']} ${worst.get('pnl', 0):+.2f}\n"

    msg += f"Equity: {equity_str}\n"

    if open_trades:
        open_syms = ", ".join(t["symbol"] for t in open_trades)
        msg += f"Open positions: {open_syms}\n"

    if sentiment_rows:
        avg_sentiment = sum(r["avg"] for r in sentiment_rows) / len(sentiment_rows)
        label = "bullish" if avg_sentiment > 0.2 else "bearish" if avg_sentiment < -0.2 else "neutral"
        msg += f"Sentiment: {label} ({avg_sentiment:+.2f})"

    _notify_slack(msg)
    logger.info("Daily journal sent to Slack")


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
    parser.add_argument("--journal", action="store_true", help="Send daily journal to Slack")
    args = parser.parse_args()

    if args.journal:
        send_daily_journal()
    elif args.report_only:
        generate_performance_report()
    elif args.cron:
        run_cron()
    else:
        run_improvement_cycle()


if __name__ == "__main__":
    main()
