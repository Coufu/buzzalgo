"""
Health Check - Daily system status report to Slack.
=====================================================
Checks all services, reports anomalies, and posts a summary.

Usage:
    python healthcheck.py          # Run once and post to Slack
    python healthcheck.py --quiet  # Only post if there are problems
"""

import argparse
import json
import logging
import os
import subprocess
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("healthcheck")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

ET = ZoneInfo("America/New_York")
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "")
PROJECT_ROOT = Path(__file__).parent


def _notify_slack(text: str):
    if not SLACK_WEBHOOK_URL:
        return
    try:
        payload = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            SLACK_WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        logger.error("Slack notification failed: %s", e)


def check_docker_containers() -> list[dict]:
    """Check Docker container health."""
    results = []
    try:
        out = subprocess.run(
            ["docker", "compose", "ps", "--format", "json"],
            capture_output=True, text=True, timeout=10,
            cwd=str(PROJECT_ROOT),
        )
        if out.returncode != 0:
            results.append({"name": "docker", "status": "error", "detail": "Docker not running"})
            return results
        containers = json.loads(out.stdout) if out.stdout.strip() else []
        if not containers:
            results.append({"name": "docker", "status": "error", "detail": "no containers found"})
            return results
        for c in containers:
            name = c.get("Service", c.get("Name", "unknown"))
            state = c.get("State", c.get("Status", "unknown"))
            health = c.get("Health", "")
            status = "ok" if "running" in state.lower() or "up" in state.lower() else "down"
            if health and "unhealthy" in health.lower():
                status = "unhealthy"
            results.append({"name": name, "status": status, "detail": f"{state} {health}".strip()})
    except FileNotFoundError:
        results.append({"name": "docker", "status": "error", "detail": "Docker CLI not found"})
    except subprocess.TimeoutExpired:
        results.append({"name": "docker", "status": "error", "detail": "Docker command timed out"})
    return results


def check_launchd_services() -> list[dict]:
    """Check launchd service status."""
    results = []
    try:
        out = subprocess.run(
            ["launchctl", "list"],
            capture_output=True, text=True, timeout=5,
        )
        for line in out.stdout.split("\n"):
            if "buzzalgo" not in line:
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                pid = parts[0].strip()
                exit_code = parts[1].strip()
                name = parts[2].strip().replace("com.buzzalgo.", "")
                if pid != "-" and pid != "0":
                    status = "running"
                    detail = f"PID {pid}"
                elif exit_code == "0":
                    status = "ok"
                    detail = "exited cleanly"
                else:
                    status = "failed"
                    detail = f"exit code {exit_code}"
                results.append({"name": name, "status": status, "detail": detail})
    except Exception as e:
        results.append({"name": "launchd", "status": "error", "detail": str(e)})
    return results


def check_ollama() -> dict:
    """Check if Ollama is reachable."""
    try:
        req = urllib.request.Request("http://localhost:11434/", method="GET")
        resp = urllib.request.urlopen(req, timeout=5)
        return {"name": "ollama", "status": "ok", "detail": "running"}
    except Exception:
        return {"name": "ollama", "status": "down", "detail": "not reachable at localhost:11434"}


def check_recent_trades() -> dict:
    """Check trade activity."""
    try:
        import db
        with db.get_db() as conn:
            # Last trade
            row = conn.execute(
                "SELECT timestamp, symbol, side, status FROM trades ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not row:
                return {"name": "trades", "status": "warning", "detail": "no trades in database"}

            last_trade_time = row["timestamp"]
            total = conn.execute("SELECT count(*) FROM trades").fetchone()[0]
            open_count = conn.execute("SELECT count(*) FROM trades WHERE status='open'").fetchone()[0]
            closed = conn.execute("SELECT count(*) FROM trades WHERE status='closed'").fetchone()[0]

            # Trades in last 24h
            since = (datetime.now(ET) - timedelta(hours=24)).isoformat()
            recent = conn.execute(
                "SELECT count(*) FROM trades WHERE timestamp >= ?", (since,)
            ).fetchone()[0]

            detail = f"{total} total, {open_count} open, {closed} closed, {recent} in last 24h, last: {last_trade_time}"
            status = "ok" if recent > 0 else "warning"
            return {"name": "trades", "status": status, "detail": detail}
    except Exception as e:
        return {"name": "trades", "status": "error", "detail": str(e)}


def check_position_sync() -> dict:
    """Compare Alpaca positions against DB open trades."""
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(
            os.environ.get("ALPACA_API_KEY", ""),
            os.environ.get("ALPACA_SECRET_KEY", ""),
            paper=True,
        )
        alpaca_positions = {p.symbol: p for p in client.get_all_positions()}

        import db
        with db.get_db() as conn:
            db_open = db.get_open_trades(conn)
        db_symbols = {t["symbol"] for t in db_open}

        # Normalize crypto symbols: Alpaca uses "BTCUSD", DB uses "BTC/USD"
        alpaca_symbols = set()
        for sym in alpaca_positions:
            if sym.endswith("USD") and len(sym) > 3 and not sym.startswith(("S", "T", "U", "E", "P", "V", "W", "D", "G", "H", "I", "A", "B", "C", "F", "J", "K", "L", "M", "N", "O", "Q", "R", "X", "Y", "Z")):
                pass  # ambiguous, skip normalization
            alpaca_symbols.add(sym)

        orphan_alpaca = alpaca_symbols - db_symbols
        orphan_db = db_symbols - alpaca_symbols

        issues = []
        if orphan_alpaca:
            issues.append(f"{len(orphan_alpaca)} on Alpaca not in DB: {', '.join(sorted(orphan_alpaca)[:5])}")
        if orphan_db:
            issues.append(f"{len(orphan_db)} in DB not on Alpaca: {', '.join(sorted(orphan_db)[:5])}")

        if issues:
            return {"name": "position sync", "status": "warning", "detail": " | ".join(issues)}
        return {"name": "position sync", "status": "ok", "detail": f"{len(alpaca_positions)} positions, all tracked"}
    except Exception as e:
        return {"name": "position sync", "status": "error", "detail": str(e)}


def check_log_freshness() -> list[dict]:
    """Check that log files are being written to."""
    results = []
    log_files = {
        "improve": PROJECT_ROOT / "improve.log",
        "sweep": PROJECT_ROOT / "sweep.log",
        "sentiment": PROJECT_ROOT / "sentiment.log",
    }
    # trader/dashboard log to Docker stdout, not files — skip them
    now = datetime.now().timestamp()
    for name, path in log_files.items():
        if not path.exists():
            results.append({"name": f"log:{name}", "status": "warning", "detail": "log file missing"})
            continue
        age_hours = (now - path.stat().st_mtime) / 3600
        if age_hours > 24:
            results.append({"name": f"log:{name}", "status": "warning", "detail": f"stale ({age_hours:.0f}h old)"})
        else:
            results.append({"name": f"log:{name}", "status": "ok", "detail": f"updated {age_hours:.1f}h ago"})
    return results


def check_dashboard() -> dict:
    """Check if dashboard is responding."""
    try:
        req = urllib.request.Request("http://localhost:8080/api/status", method="GET")
        resp = urllib.request.urlopen(req, timeout=5)
        data = json.loads(resp.read().decode())
        equity = data.get("account", {}).get("equity", "?")
        return {"name": "dashboard", "status": "ok", "detail": f"equity: ${equity}"}
    except Exception:
        return {"name": "dashboard", "status": "down", "detail": "not responding on :8080"}


def run_health_check(quiet: bool = False):
    """Run all health checks and post summary to Slack."""
    now = datetime.now(ET)
    checks = []

    # Docker
    checks.extend(check_docker_containers())

    # Launchd
    checks.extend(check_launchd_services())

    # Ollama
    checks.append(check_ollama())

    # Dashboard
    checks.append(check_dashboard())

    # Trades
    checks.append(check_recent_trades())

    # Position sync
    checks.append(check_position_sync())

    # Log freshness
    checks.extend(check_log_freshness())

    # Build report
    problems = [c for c in checks if c["status"] not in ("ok", "running")]
    all_ok = len(problems) == 0

    if quiet and all_ok:
        logger.info("All checks passed, --quiet mode, skipping Slack")
        return

    status_icon = {
        "ok": ":white_check_mark:",
        "running": ":white_check_mark:",
        "warning": ":warning:",
        "down": ":red_circle:",
        "failed": ":red_circle:",
        "error": ":red_circle:",
        "unhealthy": ":red_circle:",
    }

    lines = [f":stethoscope: *Health Check* — {now.strftime('%Y-%m-%d %I:%M %p ET')}"]
    if all_ok:
        lines.append("All systems operational.")
    else:
        lines.append(f"*{len(problems)} issue(s) detected:*")

    lines.append("")
    for c in checks:
        icon = status_icon.get(c["status"], ":grey_question:")
        lines.append(f"{icon} *{c['name']}* — {c['detail']}")

    report = "\n".join(lines)
    logger.info("\n%s", report)
    _notify_slack(report)


def main():
    parser = argparse.ArgumentParser(description="Run system health check")
    parser.add_argument("--quiet", action="store_true", help="Only post to Slack if there are problems")
    args = parser.parse_args()
    run_health_check(quiet=args.quiet)


if __name__ == "__main__":
    main()
