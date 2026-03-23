"""
Dashboard - FastAPI server on port 8080
=========================================
Live P&L, open positions, strategy evolution, and alerts.

Usage:
    uvicorn dashboard:app --host 0.0.0.0 --port 8080
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

import db

logger = logging.getLogger("dashboard")
ET = ZoneInfo("America/New_York")
PROJECT_ROOT = Path(__file__).parent
STRATEGY_JSON = PROJECT_ROOT / "strategy.json"

app = FastAPI(title="AlgoTrader Pro Dashboard", version="1.0")


def _get_account_info() -> dict:
    """Get account info from Alpaca, with fallback if unavailable."""
    try:
        from alpaca.trading.client import TradingClient
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if api_key and secret_key:
            client = TradingClient(api_key, secret_key, paper=True)
            account = client.get_account()
            return {
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
            }
    except Exception as e:
        logger.debug("Could not fetch account info: %s", e)
    return {"equity": 0, "buying_power": 0, "cash": 0, "portfolio_value": 0}


def _market_status() -> str:
    now = datetime.now(ET)
    weekday = now.weekday()
    if weekday >= 5:
        return "closed"
    hour_min = now.hour * 60 + now.minute
    if hour_min < 4 * 60:
        return "closed"
    if hour_min < 9 * 60 + 30:
        return "pre-market"
    if hour_min < 16 * 60:
        return "open"
    if hour_min < 20 * 60:
        return "after-hours"
    return "closed"


@app.get("/api/status")
def get_status():
    account = _get_account_info()
    strategy = {}
    if STRATEGY_JSON.exists():
        with open(STRATEGY_JSON) as f:
            strategy = json.load(f)

    return {
        "market_status": _market_status(),
        "timestamp": datetime.now(ET).isoformat(),
        "account": account,
        "strategy_version": strategy.get("version", "unknown"),
        "strategy_name": strategy.get("name", "unknown"),
    }


@app.get("/api/positions")
def get_positions():
    with db.get_db() as conn:
        trades = db.get_open_trades(conn)
    return {"positions": trades, "count": len(trades)}


@app.get("/api/trades")
def get_trades(days: int = 1):
    since = (datetime.now(ET) - timedelta(days=days)).isoformat()
    with db.get_db() as conn:
        trades = db.get_trades_since(conn, since)
    return {"trades": trades, "count": len(trades)}


@app.get("/api/today-pnl")
def get_today_pnl():
    """Calendar-day P&L: equity change from start of day (realized + unrealized)."""
    today = datetime.now(ET).date().isoformat()
    account = _get_account_info()
    current_equity = account.get("equity", 0)

    with db.get_db() as conn:
        dp = conn.execute(
            "SELECT starting_equity FROM daily_performance WHERE date = ?",
            (today,),
        ).fetchone()
        if not dp or not dp["starting_equity"]:
            # Fallback: use the most recent ending equity
            dp = conn.execute(
                "SELECT ending_equity as starting_equity FROM daily_performance "
                "WHERE ending_equity IS NOT NULL ORDER BY date DESC LIMIT 1",
            ).fetchone()

    if dp and dp["starting_equity"] and dp["starting_equity"] > 0:
        pnl = current_equity - dp["starting_equity"]
    else:
        pnl = 0

    return {"pnl": round(pnl, 2), "equity": round(current_equity, 2)}


@app.get("/api/daily-pnl")
def get_daily_pnl(days: int = 30):
    since = (datetime.now(ET) - timedelta(days=days)).date().isoformat()
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM daily_performance WHERE date >= ? ORDER BY date",
            (since,),
        ).fetchall()
    return {"daily_pnl": [dict(r) for r in rows]}


@app.get("/api/strategy-history")
def get_strategy_history():
    with db.get_db() as conn:
        versions = conn.execute(
            "SELECT * FROM strategy_versions ORDER BY deployed_at DESC"
        ).fetchall()
    return {"versions": [dict(v) for v in versions]}


@app.get("/api/improvements")
def get_improvements(limit: int = 20):
    with db.get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM improvement_log ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return {"improvements": [dict(r) for r in rows]}


@app.get("/api/metrics")
def get_metrics(days: int = 30):
    since = (datetime.now(ET) - timedelta(days=days)).date().isoformat()
    with db.get_db() as conn:
        daily = conn.execute(
            "SELECT * FROM daily_performance WHERE date >= ? ORDER BY date",
            (since,),
        ).fetchall()
        closed = conn.execute(
            "SELECT * FROM trades WHERE status='closed' AND closed_at >= ? ORDER BY closed_at",
            (since + "T00:00:00",),
        ).fetchall()

    daily_returns = []
    for dp in daily:
        if dp["starting_equity"] and dp["starting_equity"] > 0 and dp["daily_pnl"] is not None:
            daily_returns.append(dp["daily_pnl"] / dp["starting_equity"])

    sharpe = 0.0
    if daily_returns and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    wins = sum(1 for t in closed if (t["pnl"] or 0) > 0)
    total = len(closed)
    total_pnl = sum(t["pnl"] or 0 for t in closed)
    gross_profit = sum(t["pnl"] for t in closed if (t["pnl"] or 0) > 0)
    gross_loss = abs(sum(t["pnl"] for t in closed if (t["pnl"] or 0) <= 0)) or 1

    equities = [dp["ending_equity"] for dp in daily if dp["ending_equity"]]
    max_dd = 0.0
    if equities:
        peak = equities[0]
        for eq in equities:
            peak = max(peak, eq)
            dd = (peak - eq) / peak
            max_dd = max(max_dd, dd)

    return {
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 4),
        "win_rate": round(wins / total, 4) if total else 0,
        "profit_factor": round(gross_profit / gross_loss, 4),
        "total_pnl": round(total_pnl, 2),
        "total_trades": total,
        "period_days": days,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard_page():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AlgoTrader Pro</title>
<style>
  :root {
    --bg: #0a0e17; --card: #111827; --border: #1f2937;
    --text: #e5e7eb; --muted: #9ca3af; --accent: #3b82f6;
    --green: #10b981; --red: #ef4444; --yellow: #f59e0b;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font-family: 'SF Mono', 'Fira Code', monospace; font-size: 14px; }
  .header { padding: 20px 24px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
  .header h1 { font-size: 18px; font-weight: 600; }
  .status-badge { padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 500; }
  .status-open { background: rgba(16,185,129,0.15); color: var(--green); }
  .status-closed { background: rgba(239,68,68,0.15); color: var(--red); }
  .status-pre-market, .status-after-hours { background: rgba(245,158,11,0.15); color: var(--yellow); }
  .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; padding: 20px 24px; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .card-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .card-value { font-size: 24px; font-weight: 700; margin-top: 4px; }
  .positive { color: var(--green); }
  .negative { color: var(--red); }
  .main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; padding: 0 24px 24px; }
  .section { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; }
  .section h2 { font-size: 14px; margin-bottom: 12px; color: var(--muted); }
  table { width: 100%; border-collapse: collapse; }
  th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 13px; }
  th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; }
  .improvement-item { padding: 10px 0; border-bottom: 1px solid var(--border); }
  .improvement-item:last-child { border-bottom: none; }
  .tag-deployed { color: var(--green); }
  .tag-reverted { color: var(--red); }
  .refresh-info { font-size: 11px; color: var(--muted); }
</style>
</head>
<body>
<div class="header">
  <h1>AlgoTrader Pro</h1>
  <div>
    <span id="market-status" class="status-badge">--</span>
    <span id="strategy-version" style="margin-left:12px;color:var(--muted);font-size:12px"></span>
    <span class="refresh-info" style="margin-left:12px">Auto-refresh: 30s</span>
  </div>
</div>
<div class="grid">
  <div class="card"><div class="card-label">Equity</div><div class="card-value" id="equity">--</div></div>
  <div class="card"><div class="card-label">Daily P&L</div><div class="card-value" id="daily-pnl">--</div></div>
  <div class="card"><div class="card-label">Sharpe (30d)</div><div class="card-value" id="sharpe">--</div></div>
  <div class="card"><div class="card-label">Win Rate</div><div class="card-value" id="win-rate">--</div></div>
</div>
<div class="main-grid">
  <div>
    <div class="section" style="margin-bottom:16px">
      <h2>Open Positions</h2>
      <table><thead><tr><th>Symbol</th><th>Side</th><th>Qty</th><th>Entry</th><th>Stop</th><th>Target</th><th>Signal</th></tr></thead>
      <tbody id="positions-body"></tbody></table>
    </div>
    <div class="section">
      <h2>Recent Trades</h2>
      <table><thead><tr><th>Symbol</th><th>Side</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Signal</th><th>Time</th></tr></thead>
      <tbody id="trades-body"></tbody></table>
    </div>
  </div>
  <div>
    <div class="section" style="margin-bottom:16px">
      <h2>Performance Metrics (30d)</h2>
      <table>
        <tr><td>Sharpe Ratio</td><td id="m-sharpe">--</td></tr>
        <tr><td>Max Drawdown</td><td id="m-drawdown">--</td></tr>
        <tr><td>Profit Factor</td><td id="m-pf">--</td></tr>
        <tr><td>Total Trades</td><td id="m-trades">--</td></tr>
        <tr><td>Total P&L</td><td id="m-pnl">--</td></tr>
      </table>
    </div>
    <div class="section">
      <h2>Improvement History</h2>
      <div id="improvements"></div>
    </div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
const fmt = (v, pre='$') => v >= 0
  ? `<span class="positive">${pre}${Math.abs(v).toFixed(2)}</span>`
  : `<span class="negative">-${pre}${Math.abs(v).toFixed(2)}</span>`;

function fmtPrice(v) {
  if (v == null) return '--';
  const abs = Math.abs(v);
  if (abs === 0) return '$0';
  if (abs < 0.001) return '$' + v.toFixed(8);
  if (abs < 1) return '$' + v.toFixed(4);
  return '$' + v.toFixed(2);
}

function fmtQty(v) {
  if (v == null) return '--';
  if (v > 1000000) return (v/1000000).toFixed(1) + 'M';
  if (v > 1000) return (v/1000).toFixed(1) + 'K';
  if (Number.isInteger(v)) return v.toString();
  return v.toFixed(4);
}

async function refresh() {
  try {
    const [status, positions, trades, metrics, improvements, todayPnlData] = await Promise.all([
      fetch('/api/status').then(r=>r.json()),
      fetch('/api/positions').then(r=>r.json()),
      fetch('/api/trades?days=3').then(r=>r.json()),
      fetch('/api/metrics?days=30').then(r=>r.json()),
      fetch('/api/improvements?limit=10').then(r=>r.json()),
      fetch('/api/today-pnl').then(r=>r.json()),
    ]);

    // Status
    const ms = $('market-status');
    ms.textContent = status.market_status.replace('-', ' ');
    ms.className = 'status-badge status-' + status.market_status;
    $('strategy-version').textContent = 'Strategy ' + status.strategy_version;
    $('equity').textContent = '$' + (status.account.equity || 0).toLocaleString(undefined, {minimumFractionDigits:2});

    // Metrics
    $('sharpe').textContent = metrics.sharpe_ratio.toFixed(2);
    $('win-rate').textContent = (metrics.win_rate * 100).toFixed(1) + '%';
    $('m-sharpe').textContent = metrics.sharpe_ratio.toFixed(4);
    $('m-drawdown').textContent = (metrics.max_drawdown * 100).toFixed(2) + '%';
    $('m-pf').textContent = metrics.profit_factor.toFixed(2);
    $('m-trades').textContent = metrics.total_trades;
    $('m-pnl').innerHTML = fmt(metrics.total_pnl);

    // Daily PNL — calendar day
    $('daily-pnl').innerHTML = fmt(todayPnlData.pnl);

    // Positions table
    $('positions-body').innerHTML = positions.positions.length
      ? positions.positions.map(p => `<tr>
          <td><strong>${p.symbol}</strong></td><td>${p.side}</td><td>${fmtQty(p.qty)}</td>
          <td>${fmtPrice(p.entry_price)}</td><td>${fmtPrice(p.stop_price)}</td>
          <td>${fmtPrice(p.take_profit_price)}</td><td>${p.signal_type || '-'}</td>
        </tr>`).join('')
      : '<tr><td colspan="7" style="color:var(--muted);text-align:center">No open positions</td></tr>';

    // Trades table — only today and yesterday
    const closedTrades = trades.trades.filter(t => t.status === 'closed').slice(-15);
    $('trades-body').innerHTML = closedTrades.length
      ? closedTrades.reverse().map(t => `<tr>
          <td><strong>${t.symbol}</strong></td><td>${t.side}</td>
          <td>${fmtPrice(t.entry_price)}</td><td>${fmtPrice(t.exit_price)}</td>
          <td>${fmt(t.pnl||0)}</td><td>${t.signal_type||'-'}</td>
          <td style="color:var(--muted)">${(t.closed_at||'').slice(5,16).replace('T',' ')}</td>
        </tr>`).join('')
      : '<tr><td colspan="7" style="color:var(--muted);text-align:center">No recent trades</td></tr>';

    // Improvements — fall back to git log if DB is empty
    $('improvements').innerHTML = improvements.improvements.length
      ? improvements.improvements.map(i => `<div class="improvement-item">
          <div style="font-size:11px;color:var(--muted)">${(i.timestamp||'').slice(0,16)}</div>
          <div style="margin:4px 0">${i.hypothesis || 'N/A'}</div>
          <div><span class="${i.deployed ? 'tag-deployed' : 'tag-reverted'}">${i.deployed ? 'DEPLOYED' : 'REVERTED'}</span>
          Sharpe: ${(i.backtest_sharpe||0).toFixed(3)}</div>
        </div>`).join('')
      : '<div style="color:var(--muted)">Improvements are logged in improvement_log.md and git history</div>';
  } catch(e) { console.error('Refresh error:', e); }
}

refresh();
setInterval(refresh, 30000);
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    db.init_db()
    uvicorn.run(app, host="0.0.0.0", port=8080)
