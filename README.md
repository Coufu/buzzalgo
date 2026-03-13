# AlgoTrader Pro

Self-improving algorithmic paper trader. Combines a real-time trade execution engine with a nightly Claude Code improvement loop that analyzes performance and evolves the trading strategy autonomously.

## Architecture

Two runtime layers communicating via SQLite and git:

- **Trader** (`trader.py`) — Runs 24/7, executes trades via Alpaca paper trading on 5-min bars
- **Intelligence Loop** (`improve.py`) — Runs nightly at 5pm ET, triggers Claude Code to analyze trades, modify `strategy.py`, validate via backtest, and deploy if improved

## Setup

### Prerequisites

- macOS with Python 3.12+
- [Alpaca](https://alpaca.markets/) paper trading account
- [Claude Code](https://claude.com/claude-code) CLI (for the improvement loop)

### Install

```bash
git clone https://github.com/Coufu/buzzalgo.git
cd buzzalgo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` with your Alpaca API keys:

```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

## Usage

### Start the dashboard

```bash
python dashboard.py
```

Open http://localhost:8080 for live P&L, positions, and strategy evolution.

### Start the trader

```bash
caffeinate -i python trader.py
```

Trades during market hours (9:30am–4:00pm ET), idles otherwise.

### Run a backtest

```bash
python backtest.py              # 60-day walk-forward validation
python backtest.py --days 90    # custom window
```

### Run the improvement loop

```bash
python improve.py               # one-shot improvement cycle
python improve.py --cron        # scheduled nightly at 5pm ET
python improve.py --report-only # generate performance report without modifying strategy
```

### Run tests

```bash
python -m pytest tests/ -v
```

## Strategy

Momentum + Mean Reversion hybrid on 5-min bars:

- **Entry**: RSI(14) crosses oversold/overbought thresholds with EMA(20) direction confirmation and 1.5x volume filter
- **Universe**: SPY, QQQ, IWM + top 20 liquid large-caps
- **Risk** (locked): 2% position sizing, 1.5x ATR stop-loss, 3x ATR take-profit, max 5 positions, -3% daily circuit breaker

Claude Code modifies signal logic nightly. Risk management in `risk.py` is locked and never modified.

## File Structure

```
├── CLAUDE.md             # Claude Code operating context
├── trader.py             # Main execution loop (24/7)
├── strategy.py           # Active strategy (modified nightly by Claude Code)
├── strategy.json         # Hot-reloadable parameters
├── risk.py               # Position sizing, stop-loss, circuit breaker (LOCKED)
├── backtest.py           # Walk-forward validation harness
├── improve.py            # Nightly improvement loop orchestrator
├── dashboard.py          # FastAPI dashboard (port 8080)
├── db.py                 # SQLite persistence layer
├── trades.db             # Trade log (created at runtime)
├── improvement_log.md    # Hypothesis history
├── tests/                # Test suite
└── .env                  # Alpaca API keys (not committed)
```
