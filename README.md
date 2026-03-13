# AlgoTrader Pro

Self-improving algorithmic paper trader. Combines a real-time trade execution engine with a nightly Claude Code improvement loop that analyzes performance and evolves the trading strategy autonomously.

## Architecture

Two runtime layers communicating via SQLite and git:

- **Trader** (`trader.py`) — Runs 24/7, executes trades via Alpaca paper trading on 5-min bars
- **Intelligence Loop** (`improve.py`) — Runs nightly at 5pm ET, triggers Claude Code to analyze trades, modify `strategy.py`, validate via backtest, and deploy if improved

## Setup

### Prerequisites

- macOS (Apple Silicon)
- [Homebrew](https://brew.sh/)
- [Alpaca](https://alpaca.markets/) paper trading account
- [Claude Code](https://claude.com/claude-code) CLI (for the improvement loop)

### Install

```bash
# Install Python 3.13 if you don't have it
brew install python@3.13

# Clone and set up
git clone https://github.com/Coufu/buzzalgo.git
cd buzzalgo
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** macOS ships with an older Python (3.8/3.9) that won't work. You need Python 3.12+ installed via Homebrew. Verify with `python3.13 --version`.

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

### Docker (recommended)

```bash
docker compose up -d
```

This starts all three services (trader, dashboard, improvement loop) with auto-restart. Dashboard at http://localhost:8080.

```bash
docker compose logs -f trader    # watch trade activity
docker compose down              # stop everything
```

### Manual (without Docker)

```bash
source .venv/bin/activate
caffeinate -i python trader.py   # terminal 1: trader
python dashboard.py              # terminal 2: dashboard on :8080
python improve.py --cron         # terminal 3: nightly improvement loop
```

### Other commands

```bash
python backtest.py              # 60-day walk-forward validation
python backtest.py --days 90    # custom window
python improve.py --report-only # generate performance report only
python -m pytest tests/ -v      # run tests
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
