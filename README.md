# AlgoTrader Pro

Self-improving algorithmic paper trader. Trades equities during market hours and crypto 24/7, using Claude Code to analyze performance and evolve the strategy.

## Quick Start

### 1. Get Alpaca API keys

Sign up at [alpaca.markets](https://alpaca.markets/) and get your **paper trading** API keys.

### 2. Clone and configure

```bash
git clone https://github.com/Coufu/buzzalgo.git
cd buzzalgo
cp .env.example .env
```

Edit `.env` with your keys:

```
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Full setup

```bash
./setup.sh
```

This installs Python dependencies, sets up the venv, and loads all launchd services.

Or manually:

```bash
brew install python@3.13
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
docker compose up -d --build
```

### 4. Check on it

- **Dashboard:** http://localhost:8080
- **Trader logs:** `docker compose logs -f trader`
- **Improvement logs:** `tail -f improve.log`

## How It Works

### Trading

**Equities (9:30am-4pm ET, Mon-Fri):**
- Watches 5-minute bars for ~100 symbols (S&P 500 ETFs + large-cap stocks)
- Enters trades when RSI + EMA + volume signals align with ADX regime detection
- Manages positions with stop-losses, take-profits, and time-based exits

**Crypto (24/7):**
- Watches 5-minute bars for BTC, ETH, SOL, AVAX, DOGE, LINK, LTC, UNI
- Same signal logic with separately tuned parameters (wider RSI bands, longer time exits)
- Supports fractional quantities

### Self-Improvement

Claude Code analyzes trade history, forms hypotheses, modifies the strategy, validates via backtest, and deploys only if the Sharpe ratio improves. Every change is committed to git with an explanation.

### Safety Rules (locked, Claude Code cannot change)

- 2% of portfolio per trade
- 1.5x ATR stop-loss
- Max 5 concurrent positions
- -3% daily loss circuit breaker
- Paper trading only

## Scheduled Services

| Service | Runtime | Schedule | What it does |
|---------|---------|----------|--------------|
| Trader | Docker | 24/7 | Executes equity trades during market hours, crypto trades 24/7 |
| Dashboard | Docker | 24/7 | FastAPI dashboard on port 8080 |
| Improve | launchd | 2am, 8am, 5pm ET Mon-Fri | Claude Code analyzes + improves strategy |
| Sentiment | launchd | 2am daily | Scans headlines via Ollama deepseek-r1:14b |
| Journal | launchd | 4:30pm ET Mon-Fri | Daily trading summary to Slack |
| Scanner | launchd | 4:45pm ET Mon-Fri | Triple bottom pattern detection to Slack |
| Sweep | launchd | 2am Sat+Sun | Parameter grid search |
| Caffeinate | launchd | 24/7 | Prevents Mac sleep |

Plist templates are in `launchd/` — `setup.sh` installs them automatically.

## Slack Notifications

The system sends trade alerts, daily journals, scanner results, and improvement summaries to Slack.

### Setup

1. Go to [api.slack.com/apps](https://api.slack.com/apps) and create a new app (or use an existing one)
2. Under **Incoming Webhooks**, toggle it on and click **Add New Webhook to Workspace**
3. Pick the channel you want alerts in and copy the webhook URL
4. Add it to your `.env`:

```
SLACK_WEBHOOK_URL=<your-webhook-url>
```

Once set, these services post to Slack automatically:

| Service | What it posts |
|---------|---------------|
| Trader | Trade opens/closes with P&L |
| Journal | Daily performance summary (4:30pm ET) |
| Scanner | Triple bottom pattern alerts (4:45pm ET) |
| Improve | Strategy change results (after each cycle) |

If `SLACK_WEBHOOK_URL` is empty or unset, all services run normally — notifications are skipped silently.

## Commands

```bash
source .venv/bin/activate

# Trading
caffeinate -i python trader.py           # run trader natively (alternative to Docker)
python dashboard.py                      # run dashboard natively

# Backtesting
python backtest.py                       # 60-day equity backtest
python backtest.py --days 90             # custom window
python backtest.py --mode crypto         # crypto backtest

# Improvement
python improve.py                        # trigger one improvement cycle
python improve.py --report-only          # generate performance report only
python improve.py --journal              # send daily journal to Slack

# Scanning
python scanner.py --once                 # single triple bottom scan
python scanner.py --evaluate             # score entry rules from past detections
python sentiment.py --once               # single sentiment scan

# Parameter optimization
python sweep.py --days 60 --top 10       # parameter grid search

# Replay
python replay.py --date 2026-03-12       # replay a specific day
python replay.py --compare params.json   # compare against proposed params
```

## Stopping

```bash
docker compose down
launchctl unload ~/Library/LaunchAgents/com.buzzalgo.*.plist
```

## File Structure

```
CLAUDE.md             # Claude Code's operating instructions
trader.py             # Trading engine (runs in Docker, equities + crypto)
strategy.py           # Trading strategy (Claude Code evolves this)
strategy.json         # Strategy parameters — equity + crypto (hot-reloaded)
risk.py               # Risk management (LOCKED - never modified)
backtest.py           # Backtesting engine (supports --mode crypto)
improve.py            # Self-improvement loop + journal + reports
scanner.py            # Triple bottom pattern scanner
sentiment.py          # News sentiment via Ollama
sweep.py              # Parameter grid search
replay.py             # Day replay for analysis
dashboard.py          # FastAPI web dashboard
db.py                 # SQLite database layer
launchd/              # macOS launchd plist templates
data/                 # SQLite databases, scanner state
.env                  # API keys and webhook URL (not committed)
```
