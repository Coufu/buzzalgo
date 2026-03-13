# AlgoTrader Pro

Self-improving algorithmic paper trader. Trades automatically during market hours and uses Claude Code every evening to analyze performance and evolve the strategy.

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

### 3. Install Python (for the improvement loop)

```bash
brew install python@3.13
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 4. Start everything

```bash
docker compose up -d --build
cp com.buzzalgo.improve.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.buzzalgo.improve.plist
```

That's it. The trader and dashboard run in Docker. The improvement loop runs natively at 5pm ET every weekday (it needs Claude Code CLI access).

### 5. Check on it

- **Dashboard:** http://localhost:8080
- **Trader logs:** `docker compose logs -f trader`
- **Improvement logs:** `tail -f improve.log`

## Stopping

```bash
docker compose down
launchctl unload ~/Library/LaunchAgents/com.buzzalgo.improve.plist
```

## How It Works

**During market hours (9:30am-4pm ET):**
- Watches 5-minute price bars for SPY, QQQ, IWM and 20 large-cap stocks
- Enters trades when RSI + EMA + volume signals align
- Manages positions with stop-losses, take-profits, and trailing stops
- Halts trading if daily losses exceed 3%

**Every evening at 5pm ET:**
- Claude Code analyzes the day's trades
- Identifies what worked and what didn't
- Modifies the strategy and runs a backtest
- Deploys the change only if it improves the Sharpe ratio
- Commits every change to git with an explanation

**Safety rules (locked, Claude Code cannot change):**
- 2% of portfolio per trade
- 1.5x ATR stop-loss
- Max 5 positions at a time
- -3% daily loss circuit breaker
- Paper trading only (no real money)

## Other Commands

```bash
source .venv/bin/activate

python backtest.py              # run a 60-day backtest
python backtest.py --days 90    # custom window
python improve.py               # trigger one improvement cycle manually
python improve.py --report-only # just generate the performance report
python -m pytest tests/ -v      # run tests
```

## File Structure

```
CLAUDE.md             # Claude Code's operating instructions
trader.py             # Trading engine (runs in Docker)
strategy.py           # Trading strategy (Claude Code evolves this)
strategy.json         # Strategy parameters (hot-reloaded)
risk.py               # Risk management (LOCKED - never modified)
backtest.py           # Backtesting engine
improve.py            # Nightly improvement loop
dashboard.py          # Web dashboard
db.py                 # Database layer
.env                  # Your Alpaca API keys (not committed)
```
