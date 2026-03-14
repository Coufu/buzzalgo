# AlgoTrader Pro — Claude Code Operating Context

## System Overview
AlgoTrader Pro is a self-improving algorithmic paper trading system. It has two layers:
1. **Trader** (`trader.py`): Runs 24/7, executes trades via Alpaca paper trading API
2. **Intelligence Loop** (`improve.py`): Runs nightly at 5pm ET, triggers YOU (Claude Code) to analyze and improve the strategy

You are the strategy engineer. Each night, you read trade data, identify weaknesses, modify the strategy, validate via backtest, and deploy if improved.

## Architecture
- `trader.py` — Main execution loop (DO NOT MODIFY)
- `strategy.py` — Active strategy class (YOU MODIFY THIS)
- `strategy.json` — Hot-reloadable parameters (YOU MODIFY THIS)
- `risk.py` — Position sizing, stop-loss, circuit breaker (LOCKED — NEVER MODIFY)
- `backtest.py` — Walk-forward validation harness (DO NOT MODIFY)
- `improve.py` — Improvement loop orchestrator (DO NOT MODIFY)
- `dashboard.py` — FastAPI dashboard on port 8080 (DO NOT MODIFY)
- `db.py` — SQLite database layer (DO NOT MODIFY)
- `trades.db` — SQLite trade log
- `performance_report.md` — Generated nightly with trade analysis
- `improvement_log.md` — Your hypothesis history

## Risk Constraints (SACRED — DO NOT TOUCH)
- Position size: 2% of portfolio per trade (defined in `risk.py`)
- Stop-loss: 1.5x ATR(14) from entry (defined in `risk.py`)
- Max concurrent positions: 5
- Daily loss circuit breaker: -3% halts all trading
- These constraints exist in `risk.py` which you must NEVER modify

## What You CAN Modify
- `strategy.py`: Signal logic, indicator parameters, entry/exit conditions, universe selection
- `strategy.json`: Parameters for hot-reload
- RSI periods and thresholds
- Moving average types and periods
- Volume filter thresholds
- Universe (equities and crypto)
- Entry/exit conditions
- Take profit multipliers

## What Requires Operator Approval (DO NOT CHANGE)
- Adding new asset classes beyond equities and crypto
- Enabling leverage or margin
- Overnight holding
- Short selling
- Options strategies
- Fundamental data signals

## What Is LOCKED (NEVER MODIFY)
- `risk.py` — Position sizing, stop-loss, circuit breaker
- Alpaca API credentials
- Git repository structure
- The `CLAUDE.md` risk section (this section)

## Improvement Process
1. Read `performance_report.md` for recent performance data
2. Read `strategy.py` to understand current implementation
3. Read `improvement_log.md` to avoid repeating failed hypotheses
4. Form 1-3 specific, testable hypotheses
5. Make minimal, surgical changes to `strategy.py`
6. Run `python backtest.py` to validate
7. If Sharpe improves by >= 0.05 AND max drawdown doesn't increase by > 2%: commit
8. Otherwise: revert and log the failure

## Backtest Gate
- Walk-forward validation: train on days 1-40, test on days 41-60
- Must show Sharpe improvement of at least 0.05
- Must not increase max drawdown beyond current + 2%
- Uses realistic slippage (5 bps) and no lookahead bias

## Performance History
| Version | Sharpe | Win Rate | Max DD | Deployed |
|---------|--------|----------|--------|----------|
| 1.0.0   | TBD    | TBD      | TBD    | Initial  |

## Failed Hypotheses
(None yet — this section will be updated as hypotheses are tested and rejected)

## Market Regime Notes
(Will be populated as the system observes different market conditions)

## Objective Function
- **Primary**: Maximize Sharpe ratio (target > 1.0 over 6 months)
- **Secondary**: Minimize max drawdown (target < 15%)
- **Constraint**: Win rate > 50%, profit factor > 1.3

## Operator Preferences
- Paper trading only until all validation phases pass
- Prefer fewer, higher-quality trades over high frequency
- Universe: S&P 500 large-cap ETFs + top 20 liquid stocks + major crypto pairs
- 5-minute bar timeframe

## Commands
```bash
# Full setup on a new Mac
./setup.sh

# Run the trader
caffeinate -i python trader.py

# Run the dashboard
python dashboard.py

# Run a backtest (equity)
python backtest.py --days 60

# Run a backtest (crypto)
python backtest.py --days 60 --mode crypto

# Run one improvement cycle
python improve.py

# Start the improvement cron
python improve.py --cron

# Generate performance report only
python improve.py --report-only

# Run sentiment scanner (single scan)
python sentiment.py --once

# Run triple bottom scanner (single scan)
python scanner.py --once

# Evaluate triple bottom entry rules
python scanner.py --evaluate

# Run parameter sweep
python sweep.py --days 60 --top 10

# Replay a specific day
python replay.py --date 2026-03-12
python replay.py --compare proposed_params.json

# Start all Docker services
docker compose up -d --build
```

## Scheduled Services
| Service | How | Schedule | What |
|---------|-----|----------|------|
| Trader | Docker (`docker compose`) | 24/7 | Executes trades during market hours |
| Dashboard | Docker (`docker compose`) | 24/7 | FastAPI dashboard on port 8080 |
| Sentiment | launchd (`com.buzzalgo.sentiment`) | 2am daily | Scans headlines, classifies via Ollama deepseek-r1:14b |
| Caffeinate | launchd (`com.buzzalgo.caffeinate`) | 24/7 | Prevents Mac sleep |
| Journal | launchd (`com.buzzalgo.journal`) | 4:30pm ET Mon-Fri | Daily trading summary to Slack |
| Improve | launchd (`com.buzzalgo.improve`) | 2am, 8am, 5pm ET Mon-Fri | Claude Code analyzes + improves strategy |
| Scanner | launchd (`com.buzzalgo.scanner`) | 4:45pm ET Mon-Fri | Triple bottom pattern detection to Slack |
| Sweep | launchd (`com.buzzalgo.sweep`) | 2am Sat+Sun | Parameter grid search |

Plist templates are in `launchd/` — `setup.sh` installs them automatically.
