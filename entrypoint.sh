#!/bin/sh
echo "Running initial 60-day backtest..."
python -u backtest.py --days 60 || echo "Backtest completed (or no market data yet)"
echo "Starting trader..."
exec python -u trader.py
