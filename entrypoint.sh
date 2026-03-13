#!/bin/sh

# Validate credentials before starting
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    echo "ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"
    exit 1
fi

echo "Running initial 60-day backtest..."
if ! python -u backtest.py --days 60; then
    echo "WARNING: Backtest failed (see errors above). Starting trader anyway."
fi
echo "Starting trader..."
exec python -u trader.py
