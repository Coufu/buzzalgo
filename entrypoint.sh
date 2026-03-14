#!/bin/sh

# Validate credentials before starting
if [ -z "$ALPACA_API_KEY" ] || [ -z "$ALPACA_SECRET_KEY" ]; then
    echo "ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set"
    exit 1
fi

echo "Starting trader..."
exec python -u trader.py
