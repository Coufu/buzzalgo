#!/bin/bash
# BuzzAlgo Setup Script
# Run this on a fresh Mac to set up the full trading system.
#
# Usage: ./setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"

echo "=== BuzzAlgo Setup ==="
echo "Project dir: $SCRIPT_DIR"
echo ""

# 1. Check prerequisites
echo "Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "ERROR: Docker not installed"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: Python 3 not installed"; exit 1; }
command -v ollama >/dev/null 2>&1 || { echo "WARNING: Ollama not installed — sentiment scanner won't work"; }
echo "OK"

# 2. Python venv
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating Python venv..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    source "$SCRIPT_DIR/.venv/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "Python venv exists"
fi

# 3. Environment file
if [ ! -f "$SCRIPT_DIR/.env" ]; then
    echo ""
    echo "ERROR: .env file not found. Copy .env.example and fill in your keys:"
    echo "  cp .env.example .env"
    echo "  # Edit .env with your Alpaca API keys and Slack webhook URL"
    exit 1
else
    echo ".env file exists"
fi

# 4. Pull Ollama model
if command -v ollama >/dev/null 2>&1; then
    echo "Checking Ollama model..."
    if ! ollama list | grep -q "deepseek-r1:14b"; then
        echo "Pulling deepseek-r1:14b..."
        ollama pull deepseek-r1:14b
    else
        echo "deepseek-r1:14b already available"
    fi
fi

# 5. Install launchd agents
echo ""
echo "Installing launchd agents..."
mkdir -p "$LAUNCH_AGENTS"

for plist in "$SCRIPT_DIR/launchd"/*.plist; do
    filename=$(basename "$plist")
    # Replace /Users/dougchan/ai/buzzalgo with actual project dir
    sed "s|/Users/dougchan/ai/buzzalgo|$SCRIPT_DIR|g" "$plist" > "$LAUNCH_AGENTS/$filename"
    launchctl unload "$LAUNCH_AGENTS/$filename" 2>/dev/null || true
    launchctl load "$LAUNCH_AGENTS/$filename"
    echo "  Loaded $filename"
done

# 6. Start Docker services
echo ""
echo "Starting Docker services..."
cd "$SCRIPT_DIR"
docker compose up -d --build

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Services running:"
echo "  - Trader:    Docker (scans every 5 min during market hours)"
echo "  - Dashboard: Docker (http://localhost:8080)"
echo "  - Sentiment: Docker (scans headlines every 30 min via Ollama)"
echo "  - Caffeinate: launchd (prevents Mac sleep)"
echo "  - Improve:   launchd (5pm ET weekdays — Claude Code improvement loop)"
echo "  - Sweep:     launchd (2am Sat+Sun — parameter grid search)"
echo ""
echo "Logs:"
echo "  docker compose logs -f trader"
echo "  docker compose logs -f sentiment"
echo "  cat improve.log"
echo "  cat sweep.log"
