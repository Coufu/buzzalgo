#!/bin/bash
# BuzzAlgo Setup Script
# Run this on a fresh Mac to set up the full trading system.
#
# Usage:
#   ./setup.sh                    # Fresh setup
#   ./setup.sh backup             # Backup DB to data/backup/
#   ./setup.sh restore <file>     # Restore DB from backup

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LAUNCH_AGENTS="$HOME/Library/LaunchAgents"
DATA_DIR="$SCRIPT_DIR/data"
BACKUP_DIR="$DATA_DIR/backup"
DB_FILE="$DATA_DIR/algo.db"

# --- Backup/Restore commands ---
if [ "${1:-}" = "backup" ]; then
    mkdir -p "$BACKUP_DIR"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_FILE="$BACKUP_DIR/algo_$TIMESTAMP.db"
    echo "Stopping Docker services..."
    cd "$SCRIPT_DIR" && docker compose stop 2>/dev/null || true
    cp "$DB_FILE" "$BACKUP_FILE"
    echo "Backed up to: $BACKUP_FILE"
    echo "Restarting Docker services..."
    docker compose up -d
    echo ""
    echo "To migrate: copy $BACKUP_FILE to the new machine's data/ dir as algo.db"
    exit 0
fi

if [ "${1:-}" = "restore" ]; then
    RESTORE_FILE="${2:-}"
    if [ -z "$RESTORE_FILE" ] || [ ! -f "$RESTORE_FILE" ]; then
        echo "Usage: ./setup.sh restore <path-to-backup.db>"
        echo ""
        if [ -d "$BACKUP_DIR" ]; then
            echo "Available backups:"
            ls -lh "$BACKUP_DIR"/algo_*.db 2>/dev/null || echo "  (none)"
        fi
        exit 1
    fi
    echo "Stopping Docker services..."
    cd "$SCRIPT_DIR" && docker compose stop 2>/dev/null || true
    mkdir -p "$DATA_DIR"
    cp "$RESTORE_FILE" "$DB_FILE"
    echo "Restored from: $RESTORE_FILE"
    echo "Restarting Docker services..."
    docker compose up -d
    exit 0
fi

echo "=== BuzzAlgo Setup ==="
echo "Project dir: $SCRIPT_DIR"
echo ""

# 1. Check prerequisites
echo "Checking prerequisites..."
command -v docker >/dev/null 2>&1 || { echo "ERROR: Docker not installed. Download from docker.com"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: Python 3 not installed. Run: brew install python@3.13"; exit 1; }
command -v ollama >/dev/null 2>&1 || echo "WARNING: Ollama not installed — sentiment scanner won't work. Run: brew install ollama"
command -v claude >/dev/null 2>&1 || echo "WARNING: Claude Code CLI not installed — improvement loop won't work"
echo "OK"

# 2. Python venv
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Creating Python venv..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    source "$SCRIPT_DIR/.venv/bin/activate"
    pip install -r "$SCRIPT_DIR/requirements.txt"
else
    echo "Python venv exists"
    source "$SCRIPT_DIR/.venv/bin/activate"
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

# 4. Data directory
mkdir -p "$DATA_DIR"
echo "Data dir: $DATA_DIR"

# 5. Pull Ollama model
if command -v ollama >/dev/null 2>&1; then
    echo "Checking Ollama model..."
    if ! ollama list 2>/dev/null | grep -q "deepseek-r1:14b"; then
        echo "Pulling deepseek-r1:14b..."
        ollama pull deepseek-r1:14b
    else
        echo "deepseek-r1:14b already available"
    fi
fi

# 6. Install launchd agents
echo ""
echo "Installing launchd agents..."
mkdir -p "$LAUNCH_AGENTS"

for plist in "$SCRIPT_DIR/launchd"/*.plist; do
    [ -f "$plist" ] || continue
    filename=$(basename "$plist")
    # Replace hardcoded paths with actual project dir and home dir
    sed -e "s|/Users/dougchan/ai/buzzalgo|$SCRIPT_DIR|g" \
        -e "s|/Users/dougchan|$HOME|g" \
        "$plist" > "$LAUNCH_AGENTS/$filename"
    launchctl unload "$LAUNCH_AGENTS/$filename" 2>/dev/null || true
    launchctl load "$LAUNCH_AGENTS/$filename"
    echo "  Loaded $filename"
done

# 7. Start Docker services
echo ""
echo "Starting Docker services..."
cd "$SCRIPT_DIR"
docker compose up -d --build

# 8. Verify
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Services:"
echo "  Docker:"
echo "    Trader      24/7 (equities during market, crypto 24/7, swing daily)"
echo "    Dashboard   http://localhost:8080"
echo ""
echo "  Launchd:"
echo "    Improve     2am, 8am, 5pm ET Mon-Fri (Claude Code strategy evolution)"
echo "    Tournament  3am ET Mon-Fri (adaptive signal weighting)"
echo "    Sentiment   2am daily (Ollama headline scanning)"
echo "    Journal     4:30pm ET Mon-Fri (daily P&L summary to Slack)"
echo "    Scanner     4:45pm ET Mon-Fri (triple bottom pattern alerts)"
echo "    Healthcheck 9am+6pm ET Mon-Fri, noon weekends"
echo "    Sweep       2am Sat+Sun (parameter grid search)"
echo "    Caffeinate  24/7 (prevents Mac sleep)"
echo ""
echo "Verify:"
echo "  python3 healthcheck.py"
echo ""
echo "Logs:"
echo "  docker compose logs -f trader"
echo "  tail -f improve.log"
echo "  tail -f sweep.log"
