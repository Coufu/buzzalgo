"""
Database module - SQLite trade log persistence.
"""

import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "trades.db"


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def get_db(db_path: str | Path = DB_PATH):
    conn = get_connection(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str | Path = DB_PATH):
    """Create tables if they don't exist."""
    with get_db(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                stop_price REAL,
                take_profit_price REAL,
                pnl REAL,
                pnl_pct REAL,
                strategy_version TEXT,
                signal_type TEXT,
                signal_context TEXT,
                status TEXT DEFAULT 'open',
                opened_at TEXT,
                closed_at TEXT,
                atr_at_entry REAL,
                rsi_at_entry REAL,
                ema_at_entry REAL,
                volume_ratio REAL,
                market_regime TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT UNIQUE NOT NULL,
                starting_equity REAL,
                ending_equity REAL,
                daily_pnl REAL,
                daily_return_pct REAL,
                trades_taken INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                circuit_breaker_triggered INTEGER DEFAULT 0,
                strategy_version TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS strategy_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT UNIQUE NOT NULL,
                deployed_at TEXT NOT NULL,
                sharpe_ratio REAL,
                win_rate REAL,
                profit_factor REAL,
                max_drawdown REAL,
                description TEXT,
                commit_hash TEXT,
                backtest_result TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS improvement_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                hypothesis TEXT,
                changes_made TEXT,
                backtest_sharpe REAL,
                previous_sharpe REAL,
                deployed INTEGER DEFAULT 0,
                revert_reason TEXT,
                commit_hash TEXT
            )
        """)
    logger.info("Database initialized at %s", db_path)


_TRADE_COLUMNS = frozenset([
    "timestamp", "symbol", "side", "qty", "entry_price", "exit_price",
    "stop_price", "take_profit_price", "pnl", "pnl_pct",
    "strategy_version", "signal_type", "signal_context", "status",
    "opened_at", "closed_at", "atr_at_entry", "rsi_at_entry",
    "ema_at_entry", "volume_ratio", "market_regime",
])


def log_trade(conn: sqlite3.Connection, trade: dict):
    """Insert a new trade record."""
    present = {k: trade.get(k) for k in _TRADE_COLUMNS if k in trade}
    if "signal_context" in present and isinstance(present["signal_context"], dict):
        present["signal_context"] = json.dumps(present["signal_context"])
    keys = ", ".join(present.keys())
    placeholders = ", ".join(["?"] * len(present))
    conn.execute(
        f"INSERT INTO trades ({keys}) VALUES ({placeholders})",
        list(present.values()),
    )


def close_trade(conn: sqlite3.Connection, trade_id: int, exit_price: float, pnl: float, pnl_pct: float):
    """Close an open trade."""
    conn.execute(
        """UPDATE trades SET exit_price=?, pnl=?, pnl_pct=?, status='closed',
           closed_at=? WHERE id=?""",
        (exit_price, pnl, pnl_pct, datetime.now().isoformat(), trade_id),
    )


def get_open_trades(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM trades WHERE status='open' ORDER BY opened_at"
    ).fetchall()
    return [dict(r) for r in rows]


def get_trades_since(conn: sqlite3.Connection, since: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM trades WHERE timestamp >= ? ORDER BY timestamp",
        (since,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_daily_pnl(conn: sqlite3.Connection, trade_date: str) -> float:
    row = conn.execute(
        "SELECT COALESCE(SUM(pnl), 0) as total FROM trades WHERE date(closed_at)=? AND status='closed'",
        (trade_date,),
    ).fetchone()
    return row["total"] if row else 0.0


def log_daily_performance(conn: sqlite3.Connection, perf: dict):
    conn.execute(
        """INSERT OR REPLACE INTO daily_performance
           (date, starting_equity, ending_equity, daily_pnl, daily_return_pct,
            trades_taken, wins, losses, circuit_breaker_triggered, strategy_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            perf["date"], perf.get("starting_equity"), perf.get("ending_equity"),
            perf.get("daily_pnl"), perf.get("daily_return_pct"),
            perf.get("trades_taken", 0), perf.get("wins", 0),
            perf.get("losses", 0), perf.get("circuit_breaker_triggered", 0),
            perf.get("strategy_version"),
        ),
    )
