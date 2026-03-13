"""Tests for the database module."""

import tempfile
from pathlib import Path

import pytest
import db


@pytest.fixture
def tmp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = Path(f.name)
    db.init_db(path)
    yield path
    path.unlink(missing_ok=True)


class TestDatabase:
    def test_init_creates_tables(self, tmp_db):
        with db.get_db(tmp_db) as conn:
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            names = {t["name"] for t in tables}
            assert "trades" in names
            assert "daily_performance" in names
            assert "strategy_versions" in names
            assert "improvement_log" in names

    def test_log_and_get_trade(self, tmp_db):
        with db.get_db(tmp_db) as conn:
            db.log_trade(conn, {
                "timestamp": "2024-01-01T10:00:00",
                "symbol": "SPY",
                "side": "long",
                "qty": 10,
                "entry_price": 450.0,
                "status": "open",
                "opened_at": "2024-01-01T10:00:00",
            })
        with db.get_db(tmp_db) as conn:
            trades = db.get_open_trades(conn)
            assert len(trades) == 1
            assert trades[0]["symbol"] == "SPY"
            assert trades[0]["qty"] == 10

    def test_close_trade(self, tmp_db):
        with db.get_db(tmp_db) as conn:
            db.log_trade(conn, {
                "timestamp": "2024-01-01T10:00:00",
                "symbol": "QQQ",
                "side": "long",
                "qty": 5,
                "entry_price": 380.0,
                "status": "open",
                "opened_at": "2024-01-01T10:00:00",
            })
        with db.get_db(tmp_db) as conn:
            trades = db.get_open_trades(conn)
            db.close_trade(conn, trades[0]["id"], 385.0, 25.0, 1.32)
        with db.get_db(tmp_db) as conn:
            open_trades = db.get_open_trades(conn)
            assert len(open_trades) == 0

    def test_daily_performance(self, tmp_db):
        with db.get_db(tmp_db) as conn:
            db.log_daily_performance(conn, {
                "date": "2024-01-01",
                "starting_equity": 100000,
                "ending_equity": 100500,
                "daily_pnl": 500,
                "daily_return_pct": 0.5,
                "trades_taken": 3,
                "wins": 2,
                "losses": 1,
                "circuit_breaker_triggered": 0,
                "strategy_version": "1.0.0",
            })
        with db.get_db(tmp_db) as conn:
            rows = conn.execute("SELECT * FROM daily_performance").fetchall()
            assert len(rows) == 1
            assert rows[0]["daily_pnl"] == 500
