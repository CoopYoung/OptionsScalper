"""SQLite persistence for options trade history and portfolio state.

Adapted from poly-trader with options-specific fields:
strike, option_type, greeks_at_entry, underlying.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TradeDB:
    def __init__(self, db_path: str = "data/bot.db") -> None:
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._create_tables()
        logger.info("TradeDB connected: %s", self._db_path)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _create_tables(self) -> None:
        assert self._conn is not None
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                underlying      TEXT    NOT NULL,
                contract_symbol TEXT    NOT NULL,
                option_type     TEXT    NOT NULL,       -- call / put
                strike          TEXT    NOT NULL,
                expiration      TEXT    NOT NULL,
                side            TEXT    NOT NULL,       -- BUY_CALL / BUY_PUT
                contracts       INTEGER NOT NULL,
                entry_premium   TEXT    NOT NULL,
                entry_time      TEXT    NOT NULL,
                exit_premium    TEXT,
                exit_time       TEXT,
                pnl             TEXT,
                order_id        TEXT    DEFAULT '',
                strategy        TEXT    DEFAULT '',
                confidence      INTEGER DEFAULT 0,
                exit_reason     TEXT    DEFAULT '',
                greeks_at_entry TEXT    DEFAULT '',     -- JSON: delta, gamma, theta, vega
                quant_signals   TEXT    DEFAULT '',     -- JSON: VIX regime, GEX, sentiment at entry
                created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );

            CREATE INDEX IF NOT EXISTS idx_trades_underlying ON trades(underlying);
            CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
            CREATE INDEX IF NOT EXISTS idx_trades_exit_time ON trades(exit_time);

            CREATE TABLE IF NOT EXISTS portfolio_state (
                id                  INTEGER PRIMARY KEY CHECK (id = 1),
                portfolio_value     TEXT    NOT NULL DEFAULT '10000',
                daily_pnl           TEXT    NOT NULL DEFAULT '0',
                day_start_value     TEXT    NOT NULL DEFAULT '10000',
                daily_trades        INTEGER NOT NULL DEFAULT 0,
                day_trades_5d       TEXT    NOT NULL DEFAULT '[]',
                updated_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            );

            CREATE TABLE IF NOT EXISTS open_positions (
                contract_symbol TEXT    PRIMARY KEY,
                underlying      TEXT    NOT NULL,
                option_type     TEXT    NOT NULL,
                strike          TEXT    NOT NULL,
                side            TEXT    NOT NULL,
                contracts       INTEGER NOT NULL,
                entry_premium   TEXT    NOT NULL,
                entry_time      TEXT    NOT NULL,
                order_id        TEXT    DEFAULT '',
                entry_confidence INTEGER DEFAULT 0,
                peak_premium    TEXT    DEFAULT '0'
            );
        """)
        self._conn.execute("""
            INSERT OR IGNORE INTO portfolio_state (id, portfolio_value, daily_pnl, day_start_value, daily_trades)
            VALUES (1, '10000', '0', '10000', 0)
        """)
        self._conn.commit()
        self._migrate_analytics_columns()

    def _migrate_analytics_columns(self) -> None:
        """Add analytics columns to trades table (migration-safe)."""
        assert self._conn is not None
        new_columns = [
            ("hold_seconds", "INTEGER DEFAULT 0"),
            ("max_favorable_pnl", "TEXT DEFAULT '0'"),
            ("max_adverse_pnl", "TEXT DEFAULT '0'"),
            ("underlying_move_pct", "REAL DEFAULT 0"),
            ("slippage", "REAL DEFAULT 0"),
        ]
        for col_name, col_type in new_columns:
            try:
                self._conn.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")
            except sqlite3.OperationalError:
                pass  # Column already exists
        # Daily reports table for analytics
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_date  TEXT    NOT NULL UNIQUE,
                report_json TEXT    NOT NULL,
                created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
            )
        """)
        self._conn.commit()

    def record_trade_open(
        self, underlying: str, contract_symbol: str, option_type: str,
        strike: Decimal, expiration: str, side: str, contracts: int,
        entry_premium: Decimal, entry_time: datetime, order_id: str = "",
        strategy: str = "", confidence: int = 0,
        greeks_json: str = "", quant_json: str = "",
    ) -> int:
        assert self._conn is not None
        cur = self._conn.execute(
            """INSERT INTO trades (underlying, contract_symbol, option_type, strike,
               expiration, side, contracts, entry_premium, entry_time, order_id,
               strategy, confidence, greeks_at_entry, quant_signals)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (underlying, contract_symbol, option_type, str(strike),
             expiration, side, contracts, str(entry_premium),
             entry_time.isoformat(), order_id, strategy, confidence,
             greeks_json, quant_json),
        )
        self._conn.commit()
        return cur.lastrowid

    def record_trade_close(
        self, contract_symbol: str, exit_premium: Decimal,
        pnl: Decimal, exit_reason: str = "",
        hold_seconds: int = 0,
        max_favorable_pnl: Decimal = Decimal("0"),
        max_adverse_pnl: Decimal = Decimal("0"),
        underlying_move_pct: float = 0.0,
        slippage: float = 0.0,
    ) -> None:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """UPDATE trades SET exit_premium=?, exit_time=?, pnl=?, exit_reason=?,
               hold_seconds=?, max_favorable_pnl=?, max_adverse_pnl=?,
               underlying_move_pct=?, slippage=?
               WHERE contract_symbol=? AND exit_time IS NULL
               ORDER BY entry_time DESC LIMIT 1""",
            (str(exit_premium), now, str(pnl), exit_reason,
             hold_seconds, str(max_favorable_pnl), str(max_adverse_pnl),
             underlying_move_pct, slippage, contract_symbol),
        )
        self._conn.commit()

    def get_trade_history(self, limit: int = 200, offset: int = 0) -> list[dict]:
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_trade_stats(self) -> dict:
        assert self._conn is not None
        row = self._conn.execute("""
            SELECT COUNT(*),
                   SUM(CASE WHEN CAST(pnl AS REAL) > 0 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN CAST(pnl AS REAL) < 0 THEN 1 ELSE 0 END),
                   COALESCE(SUM(CAST(pnl AS REAL)), 0),
                   COALESCE(AVG(CAST(pnl AS REAL)), 0),
                   COALESCE(MAX(CAST(pnl AS REAL)), 0),
                   COALESCE(MIN(CAST(pnl AS REAL)), 0)
            FROM trades WHERE exit_time IS NOT NULL
        """).fetchone()
        total, wins, losses, total_pnl, avg_pnl, best, worst = row
        total = total or 0
        wins = wins or 0
        losses = losses or 0
        return {
            "total_closed": total, "wins": wins, "losses": losses,
            "win_rate": round(wins / total, 4) if total > 0 else 0,
            "total_pnl": round(total_pnl or 0, 2),
            "avg_pnl": round(avg_pnl or 0, 2),
            "best_trade": round(best or 0, 2),
            "worst_trade": round(worst or 0, 2),
        }

    def save_portfolio_state(self, portfolio_value: Decimal, daily_pnl: Decimal,
                             day_start_value: Decimal, daily_trades: int) -> None:
        assert self._conn is not None
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """UPDATE portfolio_state SET portfolio_value=?, daily_pnl=?,
               day_start_value=?, daily_trades=?, updated_at=? WHERE id=1""",
            (str(portfolio_value), str(daily_pnl), str(day_start_value), daily_trades, now),
        )
        self._conn.commit()

    def load_portfolio_state(self) -> Optional[dict]:
        assert self._conn is not None
        cur = self._conn.execute(
            "SELECT portfolio_value, daily_pnl, day_start_value, daily_trades FROM portfolio_state WHERE id=1"
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            "portfolio_value": Decimal(row[0]),
            "daily_pnl": Decimal(row[1]),
            "day_start_value": Decimal(row[2]),
            "daily_trades": int(row[3]),
        }
