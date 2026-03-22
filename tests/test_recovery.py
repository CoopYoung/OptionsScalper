"""Tests for crash recovery: position persistence and reconciliation."""

import sqlite3
import tempfile
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.engine import TradingEngine
from src.data.trade_db import TradeDB


@pytest.fixture
def db():
    """Create an in-memory TradeDB for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db") as f:
        trade_db = TradeDB(f.name)
        trade_db.connect()
        yield trade_db
        trade_db.close()


class TestPositionPersistence:
    def test_save_and_load_position(self, db):
        now = datetime.now(timezone.utc)
        db.save_open_position(
            contract_symbol="SPY260320C00550000",
            underlying="SPY",
            option_type="call",
            strike="550",
            side="BUY_CALL",
            contracts=2,
            entry_premium=Decimal("2.50"),
            entry_time=now,
            order_id="ord123",
            confidence=72,
            peak_premium=Decimal("2.80"),
            entry_spot=549.5,
            peak_spot=551.0,
        )

        positions = db.load_open_positions()
        assert len(positions) == 1
        pos = positions[0]
        assert pos["contract_symbol"] == "SPY260320C00550000"
        assert pos["underlying"] == "SPY"
        assert pos["option_type"] == "call"
        assert pos["contracts"] == 2
        assert pos["entry_premium"] == "2.50"
        assert pos["entry_confidence"] == 72
        assert pos["peak_premium"] == "2.80"
        assert pos["entry_spot"] == 549.5
        assert pos["peak_spot"] == 551.0

    def test_remove_position(self, db):
        now = datetime.now(timezone.utc)
        db.save_open_position(
            contract_symbol="SPY260320C00550000",
            underlying="SPY",
            option_type="call",
            strike="550",
            side="BUY_CALL",
            contracts=1,
            entry_premium=Decimal("2.00"),
            entry_time=now,
        )
        db.save_open_position(
            contract_symbol="QQQ260320P00450000",
            underlying="QQQ",
            option_type="put",
            strike="450",
            side="BUY_PUT",
            contracts=3,
            entry_premium=Decimal("1.50"),
            entry_time=now,
        )

        db.remove_open_position("SPY260320C00550000")
        positions = db.load_open_positions()
        assert len(positions) == 1
        assert positions[0]["contract_symbol"] == "QQQ260320P00450000"

    def test_clear_all_positions(self, db):
        now = datetime.now(timezone.utc)
        db.save_open_position(
            contract_symbol="SPY260320C00550000",
            underlying="SPY",
            option_type="call",
            strike="550",
            side="BUY_CALL",
            contracts=1,
            entry_premium=Decimal("2.00"),
            entry_time=now,
        )
        db.save_open_position(
            contract_symbol="QQQ260320P00450000",
            underlying="QQQ",
            option_type="put",
            strike="450",
            side="BUY_PUT",
            contracts=1,
            entry_premium=Decimal("1.00"),
            entry_time=now,
        )

        db.clear_open_positions()
        assert len(db.load_open_positions()) == 0

    def test_upsert_updates_peak(self, db):
        """save_open_position with same symbol should update (INSERT OR REPLACE)."""
        now = datetime.now(timezone.utc)
        db.save_open_position(
            contract_symbol="SPY260320C00550000",
            underlying="SPY",
            option_type="call",
            strike="550",
            side="BUY_CALL",
            contracts=2,
            entry_premium=Decimal("2.00"),
            entry_time=now,
            peak_premium=Decimal("2.00"),
        )
        # Update with higher peak
        db.save_open_position(
            contract_symbol="SPY260320C00550000",
            underlying="SPY",
            option_type="call",
            strike="550",
            side="BUY_CALL",
            contracts=2,
            entry_premium=Decimal("2.00"),
            entry_time=now,
            peak_premium=Decimal("3.00"),
        )

        positions = db.load_open_positions()
        assert len(positions) == 1
        assert positions[0]["peak_premium"] == "3.00"

    def test_empty_load(self, db):
        """Loading with no positions returns empty list."""
        assert db.load_open_positions() == []


class TestParseUnderlying:
    def test_spy(self):
        assert TradingEngine._parse_underlying("SPY260320C00550000") == "SPY"

    def test_qqq(self):
        assert TradingEngine._parse_underlying("QQQ260320P00450000") == "QQQ"

    def test_iwm(self):
        assert TradingEngine._parse_underlying("IWM260320C00200000") == "IWM"

    def test_spx(self):
        assert TradingEngine._parse_underlying("SPXW260320C05500000") == "SPXW"
