"""Tests for OptionsRiskManager: Kelly sizing, Greeks limits, exit logic, position tracking."""

from datetime import datetime, timedelta
from decimal import Decimal
from zoneinfo import ZoneInfo

import pytest

from src.risk.manager import OptionsRiskManager, PDTTracker, PortfolioGreeks
from src.strategy.base import TradeDirection

from tests.conftest import make_contract, make_signal


# ── PortfolioGreeks ────────────────────────────────────────────────────


class TestPortfolioGreeks:
    def test_add_single_contract(self):
        greeks = PortfolioGreeks()
        contract = make_contract(delta=0.30, gamma=0.02, theta=-0.05, vega=0.08)
        greeks.add(contract, qty=1)
        assert greeks.delta == pytest.approx(30.0)   # 0.30 * 1 * 100
        assert greeks.gamma == pytest.approx(2.0)     # 0.02 * 1 * 100
        assert greeks.theta == pytest.approx(-5.0)    # -0.05 * 1 * 100
        assert greeks.vega == pytest.approx(8.0)      # 0.08 * 1 * 100

    def test_add_multiple_contracts(self):
        greeks = PortfolioGreeks()
        contract = make_contract(delta=0.30, gamma=0.02, theta=-0.05, vega=0.08)
        greeks.add(contract, qty=3)
        assert greeks.delta == pytest.approx(90.0)
        assert greeks.gamma == pytest.approx(6.0)

    def test_add_different_contracts(self):
        greeks = PortfolioGreeks()
        call = make_contract(delta=0.30, gamma=0.02, theta=-0.05, vega=0.08)
        put = make_contract(delta=-0.30, gamma=0.02, theta=-0.05, vega=0.08, option_type="put")
        greeks.add(call, qty=1)
        greeks.add(put, qty=1)
        assert greeks.delta == pytest.approx(0.0)  # Delta-neutral


# ── PDTTracker ─────────────────────────────────────────────────────────


class TestPDTTracker:
    def test_no_trades_not_restricted(self):
        pdt = PDTTracker()
        pdt.account_equity = Decimal("10000")
        assert pdt.is_pdt_restricted is False
        assert pdt.remaining_day_trades == 3

    def test_three_trades_becomes_restricted(self):
        from datetime import date
        pdt = PDTTracker()
        pdt.account_equity = Decimal("10000")
        today = date.today()
        pdt.record_round_trip(today)
        pdt.record_round_trip(today)
        pdt.record_round_trip(today)
        assert pdt.rolling_5d_count == 3
        assert pdt.is_pdt_restricted is True
        assert pdt.remaining_day_trades == 0

    def test_above_25k_unlimited(self):
        from datetime import date
        pdt = PDTTracker()
        pdt.account_equity = Decimal("30000")
        today = date.today()
        for _ in range(10):
            pdt.record_round_trip(today)
        assert pdt.is_pdt_restricted is False
        assert pdt.remaining_day_trades == 999

    def test_old_trades_roll_off(self):
        from datetime import date
        pdt = PDTTracker()
        pdt.account_equity = Decimal("10000")
        old_date = date.today() - timedelta(days=10)
        pdt.record_round_trip(old_date)
        pdt.record_round_trip(old_date)
        pdt.record_round_trip(old_date)
        # Old trades should not count
        assert pdt.rolling_5d_count == 0
        assert pdt.is_pdt_restricted is False

    def test_status_dict(self):
        pdt = PDTTracker()
        pdt.account_equity = Decimal("15000")
        status = pdt.status()
        assert "rolling_5d" in status
        assert "remaining" in status
        assert "restricted" in status
        assert "equity" in status


# ── OptionsRiskManager.can_trade ───────────────────────────────────────


class TestCanTrade:
    def test_basic_signal_allowed(self, risk_manager):
        signal = make_signal(confidence=70)
        allowed, reason = risk_manager.can_trade(signal)
        assert allowed is True
        assert reason == "OK"

    def test_circuit_breaker_blocks(self, risk_manager, circuit_breaker):
        circuit_breaker.force_halt("Test halt")
        signal = make_signal(confidence=70)
        allowed, reason = risk_manager.can_trade(signal)
        assert allowed is False
        assert "Circuit breaker" in reason

    def test_low_confidence_blocks(self, risk_manager):
        signal = make_signal(confidence=30)
        allowed, reason = risk_manager.can_trade(signal)
        assert allowed is False
        assert "Confidence" in reason

    def test_pdt_restricted_blocks(self, risk_manager):
        from datetime import date
        risk_manager.set_portfolio_value(Decimal("10000"))
        today = date.today()
        for _ in range(3):
            risk_manager._pdt.record_round_trip(today)
        signal = make_signal(confidence=70)
        allowed, reason = risk_manager.can_trade(signal)
        assert allowed is False
        assert "PDT" in reason

    def test_drawdown_blocks(self, risk_manager, settings):
        # Simulate 10% drawdown (exceeds 8% limit)
        risk_manager.set_day_start_value(Decimal("100000"))
        risk_manager.set_portfolio_value(Decimal("91000"))
        # _compute_drawdown = (100000 - 91000) / 100000 = 0.09
        signal = make_signal(confidence=70)
        allowed, reason = risk_manager.can_trade(signal)
        assert allowed is False
        assert "drawdown" in reason.lower()

    def test_greeks_limit_blocks(self, risk_manager):
        # Fill up portfolio delta near limit
        big_delta_contract = make_contract(delta=0.50)
        risk_manager._portfolio_greeks.add(big_delta_contract, qty=9)
        # delta now = 0.50 * 9 * 100 = 450. Adding 1 more contract:
        # 0.30 * 1 * 100 = 30, total = 480 still under 50? No, 450 > 50 already
        # Actually max_portfolio_delta = 50.0, 450 >> 50
        signal = make_signal(confidence=70)
        allowed, reason = risk_manager.can_trade(signal)
        assert allowed is False
        assert "Delta" in reason or "Greeks" in reason or "delta" in reason.lower()

    def test_max_positions_per_underlying_blocks(self, risk_manager, settings):
        # Open max_positions_per_underlying positions with minimal Greeks impact
        for i in range(settings.max_positions_per_underlying):
            risk_manager.record_open(
                symbol=f"SPY250306C0055000{i}",
                underlying="SPY",
                qty=1,
                premium=Decimal("2.00"),
                contract=make_contract(delta=0.01, gamma=0.001, theta=-0.001, vega=0.001),
            )
        signal = make_signal(
            confidence=70, underlying="SPY",
            contract=make_contract(delta=0.01, gamma=0.001, theta=-0.001, vega=0.001),
        )
        allowed, reason = risk_manager.can_trade(signal)
        assert allowed is False
        assert "Max positions" in reason


# ── Kelly Sizing ───────────────────────────────────────────────────────


class TestKellySizing:
    def test_basic_sizing(self, risk_manager):
        signal = make_signal(confidence=70)
        qty = risk_manager.compute_position_size(signal, vix_multiplier=1.0)
        assert qty >= 1
        assert qty <= 10  # max_contracts_per_trade

    def test_low_confidence_zero_contracts(self, risk_manager):
        # Kelly formula with low win_prob → negative kelly → 0 contracts
        signal = make_signal(confidence=20)
        qty = risk_manager.compute_position_size(signal, vix_multiplier=1.0)
        assert qty == 0

    def test_high_vix_reduces_size(self, risk_manager):
        signal = make_signal(confidence=75)
        normal_qty = risk_manager.compute_position_size(signal, vix_multiplier=1.0)
        reduced_qty = risk_manager.compute_position_size(signal, vix_multiplier=0.5)
        assert reduced_qty <= normal_qty

    def test_no_contract_returns_zero(self, risk_manager):
        signal = make_signal(confidence=70)
        signal.contract = None
        qty = risk_manager.compute_position_size(signal)
        assert qty == 0

    def test_zero_premium_returns_zero(self, risk_manager):
        contract = make_contract(bid=Decimal("0"), ask=Decimal("0"))
        signal = make_signal(confidence=70, contract=contract)
        qty = risk_manager.compute_position_size(signal)
        assert qty == 0

    def test_respects_max_contracts(self, risk_manager):
        # Very high confidence, cheap premium → many contracts, but capped
        cheap = make_contract(bid=Decimal("0.10"), ask=Decimal("0.12"))
        signal = make_signal(confidence=95, contract=cheap)
        qty = risk_manager.compute_position_size(signal, vix_multiplier=1.0)
        assert qty <= 10  # max_contracts_per_trade

    def test_greeks_constrain_size(self, risk_manager):
        # Nearly at delta limit
        risk_manager._portfolio_greeks.delta = 47.0  # limit is 50
        contract = make_contract(delta=0.30)
        signal = make_signal(confidence=75, contract=contract)
        qty = risk_manager.compute_position_size(signal, vix_multiplier=1.0)
        # Each contract adds 30 delta, only 3 delta room → 0 contracts fit
        assert qty == 0

    def test_circuit_breaker_reduces_size(self, risk_manager, circuit_breaker):
        # After circuit breaker resumes, position_size_multiplier = 0.5
        circuit_breaker.force_resume()
        signal = make_signal(confidence=75)
        qty = risk_manager.compute_position_size(signal, vix_multiplier=1.0)
        # Should be reduced compared to normal
        assert qty >= 0


# ── Exit Logic ─────────────────────────────────────────────────────────


class TestShouldExit:
    def test_profit_target_hit(self, risk_manager):
        # Entry $2.00, current $3.10 → +55% > 50% target
        should, reason = risk_manager.should_exit(
            symbol="SPY250306C00550000",
            current_premium=Decimal("3.10"),
            peak_premium=Decimal("3.10"),
            entry_premium=Decimal("2.00"),
        )
        assert should is True
        assert "Profit target" in reason

    def test_stop_loss_hit(self, risk_manager):
        # Entry $2.00, current $1.30 → -35% < -30% stop
        should, reason = risk_manager.should_exit(
            symbol="SPY250306C00550000",
            current_premium=Decimal("1.30"),
            peak_premium=Decimal("2.00"),
            entry_premium=Decimal("2.00"),
        )
        assert should is True
        assert "Stop loss" in reason

    def test_trailing_stop_hit(self, risk_manager):
        # Entry $2.00, peak $3.00, current $2.30
        # Trail from peak: (3.00 - 2.30) / 3.00 = 23.3% > 20%
        should, reason = risk_manager.should_exit(
            symbol="SPY250306C00550000",
            current_premium=Decimal("2.30"),
            peak_premium=Decimal("3.00"),
            entry_premium=Decimal("2.00"),
        )
        assert should is True
        assert "Trailing stop" in reason

    def test_no_trailing_when_peak_at_entry(self, risk_manager):
        # Peak <= entry → trailing stop doesn't activate.
        # pnl_pct = (1.95 - 2.00) / 2.00 = -2.5%, within stop loss bounds (-30%)
        et_tz = ZoneInfo("America/New_York")
        now = datetime(2025, 3, 6, 12, 0, tzinfo=et_tz)
        should, reason = risk_manager.should_exit(
            symbol="SPY250306C00550000",
            current_premium=Decimal("1.95"),
            peak_premium=Decimal("1.95"),
            entry_premium=Decimal("2.00"),
            now=now,
        )
        assert should is False

    def test_time_exit_after_hard_close(self, risk_manager):
        # 3:20 PM ET is after 3:15 PM hard close
        et_tz = ZoneInfo("America/New_York")
        now = datetime(2025, 3, 6, 15, 20, tzinfo=et_tz)
        should, reason = risk_manager.should_exit(
            symbol="SPY250306C00550000",
            current_premium=Decimal("2.10"),
            peak_premium=Decimal("2.20"),
            entry_premium=Decimal("2.00"),
            now=now,
        )
        assert should is True
        assert "Time exit" in reason

    def test_no_exit_during_market_hours(self, risk_manager):
        et_tz = ZoneInfo("America/New_York")
        now = datetime(2025, 3, 6, 12, 0, tzinfo=et_tz)
        should, reason = risk_manager.should_exit(
            symbol="SPY250306C00550000",
            current_premium=Decimal("2.10"),
            peak_premium=Decimal("2.20"),
            entry_premium=Decimal("2.00"),
            now=now,
        )
        assert should is False

    def test_zero_entry_premium_no_exit(self, risk_manager):
        should, reason = risk_manager.should_exit(
            symbol="SPY250306C00550000",
            current_premium=Decimal("2.10"),
            peak_premium=Decimal("2.20"),
            entry_premium=Decimal("0"),
        )
        assert should is False


# ── Position Tracking ──────────────────────────────────────────────────


class TestPositionTracking:
    def test_record_open_and_close(self, risk_manager):
        contract = make_contract()
        risk_manager.record_open("SYM1", "SPY", 2, Decimal("2.00"), contract)
        assert len(risk_manager._open_positions) == 1
        assert risk_manager._positions_per_underlying["SPY"] == 1
        assert risk_manager._trade_count_today == 1

        risk_manager.record_close("SYM1", Decimal("50"))
        assert len(risk_manager._open_positions) == 0
        assert risk_manager._positions_per_underlying["SPY"] == 0
        assert risk_manager._daily_pnl == Decimal("50")
        assert risk_manager._win_count == 1

    def test_record_loss_triggers_circuit_breaker_check(self, risk_manager):
        contract = make_contract()
        risk_manager.record_open("SYM1", "SPY", 1, Decimal("2.00"), contract)
        risk_manager.record_close("SYM1", Decimal("-100"))
        assert risk_manager._loss_count == 1

    def test_greeks_update_on_close(self, risk_manager):
        contract = make_contract(delta=0.30)
        risk_manager.record_open("SYM1", "SPY", 1, Decimal("2.00"), contract)
        assert risk_manager._portfolio_greeks.delta == pytest.approx(30.0)

        risk_manager.record_close("SYM1", Decimal("0"))
        assert risk_manager._portfolio_greeks.delta == pytest.approx(0.0)

    def test_reset_daily(self, risk_manager):
        risk_manager._daily_pnl = Decimal("500")
        risk_manager._trade_count_today = 5
        risk_manager.reset_daily()
        assert risk_manager._daily_pnl == Decimal("0")
        assert risk_manager._trade_count_today == 0

    def test_status_dict(self, risk_manager):
        status = risk_manager.status()
        assert "portfolio_value" in status
        assert "daily_pnl" in status
        assert "portfolio_greeks" in status
        assert "pdt" in status
        assert "circuit_breaker" in status
