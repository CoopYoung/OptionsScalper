"""Tests for ExitManager: backtest-proven exit logic for 0DTE options.

Validates the exit hierarchy:
    1. Hard time close
    2. Max hold time (15 min)
    3. Catastrophic stop (35%)
    4. Time-scaled profit target
    5. Directional trailing (50% retracement)
    6. Time-based loser management (underwater at 10+ min)
    7. Time take profit (>2% at 10+ min)
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from src.risk.exit_manager import ExitContext, ExitDecision, ExitManager
from tests.conftest import make_contract


@pytest.fixture
def exit_mgr(settings):
    return ExitManager(settings)


def _make_ctx(
    entry_premium: float = 2.00,
    current_premium: float = 2.00,
    peak_premium: float = 2.00,
    entry_spot: float = 550.0,
    current_spot: float = 550.0,
    peak_spot: float = 550.0,
    hold_minutes: float = 0,
    direction: str = "call",
) -> tuple[ExitContext, datetime]:
    """Create an ExitContext and matching 'now' time."""
    entry_time = datetime(2026, 3, 20, 11, 0, tzinfo=None)  # 11:00 AM ET
    now = entry_time + timedelta(minutes=hold_minutes)
    ctx = ExitContext(
        symbol="SPY260320C00550000",
        current_premium=Decimal(str(current_premium)),
        entry_premium=Decimal(str(entry_premium)),
        peak_premium=Decimal(str(peak_premium)),
        entry_time=entry_time,
        entry_spot=entry_spot,
        current_spot=current_spot,
        peak_spot=peak_spot,
        contract=make_contract(),
        direction=direction,
    )
    return ctx, now


class TestHardClose:
    def test_exits_after_hard_close_time(self, exit_mgr):
        ctx, _ = _make_ctx()
        # 15:16 ET = after 15:15 hard close
        late = datetime(2026, 3, 20, 15, 16)
        decision = exit_mgr.evaluate(ctx, now=late)
        assert decision.should_exit
        assert "Time exit" in decision.reason
        assert decision.urgency == "immediate"

    def test_does_not_exit_before_hard_close(self, exit_mgr):
        ctx, now = _make_ctx(hold_minutes=1)
        decision = exit_mgr.evaluate(ctx, now=now)
        assert not decision.should_exit


class TestMaxHoldTime:
    def test_exits_at_max_hold(self, exit_mgr):
        ctx, now = _make_ctx(hold_minutes=16)
        decision = exit_mgr.evaluate(ctx, now=now)
        assert decision.should_exit
        assert "Max hold" in decision.reason

    def test_does_not_exit_before_max_hold(self, exit_mgr):
        ctx, now = _make_ctx(hold_minutes=10)
        decision = exit_mgr.evaluate(ctx, now=now)
        # Might exit for other reasons, but not max hold
        if decision.should_exit:
            assert "Max hold" not in decision.reason


class TestCatastrophicStop:
    def test_exits_on_35pct_loss(self, exit_mgr):
        ctx, now = _make_ctx(current_premium=1.20, hold_minutes=2)
        # 1.20 / 2.00 = -40% → below -35%
        decision = exit_mgr.evaluate(ctx, now=now)
        assert decision.should_exit
        assert "Catastrophic" in decision.reason
        assert decision.urgency == "immediate"

    def test_no_exit_on_25pct_loss(self, exit_mgr):
        ctx, now = _make_ctx(current_premium=1.50, hold_minutes=2)
        # 1.50 / 2.00 = -25% → above -35%
        decision = exit_mgr.evaluate(ctx, now=now)
        # Should NOT exit for catastrophic (may exit for other reasons)
        if decision.should_exit:
            assert "Catastrophic" not in decision.reason


class TestNoTightStops:
    """Verify that tight stops were removed per backtest findings."""

    def test_no_exit_on_8pct_loss_at_3min(self, exit_mgr):
        """8% loss at 3 min should NOT trigger exit (was quick stop, now removed)."""
        ctx, now = _make_ctx(current_premium=1.84, hold_minutes=3)
        decision = exit_mgr.evaluate(ctx, now=now)
        assert not decision.should_exit

    def test_no_exit_on_15pct_loss_at_5min(self, exit_mgr):
        """15% loss at 5 min should NOT trigger exit (no standard stop)."""
        ctx, now = _make_ctx(current_premium=1.70, hold_minutes=5)
        decision = exit_mgr.evaluate(ctx, now=now)
        assert not decision.should_exit

    def test_no_exit_on_20pct_loss_at_8min(self, exit_mgr):
        """20% loss at 8 min should NOT trigger exit (below catastrophic)."""
        ctx, now = _make_ctx(current_premium=1.60, hold_minutes=8)
        decision = exit_mgr.evaluate(ctx, now=now)
        # Should exit for time-based loser management (10+ min), not stop
        # At 8 min it shouldn't exit yet
        assert not decision.should_exit


class TestTimeBasedLoserManagement:
    def test_exits_underwater_at_10min(self, exit_mgr):
        ctx, now = _make_ctx(current_premium=1.90, hold_minutes=11)
        decision = exit_mgr.evaluate(ctx, now=now)
        assert decision.should_exit
        assert "Time stop" in decision.reason

    def test_no_exit_if_profitable_at_10min(self, exit_mgr):
        ctx, now = _make_ctx(current_premium=2.10, hold_minutes=11)
        decision = exit_mgr.evaluate(ctx, now=now)
        # Should take profit, not stop
        if decision.should_exit:
            assert "stop" not in decision.reason.lower() or "Time take" in decision.reason


class TestTimeTakeProfit:
    def test_takes_profit_at_10min_with_3pct_gain(self, exit_mgr):
        ctx, now = _make_ctx(current_premium=2.10, hold_minutes=11)
        decision = exit_mgr.evaluate(ctx, now=now)
        assert decision.should_exit
        assert "take profit" in decision.reason.lower() or "Profit" in decision.reason


class TestProfitTarget:
    def test_takes_profit_at_configured_target(self, exit_mgr):
        # Default target is 50% at full scale before noon
        ctx, now = _make_ctx(current_premium=3.10, hold_minutes=5)
        decision = exit_mgr.evaluate(ctx, now=now)
        assert decision.should_exit
        assert "Profit" in decision.reason


class TestDirectionalTrailing:
    def test_trails_on_50pct_retracement(self, exit_mgr):
        # Call: entered at 550, peaked at 552, now back to 551
        # peak_move = 2.0, current_move = 1.0, retracement = 50%
        ctx, now = _make_ctx(
            entry_spot=550.0, peak_spot=552.0, current_spot=551.0,
            current_premium=2.05, hold_minutes=5,
        )
        decision = exit_mgr.evaluate(ctx, now=now)
        assert decision.should_exit
        assert "trail" in decision.reason.lower()

    def test_no_trail_on_30pct_retracement(self, exit_mgr):
        # peak_move = 2.0, current_move = 1.4, retracement = 30%
        ctx, now = _make_ctx(
            entry_spot=550.0, peak_spot=552.0, current_spot=551.4,
            current_premium=2.05, hold_minutes=5,
        )
        decision = exit_mgr.evaluate(ctx, now=now)
        if decision.should_exit:
            assert "trail" not in decision.reason.lower()

    def test_put_direction_trailing(self, exit_mgr):
        # Put: entered at 550, peaked at 548 (favorable), now back to 549
        # peak_move = 2.0, current_move = 1.0, retracement = 50%
        ctx, now = _make_ctx(
            entry_spot=550.0, peak_spot=548.0, current_spot=549.0,
            current_premium=2.05, hold_minutes=5, direction="put",
        )
        decision = exit_mgr.evaluate(ctx, now=now)
        assert decision.should_exit
        assert "trail" in decision.reason.lower()
