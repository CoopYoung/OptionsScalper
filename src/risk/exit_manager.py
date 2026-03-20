"""Dynamic exit management for 0DTE options positions.

Replaces fixed percentage stops with time-aware exit logic designed for
the unique characteristics of same-day expiration options:
    - Theta accelerates exponentially as expiry approaches
    - High gamma creates rapid premium swings near ATM
    - Holding too long erodes edge faster than in multi-day trades

Key insight from backtesting (14 iterations, 2026-03-19):
    Tight stops HURT 0DTE profitability. Real-time option prices swing
    ~7% per 2-min bar from noise alone. Any stop under 25% is triggered
    by noise, not genuine adverse moves. Removing quick/standard stops
    improved profit factor from 0.65 → 0.95. Trades that hit tight stops
    often recovered when given the full hold time.

Exit hierarchy (checked in order):
    1. Hard time close (15:15 ET default)
    2. Max hold time (absolute limit)
    3. Catastrophic stop (35% premium loss — tail risk only)
    4. Profit target (time-scaled)
    5. Directional trailing stop (underlying-based, 50% retracement)
    6. Time-based loser management (if underwater at 10+ min, exit)
    7. Profitable hold timeout (take profits before theta decay)
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo

from src.infra.config import Settings
from src.strategy.base import OptionsContract

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


@dataclass
class ExitContext:
    """All data needed to evaluate exit conditions for a position."""
    symbol: str
    current_premium: Decimal
    entry_premium: Decimal
    peak_premium: Decimal
    entry_time: datetime
    entry_spot: float              # Underlying price at entry
    current_spot: float            # Current underlying price
    peak_spot: float               # Peak favorable underlying price
    contract: Optional[OptionsContract] = None  # Greeks at entry
    direction: str = "call"        # "call" or "put"


@dataclass
class ExitDecision:
    """Result of exit evaluation."""
    should_exit: bool
    reason: str
    urgency: str = "normal"  # "normal", "urgent", "immediate"


class ExitManager:
    """Dynamic exit logic for 0DTE options positions."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

        # Time-scaled profit targets (ET hour thresholds)
        # Before noon: full target; noon-1:30: reduced; 1:30-2:30: aggressive; after 2:30: take anything
        self._pt_schedule = [
            (12, 0, 1.0),     # Before noon: 100% of configured target
            (13, 30, 0.60),   # Noon–1:30 PM: 60% of target
            (14, 30, 0.30),   # 1:30–2:30 PM: 30% of target
            (15, 15, 0.10),   # 2:30–3:15 PM: 10% of target (take any profit)
        ]

        # Max hold times (minutes) — backtest-proven values
        self._max_hold_profitable = 15   # Exit winners before theta eats gains
        self._max_hold_absolute = 15     # Hard limit — 0DTE theta is exponential past this

        # Catastrophic stop (always active, never adjusted)
        # Backtest showed any stop <25% is noise. 35% catches real disasters.
        self._catastrophic_stop_pct = 0.35

    def evaluate(self, ctx: ExitContext, now: Optional[datetime] = None) -> ExitDecision:
        """Evaluate all exit conditions for a position.

        Returns ExitDecision with should_exit, reason, and urgency level.
        """
        now = now or datetime.now(ET)
        et_now = now.astimezone(ET) if now.tzinfo else now

        if ctx.entry_premium <= 0:
            return ExitDecision(False, "")

        pnl_pct = float((ctx.current_premium - ctx.entry_premium) / ctx.entry_premium)
        hold_minutes = (now - ctx.entry_time).total_seconds() / 60

        # 1. Hard time close
        decision = self._check_time_close(et_now)
        if decision.should_exit:
            return decision

        # 2. Max hold time (absolute)
        if hold_minutes >= self._max_hold_absolute:
            return ExitDecision(
                True,
                f"Max hold time ({self._max_hold_absolute}min)",
                urgency="urgent",
            )

        # 3. Catastrophic stop (immediate, no adjustment)
        if pnl_pct <= -self._catastrophic_stop_pct:
            return ExitDecision(
                True,
                f"Catastrophic stop ({pnl_pct:.1%} <= -{self._catastrophic_stop_pct:.0%})",
                urgency="immediate",
            )

        # 4. Time-scaled profit target
        decision = self._check_profit_target(et_now, pnl_pct)
        if decision.should_exit:
            return decision

        # 5. Directional trailing stop (underlying-based, 50% retracement)
        decision = self._check_directional_trail(ctx, pnl_pct)
        if decision.should_exit:
            return decision

        # 6. Time-based loser management
        # No tight stops — they're noise on 0DTE. Instead, if still underwater
        # at 10+ minutes, the momentum thesis has expired. Exit.
        if hold_minutes >= 10 and pnl_pct < 0:
            return ExitDecision(
                True,
                f"Time stop ({pnl_pct:.1%} underwater @ {hold_minutes:.0f}min)",
                urgency="normal",
            )

        # 7. Profitable hold timeout — take profit before theta eats it
        if pnl_pct > 0.02 and hold_minutes >= 10:
            return ExitDecision(
                True,
                f"Time take profit ({pnl_pct:.1%} @ {hold_minutes:.0f}min)",
                urgency="normal",
            )

        return ExitDecision(False, "")

    def _check_time_close(self, et_now: datetime) -> ExitDecision:
        """Hard close at configured time (default 15:15 ET)."""
        close_h, close_m = map(int, self._settings.hard_close.split(":"))
        if et_now.hour > close_h or (et_now.hour == close_h and et_now.minute >= close_m):
            return ExitDecision(
                True,
                f"Time exit ({self._settings.hard_close} ET)",
                urgency="immediate",
            )
        return ExitDecision(False, "")

    def _check_profit_target(self, et_now: datetime, pnl_pct: float) -> ExitDecision:
        """Time-scaled profit target — take less profit as day progresses."""
        base_target = self._settings.pt_profit_target_pct

        # Find applicable scale factor based on current time
        scale = 1.0
        for hour, minute, s in self._pt_schedule:
            if et_now.hour < hour or (et_now.hour == hour and et_now.minute < minute):
                scale = s
                break
        else:
            scale = self._pt_schedule[-1][2]  # After last threshold

        # Find the previous threshold's scale for the label
        effective_target = base_target * scale

        if pnl_pct >= effective_target:
            return ExitDecision(
                True,
                f"Profit target ({pnl_pct:.1%} >= {effective_target:.1%} "
                f"[{scale:.0%} of {base_target:.0%} @ {et_now.strftime('%H:%M')} ET])",
                urgency="normal",
            )
        return ExitDecision(False, "")

    def _check_directional_trail(self, ctx: ExitContext, pnl_pct: float) -> ExitDecision:
        """Trail based on underlying price retracing from peak favorable move.

        Uses the underlying price (clean signal) rather than noisy option premium.
        This avoids false stops from bid/ask bounce and theta decay.
        """
        if ctx.entry_spot <= 0 or ctx.current_spot <= 0 or ctx.peak_spot <= 0:
            # Fall back to premium-based trailing if no spot data
            return self._check_premium_trail(ctx)

        # Compute favorable underlying move from entry
        if ctx.direction == "call":
            peak_move = ctx.peak_spot - ctx.entry_spot
            current_move = ctx.current_spot - ctx.entry_spot
        else:
            peak_move = ctx.entry_spot - ctx.peak_spot  # Puts profit when price falls
            current_move = ctx.entry_spot - ctx.current_spot

        # Only trail if we had a meaningful favorable move
        min_favorable_move_pct = 0.001  # 0.1% of underlying
        if peak_move <= ctx.entry_spot * min_favorable_move_pct:
            return ExitDecision(False, "")

        # How much of the favorable move has been given back?
        if peak_move > 0:
            retracement = 1.0 - (current_move / peak_move) if current_move < peak_move else 0.0
        else:
            retracement = 0.0

        # Trail trigger: exit if >50% of favorable move retraced
        trail_threshold = 0.50
        if retracement >= trail_threshold and pnl_pct > -0.08:
            return ExitDecision(
                True,
                f"Directional trail ({retracement:.0%} retraced, "
                f"underlying peak move ${peak_move:.2f})",
                urgency="normal",
            )

        return ExitDecision(False, "")

    def _check_premium_trail(self, ctx: ExitContext) -> ExitDecision:
        """Fallback: standard premium-based trailing stop."""
        if ctx.peak_premium > ctx.entry_premium:
            trail_pct = float(
                (ctx.peak_premium - ctx.current_premium) / ctx.peak_premium
            )
            if trail_pct >= self._settings.sl_trailing_pct:
                return ExitDecision(
                    True,
                    f"Trailing stop ({trail_pct:.1%} from peak ${float(ctx.peak_premium):.2f})",
                    urgency="normal",
                )
        return ExitDecision(False, "")

    # NOTE: Greeks-aware stop loss was removed after backtest analysis (2026-03-19).
    # 14 iterations showed tight stops (20-45% based on gamma) are triggered by
    # noise, not real adverse moves. Catastrophic stop (35%) + time-based exit
    # at 10+ min produce better results (PF 0.95→1.07).
