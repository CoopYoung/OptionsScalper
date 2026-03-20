"""Setup detection: requires signal confirmation before entry.

Instead of entering on every strategy loop tick, this module requires:
    1. Signal must cross threshold (setup "forming")
    2. Signal must STAY above threshold for N consecutive evaluations (confirmation)
    3. Signal direction must be consistent (no flip-flopping)
    4. After a trade, cooldown period before same underlying can re-enter
    5. Signal must be STRENGTHENING, not weakening (momentum of conviction)

This prevents the machine-gun entry pattern where the bot bought
355 contracts in 2.5 hours on 2026-03-20.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SetupState:
    """Tracks the formation state of a potential trade setup."""
    underlying: str
    direction: str = ""              # "call" or "put"
    consecutive_confirms: int = 0    # How many consecutive evals above threshold
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))
    first_signal_time: Optional[datetime] = None
    last_signal_time: Optional[datetime] = None
    last_direction: str = ""
    cooldown_until: Optional[datetime] = None
    trades_today: int = 0

    @property
    def is_confirmed(self) -> bool:
        """Setup confirmed: 3+ consecutive signals in same direction."""
        return self.consecutive_confirms >= 3

    @property
    def is_strengthening(self) -> bool:
        """Signal is getting stronger, not weaker."""
        if len(self.confidence_history) < 2:
            return True  # Not enough data, allow
        recent = list(self.confidence_history)
        # Average of last 2 should be >= average of prior 2
        if len(recent) >= 4:
            recent_avg = (recent[-1] + recent[-2]) / 2
            prior_avg = (recent[-3] + recent[-4]) / 2
            return recent_avg >= prior_avg - 5  # Allow small dips (5 pts)
        return recent[-1] >= recent[-2] - 5

    @property
    def is_cooling_down(self) -> bool:
        if not self.cooldown_until:
            return False
        return datetime.now(timezone.utc) < self.cooldown_until

    @property
    def avg_confidence(self) -> float:
        if not self.confidence_history:
            return 0
        return sum(self.confidence_history) / len(self.confidence_history)


class SetupDetector:
    """Manages setup detection and confirmation for all underlyings.

    Entry requirements:
        1. Signal above threshold for 3 consecutive evaluations (~45s at 15s intervals)
        2. Direction consistent (all 3 must agree: call or put)
        3. Signal strengthening or stable (not fading)
        4. Not in cooldown from previous trade
        5. Max 3 trades per underlying per day

    This turns the strategy from "fire on every tick" to "wait for
    confirmed setups with conviction."
    """

    # Configuration
    REQUIRED_CONFIRMS = 3        # Consecutive signals needed
    COOLDOWN_MINUTES = 10        # Minutes between trades on same underlying
    MAX_TRADES_PER_UNDERLYING = 3  # Daily cap per underlying
    MIN_CONFIDENCE_INCREASE = -5   # Allow small dips, block fading signals

    def __init__(self) -> None:
        self._setups: dict[str, SetupState] = {}

    def get_or_create(self, underlying: str) -> SetupState:
        if underlying not in self._setups:
            self._setups[underlying] = SetupState(underlying=underlying)
        return self._setups[underlying]

    def record_signal(
        self, underlying: str, direction: str, confidence: int,
    ) -> None:
        """Record a signal evaluation. Call this every strategy loop."""
        setup = self.get_or_create(underlying)
        now = datetime.now(timezone.utc)

        if direction and confidence > 0:
            # Active signal
            if direction == setup.last_direction:
                # Same direction — build confirmation
                setup.consecutive_confirms += 1
            else:
                # Direction changed — reset
                setup.consecutive_confirms = 1
                setup.first_signal_time = now
                setup.direction = direction

            setup.last_direction = direction
            setup.confidence_history.append(confidence)
            setup.last_signal_time = now
        else:
            # No signal / HOLD — decay confirmation
            if setup.consecutive_confirms > 0:
                setup.consecutive_confirms -= 1
            if setup.consecutive_confirms == 0:
                setup.direction = ""
                setup.last_direction = ""

    def should_enter(self, underlying: str) -> tuple[bool, str]:
        """Check if a confirmed setup exists and entry is allowed.

        Returns (should_enter, reason).
        """
        setup = self.get_or_create(underlying)

        # Cooldown check
        if setup.is_cooling_down:
            remaining = (setup.cooldown_until - datetime.now(timezone.utc)).total_seconds()
            return False, f"Cooldown ({remaining:.0f}s remaining)"

        # Daily trade limit
        if setup.trades_today >= self.MAX_TRADES_PER_UNDERLYING:
            return False, f"Daily limit ({setup.trades_today}/{self.MAX_TRADES_PER_UNDERLYING})"

        # Confirmation check
        if not setup.is_confirmed:
            return False, (
                f"Setup forming ({setup.consecutive_confirms}/{self.REQUIRED_CONFIRMS} confirms, "
                f"dir={setup.direction or 'none'})"
            )

        # Strength check
        if not setup.is_strengthening:
            return False, "Signal fading (conviction weakening)"

        return True, "Setup confirmed"

    def record_trade(self, underlying: str) -> None:
        """Called after a trade is placed. Starts cooldown."""
        setup = self.get_or_create(underlying)
        setup.cooldown_until = datetime.now(timezone.utc) + timedelta(
            minutes=self.COOLDOWN_MINUTES
        )
        setup.trades_today += 1
        # Reset confirmation — force re-confirmation for next trade
        setup.consecutive_confirms = 0
        setup.confidence_history.clear()
        logger.info(
            "Trade recorded for %s (#%d today), cooldown until %s",
            underlying, setup.trades_today,
            setup.cooldown_until.strftime("%H:%M:%S UTC"),
        )

    def record_exit(self, underlying: str) -> None:
        """Called when a position is closed. Extends cooldown slightly."""
        setup = self.get_or_create(underlying)
        # After exit, enforce minimum cooldown before re-entry
        min_cooldown = datetime.now(timezone.utc) + timedelta(minutes=5)
        if not setup.cooldown_until or setup.cooldown_until < min_cooldown:
            setup.cooldown_until = min_cooldown

    def reset_daily(self) -> None:
        """Reset daily counters."""
        for setup in self._setups.values():
            setup.trades_today = 0
            setup.consecutive_confirms = 0
            setup.confidence_history.clear()
            setup.cooldown_until = None

    def status(self) -> dict:
        return {
            underlying: {
                "direction": s.direction,
                "confirms": s.consecutive_confirms,
                "required": self.REQUIRED_CONFIRMS,
                "confirmed": s.is_confirmed,
                "strengthening": s.is_strengthening,
                "avg_confidence": round(s.avg_confidence, 1),
                "cooling_down": s.is_cooling_down,
                "trades_today": s.trades_today,
            }
            for underlying, s in self._setups.items()
        }
