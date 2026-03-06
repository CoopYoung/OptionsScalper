"""Circuit breaker: auto-halt trading on drawdown or consecutive loss thresholds.

Forked from poly-trader — fully generic, no changes needed.
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from src.infra.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class LossEvent:
    timestamp: datetime
    pnl: Decimal


class CircuitBreaker:
    """Monitors for dangerous conditions and halts trading."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._halted = False
        self._halt_reason: str = ""
        self._halt_time: Optional[datetime] = None
        self._resume_time: Optional[datetime] = None
        self._recent_losses: deque[LossEvent] = deque(maxlen=20)
        self._position_size_multiplier: Decimal = Decimal("1.0")
        self._halt_count: int = 0

    @property
    def is_halted(self) -> bool:
        if not self._halted:
            return False
        now = datetime.now(timezone.utc)
        if self._resume_time and now >= self._resume_time:
            self._resume()
            return False
        return True

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    @property
    def position_size_multiplier(self) -> Decimal:
        return self._position_size_multiplier

    def record_loss(self, pnl: Decimal) -> None:
        if pnl >= 0:
            return
        self._recent_losses.append(LossEvent(
            timestamp=datetime.now(timezone.utc), pnl=pnl,
        ))
        self._check_consecutive_losses()

    def check_drawdown(self, daily_drawdown: Decimal) -> None:
        if daily_drawdown >= self._settings.daily_drawdown_halt:
            self._trigger(
                f"Daily drawdown {float(daily_drawdown):.2%} exceeds "
                f"limit {float(self._settings.daily_drawdown_halt):.2%}"
            )

    def _check_consecutive_losses(self) -> None:
        max_losses = self._settings.max_consecutive_losses_window
        window_minutes = self._settings.consecutive_loss_window_minutes
        if len(self._recent_losses) < max_losses:
            return
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=window_minutes)
        recent = [loss for loss in self._recent_losses if loss.timestamp >= cutoff]
        if len(recent) >= max_losses:
            self._trigger(f"{len(recent)} losses within {window_minutes} minutes")

    def _trigger(self, reason: str) -> None:
        now = datetime.now(timezone.utc)
        cooldown = timedelta(hours=self._settings.circuit_breaker_cooldown_hours)
        self._halted = True
        self._halt_reason = reason
        self._halt_time = now
        self._resume_time = now + cooldown
        self._halt_count += 1
        logger.warning("CIRCUIT BREAKER: %s. Halted until %s", reason, self._resume_time.isoformat())

    def _resume(self) -> None:
        self._halted = False
        self._position_size_multiplier = Decimal("0.5")
        logger.info("Circuit breaker cooldown expired. Resuming at %.0f%% size",
                     float(self._position_size_multiplier) * 100)
        self._halt_reason = ""
        self._halt_time = None
        self._resume_time = None

    def force_halt(self, reason: str = "Manual halt") -> None:
        self._halted = True
        self._halt_reason = reason
        self._halt_time = datetime.now(timezone.utc)
        self._resume_time = None
        self._halt_count += 1

    def force_resume(self) -> None:
        self._halted = False
        self._position_size_multiplier = Decimal("0.5")
        self._halt_reason = ""

    def reset(self) -> None:
        self._halted = False
        self._halt_reason = ""
        self._halt_time = None
        self._resume_time = None
        self._recent_losses.clear()
        self._position_size_multiplier = Decimal("1.0")
        self._halt_count = 0

    def status(self) -> dict:
        return {
            "halted": self.is_halted,
            "reason": self._halt_reason,
            "halt_time": self._halt_time.isoformat() if self._halt_time else None,
            "resume_time": self._resume_time.isoformat() if self._resume_time else None,
            "position_multiplier": str(self._position_size_multiplier),
            "total_halts": self._halt_count,
            "recent_losses": len(self._recent_losses),
        }
