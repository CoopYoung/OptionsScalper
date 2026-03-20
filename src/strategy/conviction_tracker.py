"""Conviction tracking: monitors whether the original trade thesis holds.

When we enter a trade, we snapshot the signal breakdown (8 factor scores).
On every exit check, we re-evaluate the current signal. If conviction has
degraded significantly, we exit — even if price hasn't hit stop loss.

This replaces the purely mechanical exit logic (wait for -18% stop or
+20% target) with thesis-aware exits:
    - If 3+ factors flip against us → exit immediately
    - If overall conviction drops below entry threshold → exit
    - If the direction signal reverses → exit immediately
"""

import logging
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ConvictionSnapshot:
    """Signal state at time of entry."""
    direction: str                    # "call" or "put"
    confidence: int                   # 0-100
    score_breakdown: dict             # Per-factor scores at entry
    underlying: str
    entry_time_utc: str


@dataclass
class ConvictionCheck:
    """Result of re-evaluating conviction."""
    should_exit: bool
    reason: str
    urgency: str = "normal"           # "normal", "urgent", "immediate"
    conviction_pct: float = 100.0     # Current conviction as % of entry


class ConvictionTracker:
    """Tracks conviction for open positions and signals when thesis breaks.

    Stored per-position (keyed by contract symbol).
    """

    # If conviction drops below this % of entry confidence, exit
    CONVICTION_FLOOR_PCT = 0.50       # Exit if conviction < 50% of entry
    # If this many factors flip sign, exit
    MAX_FLIPPED_FACTORS = 3
    # Minimum confidence to stay in trade (absolute)
    MIN_ABSOLUTE_CONFIDENCE = 40

    def __init__(self) -> None:
        self._snapshots: dict[str, ConvictionSnapshot] = {}

    def record_entry(
        self,
        symbol: str,
        direction: str,
        confidence: int,
        score_breakdown: dict,
        underlying: str,
    ) -> None:
        """Snapshot the signal state at entry."""
        from datetime import datetime, timezone
        self._snapshots[symbol] = ConvictionSnapshot(
            direction=direction,
            confidence=confidence,
            score_breakdown=dict(score_breakdown),
            underlying=underlying,
            entry_time_utc=datetime.now(timezone.utc).isoformat(),
        )
        logger.info(
            "Conviction snapshot for %s: dir=%s conf=%d factors=%s",
            symbol, direction, confidence,
            {k: f"{v:+.2f}" for k, v in score_breakdown.items()},
        )

    def remove(self, symbol: str) -> None:
        """Remove tracking for a closed position."""
        self._snapshots.pop(symbol, None)

    def evaluate(
        self,
        symbol: str,
        current_direction: str,
        current_confidence: int,
        current_breakdown: dict,
    ) -> ConvictionCheck:
        """Re-evaluate conviction against entry snapshot.

        Call this on every exit check with fresh signal scores.
        """
        snap = self._snapshots.get(symbol)
        if not snap:
            return ConvictionCheck(False, "No snapshot")

        # 1. Direction reversal — immediate exit
        if current_direction and current_direction != snap.direction:
            return ConvictionCheck(
                True,
                f"Direction reversed: {snap.direction} → {current_direction}",
                urgency="immediate",
                conviction_pct=0.0,
            )

        # 2. Absolute confidence floor
        if current_confidence < self.MIN_ABSOLUTE_CONFIDENCE:
            return ConvictionCheck(
                True,
                f"Confidence collapsed: {current_confidence} < {self.MIN_ABSOLUTE_CONFIDENCE}",
                urgency="urgent",
                conviction_pct=current_confidence / snap.confidence * 100 if snap.confidence else 0,
            )

        # 3. Conviction decay — current vs entry confidence
        if snap.confidence > 0:
            conviction_pct = current_confidence / snap.confidence * 100
            if conviction_pct < self.CONVICTION_FLOOR_PCT * 100:
                return ConvictionCheck(
                    True,
                    f"Conviction faded: {current_confidence} vs entry {snap.confidence} "
                    f"({conviction_pct:.0f}% < {self.CONVICTION_FLOOR_PCT * 100:.0f}%)",
                    urgency="normal",
                    conviction_pct=conviction_pct,
                )
        else:
            conviction_pct = 100.0

        # 4. Factor flip count — how many factors changed sign
        flipped = 0
        for factor, entry_score in snap.score_breakdown.items():
            current_score = current_breakdown.get(factor, 0)
            if entry_score != 0 and current_score != 0:
                # Check if sign flipped (was positive, now negative or vice versa)
                if (entry_score > 0.05 and current_score < -0.05) or \
                   (entry_score < -0.05 and current_score > 0.05):
                    flipped += 1

        if flipped >= self.MAX_FLIPPED_FACTORS:
            return ConvictionCheck(
                True,
                f"{flipped} factors flipped against position "
                f"(threshold: {self.MAX_FLIPPED_FACTORS})",
                urgency="urgent",
                conviction_pct=conviction_pct,
            )

        return ConvictionCheck(
            False,
            f"Conviction OK ({conviction_pct:.0f}%, {flipped} flipped)",
            conviction_pct=conviction_pct,
        )

    def get_snapshot(self, symbol: str) -> Optional[ConvictionSnapshot]:
        return self._snapshots.get(symbol)

    def status(self) -> dict:
        return {
            sym: {
                "direction": s.direction,
                "entry_confidence": s.confidence,
                "factors": s.score_breakdown,
            }
            for sym, s in self._snapshots.items()
        }
