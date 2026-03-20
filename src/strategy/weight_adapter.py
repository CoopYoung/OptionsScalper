"""Adaptive ensemble weight adjustment based on realized factor performance.

Instead of static weights, this module tracks which factors actually predict
profitable trades and adjusts weights proportionally. Uses rolling Sharpe
ratio per factor as the performance metric.

Key design choices:
    - Conservative smoothing (70/30 old/new) prevents whiplash
    - Minimum weight floor (2%) keeps all factors in play for evaluation
    - Maximum per-cycle change cap (±20%) prevents regime overfit
    - Requires minimum trade count before adjusting (default: 50)
    - Disabled by default — enabled after sufficient paper trading data
"""

import json
import logging
from collections import defaultdict
from typing import Optional

from src.analytics.trade_analyzer import TradeAnalyzer, FACTOR_NAMES
from src.infra.config import Settings

logger = logging.getLogger(__name__)

# Default weights from config (used as initial values)
DEFAULT_WEIGHT_MAP = {
    "technical": "weight_technical",
    "tick_momentum": "weight_tick_momentum",
    "gex": "weight_gex",
    "flow": "weight_flow",
    "vix": "weight_vix",
    "internals": "weight_internals",
    "sentiment": "weight_sentiment",
    "optionsai": "weight_optionsai",
}


class WeightAdapter:
    """Adapts ensemble weights based on realized factor Sharpe ratios.

    Usage:
        adapter = WeightAdapter(settings, analyzer)
        # In strategy evaluation:
        weight = adapter.get_weight("technical")
        # At end of day:
        adapter.maybe_recalibrate()
    """

    def __init__(
        self,
        settings: Settings,
        analyzer: TradeAnalyzer,
        min_trades: int = 50,
        smoothing: float = 0.7,
        min_weight: float = 0.02,
        max_change_pct: float = 0.20,
    ) -> None:
        self._settings = settings
        self._analyzer = analyzer
        self._min_trades = min_trades
        self._smoothing = smoothing      # 0.7 = 70% old, 30% new
        self._min_weight = min_weight
        self._max_change_pct = max_change_pct
        self._enabled = getattr(settings, "adaptive_weights", False)
        self._trade_count_at_last_calibration = 0

        # Initialize weights from config
        self._weights: dict[str, float] = {}
        for factor, config_attr in DEFAULT_WEIGHT_MAP.items():
            self._weights[factor] = getattr(settings, config_attr, 0.1)

        self._calibration_history: list[dict] = []

    def get_weight(self, factor: str) -> float:
        """Get the current weight for a factor."""
        return self._weights.get(factor, 0.1)

    @property
    def weights(self) -> dict[str, float]:
        """Get all current weights."""
        return dict(self._weights)

    @property
    def is_adapted(self) -> bool:
        """True if weights have been adapted from defaults."""
        return len(self._calibration_history) > 0

    def maybe_recalibrate(self) -> Optional[dict]:
        """Recalibrate weights if enough new trades have accumulated.

        Called at end of day. Returns the calibration result if performed,
        None if skipped.
        """
        if not self._enabled:
            return None

        # Get rolling Sharpe per factor
        sharpe_by_factor = self._analyzer.get_rolling_factor_sharpe(
            lookback_trades=self._min_trades,
        )

        # Check if we have enough data
        total_trades = sum(
            1 for _ in self._analyzer._get_recent_closed_trades(days=60)
        )

        if total_trades < self._min_trades:
            logger.info(
                "Weight adapter: insufficient trades (%d/%d), skipping",
                total_trades, self._min_trades,
            )
            return None

        # Skip if no new trades since last calibration
        if total_trades <= self._trade_count_at_last_calibration:
            return None

        # Compute new weights from Sharpe ratios
        new_weights = self._compute_weights_from_sharpe(sharpe_by_factor)

        # Apply smoothing and caps
        adjusted = self._apply_smoothing_and_caps(new_weights)

        # Log the adjustment
        old_weights = dict(self._weights)
        self._weights = adjusted
        self._trade_count_at_last_calibration = total_trades

        result = {
            "trade_count": total_trades,
            "sharpe_by_factor": sharpe_by_factor,
            "old_weights": old_weights,
            "new_weights": adjusted,
            "changes": {
                f: round(adjusted[f] - old_weights.get(f, 0), 4)
                for f in FACTOR_NAMES
            },
        }
        self._calibration_history.append(result)

        logger.info(
            "Weight adapter recalibrated (%d trades): %s",
            total_trades,
            {f: f"{adjusted[f]:.3f}" for f in FACTOR_NAMES},
        )

        return result

    def _compute_weights_from_sharpe(self, sharpe: dict[str, float]) -> dict[str, float]:
        """Convert Sharpe ratios to weight proportions.

        Factors with positive Sharpe get weight proportional to their Sharpe.
        Factors with zero/negative Sharpe get minimum weight.
        """
        raw = {}
        for factor in FACTOR_NAMES:
            s = sharpe.get(factor, 0)
            if s > 0:
                raw[factor] = s
            else:
                raw[factor] = 0.0  # Will get min_weight floor

        total_positive = sum(raw.values())

        weights = {}
        if total_positive > 0:
            # Distribute (1 - n*min_weight) among positive-Sharpe factors
            available = 1.0 - len(FACTOR_NAMES) * self._min_weight
            for factor in FACTOR_NAMES:
                if raw[factor] > 0:
                    weights[factor] = self._min_weight + available * (raw[factor] / total_positive)
                else:
                    weights[factor] = self._min_weight
        else:
            # All factors are bad — keep equal weights
            equal = 1.0 / len(FACTOR_NAMES)
            weights = {f: equal for f in FACTOR_NAMES}

        return weights

    def _apply_smoothing_and_caps(self, new_weights: dict[str, float]) -> dict[str, float]:
        """Apply exponential smoothing and cap maximum change per cycle."""
        adjusted = {}
        for factor in FACTOR_NAMES:
            old = self._weights.get(factor, 0.1)
            new = new_weights.get(factor, 0.1)

            # Exponential smoothing
            blended = self._smoothing * old + (1 - self._smoothing) * new

            # Cap maximum change
            max_change = old * self._max_change_pct
            if abs(blended - old) > max_change:
                if blended > old:
                    blended = old + max_change
                else:
                    blended = old - max_change

            # Enforce minimum
            blended = max(self._min_weight, blended)
            adjusted[factor] = blended

        # Normalize to sum to 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {f: w / total for f, w in adjusted.items()}

        return {f: round(w, 6) for f, w in adjusted.items()}

    def format_status(self) -> str:
        """Format current weight status for display."""
        lines = ["*Ensemble Weights*"]
        if self.is_adapted:
            lines.append(f"(adapted, {len(self._calibration_history)} calibrations)")
        else:
            lines.append("(default — not yet adapted)")

        for factor in FACTOR_NAMES:
            w = self._weights.get(factor, 0)
            lines.append(f"  {factor}: {w:.1%}")

        return "\n".join(lines)
