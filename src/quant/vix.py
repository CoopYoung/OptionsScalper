"""VIX regime detection, IV percentile, realized vs implied volatility.

Data sources: yfinance (^VIX), historical IV for percentile computation.
Updates every 60s during market hours.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.infra.config import Settings

logger = logging.getLogger(__name__)


class VIXRegime(str, Enum):
    LOW_VOL = "low_vol"      # VIX < 15
    NORMAL = "normal"         # 15-25
    HIGH_VOL = "high_vol"     # 25-35
    CRISIS = "crisis"         # > 35


@dataclass
class VIXSignals:
    """VIX-derived signals for strategy ensemble."""
    vix_level: float               # Current VIX value
    regime: VIXRegime
    iv_percentile: float           # 0-100: where IV sits vs last 252 days
    iv_rank: float                 # 0-100: relative to 52-week range
    rv_iv_spread: float            # Realized vol minus implied vol
    vix_roc: float                 # Rate of change (% over last session)
    size_multiplier: float         # Position size adjustment
    should_trade: bool
    block_reason: str
    updated_at: datetime


class VIXRegimeDetector:
    """VIX-based volatility regime detection and IV analysis."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._latest: Optional[VIXSignals] = None
        self._vix_history: list[float] = []
        self._last_vix: float = 0.0

    async def update(self) -> VIXSignals:
        """Fetch current VIX and compute regime signals."""
        try:
            import yfinance as yf

            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1y", interval="1d")

            if hist.empty:
                return self._make_fallback("No VIX data available")

            current_vix = float(hist["Close"].iloc[-1])
            closes = hist["Close"].values.tolist()

            # VIX regime classification
            regime = self._classify_regime(current_vix)

            # IV Percentile: % of days in last 252 where VIX was lower
            lookback = closes[-252:] if len(closes) >= 252 else closes
            below_count = sum(1 for v in lookback if v < current_vix)
            iv_percentile = (below_count / len(lookback)) * 100

            # IV Rank: (current - 52wk_low) / (52wk_high - 52wk_low)
            week52 = closes[-252:] if len(closes) >= 252 else closes
            vix_high = max(week52)
            vix_low = min(week52)
            iv_range = vix_high - vix_low
            iv_rank = ((current_vix - vix_low) / iv_range * 100) if iv_range > 0 else 50.0

            # Realized vs Implied spread
            # Use SPY realized vol as proxy
            rv_iv_spread = self._compute_rv_iv_spread(current_vix)

            # VIX rate of change
            prev_vix = closes[-2] if len(closes) >= 2 else current_vix
            vix_roc = ((current_vix - prev_vix) / prev_vix * 100) if prev_vix > 0 else 0.0

            # Size multiplier based on regime
            size_multiplier = self._get_size_multiplier(current_vix, regime)

            # Should we trade?
            should_trade = True
            block_reason = ""
            if regime == VIXRegime.CRISIS:
                should_trade = False
                block_reason = f"VIX crisis ({current_vix:.1f} > {self._settings.vix_crisis_threshold})"
            elif iv_percentile > self._settings.iv_percentile_max:
                should_trade = False
                block_reason = f"IV percentile too high ({iv_percentile:.0f}% > {self._settings.iv_percentile_max}%)"

            self._last_vix = current_vix
            self._vix_history = closes

            signals = VIXSignals(
                vix_level=round(current_vix, 2),
                regime=regime,
                iv_percentile=round(iv_percentile, 1),
                iv_rank=round(iv_rank, 1),
                rv_iv_spread=round(rv_iv_spread, 2),
                vix_roc=round(vix_roc, 2),
                size_multiplier=round(size_multiplier, 2),
                should_trade=should_trade,
                block_reason=block_reason,
                updated_at=datetime.now(timezone.utc),
            )
            self._latest = signals
            logger.info(
                "VIX=%.1f regime=%s IV_pctile=%.0f%% size_mult=%.2f trade=%s",
                current_vix, regime.value, iv_percentile, size_multiplier, should_trade,
            )
            return signals

        except Exception:
            logger.exception("VIX update failed")
            return self._make_fallback("VIX fetch error")

    def _classify_regime(self, vix: float) -> VIXRegime:
        if vix > self._settings.vix_crisis_threshold:
            return VIXRegime.CRISIS
        if vix > self._settings.vix_high_threshold:
            return VIXRegime.HIGH_VOL
        if vix < self._settings.vix_low_threshold:
            return VIXRegime.LOW_VOL
        return VIXRegime.NORMAL

    def _get_size_multiplier(self, vix: float, regime: VIXRegime) -> float:
        if regime == VIXRegime.CRISIS:
            return 0.0
        if regime == VIXRegime.HIGH_VOL:
            return 0.5
        if regime == VIXRegime.LOW_VOL:
            return 1.3
        return 1.0

    def _compute_rv_iv_spread(self, current_vix: float) -> float:
        """Estimate realized vol vs implied vol spread."""
        try:
            import yfinance as yf
            import numpy as np

            spy = yf.Ticker("SPY")
            hist = spy.history(period="30d", interval="1d")
            if len(hist) < 5:
                return 0.0

            returns = hist["Close"].pct_change().dropna().values
            realized_vol = float(np.std(returns) * np.sqrt(252) * 100)
            return realized_vol - current_vix

        except Exception:
            return 0.0

    def _make_fallback(self, reason: str) -> VIXSignals:
        return VIXSignals(
            vix_level=self._last_vix or 20.0,
            regime=VIXRegime.NORMAL,
            iv_percentile=50.0,
            iv_rank=50.0,
            rv_iv_spread=0.0,
            vix_roc=0.0,
            size_multiplier=1.0,
            should_trade=True,
            block_reason=reason,
            updated_at=datetime.now(timezone.utc),
        )

    @property
    def latest(self) -> Optional[VIXSignals]:
        return self._latest

    def get_score(self) -> float:
        """Return a -1 to +1 score for the ensemble.

        Negative = bearish environment (high vol, IV expensive)
        Positive = bullish environment (low vol, IV cheap)
        """
        if not self._latest:
            return 0.0

        score = 0.0
        regime = self._latest.regime

        # Regime contribution
        if regime == VIXRegime.LOW_VOL:
            score += 0.3   # Calm markets favor long options
        elif regime == VIXRegime.NORMAL:
            score += 0.0
        elif regime == VIXRegime.HIGH_VOL:
            score -= 0.3
        elif regime == VIXRegime.CRISIS:
            score -= 0.8

        # VIX ROC: spiking VIX = fear = potentially contrarian bullish
        if self._latest.vix_roc > 10:
            score -= 0.2  # Rapid spike = danger
        elif self._latest.vix_roc < -5:
            score += 0.1  # Falling VIX = complacency

        # RV-IV spread: positive means options cheap (buy), negative means expensive
        if self._latest.rv_iv_spread > 3:
            score += 0.2
        elif self._latest.rv_iv_spread < -5:
            score -= 0.2

        return max(-1.0, min(1.0, score))
