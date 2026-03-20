"""Gamma Exposure (GEX) analyzer for support/resistance from dealer hedging.

GEX determines how market makers hedge:
    - Positive GEX: dealers buy dips, sell rips → mean-reverting (sticky price)
    - Negative GEX: dealers sell dips, buy rips → trending (accelerating price)

Self-computed from Alpaca options chain data (gamma, OI per strike).
No external API required.

Math:
    Per-strike GEX = gamma × OI × spot² × 0.01 × 100
    Call GEX is positive (dealers long calls → buy underlying on dip)
    Put GEX is negative (dealers short puts → sell underlying on dip)

Key levels:
    Call wall: strike with max(call_gamma × call_OI) — resistance/pinning
    Put wall: strike with max(put_gamma × put_OI) — support/pinning
    Flip point: strike where net GEX crosses zero — regime transition
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from src.infra.config import Settings

logger = logging.getLogger(__name__)


class GEXRegime(str, Enum):
    POSITIVE = "positive"   # Mean-reverting: fade moves, tighter targets
    NEGATIVE = "negative"   # Trending: ride momentum, wider targets
    NEUTRAL = "neutral"


@dataclass
class GEXLevel:
    """A single GEX level (support or resistance)."""
    strike: float
    gex_value: float    # Positive = support/resistance, Negative = acceleration
    is_call_wall: bool  # Major call OI concentration
    is_put_wall: bool   # Major put OI concentration


@dataclass
class GEXSignals:
    """GEX-derived signals for strategy ensemble."""
    regime: GEXRegime
    flip_point: float              # Price where GEX switches sign
    nearest_support: float         # Closest positive GEX below price
    nearest_resistance: float      # Closest positive GEX above price
    key_levels: list[GEXLevel] = field(default_factory=list)
    total_gex: float = 0.0        # Net gamma exposure
    call_wall: float = 0.0        # Largest call gamma×OI strike
    put_wall: float = 0.0         # Largest put gamma×OI strike
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GEXAnalyzer:
    """Self-computed Gamma Exposure from options chain data."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._latest: dict[str, GEXSignals] = {}
        # Rolling history for intensity scoring (z-score normalization)
        self._gex_history: dict[str, deque] = {}
        # Cache current price per underlying for scoring
        self._prices: dict[str, float] = {}

    async def update(self, underlying: str, chain: list = None,
                     current_price: float = 0.0) -> GEXSignals:
        """Update GEX levels for an underlying.

        Primary: compute from Alpaca chain data.
        Fallback: Squeezemetrics API (if configured).
        """
        self._prices[underlying] = current_price

        # Primary: compute from chain data
        if chain:
            signals = self._compute_from_chain(underlying, chain, current_price)
            self._latest[underlying] = signals

            # Track history for intensity scoring
            if underlying not in self._gex_history:
                self._gex_history[underlying] = deque(maxlen=20)
            self._gex_history[underlying].append(signals.total_gex)

            return signals

        # Fallback: Squeezemetrics API (if key configured)
        if self._settings.squeezemetrics_api_key:
            signals = await self._fetch_squeezemetrics(underlying, current_price)
            if signals:
                self._latest[underlying] = signals
                return signals

        return self._make_fallback(underlying, current_price)

    # ── Core GEX Computation ─────────────────────────────────

    def _filter_chain(self, chain: list) -> list:
        """Remove noise contracts before GEX computation.

        Filters: OI > 10, bid > $0.05, sane IV (1%-200%), not deep ITM/OTM.
        """
        filtered = []
        for c in chain:
            oi = getattr(c, 'open_interest', 0) or 0
            if oi <= 10:
                continue

            bid = float(getattr(c, 'bid', 0) or 0)
            if bid <= 0.05:
                continue

            iv = getattr(c, 'iv', 0) or 0
            if iv < 0.01 or iv > 2.0:
                continue

            delta = getattr(c, 'delta', 0) or 0
            if abs(delta) > 0.95:
                continue

            gamma = getattr(c, 'gamma', 0) or 0
            if gamma <= 0:
                continue

            filtered.append(c)
        return filtered

    def _compute_from_chain(
        self, underlying: str, chain: list, spot: float,
    ) -> GEXSignals:
        """Compute GEX from options chain using the full formula.

        Per-strike GEX = gamma × OI × spot² × 0.01 × 100
        Calls contribute positive GEX (dealers long → buy dips).
        Puts contribute negative GEX (dealers short → sell dips).
        """
        filtered = self._filter_chain(chain)

        if not filtered or spot <= 0:
            return self._make_fallback(underlying, spot)

        # Aggregate per-strike
        strike_gex: dict[float, float] = {}
        # Track gamma×OI for wall detection (more accurate than raw OI)
        call_gamma_oi: dict[float, float] = {}
        put_gamma_oi: dict[float, float] = {}

        spot_sq = spot * spot  # Precompute spot²

        for contract in filtered:
            strike = float(contract.strike)
            gamma = contract.gamma
            oi = contract.open_interest
            is_call = contract.option_type == "call"

            # Full GEX formula: gamma × OI × spot² × 0.01 × 100
            # The ×100 is the contract multiplier (each option = 100 shares)
            gex = gamma * oi * spot_sq * 0.01 * 100

            if is_call:
                strike_gex[strike] = strike_gex.get(strike, 0.0) + gex
                call_gamma_oi[strike] = call_gamma_oi.get(strike, 0.0) + gamma * oi
            else:
                strike_gex[strike] = strike_gex.get(strike, 0.0) - gex
                put_gamma_oi[strike] = put_gamma_oi.get(strike, 0.0) + gamma * oi

        # Identify call wall and put wall by gamma×OI (hedging impact)
        call_wall = max(call_gamma_oi, key=call_gamma_oi.get) if call_gamma_oi else spot
        put_wall = max(put_gamma_oi, key=put_gamma_oi.get) if put_gamma_oi else spot

        # Build sorted GEX levels
        levels: list[GEXLevel] = []
        for strike in sorted(strike_gex):
            levels.append(GEXLevel(
                strike=strike,
                gex_value=strike_gex[strike],
                is_call_wall=(strike == call_wall),
                is_put_wall=(strike == put_wall),
            ))

        total_gex = sum(l.gex_value for l in levels)

        # Find key levels
        flip = self._find_flip_point(levels, spot)
        support = self._find_nearest_support(levels, spot)
        resistance = self._find_nearest_resistance(levels, spot)

        regime = GEXRegime.POSITIVE if total_gex > 0 else (
            GEXRegime.NEGATIVE if total_gex < 0 else GEXRegime.NEUTRAL
        )

        signals = GEXSignals(
            regime=regime,
            flip_point=flip,
            nearest_support=support,
            nearest_resistance=resistance,
            key_levels=levels[:30],  # Top 30 levels by strike
            total_gex=round(total_gex, 2),
            call_wall=call_wall,
            put_wall=put_wall,
        )
        logger.info(
            "GEX %s: regime=%s total=%.0f flip=%.2f cwall=%.2f pwall=%.2f "
            "support=%.2f resist=%.2f levels=%d",
            underlying, regime.value, total_gex, flip, call_wall, put_wall,
            support, resistance, len(levels),
        )
        return signals

    # ── Level Detection ──────────────────────────────────────

    def _find_flip_point(self, levels: list[GEXLevel], price: float) -> float:
        """Find where GEX switches sign (regime transition zone).

        Uses linear interpolation between bracketing strikes for accuracy.
        """
        if not levels:
            return price

        best_flip = price
        min_dist = float("inf")

        for i in range(1, len(levels)):
            g1 = levels[i - 1].gex_value
            g2 = levels[i].gex_value
            if g1 * g2 < 0:  # Sign change
                s1 = levels[i - 1].strike
                s2 = levels[i].strike
                # Linear interpolation: where does GEX cross zero?
                denom = abs(g1) + abs(g2)
                if denom > 0:
                    fraction = abs(g1) / denom
                    flip_strike = s1 + fraction * (s2 - s1)
                else:
                    flip_strike = (s1 + s2) / 2

                dist = abs(flip_strike - price)
                if dist < min_dist:
                    min_dist = dist
                    best_flip = flip_strike

        return best_flip

    def _find_nearest_support(self, levels: list[GEXLevel], price: float) -> float:
        """Closest positive GEX level below price (dealer-supported)."""
        supports = [l.strike for l in levels if l.gex_value > 0 and l.strike < price]
        return max(supports) if supports else price * 0.99

    def _find_nearest_resistance(self, levels: list[GEXLevel], price: float) -> float:
        """Closest positive GEX level above price (dealer-resisted)."""
        resistances = [l.strike for l in levels if l.gex_value > 0 and l.strike > price]
        return min(resistances) if resistances else price * 1.01

    # ── Scoring ──────────────────────────────────────────────

    def get_score(self, underlying: str, direction: str) -> float:
        """Score -1 to +1 for the ensemble.

        Three components:
            1. Regime (±0.3): positive GEX = mean-revert, negative = trend
            2. Intensity (±0.3): z-score of current GEX vs rolling history
            3. Proximity (±0.4): distance to call wall, put wall, flip point
        """
        signals = self._latest.get(underlying)
        if not signals:
            return 0.0

        spot = self._prices.get(underlying, 0.0)
        if spot <= 0:
            return 0.0

        score = 0.0

        # ── 1. Regime component (±0.3) ──
        score += self._score_regime(signals, direction)

        # ── 2. Intensity component (±0.3) ──
        score += self._score_intensity(underlying, signals)

        # ── 3. Proximity component (±0.4) ──
        score += self._score_proximity(signals, direction, spot)

        return max(-1.0, min(1.0, score))

    def _score_regime(self, signals: GEXSignals, direction: str) -> float:
        """Regime-based scoring: symmetric between positive and negative GEX.

        Positive GEX = mean-reverting (fade moves, favor reversals).
        Negative GEX = trending (ride momentum, favor continuation).
        Both regimes give a directional tilt, not a blanket bonus.
        """
        if signals.regime == GEXRegime.POSITIVE:
            # Mean-reverting: fading moves works.
            # Calls near support are good (bounce), puts near resistance (fade).
            if direction == "put":
                return 0.15   # Mean-reversion favors fading rallies
            else:
                return -0.05  # Calls fight the pinning effect
        elif signals.regime == GEXRegime.NEGATIVE:
            # Trending: momentum works.
            # Calls in uptrend are good (acceleration), puts in downtrend too.
            # But NOT both at once — need directional agreement from other signals.
            if direction == "call":
                return 0.10   # Trending can favor calls if momentum agrees
            else:
                return 0.10   # Trending can favor puts if momentum agrees
        return 0.0

    def _score_intensity(self, underlying: str, signals: GEXSignals) -> float:
        """Z-score of current total_gex vs rolling history.

        Strong positive GEX = high mean-reversion confidence.
        Strong negative GEX = high trend confidence.
        """
        history = self._gex_history.get(underlying)
        if not history or len(history) < 3:
            return 0.0

        values = list(history)
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(variance) if variance > 0 else 1.0

        if std == 0:
            return 0.0

        z = (signals.total_gex - mean) / std

        # Map z-score to ±0.3 using tanh for smooth compression
        return math.tanh(z * 0.5) * 0.3

    def _score_proximity(self, signals: GEXSignals, direction: str,
                         spot: float) -> float:
        """Score based on distance to key GEX levels.

        Near put wall + call direction → positive (support bounce expected)
        Near call wall + call direction → negative (resistance/pinning)
        Near flip point → mild positive (high vol zone, good for long options)
        """
        if spot <= 0:
            return 0.0

        score = 0.0
        proximity_threshold = spot * 0.003  # 0.3% of spot

        # Distance to call wall
        call_wall_dist = abs(spot - signals.call_wall)
        near_call_wall = call_wall_dist < proximity_threshold

        # Distance to put wall
        put_wall_dist = abs(spot - signals.put_wall)
        near_put_wall = put_wall_dist < proximity_threshold

        # Distance to flip point
        flip_dist = abs(spot - signals.flip_point)
        near_flip = flip_dist < proximity_threshold

        if direction == "call":
            if near_put_wall:
                score += 0.3    # Near support → bounce expected → good for calls
            if near_call_wall:
                score -= 0.2    # Near resistance → pinning expected → bad for calls
        else:  # put
            if near_call_wall:
                score += 0.3    # Near resistance → reversal expected → good for puts
            if near_put_wall:
                score -= 0.2    # Near support → bounce expected → bad for puts

        if near_flip:
            score += 0.1  # High volatility zone → good for long options

        return max(-0.4, min(0.4, score))

    # ── Accessors ────────────────────────────────────────────

    def get_latest(self, underlying: str) -> Optional[GEXSignals]:
        return self._latest.get(underlying)

    def get_target_price(self, underlying: str, direction: str,
                         entry_price: float) -> float:
        """Use GEX levels as profit targets."""
        signals = self._latest.get(underlying)
        if not signals:
            return entry_price * (1.005 if direction == "call" else 0.995)

        if direction == "call":
            return signals.nearest_resistance
        return signals.nearest_support

    def get_stop_price(self, underlying: str, direction: str,
                       entry_price: float) -> float:
        """Use GEX levels as stop loss levels."""
        signals = self._latest.get(underlying)
        if not signals:
            return entry_price * (0.995 if direction == "call" else 1.005)

        if direction == "call":
            return signals.nearest_support
        return signals.nearest_resistance

    # ── Fallbacks ────────────────────────────────────────────

    def _make_fallback(self, underlying: str, price: float) -> GEXSignals:
        signals = GEXSignals(
            regime=GEXRegime.NEUTRAL,
            flip_point=price,
            nearest_support=price * 0.995,
            nearest_resistance=price * 1.005,
        )
        self._latest[underlying] = signals
        return signals

    async def _fetch_squeezemetrics(self, underlying: str,
                                    current_price: float) -> Optional[GEXSignals]:
        """Fetch GEX data from Squeezemetrics API (fallback only)."""
        try:
            import aiohttp

            url = f"https://api.squeezemetrics.com/v1/gex/{underlying}"
            headers = {"Authorization": f"Bearer {self._settings.squeezemetrics_api_key}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers,
                                       timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()

            levels = []
            total_gex = 0.0
            max_call_oi = 0
            max_put_oi = 0
            call_wall = 0.0
            put_wall = 0.0

            for level in data.get("levels", []):
                strike = float(level["strike"])
                gex_val = float(level.get("gex", 0))
                call_oi = int(level.get("call_oi", 0))
                put_oi = int(level.get("put_oi", 0))

                total_gex += gex_val

                if call_oi > max_call_oi:
                    max_call_oi = call_oi
                    call_wall = strike
                if put_oi > max_put_oi:
                    max_put_oi = put_oi
                    put_wall = strike

                levels.append(GEXLevel(
                    strike=strike, gex_value=gex_val,
                    is_call_wall=False, is_put_wall=False,
                ))

            levels.sort(key=lambda l: l.strike)
            flip = self._find_flip_point(levels, current_price)
            support = self._find_nearest_support(levels, current_price)
            resistance = self._find_nearest_resistance(levels, current_price)

            regime = GEXRegime.POSITIVE if total_gex > 0 else (
                GEXRegime.NEGATIVE if total_gex < 0 else GEXRegime.NEUTRAL
            )

            return GEXSignals(
                regime=regime, flip_point=flip,
                nearest_support=support, nearest_resistance=resistance,
                key_levels=levels[:20], total_gex=round(total_gex, 2),
                call_wall=call_wall, put_wall=put_wall,
            )
        except Exception:
            logger.exception("Squeezemetrics fetch failed for %s", underlying)
            return None
