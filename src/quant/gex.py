"""Gamma Exposure (GEX) analyzer for support/resistance from dealer hedging.

GEX determines how market makers hedge:
    - Positive GEX: dealers buy dips, sell rips → mean-reverting (sticky price)
    - Negative GEX: dealers sell dips, buy rips → trending (accelerating price)

Data source: Squeezemetrics API ($10/mo) or computed from options chain data.
Updates every 5 minutes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
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
    call_wall: float = 0.0        # Largest call OI strike
    put_wall: float = 0.0         # Largest put OI strike
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class GEXAnalyzer:
    """Gamma Exposure level analysis for support/resistance."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._latest: dict[str, GEXSignals] = {}

    async def update(self, underlying: str, chain: list = None, current_price: float = 0.0) -> GEXSignals:
        """Update GEX levels for an underlying.

        Tries Squeezemetrics API first, falls back to chain-based estimation.
        """
        if self._settings.squeezemetrics_api_key:
            signals = await self._fetch_squeezemetrics(underlying, current_price)
            if signals:
                self._latest[underlying] = signals
                return signals

        # Fallback: estimate GEX from options chain data
        if chain:
            signals = self._estimate_from_chain(underlying, chain, current_price)
            self._latest[underlying] = signals
            return signals

        return self._make_fallback(underlying, current_price)

    async def _fetch_squeezemetrics(self, underlying: str, current_price: float) -> Optional[GEXSignals]:
        """Fetch GEX data from Squeezemetrics API."""
        try:
            import aiohttp

            url = f"https://api.squeezemetrics.com/v1/gex/{underlying}"
            headers = {"Authorization": f"Bearer {self._settings.squeezemetrics_api_key}"}

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("Squeezemetrics API returned %d for %s", resp.status, underlying)
                        return None

                    data = await resp.json()

            levels = []
            total_gex = 0.0
            call_wall = 0.0
            put_wall = 0.0
            max_call_oi = 0
            max_put_oi = 0

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
                    strike=strike,
                    gex_value=gex_val,
                    is_call_wall=(call_oi > max_call_oi * 0.8),
                    is_put_wall=(put_oi > max_put_oi * 0.8),
                ))

            # Sort by strike
            levels.sort(key=lambda l: l.strike)

            # Find flip point, support, resistance
            flip = self._find_flip_point(levels, current_price)
            support = self._find_nearest_support(levels, current_price)
            resistance = self._find_nearest_resistance(levels, current_price)

            regime = GEXRegime.POSITIVE if total_gex > 0 else (
                GEXRegime.NEGATIVE if total_gex < 0 else GEXRegime.NEUTRAL
            )

            signals = GEXSignals(
                regime=regime,
                flip_point=flip,
                nearest_support=support,
                nearest_resistance=resistance,
                key_levels=levels[:20],  # Top 20 levels
                total_gex=round(total_gex, 2),
                call_wall=call_wall,
                put_wall=put_wall,
            )
            logger.info(
                "GEX %s: regime=%s flip=%.2f support=%.2f resistance=%.2f",
                underlying, regime.value, flip, support, resistance,
            )
            return signals

        except Exception:
            logger.exception("Squeezemetrics fetch failed for %s", underlying)
            return None

    def _estimate_from_chain(
        self, underlying: str, chain: list, current_price: float,
    ) -> GEXSignals:
        """Estimate GEX from options chain OI + Greeks (no external API needed)."""
        levels: list[GEXLevel] = []
        strike_gex: dict[float, float] = {}
        max_call_oi_strike = (0.0, 0)
        max_put_oi_strike = (0.0, 0)

        for contract in chain:
            strike = float(contract.strike)
            gamma = contract.gamma if hasattr(contract, 'gamma') else 0.0
            oi = contract.open_interest if hasattr(contract, 'open_interest') else 0
            is_call = contract.option_type == "call"

            # GEX = gamma * OI * 100 * price^2 * 0.01
            # Simplified: gamma * OI (directional contribution)
            multiplier = 1 if is_call else -1
            gex_contribution = gamma * oi * multiplier * 100

            strike_gex[strike] = strike_gex.get(strike, 0) + gex_contribution

            if is_call and oi > max_call_oi_strike[1]:
                max_call_oi_strike = (strike, oi)
            elif not is_call and oi > max_put_oi_strike[1]:
                max_put_oi_strike = (strike, oi)

        # Build GEX levels
        for strike, gex_val in sorted(strike_gex.items()):
            levels.append(GEXLevel(
                strike=strike,
                gex_value=gex_val,
                is_call_wall=(strike == max_call_oi_strike[0]),
                is_put_wall=(strike == max_put_oi_strike[0]),
            ))

        total_gex = sum(l.gex_value for l in levels)
        flip = self._find_flip_point(levels, current_price)
        support = self._find_nearest_support(levels, current_price)
        resistance = self._find_nearest_resistance(levels, current_price)

        regime = GEXRegime.POSITIVE if total_gex > 0 else (
            GEXRegime.NEGATIVE if total_gex < 0 else GEXRegime.NEUTRAL
        )

        signals = GEXSignals(
            regime=regime,
            flip_point=flip,
            nearest_support=support,
            nearest_resistance=resistance,
            key_levels=levels[:20],
            total_gex=round(total_gex, 2),
            call_wall=max_call_oi_strike[0],
            put_wall=max_put_oi_strike[0],
        )
        logger.info(
            "GEX (chain-est) %s: regime=%s levels=%d support=%.2f resistance=%.2f",
            underlying, regime.value, len(levels), support, resistance,
        )
        return signals

    def _find_flip_point(self, levels: list[GEXLevel], price: float) -> float:
        """Find the price where GEX switches sign (most volatile zone)."""
        if not levels:
            return price

        # Look for sign change nearest to current price
        best_flip = price
        min_dist = float("inf")

        for i in range(1, len(levels)):
            prev_gex = levels[i - 1].gex_value
            curr_gex = levels[i].gex_value
            if prev_gex * curr_gex < 0:  # Sign change
                flip_strike = (levels[i - 1].strike + levels[i].strike) / 2
                dist = abs(flip_strike - price)
                if dist < min_dist:
                    min_dist = dist
                    best_flip = flip_strike

        return best_flip

    def _find_nearest_support(self, levels: list[GEXLevel], price: float) -> float:
        """Closest positive GEX level below price."""
        supports = [l.strike for l in levels if l.gex_value > 0 and l.strike < price]
        return max(supports) if supports else price * 0.99

    def _find_nearest_resistance(self, levels: list[GEXLevel], price: float) -> float:
        """Closest positive GEX level above price."""
        resistances = [l.strike for l in levels if l.gex_value > 0 and l.strike > price]
        return min(resistances) if resistances else price * 1.01

    def _make_fallback(self, underlying: str, price: float) -> GEXSignals:
        return GEXSignals(
            regime=GEXRegime.NEUTRAL,
            flip_point=price,
            nearest_support=price * 0.995,
            nearest_resistance=price * 1.005,
        )

    def get_latest(self, underlying: str) -> Optional[GEXSignals]:
        return self._latest.get(underlying)

    def get_score(self, underlying: str, direction: str) -> float:
        """Score -1 to +1 for the ensemble.

        Positive GEX regime favors mean reversion.
        Negative GEX regime favors trend following.
        """
        signals = self._latest.get(underlying)
        if not signals:
            return 0.0

        score = 0.0

        if signals.regime == GEXRegime.POSITIVE:
            # Mean-reverting: lean contrarian
            score += 0.2 if direction == "put" else -0.1
        elif signals.regime == GEXRegime.NEGATIVE:
            # Trending: lean with momentum
            score += 0.2

        return max(-1.0, min(1.0, score))

    def get_target_price(self, underlying: str, direction: str, entry_price: float) -> float:
        """Use GEX levels as profit targets."""
        signals = self._latest.get(underlying)
        if not signals:
            return entry_price * (1.005 if direction == "call" else 0.995)

        if direction == "call":
            return signals.nearest_resistance
        return signals.nearest_support

    def get_stop_price(self, underlying: str, direction: str, entry_price: float) -> float:
        """Use GEX levels as stop loss levels."""
        signals = self._latest.get(underlying)
        if not signals:
            return entry_price * (0.995 if direction == "call" else 1.005)

        if direction == "call":
            return signals.nearest_support
        return signals.nearest_resistance
