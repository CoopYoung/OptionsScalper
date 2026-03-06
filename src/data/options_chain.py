"""Options chain manager: strike selection, Greeks scoring, and contract filtering.

Manages the chain for each underlying, selects optimal strikes based on
delta target, spread quality, and premium bounds.
"""

import logging
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Optional

from src.data.alpaca_client import AlpacaClient
from src.infra.config import Settings
from src.strategy.base import OptionsContract

logger = logging.getLogger(__name__)


@dataclass
class StrikeCandidate:
    """Scored candidate from the chain."""
    contract: OptionsContract
    delta_score: float      # How close to target delta (0-1)
    spread_score: float     # Spread quality (0-1)
    premium_score: float    # Within premium bounds (0-1)
    greeks_score: float     # Greeks quality (0-1)
    total_score: float      # Weighted composite


class OptionsChainManager:
    """Fetches, filters, and scores the options chain for strike selection."""

    def __init__(self, settings: Settings, client: AlpacaClient) -> None:
        self._settings = settings
        self._client = client
        self._chains: dict[str, list[OptionsContract]] = {}
        self._snapshots: dict[str, OptionsContract] = {}

    async def refresh_chain(
        self, underlying: str, expiration: Optional[date] = None,
    ) -> list[OptionsContract]:
        """Fetch fresh chain from Alpaca for today's expiration."""
        exp = expiration or date.today()
        contracts = await self._client.get_options_chain(
            underlying=underlying, expiration=exp,
        )
        self._chains[underlying] = contracts
        logger.info("Chain refreshed: %s has %d contracts", underlying, len(contracts))
        return contracts

    async def refresh_snapshots(self, underlying: str) -> dict[str, OptionsContract]:
        """Fetch real-time snapshots (quotes + Greeks) for the chain."""
        chain = self._chains.get(underlying, [])
        if not chain:
            return {}

        symbols = [c.symbol for c in chain]
        snapshots = await self._client.get_snapshots(symbols)
        self._snapshots.update(snapshots)

        logger.debug("Snapshots refreshed: %d contracts for %s", len(snapshots), underlying)
        return snapshots

    def select_strike(
        self, underlying: str, direction: str, current_price: Decimal,
    ) -> Optional[StrikeCandidate]:
        """Select the best strike for a given direction.

        Args:
            underlying: "SPY", "QQQ", "IWM"
            direction: "call" or "put"
            current_price: Current underlying price

        Returns:
            Best StrikeCandidate or None if no viable strikes.
        """
        chain = self._chains.get(underlying, [])
        if not chain:
            return None

        candidates: list[StrikeCandidate] = []

        for contract in chain:
            if contract.option_type != direction:
                continue

            # Get snapshot with live quotes + Greeks
            snap = self._snapshots.get(contract.symbol)
            if snap:
                contract = snap  # Use snapshot data (has bid/ask/Greeks)

            # Filter: must have valid bid/ask
            if contract.bid <= 0 or contract.ask <= 0:
                continue

            # Filter: premium bounds
            mid = float(contract.mid)
            if mid < self._settings.min_premium or mid > self._settings.max_premium:
                continue

            # Filter: spread quality
            if contract.spread_ratio < self._settings.min_spread_ratio:
                continue

            # Score: delta proximity to target
            delta_diff = abs(abs(contract.delta) - self._settings.target_delta)
            delta_score = max(0.0, 1.0 - delta_diff / 0.30)

            # Score: spread quality (tighter = better)
            spread_score = contract.spread_ratio

            # Score: premium sweet spot (prefer mid-range)
            premium_range = self._settings.max_premium - self._settings.min_premium
            premium_center = (self._settings.min_premium + self._settings.max_premium) / 2
            premium_dist = abs(mid - premium_center) / (premium_range / 2)
            premium_score = max(0.0, 1.0 - premium_dist)

            # Score: Greeks quality
            greeks_score = self._score_greeks(contract, direction)

            # Weighted composite
            total = (
                delta_score * 0.40 +
                spread_score * 0.25 +
                premium_score * 0.15 +
                greeks_score * 0.20
            )

            candidates.append(StrikeCandidate(
                contract=contract,
                delta_score=round(delta_score, 3),
                spread_score=round(spread_score, 3),
                premium_score=round(premium_score, 3),
                greeks_score=round(greeks_score, 3),
                total_score=round(total, 3),
            ))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c.total_score, reverse=True)
        best = candidates[0]
        logger.info(
            "Selected %s %s $%s (score=%.3f, delta=%.2f, spread=%.2f, mid=$%.2f)",
            underlying, direction, best.contract.strike,
            best.total_score, best.contract.delta,
            best.contract.spread_ratio, float(best.contract.mid),
        )
        return best

    def _score_greeks(self, contract: OptionsContract, direction: str) -> float:
        """Score Greeks quality for a candidate contract."""
        score = 0.5  # baseline

        # Prefer contracts with meaningful gamma (profiting from moves)
        if contract.gamma > 0.01:
            score += 0.15
        if contract.gamma > 0.03:
            score += 0.10

        # Theta penalty: more negative theta = more time decay cost
        if contract.theta < -0.05:
            score -= 0.10
        if contract.theta < -0.10:
            score -= 0.10

        # IV: prefer moderate IV (not too expensive, not too cheap)
        if 0.15 <= contract.iv <= 0.40:
            score += 0.15
        elif contract.iv > 0.60:
            score -= 0.10

        # Vega: moderate vega is fine, excessive means IV sensitivity
        if abs(contract.vega) > 0.15:
            score -= 0.05

        return max(0.0, min(1.0, score))

    def get_chain(self, underlying: str) -> list[OptionsContract]:
        return self._chains.get(underlying, [])

    def get_snapshot(self, symbol: str) -> Optional[OptionsContract]:
        return self._snapshots.get(symbol)

    def get_chain_summary(self, underlying: str) -> dict:
        chain = self._chains.get(underlying, [])
        calls = [c for c in chain if c.option_type == "call"]
        puts = [c for c in chain if c.option_type == "put"]
        return {
            "underlying": underlying,
            "total_contracts": len(chain),
            "calls": len(calls),
            "puts": len(puts),
            "snapshots_loaded": sum(1 for c in chain if c.symbol in self._snapshots),
        }
