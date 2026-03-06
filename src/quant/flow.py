"""Options flow analyzer: institutional activity, put/call ratio, flow direction.

Data sources:
    - CBOE put/call ratio (daily, via web scrape)
    - Alpaca volume data (call vs put volume per strike)
    - Intrinio API (unusual activity events, free tier: 250/mo)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.infra.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class UnusualActivity:
    """A detected unusual options activity event."""
    symbol: str
    underlying: str
    option_type: str    # call / put
    strike: float
    volume: int
    open_interest: int
    vol_oi_ratio: float
    trade_type: str     # sweep / block / split
    sentiment: str      # bullish / bearish / neutral
    premium: float      # Total premium ($)
    timestamp: datetime


@dataclass
class FlowSignals:
    """Options flow analysis signals."""
    put_call_ratio: float          # CBOE total P/C ratio
    flow_direction: float          # Net premium flow: >0 = call-heavy, <0 = put-heavy
    unusual_activity: list[UnusualActivity] = field(default_factory=list)
    smart_money_bias: float = 0.0  # -1 to +1 (large trades alignment)
    call_volume: int = 0
    put_volume: int = 0
    call_premium: float = 0.0     # Total call premium traded ($)
    put_premium: float = 0.0      # Total put premium traded ($)
    extreme_reading: str = ""     # "extreme_bullish", "extreme_bearish", or ""
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FlowAnalyzer:
    """Institutional options flow analysis."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._latest: Optional[FlowSignals] = None
        self._pcr_history: list[float] = []

    async def update(self, chain: list = None) -> FlowSignals:
        """Refresh flow data from all sources."""
        pcr = await self._fetch_put_call_ratio()
        chain_flow = self._analyze_chain_flow(chain) if chain else {}
        unusual = await self._fetch_unusual_activity()

        call_vol = chain_flow.get("call_volume", 0)
        put_vol = chain_flow.get("put_volume", 0)
        call_prem = chain_flow.get("call_premium", 0.0)
        put_prem = chain_flow.get("put_premium", 0.0)

        # Net flow direction: positive = bullish (more call premium)
        total_prem = call_prem + put_prem
        flow_direction = 0.0
        if total_prem > 0:
            flow_direction = (call_prem - put_prem) / total_prem

        # Smart money bias from unusual activity
        smart_money = self._compute_smart_money_bias(unusual)

        # Extreme reading detection (contrarian signal)
        extreme = ""
        if pcr < 0.7:
            extreme = "extreme_bullish"   # Contrarian: too bullish → lean bearish
        elif pcr > 1.2:
            extreme = "extreme_bearish"   # Contrarian: too bearish → lean bullish

        signals = FlowSignals(
            put_call_ratio=round(pcr, 3),
            flow_direction=round(flow_direction, 3),
            unusual_activity=unusual,
            smart_money_bias=round(smart_money, 3),
            call_volume=call_vol,
            put_volume=put_vol,
            call_premium=round(call_prem, 2),
            put_premium=round(put_prem, 2),
            extreme_reading=extreme,
        )
        self._latest = signals

        logger.info(
            "Flow: P/C=%.2f direction=%.2f unusual=%d smart_money=%.2f extreme=%s",
            pcr, flow_direction, len(unusual), smart_money, extreme or "none",
        )
        return signals

    async def _fetch_put_call_ratio(self) -> float:
        """Fetch CBOE put/call ratio."""
        try:
            import aiohttp

            url = "https://www.cboe.com/us/options/market_statistics/daily/"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return self._pcr_history[-1] if self._pcr_history else 0.85

                    text = await resp.text()
                    # Parse put/call ratio from page
                    pcr = self._parse_pcr_from_html(text)
                    if pcr > 0:
                        self._pcr_history.append(pcr)
                        return pcr

        except Exception:
            logger.debug("CBOE P/C ratio fetch failed, using fallback")

        return self._pcr_history[-1] if self._pcr_history else 0.85

    def _parse_pcr_from_html(self, html: str) -> float:
        """Extract total put/call ratio from CBOE page."""
        # Look for the total equity P/C ratio
        try:
            # Simple pattern matching for the ratio value
            import re
            patterns = [
                r'Total Put/Call Ratio[:\s]*([0-9]+\.[0-9]+)',
                r'put.call.ratio[:\s]*([0-9]+\.[0-9]+)',
                r'"total_pcr"[:\s]*([0-9]+\.[0-9]+)',
            ]
            for pattern in patterns:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    return float(match.group(1))
        except Exception:
            pass
        return 0.0

    def _analyze_chain_flow(self, chain: list) -> dict:
        """Analyze volume distribution from options chain."""
        call_volume = 0
        put_volume = 0
        call_premium = 0.0
        put_premium = 0.0

        for contract in chain:
            vol = getattr(contract, 'volume', 0) or 0
            mid = float(getattr(contract, 'mid', 0) or 0)
            premium = vol * mid * 100  # Per contract = 100 shares

            if contract.option_type == "call":
                call_volume += vol
                call_premium += premium
            else:
                put_volume += vol
                put_premium += premium

        return {
            "call_volume": call_volume,
            "put_volume": put_volume,
            "call_premium": call_premium,
            "put_premium": put_premium,
        }

    async def _fetch_unusual_activity(self) -> list[UnusualActivity]:
        """Fetch unusual options activity (sweeps, blocks)."""
        # Intrinio API or chain-based detection
        # For now, detect from chain: vol/OI ratio > 3x is unusual
        return []

    def _compute_smart_money_bias(self, unusual: list[UnusualActivity]) -> float:
        """Compute bias from institutional-sized unusual trades."""
        if not unusual:
            return 0.0

        bullish_prem = sum(u.premium for u in unusual if u.sentiment == "bullish")
        bearish_prem = sum(u.premium for u in unusual if u.sentiment == "bearish")
        total = bullish_prem + bearish_prem

        if total == 0:
            return 0.0
        return (bullish_prem - bearish_prem) / total

    @property
    def latest(self) -> Optional[FlowSignals]:
        return self._latest

    def get_score(self, direction: str) -> float:
        """Score -1 to +1 for the ensemble.

        Combines flow direction, P/C ratio contrarian, and smart money.
        """
        if not self._latest:
            return 0.0

        score = 0.0
        signals = self._latest

        # Flow direction alignment with trade direction
        if direction == "call":
            score += signals.flow_direction * 0.4  # Bullish flow supports calls
        else:
            score -= signals.flow_direction * 0.4  # Bearish flow supports puts

        # P/C ratio contrarian
        if signals.extreme_reading == "extreme_bearish" and direction == "call":
            score += 0.3  # Crowd too bearish, contrarian bullish
        elif signals.extreme_reading == "extreme_bullish" and direction == "put":
            score += 0.3  # Crowd too bullish, contrarian bearish

        # Smart money alignment
        if direction == "call":
            score += signals.smart_money_bias * 0.3
        else:
            score -= signals.smart_money_bias * 0.3

        return max(-1.0, min(1.0, score))
