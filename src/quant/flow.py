"""Options flow analyzer: volume, premium, delta exposure, OI analysis.

Self-computed from Alpaca options chain data. No external API required.

Signals derived:
    - Put/call ratio (from chain volume, CBOE scrape as fallback)
    - Flow direction (net call vs put premium)
    - Volume-weighted delta exposure (institutional directional bias)
    - OI concentration skew (call vs put open interest balance)
    - Unusual activity detection (vol/OI spikes = institutional sweeps)
    - Smart money bias (large unusual trades alignment)
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
    put_call_ratio: float          # P/C ratio from volume
    flow_direction: float          # Net premium flow: >0 = call-heavy, <0 = put-heavy
    unusual_activity: list[UnusualActivity] = field(default_factory=list)
    smart_money_bias: float = 0.0  # -1 to +1 (large trades alignment)
    call_volume: int = 0
    put_volume: int = 0
    call_premium: float = 0.0     # Total call premium traded ($)
    put_premium: float = 0.0      # Total put premium traded ($)
    extreme_reading: str = ""     # "extreme_bullish", "extreme_bearish", or ""
    net_delta_exposure: float = 0.0  # Volume-weighted delta: >0 = bullish positioning
    oi_skew: float = 0.0          # (call_oi - put_oi) / total_oi
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FlowAnalyzer:
    """Self-computed options flow analysis from chain data."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._latest: Optional[FlowSignals] = None
        self._pcr_history: list[float] = []

    async def update(self, chain: list = None) -> FlowSignals:
        """Refresh flow data from chain and optional CBOE fallback."""
        # Compute all metrics from chain data (primary source)
        chain_flow = self._analyze_chain_flow(chain) if chain else {}
        unusual = self._detect_unusual_activity(chain) if chain else []

        call_vol = chain_flow.get("call_volume", 0)
        put_vol = chain_flow.get("put_volume", 0)
        call_prem = chain_flow.get("call_premium", 0.0)
        put_prem = chain_flow.get("put_premium", 0.0)

        # P/C ratio: compute from chain volume (primary), CBOE scrape (fallback)
        pcr = self._compute_chain_pcr(chain) if chain else 0.0
        if pcr <= 0:
            pcr = await self._fetch_put_call_ratio()
        else:
            self._pcr_history.append(pcr)

        # Net flow direction: positive = bullish (more call premium)
        total_prem = call_prem + put_prem
        flow_direction = 0.0
        if total_prem > 0:
            flow_direction = (call_prem - put_prem) / total_prem

        # Volume-weighted delta exposure
        delta_exposure = self._compute_delta_exposure(chain) if chain else 0.0

        # OI concentration skew
        oi_skew = self._compute_oi_skew(chain) if chain else 0.0

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
            net_delta_exposure=round(delta_exposure, 4),
            oi_skew=round(oi_skew, 4),
        )
        self._latest = signals

        logger.info(
            "Flow: P/C=%.2f dir=%.2f delta_exp=%.3f oi_skew=%.3f "
            "unusual=%d smart=%.2f extreme=%s",
            pcr, flow_direction, delta_exposure, oi_skew,
            len(unusual), smart_money, extreme or "none",
        )
        return signals

    # ── Chain-Based Analytics ────────────────────────────────

    def _compute_chain_pcr(self, chain: list) -> float:
        """Compute put/call ratio from chain volume (replaces CBOE scrape)."""
        if not chain:
            return 0.0

        call_vol = 0
        put_vol = 0
        for c in chain:
            vol = getattr(c, 'volume', 0) or 0
            if vol <= 0:
                continue
            if c.option_type == "call":
                call_vol += vol
            else:
                put_vol += vol

        if call_vol == 0:
            return 1.0  # Neutral default
        return put_vol / call_vol

    def _analyze_chain_flow(self, chain: list) -> dict:
        """Analyze volume and premium distribution from options chain."""
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

    def _compute_delta_exposure(self, chain: list) -> float:
        """Volume-weighted delta exposure: net directional bias from options flow.

        Positive = market tilting bullish via options (more call delta traded).
        Negative = market tilting bearish (more put delta traded).
        Weighted by premium to emphasize institutional-sized trades.
        """
        if not chain:
            return 0.0

        weighted_delta_sum = 0.0
        total_premium = 0.0

        for c in chain:
            vol = getattr(c, 'volume', 0) or 0
            delta = getattr(c, 'delta', 0) or 0
            mid = float(getattr(c, 'mid', 0) or 0)

            if vol == 0 or mid <= 0:
                continue

            premium = vol * mid * 100
            weighted_delta_sum += delta * premium
            total_premium += premium

        if total_premium == 0:
            return 0.0

        # Normalize: result ranges roughly -1 to +1
        return weighted_delta_sum / total_premium

    def _compute_oi_skew(self, chain: list) -> float:
        """OI concentration skew: (call_oi - put_oi) / total_oi.

        Positive = more call OI → bullish positioning.
        Negative = more put OI → bearish positioning / hedging.
        """
        if not chain:
            return 0.0

        call_oi = 0
        put_oi = 0

        for c in chain:
            oi = getattr(c, 'open_interest', 0) or 0
            if oi <= 0:
                continue
            if c.option_type == "call":
                call_oi += oi
            else:
                put_oi += oi

        total = call_oi + put_oi
        if total == 0:
            return 0.0

        return (call_oi - put_oi) / total

    # ── Unusual Activity Detection ───────────────────────────

    def _detect_unusual_activity(self, chain: list) -> list[UnusualActivity]:
        """Detect unusual options activity from chain data.

        Flags contracts where volume/OI ratio > 3x and premium > $25k.
        This indicates institutional sweeps, blocks, or split orders.
        """
        if not chain:
            return []

        # Compute median premium for relative thresholds
        premiums = []
        for c in chain:
            vol = getattr(c, 'volume', 0) or 0
            mid = float(getattr(c, 'mid', 0) or 0)
            if vol > 0 and mid > 0:
                premiums.append(vol * mid * 100)
        median_premium = sorted(premiums)[len(premiums) // 2] if premiums else 50_000

        # Dynamic threshold: 10x median or $25k, whichever is lower
        premium_threshold = min(median_premium * 10, 50_000)
        premium_threshold = max(premium_threshold, 25_000)  # Floor at $25k

        unusual: list[UnusualActivity] = []

        for contract in chain:
            vol = getattr(contract, 'volume', 0) or 0
            oi = getattr(contract, 'open_interest', 0) or 0
            if oi == 0 or vol < 100:
                continue

            vol_oi = vol / oi
            if vol_oi < 3.0:
                continue

            mid = float(getattr(contract, 'mid', 0) or 0)
            total_premium = vol * mid * 100

            if total_premium < premium_threshold:
                continue

            # Determine sentiment from delta direction
            delta = getattr(contract, 'delta', 0) or 0
            opt_type = getattr(contract, 'option_type', 'call')
            if opt_type == "call":
                sentiment = "bullish" if delta > 0 else "bearish"
            else:
                sentiment = "bearish" if delta < 0 else "bullish"

            # Classify trade type by vol/OI ratio
            if vol_oi > 10:
                trade_type = "sweep"
            elif vol_oi > 5:
                trade_type = "block"
            else:
                trade_type = "split"

            unusual.append(UnusualActivity(
                symbol=getattr(contract, 'symbol', ''),
                underlying=getattr(contract, 'underlying', ''),
                option_type=opt_type,
                strike=float(getattr(contract, 'strike', 0)),
                volume=vol,
                open_interest=oi,
                vol_oi_ratio=round(vol_oi, 2),
                trade_type=trade_type,
                sentiment=sentiment,
                premium=round(total_premium, 2),
                timestamp=datetime.now(timezone.utc),
            ))

        # Sort by premium descending — largest trades first
        unusual.sort(key=lambda u: u.premium, reverse=True)
        return unusual[:20]  # Top 20

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

    # ── CBOE Fallback ────────────────────────────────────────

    async def _fetch_put_call_ratio(self) -> float:
        """Fetch CBOE put/call ratio (fallback when chain data unavailable)."""
        try:
            import aiohttp

            url = "https://www.cboe.com/us/options/market_statistics/daily/"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return self._pcr_history[-1] if self._pcr_history else 0.85

                    text = await resp.text()
                    pcr = self._parse_pcr_from_html(text)
                    if pcr > 0:
                        self._pcr_history.append(pcr)
                        return pcr

        except Exception:
            logger.debug("CBOE P/C ratio fetch failed, using fallback")

        return self._pcr_history[-1] if self._pcr_history else 0.85

    def _parse_pcr_from_html(self, html: str) -> float:
        """Extract total put/call ratio from CBOE page."""
        try:
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

    # ── Scoring ──────────────────────────────────────────────

    @property
    def latest(self) -> Optional[FlowSignals]:
        return self._latest

    def get_score(self, direction: str) -> float:
        """Score -1 to +1 for the ensemble.

        Combines 5 sub-signals:
            1. Flow direction alignment (±0.3)
            2. P/C ratio contrarian (±0.2)
            3. Smart money alignment (±0.2)
            4. Delta exposure alignment (±0.15)
            5. OI skew — contrarian if extreme (±0.15)
        """
        if not self._latest:
            return 0.0

        score = 0.0
        s = self._latest

        # 1. Flow direction alignment with trade direction
        if direction == "call":
            score += s.flow_direction * 0.3  # Bullish flow supports calls
        else:
            score -= s.flow_direction * 0.3  # Bearish flow supports puts

        # 2. P/C ratio contrarian
        if s.extreme_reading == "extreme_bearish" and direction == "call":
            score += 0.2  # Crowd too bearish, contrarian bullish
        elif s.extreme_reading == "extreme_bullish" and direction == "put":
            score += 0.2  # Crowd too bullish, contrarian bearish

        # 3. Smart money alignment
        if direction == "call":
            score += s.smart_money_bias * 0.2
        else:
            score -= s.smart_money_bias * 0.2

        # 4. Delta exposure alignment
        if direction == "call":
            score += s.net_delta_exposure * 0.15
        else:
            score -= s.net_delta_exposure * 0.15

        # 5. OI skew — contrarian if extreme, else directional
        oi_component = s.oi_skew
        if abs(oi_component) > 0.7:
            # Extreme OI skew = contrarian (too many calls → bearish reversion)
            oi_component = -oi_component * 0.5
        if direction == "call":
            score += oi_component * 0.15
        else:
            score -= oi_component * 0.15

        return max(-1.0, min(1.0, score))
