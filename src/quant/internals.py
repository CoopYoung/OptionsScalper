"""Market internals: NYSE TICK, advance/decline, VWAP, cumulative delta.

Real-time market breadth indicators for confirming or denying momentum.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from src.infra.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class InternalsSignals:
    """Market breadth/internals signals."""
    nyse_tick: int                   # NYSE TICK index (-1000 to +1000)
    tick_extreme: str                # "high" (>500), "low" (<-500), or ""
    advance_decline_ratio: float     # A/D ratio (>1 = more advancers)
    vwap: dict[str, float] = field(default_factory=dict)  # symbol → VWAP
    vwap_deviation: dict[str, float] = field(default_factory=dict)  # symbol → % from VWAP
    cumulative_delta: float = 0.0    # Net buy-sell volume intraday
    breadth_score: float = 0.0       # -1 to +1 composite
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MarketInternals:
    """Real-time market breadth indicators."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._latest: Optional[InternalsSignals] = None
        self._vwap_data: dict[str, _VWAPTracker] = {}
        self._tick_history: deque[int] = deque(maxlen=100)
        self._cumulative_delta: float = 0.0

        for symbol in settings.underlying_list:
            self._vwap_data[symbol] = _VWAPTracker()

    async def update(self) -> InternalsSignals:
        """Refresh all market internals."""
        tick = await self._fetch_nyse_tick()
        ad_ratio = await self._fetch_advance_decline()

        self._tick_history.append(tick)

        tick_extreme = ""
        if tick > 500:
            tick_extreme = "high"
        elif tick < -500:
            tick_extreme = "low"

        # VWAP and deviations
        vwap_dict = {}
        vwap_dev = {}
        for symbol, tracker in self._vwap_data.items():
            if tracker.vwap > 0:
                vwap_dict[symbol] = round(tracker.vwap, 2)
                if tracker.last_price > 0:
                    dev = (tracker.last_price - tracker.vwap) / tracker.vwap * 100
                    vwap_dev[symbol] = round(dev, 3)

        # Breadth score composite
        breadth = self._compute_breadth_score(tick, ad_ratio, vwap_dev)

        signals = InternalsSignals(
            nyse_tick=tick,
            tick_extreme=tick_extreme,
            advance_decline_ratio=round(ad_ratio, 3),
            vwap=vwap_dict,
            vwap_deviation=vwap_dev,
            cumulative_delta=round(self._cumulative_delta, 2),
            breadth_score=round(breadth, 3),
        )
        self._latest = signals

        logger.debug(
            "Internals: TICK=%d A/D=%.2f breadth=%.2f",
            tick, ad_ratio, breadth,
        )
        return signals

    def update_vwap(self, symbol: str, price: float, volume: float) -> None:
        """Feed tick data into VWAP calculation."""
        tracker = self._vwap_data.get(symbol)
        if tracker:
            tracker.add_tick(price, volume)

    def update_delta(self, buy_volume: float, sell_volume: float) -> None:
        """Update cumulative delta with new volume data."""
        self._cumulative_delta += buy_volume - sell_volume

    def reset_daily(self) -> None:
        """Reset daily VWAP and cumulative delta at market open."""
        for tracker in self._vwap_data.values():
            tracker.reset()
        self._cumulative_delta = 0.0
        self._tick_history.clear()

    async def _fetch_nyse_tick(self) -> int:
        """Fetch NYSE TICK index."""
        try:
            import yfinance as yf

            tick = yf.Ticker("^TICK")
            data = tick.history(period="1d", interval="1m")
            if not data.empty:
                return int(data["Close"].iloc[-1])

        except Exception:
            logger.debug("NYSE TICK fetch failed")

        return 0

    async def _fetch_advance_decline(self) -> float:
        """Estimate advance/decline ratio using sector ETF breadth.

        Uses a basket of sector ETFs as a proxy for market breadth:
        if most sectors are up, A/D ratio > 1.0.
        """
        try:
            import yfinance as yf

            # Sector ETFs as breadth proxy
            sectors = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE", "XLC"]
            data = yf.download(sectors, period="2d", interval="1d", progress=False, threads=True)

            if data.empty or len(data) < 2:
                return 1.0

            closes = data["Close"]
            if len(closes) < 2:
                return 1.0

            # Count how many sectors advanced vs declined today
            prev = closes.iloc[-2]
            curr = closes.iloc[-1]
            changes = curr - prev

            advancers = int((changes > 0).sum())
            decliners = int((changes < 0).sum())

            if decliners == 0:
                return float(advancers) if advancers > 0 else 1.0
            return advancers / decliners

        except Exception:
            logger.debug("A/D ratio fetch failed")

        return 1.0

    def _compute_breadth_score(
        self, tick: int, ad_ratio: float, vwap_dev: dict[str, float],
    ) -> float:
        """Composite breadth score: -1 (bearish) to +1 (bullish)."""
        score = 0.0

        # TICK contribution: normalize -1000 to +1000 → -0.4 to +0.4
        tick_component = max(-1000, min(1000, tick)) / 1000 * 0.4
        score += tick_component

        # A/D ratio: >1.5 = strong bullish, <0.67 = strong bearish
        if ad_ratio > 1.5:
            score += 0.3
        elif ad_ratio > 1.0:
            score += 0.1
        elif ad_ratio < 0.67:
            score -= 0.3
        elif ad_ratio < 1.0:
            score -= 0.1

        # VWAP: average deviation across underlyings
        if vwap_dev:
            avg_dev = sum(vwap_dev.values()) / len(vwap_dev)
            # Normalize: ±0.5% → ±0.3 score
            vwap_component = max(-0.5, min(0.5, avg_dev)) / 0.5 * 0.3
            score += vwap_component

        return max(-1.0, min(1.0, score))

    @property
    def latest(self) -> Optional[InternalsSignals]:
        return self._latest

    def get_score(self, direction: str) -> float:
        """Score for the ensemble. Breadth confirms or denies direction."""
        if not self._latest:
            return 0.0

        breadth = self._latest.breadth_score

        if direction == "call":
            return breadth  # Positive breadth confirms bullish
        return -breadth      # Negative breadth confirms bearish


class _VWAPTracker:
    """Volume-weighted average price tracker for a single symbol."""

    def __init__(self) -> None:
        self.cumulative_pv: float = 0.0   # Price * Volume
        self.cumulative_vol: float = 0.0
        self.last_price: float = 0.0

    @property
    def vwap(self) -> float:
        if self.cumulative_vol == 0:
            return 0.0
        return self.cumulative_pv / self.cumulative_vol

    def add_tick(self, price: float, volume: float) -> None:
        self.cumulative_pv += price * volume
        self.cumulative_vol += volume
        self.last_price = price

    def reset(self) -> None:
        self.cumulative_pv = 0.0
        self.cumulative_vol = 0.0
        self.last_price = 0.0
