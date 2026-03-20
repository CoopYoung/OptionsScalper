"""OptionsAI API integration: expected moves, IV skew, AI strategy validation.

Data sources: tools.optionsai.com (unauthenticated, free)
    - Expected move: implied daily range, call/put IV
    - Trade generator: AI strategy suggestions for directional bias
    - Earnings calendar: blackout enhancement for underlying-specific events

Sub-signals (weights within this factor):
    IV Skew           40%   callIv - putIv, normalized
    Move Proximity    35%   price vs implied range (mean reversion)
    AI Strategy Bias  25%   bullish vs bearish strategy count
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import aiohttp

from src.infra.config import Settings

logger = logging.getLogger(__name__)

BASE_URL = "https://tools.optionsai.com/api"
REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=10)

# Strategy classification for directional bias
BULLISH_STRATEGIES = {"LONG_CALL", "LONG_CALL_SPREAD", "SHORT_PUT_SPREAD"}
BEARISH_STRATEGIES = {"LONG_PUT", "LONG_PUT_SPREAD", "SHORT_CALL_SPREAD"}
NEUTRAL_STRATEGIES = {"SHORT_IRON_CONDOR", "SHORT_IRON_BUTTERFLY"}


@dataclass
class OptionsAISignals:
    """OptionsAI-derived signals for strategy ensemble."""
    move_amount: float = 0.0
    move_percent: float = 0.0
    implied_high: float = 0.0
    implied_low: float = 0.0
    overall_iv: float = 0.0
    call_iv: float = 0.0
    put_iv: float = 0.0
    iv_skew: float = 0.0          # callIv - putIv; positive = bullish
    bullish_strategies: int = 0
    bearish_strategies: int = 0
    neutral_strategies: int = 0
    strategy_bias: float = 0.0    # -1.0 to +1.0
    strategy_names: list[str] = field(default_factory=list)
    price_vs_implied_range: float = 0.0  # -1.0 (at low) to +1.0 (at high)
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class EarningsEvent:
    """An upcoming earnings event."""
    symbol: str
    date: str
    time: str  # "bmo" (before market open) or "amc" (after market close)
    eps_estimated: Optional[float] = None


class OptionsAIAnalyzer:
    """OptionsAI API integration for expected moves, IV skew, and AI strategy validation."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._latest: dict[str, OptionsAISignals] = {}
        self._earnings_events: list[EarningsEvent] = []
        self._earnings_symbols: set[str] = set()
        self._update_cycle: int = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create a persistent aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=REQUEST_TIMEOUT)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def update(self, underlying: str, current_price: float = 0.0) -> OptionsAISignals:
        """Fetch expected move + trade generator for one underlying.

        Expected move is fetched every cycle (30s).
        Trade generator is fetched every 2nd cycle (~60s).
        """
        self._update_cycle += 1

        # Always fetch expected move
        move_data = await self._fetch_expected_move(underlying)

        # Trade generator every 2nd cycle (slower changing data)
        strategy_data = None
        if self._update_cycle % 2 == 0:
            strategy_data = await self._fetch_trade_generator(underlying)

        signals = self._build_signals(underlying, move_data, strategy_data, current_price)
        self._latest[underlying] = signals

        logger.info(
            "OptionsAI %s: skew=%.4f move=$%.2f bias=%.2f range_pos=%.2f",
            underlying, signals.iv_skew, signals.move_amount,
            signals.strategy_bias, signals.price_vs_implied_range,
        )
        return signals

    async def load_earnings(self) -> list[EarningsEvent]:
        """Load earnings calendar for today and tomorrow. Call once at pre-market."""
        today = date.today()
        tomorrow = today + timedelta(days=2)
        events: list[EarningsEvent] = []

        try:
            session = await self._get_session()
            url = f"{BASE_URL}/earnings-calendar?from={today.isoformat()}&to={tomorrow.isoformat()}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.debug("OptionsAI earnings fetch failed: HTTP %d", resp.status)
                    return events
                data = await resp.json()

            if not isinstance(data, list):
                return events

            underlyings = set(self._settings.underlying_list)
            for item in data:
                symbol = item.get("symbol", "")
                if symbol in underlyings:
                    event = EarningsEvent(
                        symbol=symbol,
                        date=item.get("date", ""),
                        time=item.get("time", ""),
                        eps_estimated=item.get("epsEstimated"),
                    )
                    events.append(event)
                    self._earnings_symbols.add(symbol)

            self._earnings_events = events
            if events:
                logger.warning(
                    "EARNINGS NEARBY: %s",
                    [f"{e.symbol} ({e.date} {e.time})" for e in events],
                )

        except Exception:
            logger.debug("OptionsAI earnings calendar fetch failed", exc_info=True)

        return events

    def has_earnings(self, underlying: str) -> bool:
        """Check if underlying has earnings within blackout window."""
        return underlying in self._earnings_symbols

    def get_score(self, underlying: str, direction: str) -> float:
        """Compute -1.0 to +1.0 score for the ensemble.

        Sub-signals:
            IV Skew (40%): callIv - putIv normalized. Positive skew → bullish.
            Move Proximity (35%): mean reversion from implied range edges.
            Strategy Bias (25%): bullish vs bearish AI strategy count.
        """
        signals = self._latest.get(underlying)
        if not signals:
            return 0.0

        # Sub-signal 1: IV Skew (weight 0.40)
        # Normalize skew over [-0.10, +0.10] range
        skew_normalized = max(-1.0, min(1.0, signals.iv_skew / 0.10))
        skew_score = skew_normalized if direction == "call" else -skew_normalized

        # Sub-signal 2: Expected Move Proximity (weight 0.35)
        # price_vs_implied_range: -1.0 at implied low, +1.0 at implied high
        # Near implied low → bullish (mean reversion), near high → bearish
        range_score = -signals.price_vs_implied_range  # Negate: low = bullish
        if direction == "put":
            range_score = signals.price_vs_implied_range  # High = bearish for puts

        # Sub-signal 3: AI Strategy Bias (weight 0.25)
        bias_score = signals.strategy_bias if direction == "call" else -signals.strategy_bias

        score = skew_score * 0.40 + range_score * 0.35 + bias_score * 0.25
        return max(-1.0, min(1.0, score))

    def get_latest(self, underlying: str) -> Optional[OptionsAISignals]:
        """Get latest signals for a specific underlying."""
        return self._latest.get(underlying)

    @property
    def latest(self) -> dict[str, OptionsAISignals]:
        """All per-underlying signals."""
        return self._latest

    # ── Private: API Fetchers ─────────────────────────────────

    async def _fetch_expected_move(self, symbol: str) -> Optional[dict]:
        """GET /api/moves/nearest — expected move, call/put IV."""
        try:
            session = await self._get_session()
            today_str = date.today().isoformat()
            url = f"{BASE_URL}/moves/nearest?symbol={symbol}&expirationDate={today_str}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception:
            logger.debug("OptionsAI moves fetch failed for %s", symbol)
            return None

    async def _fetch_trade_generator(self, symbol: str) -> Optional[list]:
        """GET /api/trade-generator — AI strategy suggestions."""
        try:
            session = await self._get_session()
            url = f"{BASE_URL}/trade-generator?symbol={symbol}"
            async with session.get(url) as resp:
                if resp.status != 200:
                    return None
                return await resp.json()
        except Exception:
            logger.debug("OptionsAI trade-generator fetch failed for %s", symbol)
            return None

    # ── Private: Signal Building ──────────────────────────────

    def _build_signals(
        self, underlying: str, move_data: Optional[dict],
        strategy_data: Optional[list], current_price: float,
    ) -> OptionsAISignals:
        """Combine raw API data into OptionsAISignals."""
        # Start with previous signals (retain strategy data across cycles)
        prev = self._latest.get(underlying)
        signals = OptionsAISignals(updated_at=datetime.now(timezone.utc))

        # Parse expected move data
        if move_data and isinstance(move_data, dict):
            signals.move_amount = float(move_data.get("moveAmount", 0) or 0)
            signals.move_percent = float(move_data.get("movePercent", 0) or 0)
            signals.overall_iv = float(move_data.get("iv", 0) or 0)
            signals.call_iv = float(move_data.get("callIv", 0) or 0)
            signals.put_iv = float(move_data.get("putIv", 0) or 0)
            signals.iv_skew = signals.call_iv - signals.put_iv

            # Compute implied range from current price + expected move
            if current_price > 0 and signals.move_amount > 0:
                signals.implied_high = current_price + signals.move_amount
                signals.implied_low = current_price - signals.move_amount

                # Where is price within the implied range? (-1 to +1)
                range_width = signals.implied_high - signals.implied_low
                if range_width > 0:
                    signals.price_vs_implied_range = (
                        (current_price - signals.implied_low) / range_width * 2 - 1
                    )
                    signals.price_vs_implied_range = max(
                        -1.0, min(1.0, signals.price_vs_implied_range)
                    )

        # Parse strategy data (or retain from previous cycle)
        if strategy_data and isinstance(strategy_data, list):
            bullish = 0
            bearish = 0
            neutral = 0
            names: list[str] = []

            for strat in strategy_data:
                name = strat.get("strategy", "")
                names.append(name)
                if name in BULLISH_STRATEGIES:
                    bullish += 1
                elif name in BEARISH_STRATEGIES:
                    bearish += 1
                elif name in NEUTRAL_STRATEGIES:
                    neutral += 1

            signals.bullish_strategies = bullish
            signals.bearish_strategies = bearish
            signals.neutral_strategies = neutral
            signals.strategy_names = names

            total = bullish + bearish + neutral
            if total > 0:
                signals.strategy_bias = (bullish - bearish) / total
            else:
                signals.strategy_bias = 0.0

        elif prev:
            # Retain previous cycle's strategy data
            signals.bullish_strategies = prev.bullish_strategies
            signals.bearish_strategies = prev.bearish_strategies
            signals.neutral_strategies = prev.neutral_strategies
            signals.strategy_bias = prev.strategy_bias
            signals.strategy_names = prev.strategy_names

        return signals
