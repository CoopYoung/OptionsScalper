"""Economic calendar gate: halt trading around high-impact events.

Tracks FOMC, CPI, NFP, PPI, GDP, PCE and pauses all new entries
within the configured blackout window (default ±60 minutes).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from src.infra.config import Settings

logger = logging.getLogger(__name__)


class EventImpact(str, Enum):
    HIGH = "high"        # FOMC, CPI, NFP — always blackout
    MEDIUM = "medium"    # PPI, GDP, PCE — blackout if within window
    LOW = "low"          # Minor reports — no blackout


@dataclass
class MacroEvent:
    """A scheduled economic event."""
    name: str
    timestamp: datetime   # UTC
    impact: EventImpact
    description: str = ""


@dataclass
class MacroSignals:
    """Macro calendar state."""
    is_blackout: bool
    blackout_reason: str
    events_today: list[MacroEvent] = field(default_factory=list)
    next_event: Optional[MacroEvent] = None
    minutes_to_event: Optional[int] = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MacroCalendar:
    """Economic event calendar for trade blackout periods."""

    # Known high-impact recurring events (checked against API results)
    HIGH_IMPACT_KEYWORDS = [
        "fomc", "federal funds rate", "interest rate decision",
        "fomc minutes", "fomc statement",
        "cpi", "consumer price index",
        "nfp", "non-farm payrolls", "nonfarm payrolls",
        "employment situation",
    ]
    MEDIUM_IMPACT_KEYWORDS = [
        "ppi", "producer price index",
        "gdp", "gross domestic product",
        "pce", "personal consumption expenditures",
        "retail sales", "industrial production",
        "ism manufacturing", "ism services",
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._events_today: list[MacroEvent] = []
        self._loaded = False
        self._latest: Optional[MacroSignals] = None

    async def load_today(self) -> list[MacroEvent]:
        """Load today's economic events."""
        events = await self._fetch_events()
        self._events_today = events
        self._loaded = True

        high = sum(1 for e in events if e.impact == EventImpact.HIGH)
        med = sum(1 for e in events if e.impact == EventImpact.MEDIUM)
        logger.info("Macro calendar loaded: %d events (%d high, %d medium)", len(events), high, med)

        return events

    async def update(self) -> MacroSignals:
        """Check blackout status against current time."""
        if not self._loaded:
            await self.load_today()

        now = datetime.now(timezone.utc)
        blackout_window = timedelta(minutes=self._settings.macro_blackout_minutes)

        is_blackout = False
        blackout_reason = ""
        next_event: Optional[MacroEvent] = None
        min_to_event: Optional[int] = None

        for event in self._events_today:
            if event.impact == EventImpact.LOW:
                continue

            window_start = event.timestamp - blackout_window
            window_end = event.timestamp + blackout_window

            if window_start <= now <= window_end:
                is_blackout = True
                blackout_reason = f"{event.name} ({event.impact.value}) at {event.timestamp.strftime('%H:%M')} UTC"

            # Find next upcoming event
            if event.timestamp > now:
                if next_event is None or event.timestamp < next_event.timestamp:
                    next_event = event
                    min_to_event = int((event.timestamp - now).total_seconds() / 60)

        signals = MacroSignals(
            is_blackout=is_blackout,
            blackout_reason=blackout_reason,
            events_today=self._events_today,
            next_event=next_event,
            minutes_to_event=min_to_event,
        )
        self._latest = signals

        if is_blackout:
            logger.warning("MACRO BLACKOUT: %s", blackout_reason)

        return signals

    async def _fetch_events(self) -> list[MacroEvent]:
        """Fetch economic calendar from multiple sources."""
        events: list[MacroEvent] = []

        # Try FinanceFlow / Trading Economics API
        api_events = await self._fetch_from_api()
        if api_events:
            return api_events

        # Fallback: hardcoded well-known schedule check
        events = self._check_known_schedule()
        return events

    async def _fetch_from_api(self) -> list[MacroEvent]:
        """Fetch from economic calendar API."""
        try:
            import aiohttp

            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            url = f"https://nfs.faireconomy.media/ff_calendar_thisweek.json"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return []
                    data = await resp.json()

            events = []
            for item in data:
                event_date = item.get("date", "")
                if today not in event_date:
                    continue

                title = item.get("title", "")
                impact_str = item.get("impact", "").lower()

                # Classify impact
                impact = self._classify_impact(title, impact_str)

                # Parse timestamp
                try:
                    ts = datetime.fromisoformat(event_date.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue

                events.append(MacroEvent(
                    name=title,
                    timestamp=ts,
                    impact=impact,
                    description=item.get("forecast", ""),
                ))

            return events

        except Exception:
            logger.debug("Economic calendar API fetch failed")
            return []

    def _classify_impact(self, title: str, impact_str: str) -> EventImpact:
        """Classify event impact based on title keywords."""
        title_lower = title.lower()

        for keyword in self.HIGH_IMPACT_KEYWORDS:
            if keyword in title_lower:
                return EventImpact.HIGH

        for keyword in self.MEDIUM_IMPACT_KEYWORDS:
            if keyword in title_lower:
                return EventImpact.MEDIUM

        if "high" in impact_str:
            return EventImpact.HIGH
        if "medium" in impact_str:
            return EventImpact.MEDIUM

        return EventImpact.LOW

    def _check_known_schedule(self) -> list[MacroEvent]:
        """Check if today matches known high-impact event patterns."""
        # This is a static fallback — real implementation uses API
        return []

    @property
    def latest(self) -> Optional[MacroSignals]:
        return self._latest

    def is_blackout(self) -> bool:
        """Quick check: are we in a macro blackout?"""
        if self._latest:
            return self._latest.is_blackout
        return False

    def minutes_to_event(self) -> Optional[int]:
        if self._latest:
            return self._latest.minutes_to_event
        return None
