"""Alpaca WebSocket streaming for real-time underlying + options data.

Two separate streams:
    1. Equity stream — SPY/QQQ/IWM trades + bars for tick momentum
    2. Option stream — subscribed option contract quotes for exit management
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Optional

from src.infra.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class TickMomentum:
    """Rolling tick-level momentum from underlying price feed."""
    prices: deque = field(default_factory=lambda: deque(maxlen=60))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=60))

    def add_tick(self, price: float, ts: float) -> None:
        self.prices.append(price)
        self.timestamps.append(ts)

    @property
    def direction(self) -> float:
        """Returns -1.0 (strong down), 0.0 (flat), +1.0 (strong up)."""
        if len(self.prices) < 3:
            return 0.0
        recent = list(self.prices)[-10:]
        if len(recent) < 2:
            return 0.0
        moves = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
        up = sum(1 for m in moves if m > 0)
        down = sum(1 for m in moves if m < 0)
        total = up + down
        if total == 0:
            return 0.0
        return (up - down) / total

    @property
    def speed(self) -> float:
        """Price change per second over recent ticks."""
        if len(self.prices) < 2:
            return 0.0
        dt = self.timestamps[-1] - self.timestamps[-2]
        if dt <= 0:
            return 0.0
        return abs(self.prices[-1] - self.prices[-2]) / dt

    @property
    def roc_pct(self) -> float:
        """Rate of change (%) over the window."""
        if len(self.prices) < 5:
            return 0.0
        oldest = self.prices[0]
        if oldest == 0:
            return 0.0
        return ((self.prices[-1] - oldest) / oldest) * 100

    @property
    def latest_price(self) -> Optional[float]:
        return self.prices[-1] if self.prices else None


class AlpacaStream:
    """Manages Alpaca WebSocket connections for real-time data."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._running = False
        self._equity_ws: Any = None
        self._option_ws: Any = None
        self._momentum: dict[str, TickMomentum] = {}
        self._option_quotes: dict[str, dict] = {}
        self._on_trade_callbacks: list[Callable] = []
        self._on_option_quote_callbacks: list[Callable] = []
        self._subscribed_options: set[str] = set()
        self._reconnect_delay = 1.0

        for symbol in settings.underlying_list:
            self._momentum[symbol] = TickMomentum()

    def on_trade(self, callback: Callable) -> None:
        self._on_trade_callbacks.append(callback)

    def on_option_quote(self, callback: Callable) -> None:
        self._on_option_quote_callbacks.append(callback)

    def get_momentum(self, symbol: str) -> Optional[TickMomentum]:
        return self._momentum.get(symbol)

    def get_option_quote(self, contract_symbol: str) -> Optional[dict]:
        return self._option_quotes.get(contract_symbol)

    async def start(self) -> None:
        """Start both equity and option WebSocket streams."""
        self._running = True
        logger.info("Starting Alpaca streams for %s", self._settings.underlying_list)

        await asyncio.gather(
            self._run_equity_stream(),
            self._run_option_stream(),
        )

    async def stop(self) -> None:
        self._running = False
        if self._equity_ws:
            try:
                await self._equity_ws.close()
            except Exception:
                pass
        if self._option_ws:
            try:
                await self._option_ws.close()
            except Exception:
                pass
        logger.info("Alpaca streams stopped")

    async def subscribe_options(self, symbols: list[str]) -> None:
        """Subscribe to real-time quotes for specific option contracts."""
        new_symbols = set(symbols) - self._subscribed_options
        if not new_symbols:
            return

        self._subscribed_options.update(new_symbols)
        logger.info("Subscribing to %d option contracts", len(new_symbols))

        if self._option_ws:
            try:
                from alpaca.data.live.option import OptionDataStream
                self._option_ws.subscribe_quotes(
                    self._handle_option_quote, *list(new_symbols),
                )
            except Exception:
                logger.exception("Failed to subscribe option quotes")

    async def unsubscribe_options(self, symbols: list[str]) -> None:
        """Unsubscribe from option contract quotes."""
        for sym in symbols:
            self._subscribed_options.discard(sym)
            self._option_quotes.pop(sym, None)

    # ── Equity Stream ────────────────────────────────────────────

    async def _run_equity_stream(self) -> None:
        """Connect to Alpaca equity data stream for underlying trades."""
        while self._running:
            try:
                from alpaca.data.live.stock import StockDataStream
                from alpaca.data.enums import DataFeed

                # alpaca-py expects DataFeed enum, not raw string
                feed_str = self._settings.alpaca_data_feed.lower()
                feed_enum = DataFeed.SIP if feed_str == "sip" else DataFeed.IEX

                self._equity_ws = StockDataStream(
                    api_key=self._settings.alpaca_api_key,
                    secret_key=self._settings.alpaca_secret_key,
                    feed=feed_enum,
                )

                # Subscribe to trades for all underlyings
                self._equity_ws.subscribe_trades(
                    self._handle_equity_trade,
                    *self._settings.underlying_list,
                )

                # Subscribe to 1-min bars
                self._equity_ws.subscribe_bars(
                    self._handle_equity_bar,
                    *self._settings.underlying_list,
                )

                logger.info("Equity stream connecting...")
                await self._equity_ws._run_forever()

            except Exception:
                if not self._running:
                    break
                logger.exception("Equity stream error, reconnecting in %.0fs", self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def _handle_equity_trade(self, trade: Any) -> None:
        """Process a real-time equity trade tick."""
        symbol = str(trade.symbol)
        price = float(trade.price)
        ts = time.time()

        momentum = self._momentum.get(symbol)
        if momentum:
            momentum.add_tick(price, ts)

        self._reconnect_delay = 1.0  # Reset on successful data

        for cb in self._on_trade_callbacks:
            try:
                await cb(symbol, price, ts)
            except Exception:
                logger.exception("Trade callback error")

    async def _handle_equity_bar(self, bar: Any) -> None:
        """Process a 1-minute bar."""
        # Bars handled by engine for candle aggregation
        pass

    # ── Option Stream ────────────────────────────────────────────

    async def _run_option_stream(self) -> None:
        """Connect to Alpaca option data stream for contract quotes."""
        while self._running:
            try:
                from alpaca.data.live.option import OptionDataStream

                self._option_ws = OptionDataStream(
                    api_key=self._settings.alpaca_api_key,
                    secret_key=self._settings.alpaca_secret_key,
                )

                # Re-subscribe to any previously subscribed symbols
                if self._subscribed_options:
                    self._option_ws.subscribe_quotes(
                        self._handle_option_quote,
                        *list(self._subscribed_options),
                    )

                logger.info("Option stream connecting...")
                await self._option_ws._run_forever()

            except Exception:
                if not self._running:
                    break
                logger.exception("Option stream error, reconnecting in %.0fs", self._reconnect_delay)
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)

    async def _handle_option_quote(self, quote: Any) -> None:
        """Process a real-time option quote."""
        symbol = str(quote.symbol)
        data = {
            "symbol": symbol,
            "bid": float(quote.bid_price) if quote.bid_price else 0.0,
            "ask": float(quote.ask_price) if quote.ask_price else 0.0,
            "bid_size": int(quote.bid_size) if quote.bid_size else 0,
            "ask_size": int(quote.ask_size) if quote.ask_size else 0,
            "ts": time.time(),
        }
        self._option_quotes[symbol] = data

        for cb in self._on_option_quote_callbacks:
            try:
                await cb(symbol, data)
            except Exception:
                logger.exception("Option quote callback error")

    # ── Status ───────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "running": self._running,
            "underlyings": {
                sym: {
                    "latest_price": mom.latest_price,
                    "direction": round(mom.direction, 2),
                    "speed": round(mom.speed, 4),
                    "roc_pct": round(mom.roc_pct, 4),
                    "tick_count": len(mom.prices),
                }
                for sym, mom in self._momentum.items()
            },
            "subscribed_options": len(self._subscribed_options),
            "active_option_quotes": len(self._option_quotes),
        }
