"""Adaptive order execution for 0DTE options.

Handles the cancel-replace flow that Alpaca requires (no IOC for options):
    - Entry orders: start at mid, walk toward aggressive side over 30s
    - Exit orders: more aggressive pricing based on urgency
    - Tracks fill latency and slippage for analytics

The order manager wraps the engine's pending order tracking and provides
a higher-level interface for order lifecycle management.
"""

import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

from src.data.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


@dataclass
class ManagedOrder:
    """An order being actively managed by the OrderManager."""
    order_id: str
    symbol: str
    side: str                       # "buy" or "sell"
    qty: int
    initial_mid: float              # Mid price at signal time
    spread: float                   # Bid-ask spread at signal time
    current_limit: float            # Current limit price
    created_at: float               # time.monotonic()
    walk_step: int = 0              # How many times we've walked the price
    max_walk_steps: int = 3         # Max price walks before cancel
    walk_interval_secs: float = 10  # Seconds between walks
    urgency: str = "normal"         # "normal", "urgent", "immediate"
    metadata: dict = field(default_factory=dict)  # Passthrough data for engine

    @property
    def age_seconds(self) -> float:
        return time.monotonic() - self.created_at

    @property
    def slippage(self) -> float:
        """Slippage = how much worse than mid we're willing to pay."""
        return abs(self.current_limit - self.initial_mid)


class OrderManager:
    """Manages order lifecycle with adaptive limit pricing.

    Entry flow (buying options):
        1. Place limit at mid price
        2. After 10s unfilled: cancel, replace at mid + 25% of spread
        3. After 20s unfilled: cancel, replace at mid + 50% of spread
        4. After 30s unfilled: cancel entirely (signal is stale)

    Exit flow (selling options):
        normal:    mid - 1% of spread → walk down
        urgent:    start at mid - 25% of spread
        immediate: accept bid price
    """

    def __init__(self, client: AlpacaClient) -> None:
        self._client = client
        self._managed_orders: dict[str, ManagedOrder] = {}

    async def submit_entry(
        self,
        symbol: str,
        qty: int,
        bid: float,
        ask: float,
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Submit an entry (buy) order with adaptive pricing.

        Returns the order_id if placed successfully.
        """
        if ask <= 0 or bid <= 0:
            return None

        mid = (bid + ask) / 2
        spread = ask - bid

        # Start at mid for entries
        limit_price = round(mid, 2)

        order = await self._client.place_order(
            symbol=symbol,
            side="buy",
            qty=qty,
            limit_price=limit_price,
        )

        if not order:
            return None

        order_id = order["id"]
        self._managed_orders[order_id] = ManagedOrder(
            order_id=order_id,
            symbol=symbol,
            side="buy",
            qty=qty,
            initial_mid=mid,
            spread=spread,
            current_limit=limit_price,
            created_at=time.monotonic(),
            metadata=metadata or {},
        )

        logger.info(
            "Entry order: %s %d @ $%.2f (mid=$%.2f, spread=$%.2f)",
            symbol, qty, limit_price, mid, spread,
        )
        return order_id

    async def submit_exit(
        self,
        symbol: str,
        qty: int,
        bid: float,
        ask: float,
        urgency: str = "normal",
        metadata: Optional[dict] = None,
    ) -> Optional[str]:
        """Submit an exit (sell) order with urgency-based pricing.

        Urgency levels:
            normal:    mid - small buffer
            urgent:    mid - 25% of spread
            immediate: bid price (accept market)
        """
        if bid <= 0:
            return None

        mid = (bid + ask) / 2
        spread = ask - bid

        if urgency == "immediate":
            limit_price = round(bid, 2)
        elif urgency == "urgent":
            limit_price = round(mid - spread * 0.25, 2)
        else:
            limit_price = round(mid * 0.99, 2)

        limit_price = max(0.01, limit_price)

        order = await self._client.place_order(
            symbol=symbol,
            side="sell",
            qty=qty,
            limit_price=limit_price,
        )

        if not order:
            return None

        order_id = order["id"]
        self._managed_orders[order_id] = ManagedOrder(
            order_id=order_id,
            symbol=symbol,
            side="sell",
            qty=qty,
            initial_mid=mid,
            spread=spread,
            current_limit=limit_price,
            created_at=time.monotonic(),
            urgency=urgency,
            max_walk_steps=2 if urgency != "immediate" else 0,
            walk_interval_secs=5,  # Faster walks for exits
            metadata=metadata or {},
        )

        logger.info(
            "Exit order: %s %d @ $%.2f (mid=$%.2f, urgency=%s)",
            symbol, qty, limit_price, mid, urgency,
        )
        return order_id

    async def check_and_walk(self) -> list[dict]:
        """Check all managed orders and walk prices if needed.

        Returns list of events: filled, cancelled, walked.
        Call this from the engine's fast loop.
        """
        events = []

        for order_id in list(self._managed_orders.keys()):
            managed = self._managed_orders[order_id]

            # Check fill status
            order_status = await self._client.get_order(order_id)
            if not order_status:
                continue

            status = order_status["status"]

            if status == "filled":
                fill_price = float(order_status.get("filled_avg_price") or 0)
                slippage = fill_price - managed.initial_mid if managed.side == "buy" else managed.initial_mid - fill_price
                events.append({
                    "type": "filled",
                    "order_id": order_id,
                    "symbol": managed.symbol,
                    "side": managed.side,
                    "qty": int(order_status.get("filled_qty") or managed.qty),
                    "fill_price": fill_price,
                    "initial_mid": managed.initial_mid,
                    "slippage": round(slippage, 4),
                    "fill_latency_secs": round(managed.age_seconds, 1),
                    "walk_steps": managed.walk_step,
                    "metadata": managed.metadata,
                })
                self._managed_orders.pop(order_id, None)
                continue

            if status in ("cancelled", "expired", "rejected"):
                events.append({
                    "type": "cancelled",
                    "order_id": order_id,
                    "symbol": managed.symbol,
                    "side": managed.side,
                    "reason": status,
                    "metadata": managed.metadata,
                })
                self._managed_orders.pop(order_id, None)
                continue

            # Still open — should we walk the price?
            if managed.age_seconds >= (managed.walk_step + 1) * managed.walk_interval_secs:
                if managed.walk_step < managed.max_walk_steps:
                    # Walk the price
                    walked = await self._walk_price(managed)
                    if walked:
                        events.append({
                            "type": "walked",
                            "order_id": order_id,
                            "symbol": managed.symbol,
                            "new_limit": managed.current_limit,
                            "walk_step": managed.walk_step,
                        })
                else:
                    # Max walks reached — cancel
                    cancelled = await self._client.cancel_order(order_id)
                    if cancelled:
                        events.append({
                            "type": "timeout_cancelled",
                            "order_id": order_id,
                            "symbol": managed.symbol,
                            "side": managed.side,
                            "age_secs": round(managed.age_seconds, 1),
                            "metadata": managed.metadata,
                        })
                        self._managed_orders.pop(order_id, None)

        return events

    async def _walk_price(self, managed: ManagedOrder) -> bool:
        """Cancel current order and replace at a more aggressive price."""
        cancelled = await self._client.cancel_order(managed.order_id)
        if not cancelled:
            return False

        managed.walk_step += 1
        walk_pct = managed.walk_step / managed.max_walk_steps  # 0.33, 0.67, 1.0

        if managed.side == "buy":
            # Walk up toward ask
            new_limit = managed.initial_mid + managed.spread * walk_pct * 0.5
        else:
            # Walk down toward bid
            new_limit = managed.initial_mid - managed.spread * walk_pct * 0.5

        new_limit = round(max(0.01, new_limit), 2)

        order = await self._client.place_order(
            symbol=managed.symbol,
            side=managed.side,
            qty=managed.qty,
            limit_price=new_limit,
        )

        if order:
            managed.order_id = order["id"]
            managed.current_limit = new_limit
            # Re-key in dict
            self._managed_orders.pop(managed.order_id, None)
            self._managed_orders[order["id"]] = managed
            logger.info(
                "Walked %s order: %s step %d @ $%.2f",
                managed.side, managed.symbol, managed.walk_step, new_limit,
            )
            return True

        return False

    async def cancel_all(self) -> None:
        """Cancel all managed orders."""
        for order_id in list(self._managed_orders.keys()):
            await self._client.cancel_order(order_id)
            self._managed_orders.pop(order_id, None)

    @property
    def pending_count(self) -> int:
        return len(self._managed_orders)

    def get_managed_order(self, order_id: str) -> Optional[ManagedOrder]:
        return self._managed_orders.get(order_id)
