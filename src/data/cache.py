"""In-memory and Redis cache layer with cross-asset tick momentum sharing.

Adapted from poly-trader for SPY/QQQ/IWM instead of BTC/ETH/SOL.
"""

import json
import logging
import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from src.infra.config import Settings

logger = logging.getLogger(__name__)


class PriceCache:
    """Two-tier cache: in-memory dict + optional Redis backing."""

    ALL_SYMBOLS = ("SPY", "QQQ", "IWM")

    def __init__(self, settings: Settings) -> None:
        self._memory: dict[str, Any] = {}
        self._redis_url = settings.redis_url
        self._redis: Any = None

    async def connect_redis(self) -> None:
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self._redis_url, decode_responses=True, socket_timeout=5,
            )
            await self._redis.ping()
            logger.info("Redis connected at %s", self._redis_url)
        except Exception:
            logger.warning("Redis unavailable, using memory-only cache")
            self._redis = None

    async def get(self, key: str) -> Optional[str]:
        if key in self._memory:
            return self._memory[key]
        if self._redis:
            try:
                val = await self._redis.get(key)
                if val is not None:
                    self._memory[key] = val
                return val
            except Exception:
                pass
        return None

    async def set(self, key: str, value: str, ttl_seconds: int = 300) -> None:
        self._memory[key] = value
        if self._redis:
            try:
                await self._redis.set(key, value, ex=ttl_seconds)
            except Exception:
                pass

    async def get_price(self, symbol: str) -> Optional[Decimal]:
        val = await self.get(f"price:{symbol}")
        return Decimal(val) if val else None

    async def set_price(self, symbol: str, price: Decimal) -> None:
        await self.set(f"price:{symbol}", str(price), ttl_seconds=60)

    # ── Cross-Asset Tick Momentum (SPY/QQQ/IWM consensus) ──────

    async def publish_tick_momentum(
        self, symbol: str, direction: float, speed: float,
        roc_pct: float, price: float,
    ) -> None:
        data = json.dumps({
            "direction": direction, "speed": speed,
            "roc_pct": roc_pct, "price": price, "ts": time.time(),
        })
        if self._redis:
            try:
                await self._redis.set(f"cross_asset:tick:{symbol}", data, ex=30)
            except Exception:
                pass

    async def get_cross_asset_consensus(
        self, own_symbol: str, max_age_seconds: float = 15.0,
    ) -> Optional[dict]:
        if not self._redis:
            return None

        now = time.time()
        assets: dict[str, dict] = {}

        for symbol in self.ALL_SYMBOLS:
            try:
                raw = await self._redis.get(f"cross_asset:tick:{symbol}")
                if raw is None:
                    continue
                data = json.loads(raw)
                if now - data.get("ts", 0) > max_age_seconds:
                    continue
                assets[symbol] = data
            except Exception:
                continue

        if len(assets) < 2:
            return None

        directions = [d["direction"] for d in assets.values()]
        avg_direction = sum(directions) / len(directions)
        strength = abs(avg_direction)
        majority_dir = 1.0 if avg_direction > 0 else (-1.0 if avg_direction < 0 else 0.0)
        aligned = sum(1 for d in directions if d == majority_dir)

        return {
            "direction": avg_direction,
            "strength": strength,
            "aligned_count": aligned,
            "total_count": len(assets),
            "assets": {s: {"direction": d["direction"], "speed": d["speed"]} for s, d in assets.items()},
        }

    # ── Quant Signal Cache ─────────────────────────────────────

    async def set_quant_signal(self, signal_type: str, data: dict, ttl: int = 120) -> None:
        await self.set(f"quant:{signal_type}", json.dumps(data), ttl_seconds=ttl)

    async def get_quant_signal(self, signal_type: str) -> Optional[dict]:
        val = await self.get(f"quant:{signal_type}")
        return json.loads(val) if val else None

    async def close(self) -> None:
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:
                pass
