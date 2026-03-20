"""Trading engine: orchestrates 3 async loops for 0DTE scalping.

Loops:
    Fast loop (5s):   Poll options quotes, check exits, update tick momentum
    Quant loop (30s):  Refresh VIX, GEX, flow, sentiment, internals
    Strategy loop (15s): Evaluate entries using full signal ensemble
"""

import asyncio
import json
import logging
from collections import deque
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo

from src.data.alpaca_client import AlpacaClient
from src.data.alpaca_stream import AlpacaStream
from src.data.cache import PriceCache
from src.data.options_chain import OptionsChainManager
from src.data.trade_db import TradeDB
from src.infra.alerts import TelegramAlerts
from src.infra.config import Settings
from src.quant.flow import FlowAnalyzer
from src.quant.gex import GEXAnalyzer
from src.quant.internals import MarketInternals
from src.quant.macro import MacroCalendar
from src.quant.optionsai import OptionsAIAnalyzer
from src.quant.sentiment import SentimentAggregator
from src.quant.vix import VIXRegimeDetector
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.manager import OptionsRiskManager
from src.strategy.base import TradeDirection
from src.strategy.signals import compute_all_signals
from src.strategy.zero_dte import ZeroDTEStrategy

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


class TradingEngine:
    """Main trading engine for 0DTE options scalping."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._running = False

        # ── Core Components ────────────────────────────────────
        self._client = AlpacaClient(settings)
        self._stream = AlpacaStream(settings)
        self._cache = PriceCache(settings)
        self._db = TradeDB(settings.sqlite_db_path)
        self._alerts = TelegramAlerts(settings)
        self._cb = CircuitBreaker(settings)
        self._risk = OptionsRiskManager(settings, self._cb)
        self._chain_mgr = OptionsChainManager(settings, self._client)

        # ── Quant Layer ────────────────────────────────────────
        self._vix = VIXRegimeDetector(settings)
        self._gex = GEXAnalyzer(settings)
        self._flow = FlowAnalyzer(settings)
        self._sentiment = SentimentAggregator(settings)
        self._macro = MacroCalendar(settings)
        self._internals = MarketInternals(settings)
        self._optionsai = OptionsAIAnalyzer(settings)

        # ── Strategy ───────────────────────────────────────────
        self._strategy = ZeroDTEStrategy(
            settings, self._chain_mgr,
            self._vix, self._gex, self._flow,
            self._sentiment, self._macro, self._internals,
            self._optionsai,
        )

        # ── State ──────────────────────────────────────────────
        self._candles: dict[str, deque] = {
            sym: deque(maxlen=settings.candle_cache_size)
            for sym in settings.underlying_list
        }
        self._open_positions: dict[str, dict] = {}
        self._pending_orders: dict[str, dict] = {}
        self._last_prices: dict[str, Decimal] = {}

    async def start(self) -> None:
        """Initialize all components and start trading loops."""
        logger.info("=" * 60)
        logger.info("Zero-DTE Scalper starting (%s mode)", self._settings.trading_mode.value)
        logger.info("Underlyings: %s", self._settings.underlying_list)
        logger.info("=" * 60)

        # Connect to services
        self._db.connect()
        await self._cache.connect_redis()

        # Load persisted state
        state = self._db.load_portfolio_state()
        if state:
            self._risk.set_portfolio_value(state["portfolio_value"])
            self._risk.set_day_start_value(state["day_start_value"])
            logger.info("Restored portfolio: $%s", state["portfolio_value"])

        # Connect to Alpaca
        connected = await self._client.connect()
        if not connected:
            logger.error("Failed to connect to Alpaca — aborting")
            return

        # Load account info
        account = await self._client.get_account()
        if account:
            self._risk.set_portfolio_value(Decimal(account["equity"]))
            logger.info("Account equity: $%s", account["equity"])

        # Pre-market setup
        await self._pre_market_setup()

        # Register stream callbacks
        self._stream.on_trade(self._on_underlying_trade)

        # Start web dashboard
        if self._settings.web_enabled:
            from src.web.app import Dashboard
            self._dashboard = Dashboard(self, port=self._settings.web_port)
            await self._dashboard.start()

        # Start all loops
        self._running = True
        logger.info("Starting trading loops...")

        await asyncio.gather(
            self._stream.start(),
            self._fast_loop(),
            self._quant_loop(),
            self._strategy_loop(),
            self._chain_refresh_loop(),
        )

    async def stop(self) -> None:
        """Graceful shutdown."""
        logger.info("Shutting down...")
        self._running = False

        # Close all positions
        await self._close_all_positions("Engine shutdown")

        # Persist state
        self._persist_state()

        await self._optionsai.close()
        await self._stream.stop()
        await self._cache.close()
        self._db.close()
        logger.info("Shutdown complete")

    # ── Pre-Market Setup ──────────────────────────────────────

    async def _pre_market_setup(self) -> None:
        """Load macro calendar, refresh initial data."""
        logger.info("Pre-market setup...")

        # Load macro calendar
        events = await self._macro.load_today()
        if events:
            high = [e for e in events if e.impact.value == "high"]
            if high:
                logger.warning("HIGH IMPACT EVENTS TODAY: %s", [e.name for e in high])
                await self._alerts.send(
                    f"HIGH IMPACT: {', '.join(e.name for e in high)}",
                )

        # Initial VIX check
        vix_signals = await self._vix.update()
        logger.info("VIX regime: %s (%.1f)", vix_signals.regime.value, vix_signals.vix_level)

        # Load OptionsAI earnings calendar
        earnings = await self._optionsai.load_earnings()
        if earnings:
            logger.warning(
                "EARNINGS TODAY: %s",
                [f"{e.symbol} ({e.time})" for e in earnings],
            )

        # Load options chains
        for underlying in self._settings.underlying_list:
            await self._chain_mgr.refresh_chain(underlying)

        # Reset daily state
        self._risk.reset_daily()
        self._internals.reset_daily()

    # ── Fast Loop (5s): Exits + Tick Momentum ─────────────────

    async def _fast_loop(self) -> None:
        """Check exits, update option quotes, tick momentum."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                # Check exit conditions for all open positions
                await self._check_exits(now)

                # Check pending orders for fills
                await self._check_pending_orders()

                # Publish tick momentum to Redis for cross-asset consensus
                await self._publish_tick_momentum()

            except Exception:
                logger.exception("Fast loop error")

            await asyncio.sleep(self._settings.fast_loop_seconds)

    async def _check_exits(self, now: datetime) -> None:
        """Check all open positions for exit conditions."""
        for symbol, pos in list(self._open_positions.items()):
            try:
                # Get current quote
                quote = self._stream.get_option_quote(symbol)
                if not quote:
                    # Fallback: REST snapshot
                    snapshots = await self._client.get_snapshots([symbol])
                    snap = snapshots.get(symbol)
                    if snap:
                        current_premium = snap.mid
                    else:
                        continue
                else:
                    bid = Decimal(str(quote["bid"]))
                    ask = Decimal(str(quote["ask"]))
                    current_premium = (bid + ask) / 2 if bid > 0 and ask > 0 else Decimal("0")

                if current_premium <= 0:
                    continue

                entry_premium = pos["entry_premium"]
                peak_premium = pos.get("peak_premium", entry_premium)

                # Update peak
                if current_premium > peak_premium:
                    pos["peak_premium"] = current_premium

                # Check exit conditions
                should_exit, reason = self._risk.should_exit(
                    symbol, current_premium, peak_premium, entry_premium, now,
                )

                if should_exit:
                    await self._close_position(symbol, current_premium, reason)

            except Exception:
                logger.exception("Exit check error for %s", symbol)

    async def _check_pending_orders(self) -> None:
        """Check if pending orders have been filled."""
        for order_id, order_info in list(self._pending_orders.items()):
            try:
                order = await self._client.get_order(order_id)
                if not order:
                    continue

                status = order["status"]

                if status == "filled":
                    await self._handle_fill(order_id, order, order_info)
                elif status in ("cancelled", "expired", "rejected"):
                    logger.info("Order %s: %s", order_id, status)
                    self._pending_orders.pop(order_id, None)

            except Exception:
                logger.exception("Order check error for %s", order_id)

    async def _handle_fill(self, order_id: str, order: dict, info: dict) -> None:
        """Handle a filled order."""
        symbol = info["symbol"]
        side = info["side"]
        filled_price = Decimal(order.get("filled_avg_price", "0"))
        qty = int(order.get("filled_qty", info.get("qty", 0)))

        if side == "buy":
            # Opening position
            self._open_positions[symbol] = {
                "underlying": info["underlying"],
                "option_type": info["option_type"],
                "strike": info["strike"],
                "entry_premium": filled_price,
                "peak_premium": filled_price,
                "qty": qty,
                "entry_time": datetime.now(timezone.utc),
                "order_id": order_id,
                "contract": info.get("contract"),
                "confidence": info.get("confidence", 0),
            }

            # Subscribe to option quotes
            await self._stream.subscribe_options([symbol])

            # Record in DB
            self._db.record_trade_open(
                underlying=info["underlying"],
                contract_symbol=symbol,
                option_type=info["option_type"],
                strike=info["strike"],
                expiration=info.get("expiration", ""),
                side=info["direction"],
                contracts=qty,
                entry_premium=filled_price,
                entry_time=datetime.now(timezone.utc),
                order_id=order_id,
                strategy=self._strategy.name,
                confidence=info.get("confidence", 0),
                greeks_json=info.get("greeks_json", ""),
                quant_json=info.get("quant_json", ""),
            )

            # Risk tracking
            self._risk.record_open(
                symbol, info["underlying"], qty, filled_price,
                info.get("contract"),
            )

            logger.info(
                "FILLED BUY: %s %d @ $%.2f (order=%s)",
                symbol, qty, float(filled_price), order_id,
            )
            await self._alerts.trade_opened(
                underlying=info["underlying"],
                strike=str(info["strike"]),
                option_type=info["option_type"],
                contracts=qty,
                premium=float(filled_price),
                confidence=info.get("confidence", 0),
                reason=info.get("reason", ""),
            )

        self._pending_orders.pop(order_id, None)

    # ── Quant Loop (30s): Refresh Quant Signals ───────────────

    async def _quant_loop(self) -> None:
        """Refresh VIX, GEX, flow, sentiment, macro, internals."""
        while self._running:
            try:
                # Run quant updates in parallel
                await asyncio.gather(
                    self._vix.update(),
                    self._macro.update(),
                    self._internals.update(),
                    self._update_gex(),
                    self._flow.update(),
                    self._update_optionsai(),
                    return_exceptions=True,
                )

                # Sentiment is slower (FinBERT), update less frequently
                # Run every 4th quant cycle (~2 min)
                if not hasattr(self, '_quant_cycle'):
                    self._quant_cycle = 0
                self._quant_cycle += 1
                if self._quant_cycle % 4 == 0:
                    await self._sentiment.update()

                # Cache quant signals for dashboard
                await self._cache_quant_signals()

            except Exception:
                logger.exception("Quant loop error")

            await asyncio.sleep(self._settings.quant_loop_seconds)

    async def _update_gex(self) -> None:
        """Update GEX for all underlyings."""
        for underlying in self._settings.underlying_list:
            chain = self._chain_mgr.get_chain(underlying)
            price = self._last_prices.get(underlying, Decimal("0"))
            await self._gex.update(underlying, chain, float(price))

    async def _update_optionsai(self) -> None:
        """Update OptionsAI signals for all underlyings."""
        for underlying in self._settings.underlying_list:
            price = self._last_prices.get(underlying, Decimal("0"))
            await self._optionsai.update(underlying, float(price))

    async def _cache_quant_signals(self) -> None:
        """Cache quant signals in Redis for dashboard."""
        vix = self._vix.latest
        if vix:
            await self._cache.set_quant_signal("vix", {
                "level": vix.vix_level,
                "regime": vix.regime.value,
                "iv_percentile": vix.iv_percentile,
                "size_multiplier": vix.size_multiplier,
            })

        macro = self._macro.latest
        if macro:
            await self._cache.set_quant_signal("macro", {
                "is_blackout": macro.is_blackout,
                "reason": macro.blackout_reason,
                "minutes_to_event": macro.minutes_to_event,
            })

        for sym in self._settings.underlying_list:
            oai = self._optionsai.get_latest(sym)
            if oai:
                await self._cache.set_quant_signal(f"optionsai_{sym}", {
                    "iv_skew": oai.iv_skew,
                    "move_amount": oai.move_amount,
                    "strategy_bias": oai.strategy_bias,
                    "implied_high": oai.implied_high,
                    "implied_low": oai.implied_low,
                })

    # ── Strategy Loop (15s): Entry Evaluation ─────────────────

    async def _strategy_loop(self) -> None:
        """Evaluate entry signals for each underlying."""
        while self._running:
            try:
                for underlying in self._settings.underlying_list:
                    await self._evaluate_entry(underlying)

            except Exception:
                logger.exception("Strategy loop error")

            await asyncio.sleep(self._settings.strategy_loop_seconds)

    async def _evaluate_entry(self, underlying: str) -> None:
        """Evaluate entry for a single underlying."""
        price = self._last_prices.get(underlying)
        if not price:
            return

        # Compute technical signals from candle data
        candles = list(self._candles.get(underlying, []))
        if len(candles) < 20:
            return

        closes = [Decimal(str(c["close"])) for c in candles]
        tech_signals = compute_all_signals(
            closes, candles,
            rsi_period=self._settings.rsi_period,
            rsi_overbought=self._settings.rsi_overbought,
            rsi_oversold=self._settings.rsi_oversold,
            macd_fast=self._settings.macd_fast,
            macd_slow=self._settings.macd_slow,
            macd_signal=self._settings.macd_signal,
            bb_period=self._settings.bb_period,
            bb_std=self._settings.bb_std,
        )

        # Get tick momentum
        momentum = self._stream.get_momentum(underlying)

        # Run strategy evaluation
        signal = self._strategy.evaluate(
            underlying=underlying,
            current_price=price,
            signals=tech_signals,
            momentum=momentum,
        )

        if not signal.should_trade:
            return

        # Risk check
        can_trade, reason = self._risk.can_trade(signal)
        if not can_trade:
            logger.debug("Risk blocked %s %s: %s", underlying, signal.direction.value, reason)
            return

        # Compute position size
        vix_mult = self._vix.latest.size_multiplier if self._vix.latest else 1.0
        contracts = self._risk.compute_position_size(signal, vix_mult)
        if contracts <= 0:
            return

        # Place order
        await self._place_entry_order(signal, contracts)

    async def _place_entry_order(self, signal, contracts: int) -> None:
        """Place a limit order for entry."""
        contract = signal.contract
        if not contract:
            return

        # Limit price: slightly above mid for better fill probability
        limit_price = float(contract.mid * Decimal("1.01"))

        order = await self._client.place_order(
            symbol=contract.symbol,
            side="buy",
            qty=contracts,
            limit_price=limit_price,
        )

        if order:
            self._pending_orders[order["id"]] = {
                "symbol": contract.symbol,
                "underlying": signal.underlying,
                "option_type": contract.option_type,
                "strike": contract.strike,
                "expiration": contract.expiration,
                "side": "buy",
                "direction": signal.direction.value,
                "qty": contracts,
                "limit_price": limit_price,
                "confidence": signal.confidence,
                "reason": signal.reason,
                "contract": contract,
                "greeks_json": json.dumps({
                    "delta": contract.delta,
                    "gamma": contract.gamma,
                    "theta": contract.theta,
                    "vega": contract.vega,
                    "iv": contract.iv,
                }),
                "quant_json": json.dumps(signal.score_breakdown),
            }
            logger.info(
                "ORDER: %s %s %d x $%s %s @ $%.2f (conf=%d)",
                signal.direction.value, signal.underlying, contracts,
                contract.strike, contract.option_type, limit_price, signal.confidence,
            )

    # ── Chain Refresh Loop ────────────────────────────────────

    async def _chain_refresh_loop(self) -> None:
        """Refresh options chains every 5 minutes."""
        while self._running:
            try:
                for underlying in self._settings.underlying_list:
                    await self._chain_mgr.refresh_chain(underlying)
                    await self._chain_mgr.refresh_snapshots(underlying)

            except Exception:
                logger.exception("Chain refresh error")

            await asyncio.sleep(300)  # 5 minutes

    # ── Position Management ───────────────────────────────────

    async def _close_position(self, symbol: str, current_premium: Decimal, reason: str) -> None:
        """Close a position by selling."""
        pos = self._open_positions.get(symbol)
        if not pos:
            return

        qty = pos["qty"]
        entry_premium = pos["entry_premium"]

        # Place sell order
        order = await self._client.place_order(
            symbol=symbol,
            side="sell",
            qty=qty,
            limit_price=float(current_premium * Decimal("0.99")),  # Slightly below mid
        )

        if order:
            # Estimate P&L
            pnl = (current_premium - entry_premium) * qty * 100
            self._risk.record_close(symbol, pnl)

            self._db.record_trade_close(
                contract_symbol=symbol,
                exit_premium=current_premium,
                pnl=pnl,
                exit_reason=reason,
            )

            # Unsubscribe from quotes
            await self._stream.unsubscribe_options([symbol])
            self._open_positions.pop(symbol, None)

            hold_time = datetime.now(timezone.utc) - pos["entry_time"]
            logger.info(
                "CLOSED: %s %d @ $%.2f PnL=$%.2f (%s) hold=%s",
                symbol, qty, float(current_premium), float(pnl), reason,
                str(hold_time).split(".")[0],
            )
            await self._alerts.trade_closed(
                underlying=pos["underlying"],
                strike=str(pos["strike"]),
                option_type=pos["option_type"],
                contracts=qty,
                entry_premium=float(entry_premium),
                exit_premium=float(current_premium),
                pnl=float(pnl),
                hold_time=str(hold_time).split(".")[0],
                reason=reason,
            )

    async def _close_all_positions(self, reason: str) -> None:
        """Close all open positions."""
        for symbol in list(self._open_positions.keys()):
            try:
                snapshots = await self._client.get_snapshots([symbol])
                snap = snapshots.get(symbol)
                if snap:
                    await self._close_position(symbol, snap.mid, reason)
                else:
                    # Force close via Alpaca
                    await self._client.close_position(symbol)
                    self._open_positions.pop(symbol, None)
            except Exception:
                logger.exception("Error closing position %s", symbol)

    # ── Callbacks ─────────────────────────────────────────────

    async def _on_underlying_trade(self, symbol: str, price: float, ts: float) -> None:
        """Handle underlying price tick."""
        self._last_prices[symbol] = Decimal(str(price))
        await self._cache.set_price(symbol, Decimal(str(price)))

        # Update VWAP
        self._internals.update_vwap(symbol, price, 1.0)

        # Build candle data (simple 1-min aggregation)
        self._update_candle(symbol, price, ts)

    def _update_candle(self, symbol: str, price: float, ts: float) -> None:
        """Simple 1-minute candle aggregation from ticks."""
        candles = self._candles[symbol]
        minute = int(ts / 60) * 60

        if candles and candles[-1].get("minute") == minute:
            # Update existing candle
            candle = candles[-1]
            candle["high"] = max(candle["high"], price)
            candle["low"] = min(candle["low"], price)
            candle["close"] = price
            candle["volume"] = candle.get("volume", 0) + 1
        else:
            # New candle
            candles.append({
                "minute": minute,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1,
            })

    async def _publish_tick_momentum(self) -> None:
        """Publish tick momentum to Redis for cross-asset consensus."""
        for symbol in self._settings.underlying_list:
            momentum = self._stream.get_momentum(symbol)
            if momentum and momentum.latest_price:
                await self._cache.publish_tick_momentum(
                    symbol=symbol,
                    direction=momentum.direction,
                    speed=momentum.speed,
                    roc_pct=momentum.roc_pct,
                    price=momentum.latest_price,
                )

    def _persist_state(self) -> None:
        """Save portfolio state to DB."""
        try:
            status = self._risk.status()
            self._db.save_portfolio_state(
                portfolio_value=Decimal(status["portfolio_value"]),
                daily_pnl=Decimal(status["daily_pnl"]),
                day_start_value=self._risk._day_start_value,
                daily_trades=int(status["trades_today"]),
            )
        except Exception:
            logger.exception("Failed to persist state")

    # ── Status ────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "running": self._running,
            "strategy": self._strategy.name,
            "underlyings": self._settings.underlying_list,
            "risk": self._risk.status(),
            "stream": self._stream.status(),
            "open_positions": len(self._open_positions),
            "pending_orders": len(self._pending_orders),
            "vix": {
                "level": self._vix.latest.vix_level if self._vix.latest else None,
                "regime": self._vix.latest.regime.value if self._vix.latest else None,
            },
            "macro_blackout": self._macro.is_blackout(),
            "optionsai": {
                sym: {
                    "iv_skew": oai.iv_skew,
                    "move_amount": oai.move_amount,
                    "strategy_bias": oai.strategy_bias,
                } if (oai := self._optionsai.get_latest(sym)) else None
                for sym in self._settings.underlying_list
            },
        }
