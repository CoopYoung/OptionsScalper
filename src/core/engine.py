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
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional
from zoneinfo import ZoneInfo

from src.analytics.trade_analyzer import TradeAnalyzer
from src.data.alpaca_client import AlpacaClient
from src.execution.order_manager import OrderManager
from src.data.alpaca_stream import AlpacaStream
from src.data.cache import PriceCache
from src.data.options_chain import OptionsChainManager
from src.data.trade_db import TradeDB
from src.infra.alerts import AlertManager
from src.infra.config import Settings
from src.quant.flow import FlowAnalyzer
from src.quant.gex import GEXAnalyzer
from src.quant.internals import MarketInternals
from src.quant.macro import MacroCalendar
from src.quant.optionsai import OptionsAIAnalyzer
from src.quant.sentiment import SentimentAggregator
from src.quant.vix import VIXRegimeDetector
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.exit_manager import ExitContext, ExitManager
from src.risk.manager import OptionsRiskManager
from src.strategy.base import TradeDirection
from src.strategy.signals import compute_all_signals
from src.strategy.weight_adapter import WeightAdapter
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
        self._alerts = AlertManager(settings)
        self._cb = CircuitBreaker(settings)
        self._risk = OptionsRiskManager(settings, self._cb)
        self._exit_mgr = ExitManager(settings)
        self._chain_mgr = OptionsChainManager(settings, self._client)
        self._analyzer = TradeAnalyzer(self._db)
        self._order_mgr = OrderManager(self._client)

        # ── Quant Layer ────────────────────────────────────────
        self._vix = VIXRegimeDetector(settings)
        self._gex = GEXAnalyzer(settings)
        self._flow = FlowAnalyzer(settings)
        self._sentiment = SentimentAggregator(settings)
        self._macro = MacroCalendar(settings)
        self._internals = MarketInternals(settings)
        self._optionsai = OptionsAIAnalyzer(settings)

        # ── Strategy ───────────────────────────────────────────
        self._weight_adapter = WeightAdapter(
            settings, self._analyzer,
            min_trades=settings.adaptive_min_trades,
        )
        self._strategy = ZeroDTEStrategy(
            settings, self._chain_mgr,
            self._vix, self._gex, self._flow,
            self._sentiment, self._macro, self._internals,
            self._optionsai,
            weight_adapter=self._weight_adapter,
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
        """Initialize components and run the daily trading cycle.

        Designed for overnight Docker deployment:
        - Starts dashboard + Telegram bot immediately (always available)
        - Waits for market open if started outside trading hours
        - Runs trading loops during market hours
        - Closes positions at hard_close, then idles until next trading day
        - Repeats indefinitely until stopped
        """
        logger.info("=" * 60)
        logger.info("Zero-DTE Scalper starting (%s mode)", self._settings.trading_mode.value)
        logger.info("Underlyings: %s", self._settings.underlying_list)
        logger.info("=" * 60)

        # Connect to services (persistent — survive across trading days)
        self._db.connect()
        await self._cache.connect_redis()

        # Load persisted state
        state = self._db.load_portfolio_state()
        if state:
            self._risk.set_portfolio_value(state["portfolio_value"])
            self._risk.set_day_start_value(state["day_start_value"])
            logger.info("Restored portfolio: $%s", state["portfolio_value"])

        # Recover orphaned positions from previous crash/restart
        await self._reconcile_positions()

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

        # Start web dashboard (always on, even outside market hours)
        if self._settings.web_enabled:
            from src.web.engine_dashboard import Dashboard
            self._dashboard = Dashboard(self, port=self._settings.web_port)
            await self._dashboard.start()

        # Start Telegram bot (always on for remote monitoring)
        await self._start_telegram_bot()

        # Register stream callbacks
        self._stream.on_trade(self._on_underlying_trade)

        # ── Daily trading cycle (runs forever) ────────────────
        self._running = True
        while self._running:
            try:
                await self._wait_for_market_open()
                if not self._running:
                    break

                # Fresh daily setup
                await self._pre_market_setup()

                logger.info("Market open — starting trading loops")
                await self._send_alert("Market open — trading loops started")

                # Run trading loops until market close
                await self._run_trading_session()

                # End of day cleanup
                await self._end_of_day()

            except Exception:
                logger.exception("Daily cycle error — will retry next trading day")
                await asyncio.sleep(60)  # Brief pause before retrying

    async def _wait_for_market_open(self) -> None:
        """Sleep until 15 minutes before market open (9:15 AM ET).

        This gives time for pre-market setup before entry_start.
        """
        while self._running:
            now_et = datetime.now(ET)
            weekday = now_et.weekday()  # 0=Mon, 6=Sun

            # Skip weekends
            if weekday >= 5:
                next_monday = now_et + timedelta(days=(7 - weekday))
                wake_at = next_monday.replace(hour=9, minute=15, second=0, microsecond=0)
                wait_secs = (wake_at - now_et).total_seconds()
                logger.info("Weekend — sleeping until Monday %s ET (%.0f hours)",
                            wake_at.strftime("%H:%M"), wait_secs / 3600)
                await self._sleep_interruptible(wait_secs)
                continue

            current_time = now_et.time()
            pre_market = datetime.strptime("09:15", "%H:%M").time()
            market_close = datetime.strptime("16:00", "%H:%M").time()

            # If we're before pre-market, sleep until 9:15 AM ET
            if current_time < pre_market:
                wake_at = now_et.replace(hour=9, minute=15, second=0, microsecond=0)
                wait_secs = (wake_at - now_et).total_seconds()
                logger.info("Pre-market — sleeping until %s ET (%.0f min)",
                            wake_at.strftime("%H:%M"), wait_secs / 60)
                await self._sleep_interruptible(wait_secs)
                return

            # If market is already open, start immediately
            if current_time < market_close:
                logger.info("Market is open — starting immediately")
                return

            # After market close — sleep until tomorrow 9:15 AM ET
            tomorrow = now_et + timedelta(days=1)
            wake_at = tomorrow.replace(hour=9, minute=15, second=0, microsecond=0)
            wait_secs = (wake_at - now_et).total_seconds()
            logger.info("After hours — sleeping until tomorrow %s ET (%.0f hours)",
                        wake_at.strftime("%H:%M"), wait_secs / 3600)
            await self._sleep_interruptible(wait_secs)

    async def _sleep_interruptible(self, seconds: float) -> None:
        """Sleep that can be interrupted by engine stop."""
        try:
            # Sleep in 30-second chunks so we can respond to shutdown
            remaining = seconds
            while remaining > 0 and self._running:
                chunk = min(remaining, 30)
                await asyncio.sleep(chunk)
                remaining -= chunk
        except asyncio.CancelledError:
            pass

    async def _run_trading_session(self) -> None:
        """Run all trading loops until market close (4 PM ET)."""
        # Start WebSocket stream
        stream_task = asyncio.create_task(self._stream.start())

        # Run loops concurrently
        loop_tasks = [
            asyncio.create_task(self._fast_loop()),
            asyncio.create_task(self._quant_loop()),
            asyncio.create_task(self._strategy_loop()),
            asyncio.create_task(self._chain_refresh_loop()),
        ]

        # Wait until market close
        while self._running:
            now_et = datetime.now(ET)
            if now_et.time() >= datetime.strptime("16:00", "%H:%M").time():
                logger.info("Market closed (4:00 PM ET) — stopping trading loops")
                break
            await asyncio.sleep(10)

        # Stop all loops
        self._running = False  # Temporarily — signals loops to exit
        for task in loop_tasks:
            task.cancel()
        stream_task.cancel()

        # Wait for tasks to finish
        await asyncio.gather(*loop_tasks, stream_task, return_exceptions=True)
        await self._stream.stop()

        # Re-enable for next day
        self._running = True

    async def _end_of_day(self) -> None:
        """End-of-day cleanup: close positions, persist state, send analytics."""
        logger.info("End of day — closing all positions")

        # Close any remaining positions
        await self._close_all_positions("End of day")

        # Persist state
        self._persist_state()

        # Generate analytics report
        analytics_report_text = ""
        try:
            report = self._analyzer.daily_report()
            if report.get("total_trades", 0) > 0:
                analytics_report_text = self._analyzer.format_telegram_summary(report)
                logger.info(
                    "Daily analytics: %d trades, %.1f%% win rate, $%.2f P&L",
                    report["total_trades"],
                    report["performance"]["win_rate"] * 100,
                    report["performance"]["total_pnl"],
                )
        except Exception:
            logger.exception("Failed to generate daily analytics report")

        # Send daily P&L summary via Telegram (with analytics appended)
        risk_status = self._risk.status()
        daily_pnl = Decimal(str(risk_status.get("daily_pnl", 0)))
        trades_today = risk_status.get("trades_today", 0)
        win_rate_val = risk_status.get("win_rate", 0)
        if isinstance(win_rate_val, str):
            win_rate_val = 0.0
        portfolio_val = Decimal(str(risk_status.get("portfolio_value", 0)))

        await self._alerts.daily_summary(
            total_pnl=daily_pnl,
            trades=trades_today,
            win_rate=float(win_rate_val),
            portfolio=portfolio_val,
            report=analytics_report_text,
        )

        # Attempt weight recalibration (no-op if disabled or insufficient data)
        try:
            result = self._weight_adapter.maybe_recalibrate()
            if result:
                await self._send_alert(self._weight_adapter.format_status())
        except Exception:
            logger.exception("Weight recalibration failed")

        logger.info("End of day complete — will resume tomorrow")

    async def _start_telegram_bot(self) -> None:
        """Start the Telegram command bot if credentials are configured."""
        if not self._settings.telegram_bot_token:
            logger.info("Telegram bot token not set — skipping")
            return

        try:
            from src.infra.telegram_bot import TelegramBot
            self._telegram_bot = TelegramBot(self._settings)
            self._telegram_bot.set_engine(self)
            asyncio.create_task(self._telegram_bot.start())
            logger.info("Telegram bot started")
        except ImportError:
            logger.warning("Telegram bot module not available")
        except Exception:
            logger.exception("Failed to start Telegram bot")

    async def _send_alert(self, message: str) -> None:
        """Send an alert via Telegram (best-effort)."""
        try:
            await self._alerts.send(message)
        except Exception:
            logger.debug("Alert send failed: %s", message[:50])

    async def stop(self) -> None:
        """Graceful shutdown — persist positions first, then attempt to close."""
        logger.info("Shutting down...")
        self._running = False

        # Persist open positions FIRST (in case close fails or is interrupted)
        self._persist_open_positions()
        self._persist_state()

        # Attempt to close all positions (best-effort)
        try:
            await self._close_all_positions("Engine shutdown")
        except Exception:
            logger.exception("Error closing positions during shutdown — positions persisted for recovery")

        # If all positions were closed, clear the persisted records
        if not self._open_positions:
            self._db.clear_open_positions()

        try:
            await self._optionsai.close()
        except Exception:
            pass
        try:
            await self._stream.stop()
        except Exception:
            pass
        try:
            await self._cache.close()
        except Exception:
            pass

        # Stop Telegram bot
        if hasattr(self, '_telegram_bot'):
            try:
                await self._telegram_bot.stop()
            except Exception:
                pass

        await self._alerts.send("🔴 Bot stopped — shutdown complete")
        await self._alerts.close()
        self._db.close()
        logger.info("Shutdown complete")

    # ── Pre-Market Setup ──────────────────────────────────────

    async def _pre_market_setup(self) -> None:
        """Run health checks, load macro calendar, refresh initial data."""
        logger.info("Pre-market setup...")

        # ── Health checks ──────────────────────────────────────
        checks: dict[str, bool] = {}

        # 1. Alpaca connection + equity
        try:
            account = await self._client.get_account()
            if account:
                equity = Decimal(account["equity"])
                self._risk.set_portfolio_value(equity)
                checks["Alpaca connected"] = True
                checks[f"Equity ${equity:,.2f}"] = float(equity) >= 100
                # PDT check (only relevant if <$25k)
                if float(equity) < 25000:
                    pdt_status = self._risk.status().get("pdt", {})
                    pdt_remaining = pdt_status.get("remaining", 0)
                    checks[f"PDT budget ({pdt_remaining} remaining)"] = pdt_remaining > 0
                else:
                    checks["PDT exempt (>$25k)"] = True
            else:
                checks["Alpaca connected"] = False
        except Exception:
            logger.exception("Alpaca health check failed")
            checks["Alpaca connected"] = False

        # 2. Redis connectivity
        try:
            redis_ok = await self._cache.ping()
            checks["Redis connected"] = redis_ok
        except Exception:
            checks["Redis connected"] = False

        # 3. Load macro calendar + check
        try:
            events = await self._macro.load_today()
            if events:
                high = [e for e in events if e.impact.value == "high"]
                if high:
                    logger.warning("HIGH IMPACT EVENTS TODAY: %s", [e.name for e in high])
                    checks[f"Macro events ({len(high)} high-impact)"] = True
                else:
                    checks["Macro calendar loaded"] = True
            else:
                checks["Macro calendar loaded"] = True
            checks["No macro blackout"] = not self._macro.is_blackout()
        except Exception:
            logger.exception("Macro calendar failed")
            checks["Macro calendar loaded"] = False

        # 4. VIX regime check
        try:
            vix_signals = await self._vix.update()
            regime = vix_signals.regime.value
            vix_level = vix_signals.vix_level
            checks[f"VIX {vix_level:.1f} ({regime})"] = regime != "crisis"
            logger.info("VIX regime: %s (%.1f)", regime, vix_level)
        except Exception:
            logger.exception("VIX check failed")
            checks["VIX data"] = False

        # 5. Load OptionsAI earnings calendar
        try:
            earnings = await self._optionsai.load_earnings()
            if earnings:
                symbols = [f"{e.symbol} ({e.time})" for e in earnings]
                logger.warning("EARNINGS TODAY: %s", symbols)
                checks[f"Earnings ({len(earnings)} today)"] = True
            else:
                checks["No earnings conflicts"] = True
        except Exception:
            checks["OptionsAI earnings"] = False

        # 6. Load options chains
        chains_loaded = 0
        for underlying in self._settings.underlying_list:
            try:
                await self._chain_mgr.refresh_chain(underlying)
                chain = self._chain_mgr.get_chain(underlying)
                if chain:
                    chains_loaded += 1
            except Exception:
                logger.exception("Failed to load chain for %s", underlying)
        total = len(self._settings.underlying_list)
        checks[f"Options chains ({chains_loaded}/{total})"] = chains_loaded == total

        # ── Report results ─────────────────────────────────────
        all_ok = all(checks.values())
        for check, passed in checks.items():
            log_fn = logger.info if passed else logger.warning
            log_fn("Health check: %s %s", "✓" if passed else "✗", check)

        await self._alerts.startup_status(checks)

        if not all_ok:
            failed = [k for k, v in checks.items() if not v]
            logger.warning("Pre-market issues: %s", failed)

        # Reset daily state
        self._risk.reset_daily()
        self._internals.reset_daily()

    # ── Position Recovery ──────────────────────────────────────

    async def _reconcile_positions(self) -> None:
        """Reconcile local state with Alpaca + DB after restart.

        Handles three scenarios:
        1. Position in DB + on Alpaca → restore to _open_positions (normal recovery)
        2. Position in DB + NOT on Alpaca → already closed, clean up DB
        3. Position on Alpaca + NOT in DB → orphan, adopt for exit management
        """
        # Load persisted positions from DB
        db_positions = self._db.load_open_positions()
        db_symbols = {p["contract_symbol"] for p in db_positions}

        # Get actual positions from Alpaca
        alpaca_positions = await self._client.get_positions() or []
        # Filter to option positions only (symbol format: SPY260320C00550000)
        alpaca_option_positions = [
            p for p in alpaca_positions
            if len(p.get("symbol", "")) > 10  # Option symbols are long
        ]
        alpaca_symbols = {p["symbol"] for p in alpaca_option_positions}

        recovered = 0
        cleaned = 0
        orphaned = 0

        # Scenario 1: Restore positions that exist in both DB and Alpaca
        for pos_data in db_positions:
            symbol = pos_data["contract_symbol"]
            if symbol in alpaca_symbols:
                entry_time = datetime.fromisoformat(pos_data["entry_time"])
                entry_premium = Decimal(pos_data["entry_premium"])
                peak_premium = Decimal(pos_data.get("peak_premium", pos_data["entry_premium"]))
                entry_spot = pos_data.get("entry_spot", 0) or 0
                peak_spot = pos_data.get("peak_spot", entry_spot) or entry_spot

                self._open_positions[symbol] = {
                    "underlying": pos_data["underlying"],
                    "option_type": pos_data["option_type"],
                    "strike": pos_data["strike"],
                    "entry_premium": entry_premium,
                    "peak_premium": peak_premium,
                    "qty": pos_data["contracts"],
                    "entry_time": entry_time,
                    "order_id": pos_data.get("order_id", ""),
                    "contract": None,  # Greeks not persisted — will update on next chain refresh
                    "confidence": pos_data.get("entry_confidence", 0),
                    "entry_spot": entry_spot,
                    "peak_spot": peak_spot,
                    "max_favorable_pnl": Decimal("0"),
                    "max_adverse_pnl": Decimal("0"),
                }
                recovered += 1
                logger.info("Recovered position: %s (%d contracts @ $%s)",
                            symbol, pos_data["contracts"], pos_data["entry_premium"])

        # Scenario 2: Clean up DB positions that no longer exist on Alpaca
        for pos_data in db_positions:
            symbol = pos_data["contract_symbol"]
            if symbol not in alpaca_symbols:
                self._db.remove_open_position(symbol)
                cleaned += 1
                logger.warning("Cleaned stale DB position: %s (not on Alpaca)", symbol)

        # Scenario 3: Adopt orphaned Alpaca positions not tracked in DB
        for alpaca_pos in alpaca_option_positions:
            symbol = alpaca_pos["symbol"]
            if symbol not in db_symbols and symbol not in self._open_positions:
                qty = int(alpaca_pos.get("qty", 0))
                if qty <= 0:
                    continue

                avg_entry = Decimal(alpaca_pos.get("avg_entry_price", "0"))
                # Parse underlying from symbol (first 3-4 chars before date)
                underlying = self._parse_underlying(symbol)
                option_type = "call" if "C" in symbol[6:] else "put"

                self._open_positions[symbol] = {
                    "underlying": underlying,
                    "option_type": option_type,
                    "strike": "0",  # Not available from Alpaca position API
                    "entry_premium": avg_entry,
                    "peak_premium": avg_entry,
                    "qty": qty,
                    "entry_time": datetime.now(timezone.utc),  # Approximate
                    "order_id": "",
                    "contract": None,
                    "confidence": 0,
                    "entry_spot": 0,
                    "peak_spot": 0,
                    "max_favorable_pnl": Decimal("0"),
                    "max_adverse_pnl": Decimal("0"),
                }
                orphaned += 1
                logger.warning(
                    "Adopted orphan position: %s (%d contracts @ $%s) — will manage exits",
                    symbol, qty, avg_entry,
                )

        if recovered or cleaned or orphaned:
            logger.info(
                "Position reconciliation: %d recovered, %d cleaned, %d orphans adopted",
                recovered, cleaned, orphaned,
            )
            if orphaned:
                await self._send_alert(
                    f"⚠️ Adopted {orphaned} orphan position(s) from Alpaca — "
                    "managing exits with approximate entry data"
                )
        else:
            logger.info("Position reconciliation: no positions to recover")

    @staticmethod
    def _parse_underlying(option_symbol: str) -> str:
        """Extract underlying ticker from OCC option symbol.

        Format: SPY260320C00550000 → SPY
        The underlying is everything before the 6-digit date.
        """
        # Find where the date starts (6 consecutive digits)
        for i in range(len(option_symbol)):
            if option_symbol[i:i+6].isdigit():
                return option_symbol[:i]
        return option_symbol[:3]  # Fallback

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
                    snapshots = await self._client.get_snapshots([symbol])
                    snap = snapshots.get(symbol)
                    if snap:
                        current_premium = snap.mid
                    else:
                        # No quote available — still check time-based exits
                        # (hard close, max hold) using last known premium
                        current_premium = pos.get("last_premium", pos["entry_premium"])
                        logger.warning(
                            "No quote for %s, using last known premium $%.2f",
                            symbol, float(current_premium),
                        )
                else:
                    bid = Decimal(str(quote["bid"]))
                    ask = Decimal(str(quote["ask"]))
                    current_premium = (bid + ask) / 2 if bid > 0 and ask > 0 else Decimal("0")

                # Track last known premium for fallback
                if current_premium > 0:
                    pos["last_premium"] = current_premium
                else:
                    # Zero premium: use last known, or entry as final fallback
                    current_premium = pos.get("last_premium", pos["entry_premium"])

                entry_premium = pos["entry_premium"]
                peak_premium = pos.get("peak_premium", entry_premium)
                underlying = pos["underlying"]
                current_spot = float(self._last_prices.get(underlying, Decimal("0")))
                option_type = pos.get("option_type", "call")

                # Update peak premium
                if current_premium > peak_premium:
                    pos["peak_premium"] = current_premium

                # Update peak spot (favorable direction)
                entry_spot = pos.get("entry_spot", 0)
                peak_spot = pos.get("peak_spot", entry_spot)
                if option_type == "call" and current_spot > peak_spot:
                    pos["peak_spot"] = current_spot
                elif option_type == "put" and current_spot < peak_spot:
                    pos["peak_spot"] = current_spot

                # Track max favorable / adverse P&L for analytics
                pnl = (current_premium - entry_premium) * pos["qty"] * 100
                if pnl > pos.get("max_favorable_pnl", Decimal("0")):
                    pos["max_favorable_pnl"] = pnl
                if pnl < pos.get("max_adverse_pnl", Decimal("0")):
                    pos["max_adverse_pnl"] = pnl

                # Build exit context and evaluate
                ctx = ExitContext(
                    symbol=symbol,
                    current_premium=current_premium,
                    entry_premium=entry_premium,
                    peak_premium=pos["peak_premium"],
                    entry_time=pos["entry_time"],
                    entry_spot=entry_spot,
                    current_spot=current_spot,
                    peak_spot=pos.get("peak_spot", entry_spot),
                    contract=pos.get("contract"),
                    direction=option_type,
                )
                decision = self._exit_mgr.evaluate(ctx, now)

                if decision.should_exit:
                    await self._close_position(
                        symbol, current_premium, decision.reason,
                        urgency=decision.urgency,
                    )

            except Exception:
                logger.exception("Exit check error for %s", symbol)

    async def _check_pending_orders(self) -> None:
        """Check managed orders: walk prices, handle fills/cancels."""
        try:
            events = await self._order_mgr.check_and_walk()
        except Exception:
            logger.exception("Order manager check error")
            return

        for event in events:
            try:
                etype = event["type"]

                if etype == "filled":
                    info = event.get("metadata", {})
                    info["slippage"] = event.get("slippage", 0)
                    await self._handle_fill(
                        event["order_id"],
                        {
                            "filled_avg_price": str(event["fill_price"]),
                            "filled_qty": str(event["qty"]),
                            "status": "filled",
                        },
                        info,
                    )
                    logger.info(
                        "Order filled: %s %s %d @ $%.2f (slippage=$%.4f, latency=%.1fs, walks=%d)",
                        event["side"], event["symbol"], event["qty"],
                        event["fill_price"], event.get("slippage", 0),
                        event.get("fill_latency_secs", 0), event.get("walk_steps", 0),
                    )

                elif etype == "timeout_cancelled":
                    logger.info(
                        "Order timed out: %s %s after %.1fs",
                        event["side"], event["symbol"], event.get("age_secs", 0),
                    )

                elif etype == "cancelled":
                    logger.info("Order %s: %s", event["order_id"], event.get("reason", "cancelled"))

                elif etype == "walked":
                    logger.debug(
                        "Order walked: %s step %d → $%.2f",
                        event["symbol"], event.get("walk_step", 0), event.get("new_limit", 0),
                    )

            except Exception:
                logger.exception("Error handling order event: %s", event)

    async def _handle_fill(self, order_id: str, order: dict, info: dict) -> None:
        """Handle a filled order."""
        symbol = info["symbol"]
        side = info["side"]
        filled_price = Decimal(order.get("filled_avg_price", "0"))
        qty = int(order.get("filled_qty", info.get("qty", 0)))

        if side == "buy":
            # Opening position — capture underlying price for directional trailing
            underlying = info["underlying"]
            entry_spot = float(self._last_prices.get(underlying, Decimal("0")))

            self._open_positions[symbol] = {
                "underlying": underlying,
                "option_type": info["option_type"],
                "strike": info["strike"],
                "entry_premium": filled_price,
                "peak_premium": filled_price,
                "qty": qty,
                "entry_time": datetime.now(timezone.utc),
                "order_id": order_id,
                "contract": info.get("contract"),
                "confidence": info.get("confidence", 0),
                "entry_spot": entry_spot,
                "peak_spot": entry_spot,
                "max_favorable_pnl": Decimal("0"),
                "max_adverse_pnl": Decimal("0"),
                "signal_mid": info.get("limit_price", 0),
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

            # Persist to DB for crash recovery
            self._db.save_open_position(
                contract_symbol=symbol,
                underlying=info["underlying"],
                option_type=info["option_type"],
                strike=str(info["strike"]),
                side=info["direction"],
                contracts=qty,
                entry_premium=filled_price,
                entry_time=datetime.now(timezone.utc),
                order_id=order_id,
                confidence=info.get("confidence", 0),
                peak_premium=filled_price,
                entry_spot=entry_spot,
                peak_spot=entry_spot,
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
                score_breakdown=info.get("score_breakdown", {}),
                delta=info.get("delta", 0),
                spot_price=info.get("spot_price", 0),
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
                    self._update_flow(),
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
        """Update GEX and chain IV for all underlyings."""
        for underlying in self._settings.underlying_list:
            chain = self._chain_mgr.get_chain(underlying)
            price = self._last_prices.get(underlying, Decimal("0"))
            await self._gex.update(underlying, chain, float(price))
            # Piggyback: update per-underlying chain IV percentile
            if chain:
                self._vix.update_chain_iv(underlying, chain)

    async def _update_flow(self) -> None:
        """Update flow for all underlyings, passing aggregated chain data."""
        all_chains = []
        for underlying in self._settings.underlying_list:
            chain = self._chain_mgr.get_chain(underlying)
            if chain:
                all_chains.extend(chain)
        await self._flow.update(all_chains)

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
            logger.info("Risk blocked %s %s: %s", underlying, signal.direction.value, reason)
            await self._alerts.signal_rejected(
                underlying, reason, confidence=signal.confidence,
            )
            return

        # Compute position size
        vix_mult = self._vix.latest.size_multiplier if self._vix.latest else 1.0
        contracts = self._risk.compute_position_size(signal, vix_mult)
        if contracts <= 0:
            return

        # Place order
        await self._place_entry_order(signal, contracts)

    async def _place_entry_order(self, signal, contracts: int) -> None:
        """Place an entry order via OrderManager with adaptive pricing."""
        contract = signal.contract
        if not contract:
            return

        bid = float(contract.bid) if contract.bid else 0
        ask = float(contract.ask) if contract.ask else 0
        if bid <= 0 or ask <= 0:
            bid = float(contract.mid) * 0.95
            ask = float(contract.mid) * 1.05

        metadata = {
            "symbol": contract.symbol,
            "underlying": signal.underlying,
            "option_type": contract.option_type,
            "strike": contract.strike,
            "expiration": contract.expiration,
            "side": "buy",
            "direction": signal.direction.value,
            "qty": contracts,
            "limit_price": (bid + ask) / 2,
            "confidence": signal.confidence,
            "reason": signal.reason,
            "score_breakdown": signal.score_breakdown or {},
            "delta": contract.delta,
            "spot_price": float(self._last_prices.get(signal.underlying, 0)),
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

        order_id = await self._order_mgr.submit_entry(
            symbol=contract.symbol,
            qty=contracts,
            bid=bid,
            ask=ask,
            metadata=metadata,
        )

        if order_id:
            logger.info(
                "ORDER: %s %s %d x $%s %s mid=$%.2f (conf=%d)",
                signal.direction.value, signal.underlying, contracts,
                contract.strike, contract.option_type,
                (bid + ask) / 2, signal.confidence,
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

    async def _close_position(
        self, symbol: str, current_premium: Decimal, reason: str,
        urgency: str = "normal",
    ) -> None:
        """Close a position by selling.

        Urgency levels affect limit price:
            normal:    mid * 0.99 (slightly below mid)
            urgent:    mid * 0.97 (more aggressive)
            immediate: mid * 0.95 (accept worse fill for speed)
        """
        pos = self._open_positions.get(symbol)
        if not pos:
            return

        qty = pos["qty"]
        entry_premium = pos["entry_premium"]

        # Urgency-based limit pricing
        price_multiplier = {"normal": "0.99", "urgent": "0.97", "immediate": "0.95"}
        mult = Decimal(price_multiplier.get(urgency, "0.99"))
        limit_price = float(current_premium * mult)

        order = await self._client.place_order(
            symbol=symbol,
            side="sell",
            qty=qty,
            limit_price=limit_price,
        )

        if not order:
            # Sell order failed — escalate to force close via Alpaca
            logger.warning(
                "Limit sell failed for %s, attempting force close", symbol,
            )
            result = await self._client.close_position(symbol)
            if result:
                # Force close accepted — clean up position tracking
                pnl = (current_premium - entry_premium) * qty * 100
                self._risk.record_close(symbol, pnl)
                await self._stream.unsubscribe_options([symbol])
                self._open_positions.pop(symbol, None)
                self._db.remove_open_position(symbol)
                logger.info(
                    "FORCE CLOSED: %s PnL=$%.2f (%s)", symbol, float(pnl), reason,
                )
            else:
                logger.error(
                    "STUCK POSITION: %s — both limit sell and force close failed",
                    symbol,
                )
            return

        if order:
            pnl = (current_premium - entry_premium) * qty * 100
            self._risk.record_close(symbol, pnl)

            # Compute enriched analytics data
            hold_time = datetime.now(timezone.utc) - pos["entry_time"]
            hold_seconds = int(hold_time.total_seconds())
            underlying = pos["underlying"]
            current_spot = float(self._last_prices.get(underlying, Decimal("0")))
            entry_spot = pos.get("entry_spot", 0)
            underlying_move_pct = (
                (current_spot - entry_spot) / entry_spot
                if entry_spot > 0 else 0
            )

            self._db.record_trade_close(
                contract_symbol=symbol,
                exit_premium=current_premium,
                pnl=pnl,
                exit_reason=reason,
                hold_seconds=hold_seconds,
                max_favorable_pnl=pos.get("max_favorable_pnl", Decimal("0")),
                max_adverse_pnl=pos.get("max_adverse_pnl", Decimal("0")),
                underlying_move_pct=round(underlying_move_pct, 6),
            )

            await self._stream.unsubscribe_options([symbol])
            self._open_positions.pop(symbol, None)
            self._db.remove_open_position(symbol)

            logger.info(
                "CLOSED: %s %d @ $%.2f PnL=$%.2f (%s) hold=%s urgency=%s",
                symbol, qty, float(current_premium), float(pnl), reason,
                str(hold_time).split(".")[0], urgency,
            )
            # Compute day P&L for alert context
            day_pnl = sum(
                float(p.get("pnl", 0))
                for p in self._open_positions.values()
                if "pnl" in p
            ) + float(pnl)

            await self._alerts.trade_closed(
                underlying=underlying,
                strike=str(pos["strike"]),
                option_type=pos["option_type"],
                contracts=qty,
                entry_premium=float(entry_premium),
                exit_premium=float(current_premium),
                pnl=float(pnl),
                pnl_pct=float((current_premium - entry_premium) / entry_premium) if entry_premium else 0,
                hold_time=str(hold_time).split(".")[0],
                reason=reason,
                underlying_move_pct=round(underlying_move_pct, 4),
                day_pnl=day_pnl,
            )

    async def _close_all_positions(self, reason: str) -> None:
        """Close all open positions — escalates to force close on failure."""
        for symbol in list(self._open_positions.keys()):
            try:
                snapshots = await self._client.get_snapshots([symbol])
                snap = snapshots.get(symbol)
                if snap:
                    await self._close_position(
                        symbol, snap.mid, reason, urgency="immediate",
                    )
                else:
                    # No quote — force close via Alpaca with cleanup
                    logger.warning("No snapshot for %s, force closing", symbol)
                    await self._client.close_position(symbol)
                    pos = self._open_positions.get(symbol, {})
                    entry_premium = pos.get("entry_premium", Decimal("0"))
                    last_premium = pos.get("last_premium", entry_premium)
                    qty = pos.get("qty", 1)
                    pnl = (last_premium - entry_premium) * qty * 100
                    self._risk.record_close(symbol, pnl)
                    self._db.remove_open_position(symbol)
                    self._open_positions.pop(symbol, None)
                    logger.info(
                        "FORCE CLOSED: %s est_PnL=$%.2f (%s)",
                        symbol, float(pnl), reason,
                    )
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

    def _persist_open_positions(self) -> None:
        """Persist all open positions to DB for crash recovery."""
        for symbol, pos in self._open_positions.items():
            try:
                self._db.save_open_position(
                    contract_symbol=symbol,
                    underlying=pos["underlying"],
                    option_type=pos["option_type"],
                    strike=str(pos["strike"]),
                    side=pos.get("direction", "BUY_CALL"),
                    contracts=pos["qty"],
                    entry_premium=pos["entry_premium"],
                    entry_time=pos["entry_time"],
                    order_id=pos.get("order_id", ""),
                    confidence=pos.get("confidence", 0),
                    peak_premium=pos.get("peak_premium", pos["entry_premium"]),
                    entry_spot=pos.get("entry_spot", 0),
                    peak_spot=pos.get("peak_spot", 0),
                )
            except Exception:
                logger.exception("Failed to persist position %s", symbol)

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
