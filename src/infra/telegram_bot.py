"""Telegram bot for interactive control of the Zero-DTE Options Scalper.

Commands:
    /status     — Account overview, engine status, circuit breaker state
    /positions  — Open positions with live P&L
    /orders     — Recent order history
    /prices     — Live underlying prices
    /vix        — VIX regime and IV analysis
    /pnl        — Daily P&L summary and trade stats
    /halt       — Manually halt trading (circuit breaker)
    /resume     — Resume trading after halt
    /config     — Show current strategy configuration
    /trades     — Recent trade history with P&L
    /chain <SYM> — 0DTE options chain snapshot (default: SPY)
    /help       — List all commands
"""

import asyncio
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Optional

import aiohttp

from src.infra.config import Settings

logger = logging.getLogger(__name__)

# Max message length for Telegram
MAX_MSG = 4000


def _truncate(text: str, limit: int = MAX_MSG) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 20] + "\n...(truncated)"


class TelegramBot:
    """Polling-based Telegram bot for trade management commands."""

    def __init__(self, settings: Settings) -> None:
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._settings = settings
        self._enabled = bool(self._token and self._chat_id)
        self._session: Optional[aiohttp.ClientSession] = None
        self._offset = 0
        self._running = False

        # These get injected by the engine or set up standalone
        self._trading_client = None
        self._stock_client = None
        self._data_client = None
        self._engine = None
        self._halted = False
        self._halt_reason = ""

    def set_engine(self, engine: Any) -> None:
        self._engine = engine

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _init_alpaca(self) -> None:
        """Initialize Alpaca clients for standalone mode."""
        if self._trading_client:
            return
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.historical.option import OptionHistoricalDataClient

            self._trading_client = TradingClient(
                api_key=self._settings.alpaca_api_key,
                secret_key=self._settings.alpaca_secret_key,
                paper=self._settings.alpaca_paper,
            )
            self._stock_client = StockHistoricalDataClient(
                api_key=self._settings.alpaca_api_key,
                secret_key=self._settings.alpaca_secret_key,
            )
            self._data_client = OptionHistoricalDataClient(
                api_key=self._settings.alpaca_api_key,
                secret_key=self._settings.alpaca_secret_key,
            )
            logger.info("Telegram bot: Alpaca clients initialized")
        except Exception:
            logger.exception("Telegram bot: Failed to init Alpaca")

    # ── Core API ──────────────────────────────────────────────

    async def send(self, text: str, parse_mode: str = "Markdown") -> None:
        if not self._enabled:
            return
        try:
            session = await self._get_session()
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload = {
                "chat_id": self._chat_id,
                "text": _truncate(text),
                "parse_mode": parse_mode,
            }
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 400:
                    # Markdown parse error — retry without formatting
                    payload.pop("parse_mode")
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)):
                        pass
                elif resp.status != 200:
                    body = await resp.text()
                    logger.error("Telegram send failed (%d): %s", resp.status, body)
        except Exception:
            logger.exception("Failed to send Telegram message")

    async def _get_updates(self) -> list[dict]:
        if not self._enabled:
            return []
        try:
            session = await self._get_session()
            url = f"https://api.telegram.org/bot{self._token}/getUpdates"
            params = {"offset": self._offset, "timeout": 10, "allowed_updates": '["message"]'}
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                return data.get("result", [])
        except asyncio.TimeoutError:
            return []
        except Exception:
            logger.exception("Failed to get Telegram updates")
            return []

    # ── Polling Loop ──────────────────────────────────────────

    async def start(self) -> None:
        """Start the Telegram bot polling loop."""
        if not self._enabled:
            logger.info("Telegram bot disabled (no token/chat_id)")
            return

        await self._init_alpaca()
        self._running = True
        logger.info("Telegram bot started (chat_id=%s)", self._chat_id)
        await self.send("*Zero-DTE Scalper Bot Online*\nSend /help for commands.")

        while self._running:
            try:
                updates = await self._get_updates()
                for update in updates:
                    self._offset = update["update_id"] + 1
                    msg = update.get("message", {})
                    text = msg.get("text", "").strip()
                    chat_id = str(msg.get("chat", {}).get("id", ""))

                    # Only respond to our configured chat
                    if chat_id != self._chat_id:
                        continue

                    if text.startswith("/"):
                        await self._handle_command(text)
            except Exception:
                logger.exception("Telegram polling error")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Command Router ────────────────────────────────────────

    async def _handle_command(self, text: str) -> None:
        parts = text.split()
        cmd = parts[0].lower().split("@")[0]  # Strip @botname
        args = parts[1:] if len(parts) > 1 else []

        handlers = {
            "/start": self._cmd_help,
            "/help": self._cmd_help,
            "/status": self._cmd_status,
            "/positions": self._cmd_positions,
            "/pos": self._cmd_positions,
            "/orders": self._cmd_orders,
            "/prices": self._cmd_prices,
            "/price": self._cmd_prices,
            "/vix": self._cmd_vix,
            "/pnl": self._cmd_pnl,
            "/halt": self._cmd_halt,
            "/resume": self._cmd_resume,
            "/config": self._cmd_config,
            "/trades": self._cmd_trades,
            "/chain": self._cmd_chain,
        }

        handler = handlers.get(cmd)
        if handler:
            try:
                await handler(args)
            except Exception as e:
                await self.send(f"Error: {e}")
                logger.exception("Command error: %s", cmd)
        else:
            await self.send(f"Unknown command: `{cmd}`\nSend /help for available commands.")

    # ── Commands ──────────────────────────────────────────────

    async def _cmd_help(self, args: list[str]) -> None:
        await self.send(
            "*Zero-DTE Scalper Commands*\n\n"
            "/status — Account + engine status\n"
            "/positions — Open positions\n"
            "/orders — Recent orders\n"
            "/prices — Underlying prices\n"
            "/vix — VIX regime analysis\n"
            "/pnl — Daily P&L summary\n"
            "/halt — Halt all trading\n"
            "/resume — Resume trading\n"
            "/config — Strategy config\n"
            "/trades — Trade history\n"
            "/chain SPY — Options chain\n"
            "/help — This message"
        )

    async def _cmd_status(self, args: list[str]) -> None:
        lines = ["*Account Status*\n"]

        # Account info
        if self._trading_client:
            try:
                acct = self._trading_client.get_account()
                equity = float(acct.equity)
                last_eq = float(acct.last_equity) if acct.last_equity else equity
                daily_pnl = equity - last_eq
                pnl_pct = (daily_pnl / last_eq * 100) if last_eq else 0
                sign = "+" if daily_pnl >= 0 else ""

                lines.append(f"Equity: `${equity:,.2f}`")
                lines.append(f"Buying Power: `${float(acct.buying_power):,.2f}`")
                lines.append(f"Cash: `${float(acct.cash):,.2f}`")
                lines.append(f"Day P&L: `{sign}${daily_pnl:,.2f} ({sign}{pnl_pct:.2f}%)`")
                lines.append(f"Day Trades: `{acct.daytrade_count}`")
                lines.append(f"PDT: `{'Yes' if acct.pattern_day_trader else 'No'}`")
                lines.append(f"Status: `{str(acct.status).split('.')[-1]}`")
            except Exception as e:
                lines.append(f"Error fetching account: {e}")

        # Engine state
        lines.append("\n*Engine State*\n")
        if self._engine:
            st = self._engine.status()
            lines.append(f"Running: `{st['running']}`")
            lines.append(f"Open Positions: `{st['open_positions']}`")
            lines.append(f"Pending Orders: `{st['pending_orders']}`")
            if st.get("vix", {}).get("level"):
                lines.append(f"VIX: `{st['vix']['level']} ({st['vix']['regime']})`")
            lines.append(f"Macro Blackout: `{st['macro_blackout']}`")
        else:
            lines.append("Engine: `standalone mode`")

        # Halt status
        if self._halted:
            lines.append(f"\n*HALTED*: {self._halt_reason}")

        await self.send("\n".join(lines))

    async def _cmd_positions(self, args: list[str]) -> None:
        if not self._trading_client:
            await self.send("Not connected to Alpaca")
            return

        try:
            positions = self._trading_client.get_all_positions()
            if not positions:
                await self.send("*Positions*\nNo open positions.")
                return

            lines = ["*Open Positions*\n"]
            total_pl = 0.0
            for p in positions:
                pl = float(p.unrealized_pl) if p.unrealized_pl else 0
                total_pl += pl
                sign = "+" if pl >= 0 else ""
                plpc = float(p.unrealized_plpc) * 100 if p.unrealized_plpc else 0
                lines.append(
                    f"`{p.symbol}`\n"
                    f"  Qty: {p.qty} | Entry: ${float(p.avg_entry_price):.2f}\n"
                    f"  Current: ${float(p.current_price):.2f} | P&L: {sign}${pl:.2f} ({sign}{plpc:.1f}%)"
                )

            sign = "+" if total_pl >= 0 else ""
            lines.append(f"\nTotal Unrealized: `{sign}${total_pl:,.2f}`")
            await self.send("\n".join(lines))
        except Exception as e:
            await self.send(f"Error: {e}")

    async def _cmd_orders(self, args: list[str]) -> None:
        if not self._trading_client:
            await self.send("Not connected to Alpaca")
            return

        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus

            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=15)
            orders = self._trading_client.get_orders(req)
            if not orders:
                await self.send("*Orders*\nNo recent orders.")
                return

            lines = ["*Recent Orders*\n"]
            for o in orders[:15]:
                side = str(o.side).split(".")[-1].upper()
                status = str(o.status).split(".")[-1]
                price = f"${float(o.limit_price):.2f}" if o.limit_price else "MKT"
                filled = f"@ ${float(o.filled_avg_price):.2f}" if o.filled_avg_price else ""
                time_str = str(o.created_at)[:16] if o.created_at else ""
                lines.append(f"`{side}` {o.symbol} x{o.qty} {price} {filled} [{status}] {time_str}")

            await self.send("\n".join(lines))
        except Exception as e:
            await self.send(f"Error: {e}")

    async def _cmd_prices(self, args: list[str]) -> None:
        if not self._stock_client:
            await self.send("Not connected to Alpaca")
            return

        try:
            from alpaca.data.requests import StockSnapshotRequest

            symbols = self._settings.underlying_list
            snapshots = self._stock_client.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=symbols)
            )

            lines = ["*Underlying Prices*\n"]
            for sym in symbols:
                snap = snapshots.get(sym)
                if not snap:
                    lines.append(f"`{sym}`: no data")
                    continue
                price = float(snap.latest_trade.price) if snap.latest_trade else 0
                prev = float(snap.previous_daily_bar.close) if snap.previous_daily_bar else price
                chg = price - prev
                pct = (chg / prev * 100) if prev else 0
                sign = "+" if chg >= 0 else ""
                bar = snap.daily_bar
                vol = f"{int(bar.volume):,}" if bar else "?"
                lines.append(
                    f"`{sym}`: ${price:.2f} ({sign}{chg:.2f}, {sign}{pct:.2f}%)\n"
                    f"  Vol: {vol}"
                )

            await self.send("\n".join(lines))
        except Exception as e:
            await self.send(f"Error: {e}")

    async def _cmd_vix(self, args: list[str]) -> None:
        try:
            import yfinance as yf
            import numpy as np

            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1y", interval="1d")
            if hist.empty:
                await self.send("Failed to fetch VIX data")
                return

            current = float(hist["Close"].iloc[-1])
            closes = hist["Close"].values.tolist()
            lookback = closes[-252:] if len(closes) >= 252 else closes
            below = sum(1 for v in lookback if v < current)
            iv_pctile = (below / len(lookback)) * 100
            vix_high = max(lookback)
            vix_low = min(lookback)
            iv_range = vix_high - vix_low
            iv_rank = ((current - vix_low) / iv_range * 100) if iv_range > 0 else 50
            prev = closes[-2] if len(closes) >= 2 else current
            roc = ((current - prev) / prev * 100) if prev > 0 else 0

            if current > 35:
                regime = "CRISIS"
            elif current > 30:
                regime = "HIGH VOL"
            elif current < 12:
                regime = "LOW VOL"
            else:
                regime = "NORMAL"

            # RV-IV spread
            try:
                spy = yf.Ticker("SPY")
                spy_hist = spy.history(period="30d", interval="1d")
                returns = spy_hist["Close"].pct_change().dropna().values
                rv = float(np.std(returns) * np.sqrt(252) * 100)
                rv_iv = rv - current
            except Exception:
                rv_iv = 0

            sign_roc = "+" if roc >= 0 else ""
            sign_rv = "+" if rv_iv >= 0 else ""

            await self.send(
                f"*VIX Analysis*\n\n"
                f"Level: `{current:.2f}`\n"
                f"Regime: `{regime}`\n"
                f"IV Percentile: `{iv_pctile:.1f}%`\n"
                f"IV Rank: `{iv_rank:.1f}%`\n"
                f"1d Change: `{sign_roc}{roc:.2f}%`\n"
                f"RV-IV Spread: `{sign_rv}{rv_iv:.2f}`\n"
                f"52w Range: `{vix_low:.2f} — {vix_high:.2f}`"
            )
        except Exception as e:
            await self.send(f"Error: {e}")

    async def _cmd_pnl(self, args: list[str]) -> None:
        lines = ["*Daily P&L Summary*\n"]

        # Account P&L
        if self._trading_client:
            try:
                acct = self._trading_client.get_account()
                equity = float(acct.equity)
                last_eq = float(acct.last_equity) if acct.last_equity else equity
                daily_pnl = equity - last_eq
                pnl_pct = (daily_pnl / last_eq * 100) if last_eq else 0
                sign = "+" if daily_pnl >= 0 else ""
                lines.append(f"Account P&L: `{sign}${daily_pnl:,.2f} ({sign}{pnl_pct:.2f}%)`")
                lines.append(f"Portfolio: `${equity:,.2f}`")
            except Exception as e:
                lines.append(f"Error: {e}")

        # Trade stats from DB
        try:
            from src.data.trade_db import TradeDB
            db = TradeDB(self._settings.sqlite_db_path)
            db.connect()
            stats = db.get_trade_stats()
            db.close()

            if stats.get("total_closed", 0) > 0:
                lines.append(f"\n*Trade Stats*\n")
                lines.append(f"Total Closed: `{stats['total_closed']}`")
                lines.append(f"Wins: `{stats['wins']}` | Losses: `{stats['losses']}`")
                lines.append(f"Win Rate: `{stats['win_rate']:.1%}`")
                total_sign = "+" if stats['total_pnl'] >= 0 else ""
                lines.append(f"Total P&L: `{total_sign}${stats['total_pnl']:,.2f}`")
                lines.append(f"Avg P&L: `${stats['avg_pnl']:,.2f}`")
                lines.append(f"Best: `+${stats['best_trade']:,.2f}`")
                lines.append(f"Worst: `${stats['worst_trade']:,.2f}`")
        except Exception:
            pass

        await self.send("\n".join(lines))

    async def _cmd_halt(self, args: list[str]) -> None:
        reason = " ".join(args) if args else "Manual halt via Telegram"
        self._halted = True
        self._halt_reason = reason

        if self._engine:
            self._engine._cb.force_halt(reason)

        await self.send(f"*TRADING HALTED*\nReason: {reason}\n\nSend /resume to restart.")

    async def _cmd_resume(self, args: list[str]) -> None:
        self._halted = False
        self._halt_reason = ""

        if self._engine:
            self._engine._cb.force_resume()

        await self.send("*Trading Resumed*\nPosition size at 50% until full confidence.")

    async def _cmd_config(self, args: list[str]) -> None:
        s = self._settings
        await self.send(
            f"*Strategy Configuration*\n\n"
            f"Mode: `{s.trading_mode.value}`\n"
            f"Underlyings: `{', '.join(s.underlying_list)}`\n"
            f"Target Delta: `{s.target_delta}`\n"
            f"Premium: `${s.min_premium} — ${s.max_premium}`\n"
            f"Spread Ratio: `>= {s.min_spread_ratio}`\n"
            f"Confidence: `>= {s.signal_confidence_threshold}`\n\n"
            f"*Entry/Exit*\n"
            f"Window: `{s.entry_start} — {s.entry_cutoff} ET`\n"
            f"Hard Close: `{s.hard_close} ET`\n"
            f"Profit Target: `{s.pt_profit_target_pct:.0%}`\n"
            f"Stop Loss: `{s.sl_stop_loss_pct:.0%}`\n"
            f"Trailing: `{s.sl_trailing_pct:.0%}`\n\n"
            f"*Risk*\n"
            f"Kelly: `{s.kelly_fraction}`\n"
            f"Max Position: `{s.max_position_pct}`\n"
            f"Drawdown Halt: `{s.daily_drawdown_halt}`\n"
            f"Max Delta: `±{s.max_portfolio_delta}`\n"
            f"Max Gamma: `±{s.max_portfolio_gamma}`\n\n"
            f"*Ensemble Weights*\n"
            f"Technical: `{s.weight_technical:.0%}`\n"
            f"Tick Mom.: `{s.weight_tick_momentum:.0%}`\n"
            f"GEX: `{s.weight_gex:.0%}`\n"
            f"Flow: `{s.weight_flow:.0%}`\n"
            f"VIX: `{s.weight_vix:.0%}`\n"
            f"Internals: `{s.weight_internals:.0%}`\n"
            f"Sentiment: `{s.weight_sentiment:.0%}`"
        )

    async def _cmd_trades(self, args: list[str]) -> None:
        try:
            from src.data.trade_db import TradeDB
            db = TradeDB(self._settings.sqlite_db_path)
            db.connect()
            trades = db.get_trade_history(limit=10)
            db.close()

            if not trades:
                await self.send("*Trade History*\nNo trades recorded yet.")
                return

            lines = ["*Recent Trades*\n"]
            for t in trades:
                pnl = float(t.get("pnl") or 0)
                sign = "+" if pnl >= 0 else ""
                entry = t.get("entry_time", "")[:16]
                sym = t.get("contract_symbol", "?")
                otype = t.get("option_type", "")
                strike = t.get("strike", "")
                entry_p = t.get("entry_premium", "?")
                exit_p = t.get("exit_premium", "—")
                reason = t.get("exit_reason", "")
                status = f"{sign}${pnl:.2f}" if t.get("exit_time") else "OPEN"
                lines.append(
                    f"`{entry}` {otype.upper()} ${strike}\n"
                    f"  Entry: ${entry_p} | Exit: ${exit_p} | {status}\n"
                    f"  {reason}"
                )

            await self.send("\n".join(lines))
        except Exception as e:
            await self.send(f"Error: {e}")

    async def _cmd_chain(self, args: list[str]) -> None:
        underlying = args[0].upper() if args else "SPY"
        if underlying not in self._settings.underlying_list:
            await self.send(f"Unknown underlying: `{underlying}`\nAvailable: {', '.join(self._settings.underlying_list)}")
            return

        if not self._trading_client or not self._data_client:
            await self.send("Not connected to Alpaca")
            return

        try:
            from alpaca.trading.requests import GetOptionContractsRequest

            today = date.today()
            params = GetOptionContractsRequest(
                underlying_symbols=[underlying],
                expiration_date=today.isoformat(),
                status="active",
            )
            response = self._trading_client.get_option_contracts(params)
            contracts = response.option_contracts if response and hasattr(response, 'option_contracts') else []

            if not contracts:
                await self.send(f"*{underlying} Chain*\nNo 0DTE contracts available today.")
                return

            # Get snapshots for a subset
            symbols = [c.symbol for c in contracts[:40]]
            snapshots = {}
            try:
                from alpaca.data.requests import OptionSnapshotRequest
                snapshots = self._data_client.get_option_snapshot(
                    OptionSnapshotRequest(symbol_or_symbols=symbols)
                )
            except Exception:
                pass

            calls = []
            puts = []
            for c in contracts:
                snap = snapshots.get(c.symbol)
                quote = snap.latest_quote if snap else None
                greeks = snap.greeks if snap and hasattr(snap, 'greeks') and snap.greeks else None
                bid = f"{float(quote.bid_price):.2f}" if quote and quote.bid_price else "—"
                ask = f"{float(quote.ask_price):.2f}" if quote and quote.ask_price else "—"
                delta = f"{float(greeks.delta):.2f}" if greeks and greeks.delta else "—"
                ctype = str(c.type).split(".")[-1].lower()
                entry = {
                    "strike": str(c.strike_price),
                    "bid": bid, "ask": ask, "delta": delta
                }
                if ctype == "call":
                    calls.append(entry)
                else:
                    puts.append(entry)

            calls.sort(key=lambda x: float(x["strike"]))
            puts.sort(key=lambda x: float(x["strike"]))

            lines = [f"*{underlying} 0DTE Chain* ({today.isoformat()})\n"]

            if calls:
                lines.append("*Calls*")
                lines.append("`Strike  Bid    Ask    Delta`")
                for c in calls[:15]:
                    lines.append(f"`{c['strike']:>6s}  {c['bid']:>5s}  {c['ask']:>5s}  {c['delta']:>5s}`")

            if puts:
                lines.append("\n*Puts*")
                lines.append("`Strike  Bid    Ask    Delta`")
                for p in puts[:15]:
                    lines.append(f"`{p['strike']:>6s}  {p['bid']:>5s}  {p['ask']:>5s}  {p['delta']:>5s}`")

            await self.send("\n".join(lines))
        except Exception as e:
            await self.send(f"Error: {e}")

    # ── Alert Convenience Methods ─────────────────────────────

    async def alert_trade_opened(self, underlying: str, option_type: str,
                                  strike: str, contracts: int, premium: float,
                                  confidence: int, reason: str) -> None:
        sign = "C" if option_type == "call" else "P"
        await self.send(
            f"*TRADE OPENED*\n\n"
            f"`{underlying}` ${strike}{sign} x{contracts}\n"
            f"Premium: `${premium:.2f}`\n"
            f"Confidence: `{confidence}/100`\n"
            f"Reason: {reason}"
        )

    async def alert_trade_closed(self, underlying: str, option_type: str,
                                  strike: str, contracts: int,
                                  entry_premium: float, exit_premium: float,
                                  pnl: float, hold_time: str, reason: str) -> None:
        sign = "+" if pnl >= 0 else ""
        emoji = "W" if pnl >= 0 else "L"
        await self.send(
            f"*TRADE CLOSED [{emoji}]*\n\n"
            f"`{underlying}` ${strike} {option_type}\n"
            f"Entry: `${entry_premium:.2f}` | Exit: `${exit_premium:.2f}`\n"
            f"P&L: `{sign}${pnl:.2f}`\n"
            f"Hold: `{hold_time}`\n"
            f"Reason: {reason}"
        )

    async def alert_circuit_breaker(self, reason: str) -> None:
        await self.send(f"*CIRCUIT BREAKER TRIGGERED*\n\nReason: {reason}\n\nSend /resume to restart.")

    async def alert_daily_summary(self, total_pnl: float, trades: int,
                                   win_rate: float, portfolio: float) -> None:
        sign = "+" if total_pnl >= 0 else ""
        await self.send(
            f"*End of Day Summary*\n\n"
            f"P&L: `{sign}${total_pnl:,.2f}`\n"
            f"Trades: `{trades}`\n"
            f"Win Rate: `{win_rate:.1%}`\n"
            f"Portfolio: `${portfolio:,.2f}`"
        )


def run_bot() -> None:
    """Run the Telegram bot standalone."""
    import os
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.infra.config import get_settings
    settings = get_settings()
    bot = TelegramBot(settings)

    print(f"\n  Zero-DTE Telegram Bot")
    print(f"  Chat ID: {settings.telegram_chat_id}")
    print(f"  Mode: {settings.trading_mode.value}\n")

    asyncio.run(bot.start())


if __name__ == "__main__":
    run_bot()
