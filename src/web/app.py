"""Web dashboard for Zero-DTE Options Scalper.

Standalone aiohttp server that displays:
    - Account info (equity, buying power, PDT status)
    - Live underlying prices (SPY, QQQ, IWM)
    - Options chain snapshot
    - Quant signals panel (VIX, GEX, flow, internals, sentiment, macro)
    - Risk gauges (portfolio Greeks, drawdown, circuit breaker)
    - Open positions with P&L
    - Trade history / activity log
"""

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

import aiohttp
from aiohttp import web
import aiohttp_jinja2
import jinja2

from src.infra.config import Settings, get_settings

logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).parent


class DashboardData:
    """Fetches and caches data for the dashboard from Alpaca + quant modules."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._trading_client = None
        self._data_client = None
        self._stock_client = None
        self._connected = False
        self._cache: dict[str, Any] = {}
        self._cache_ts: dict[str, float] = {}

    async def connect(self) -> bool:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.option import OptionHistoricalDataClient
            from alpaca.data.historical.stock import StockHistoricalDataClient

            self._trading_client = TradingClient(
                api_key=self._settings.alpaca_api_key,
                secret_key=self._settings.alpaca_secret_key,
                paper=self._settings.alpaca_paper,
            )
            self._data_client = OptionHistoricalDataClient(
                api_key=self._settings.alpaca_api_key,
                secret_key=self._settings.alpaca_secret_key,
            )
            self._stock_client = StockHistoricalDataClient(
                api_key=self._settings.alpaca_api_key,
                secret_key=self._settings.alpaca_secret_key,
            )
            account = self._trading_client.get_account()
            self._connected = True
            logger.info("Dashboard connected to Alpaca: equity=$%s", account.equity)
            return True
        except Exception:
            logger.exception("Dashboard failed to connect to Alpaca")
            return False

    def _cache_valid(self, key: str, max_age: float = 10.0) -> bool:
        ts = self._cache_ts.get(key, 0)
        return (datetime.now(timezone.utc).timestamp() - ts) < max_age

    def _set_cache(self, key: str, data: Any) -> None:
        self._cache[key] = data
        self._cache_ts[key] = datetime.now(timezone.utc).timestamp()

    async def get_account(self) -> dict:
        if self._cache_valid("account", 5.0):
            return self._cache["account"]
        if not self._trading_client:
            return {}
        try:
            acct = self._trading_client.get_account()
            data = {
                "equity": str(acct.equity),
                "buying_power": str(acct.buying_power),
                "cash": str(acct.cash),
                "portfolio_value": str(acct.portfolio_value),
                "pattern_day_trader": acct.pattern_day_trader,
                "daytrade_count": acct.daytrade_count,
                "status": str(acct.status).split(".")[-1],
                "currency": str(acct.currency) if hasattr(acct, 'currency') else "USD",
                "last_equity": str(acct.last_equity) if hasattr(acct, 'last_equity') else str(acct.equity),
            }
            # Compute daily P&L
            equity = float(acct.equity)
            last_equity = float(acct.last_equity) if hasattr(acct, 'last_equity') and acct.last_equity else equity
            data["daily_pnl"] = f"{equity - last_equity:.2f}"
            data["daily_pnl_pct"] = f"{((equity - last_equity) / last_equity * 100) if last_equity else 0:.2f}"
            self._set_cache("account", data)
            return data
        except Exception:
            logger.exception("Failed to get account")
            return self._cache.get("account", {})

    async def get_positions(self) -> list[dict]:
        if self._cache_valid("positions", 5.0):
            return self._cache["positions"]
        if not self._trading_client:
            return []
        try:
            positions = self._trading_client.get_all_positions()
            result = []
            for p in positions:
                result.append({
                    "symbol": str(p.symbol),
                    "qty": str(p.qty),
                    "side": str(p.side),
                    "avg_entry_price": str(p.avg_entry_price),
                    "current_price": str(p.current_price),
                    "unrealized_pl": str(p.unrealized_pl),
                    "unrealized_plpc": str(p.unrealized_plpc) if hasattr(p, 'unrealized_plpc') else "0",
                    "market_value": str(p.market_value),
                    "asset_class": str(p.asset_class),
                })
            self._set_cache("positions", result)
            return result
        except Exception:
            logger.exception("Failed to get positions")
            return self._cache.get("positions", [])

    async def get_orders(self, limit: int = 50) -> list[dict]:
        if self._cache_valid("orders", 5.0):
            return self._cache["orders"]
        if not self._trading_client:
            return []
        try:
            from alpaca.trading.requests import GetOrdersRequest
            from alpaca.trading.enums import QueryOrderStatus
            req = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=limit)
            orders = self._trading_client.get_orders(req)
            result = []
            for o in orders:
                result.append({
                    "id": str(o.id),
                    "symbol": str(o.symbol),
                    "side": str(o.side).split(".")[-1],
                    "qty": str(o.qty),
                    "filled_qty": str(o.filled_qty) if o.filled_qty else "0",
                    "type": str(o.type).split(".")[-1] if o.type else "",
                    "status": str(o.status).split(".")[-1],
                    "limit_price": str(o.limit_price) if o.limit_price else "",
                    "filled_avg_price": str(o.filled_avg_price) if o.filled_avg_price else "",
                    "created_at": str(o.created_at)[:19] if o.created_at else "",
                    "filled_at": str(o.filled_at)[:19] if o.filled_at else "",
                })
            self._set_cache("orders", result)
            return result
        except Exception:
            logger.exception("Failed to get orders")
            return self._cache.get("orders", [])

    async def get_prices(self) -> dict[str, dict]:
        if self._cache_valid("prices", 8.0):
            return self._cache["prices"]
        if not self._stock_client:
            return {}
        try:
            from alpaca.data.requests import StockLatestQuoteRequest, StockSnapshotRequest
            symbols = self._settings.underlying_list
            snapshots = self._stock_client.get_stock_snapshot(
                StockSnapshotRequest(symbol_or_symbols=symbols)
            )
            result = {}
            for sym, snap in snapshots.items():
                trade = snap.latest_trade
                quote = snap.latest_quote
                bar = snap.daily_bar
                prev_bar = snap.previous_daily_bar
                prev_close = float(prev_bar.close) if prev_bar else 0
                current = float(trade.price) if trade else 0
                change = current - prev_close if prev_close else 0
                change_pct = (change / prev_close * 100) if prev_close else 0
                result[sym] = {
                    "price": f"{current:.2f}",
                    "change": f"{change:+.2f}",
                    "change_pct": f"{change_pct:+.2f}",
                    "bid": f"{float(quote.bid_price):.2f}" if quote else "0",
                    "ask": f"{float(quote.ask_price):.2f}" if quote else "0",
                    "volume": f"{int(bar.volume):,}" if bar else "0",
                    "high": f"{float(bar.high):.2f}" if bar else "0",
                    "low": f"{float(bar.low):.2f}" if bar else "0",
                    "open": f"{float(bar.open):.2f}" if bar else "0",
                    "prev_close": f"{prev_close:.2f}",
                    "vwap": f"{float(bar.vwap):.2f}" if bar and hasattr(bar, 'vwap') and bar.vwap else "0",
                }
            self._set_cache("prices", result)
            return result
        except Exception:
            logger.exception("Failed to get prices")
            return self._cache.get("prices", {})

    async def get_options_chain(self, underlying: str) -> list[dict]:
        cache_key = f"chain_{underlying}"
        if self._cache_valid(cache_key, 30.0):
            return self._cache[cache_key]
        if not self._trading_client or not self._data_client:
            return []
        try:
            from alpaca.trading.requests import GetOptionContractsRequest

            today = date.today()
            params = GetOptionContractsRequest(
                underlying_symbols=[underlying],
                expiration_date=today.isoformat(),
                status="active",
            )
            response = self._trading_client.get_option_contracts(params)
            if not response or not hasattr(response, 'option_contracts'):
                return []

            contracts = response.option_contracts or []
            if not contracts:
                return []

            # Get snapshots for top contracts
            symbols = [c.symbol for c in contracts[:80]]
            snapshots = {}
            if symbols:
                from alpaca.data.requests import OptionSnapshotRequest
                try:
                    snapshots = self._data_client.get_option_snapshot(
                        OptionSnapshotRequest(symbol_or_symbols=symbols)
                    )
                except Exception:
                    logger.warning("Failed to get option snapshots for %s", underlying)

            result = []
            for c in contracts:
                snap = snapshots.get(c.symbol)
                quote = snap.latest_quote if snap else None
                greeks = snap.greeks if snap and hasattr(snap, 'greeks') and snap.greeks else None
                result.append({
                    "symbol": c.symbol,
                    "type": str(c.type).split(".")[-1].lower() if c.type else "",
                    "strike": str(c.strike_price),
                    "expiration": str(c.expiration_date),
                    "bid": f"{float(quote.bid_price):.2f}" if quote and quote.bid_price else "0.00",
                    "ask": f"{float(quote.ask_price):.2f}" if quote and quote.ask_price else "0.00",
                    "last": f"{float(snap.latest_trade.price):.2f}" if snap and snap.latest_trade else "0.00",
                    "iv": f"{float(snap.implied_volatility)*100:.1f}" if snap and hasattr(snap, 'implied_volatility') and snap.implied_volatility else "—",
                    "delta": f"{float(greeks.delta):.3f}" if greeks and greeks.delta else "—",
                    "gamma": f"{float(greeks.gamma):.4f}" if greeks and greeks.gamma else "—",
                    "theta": f"{float(greeks.theta):.4f}" if greeks and greeks.theta else "—",
                    "vega": f"{float(greeks.vega):.4f}" if greeks and greeks.vega else "—",
                    "volume": str(snap.latest_trade.size) if snap and snap.latest_trade and hasattr(snap.latest_trade, 'size') else "0",
                    "open_interest": str(c.open_interest) if hasattr(c, 'open_interest') and c.open_interest else "—",
                })
            result.sort(key=lambda x: (x["type"], float(x["strike"])))
            self._set_cache(cache_key, result)
            return result
        except Exception:
            logger.exception("Failed to get options chain for %s", underlying)
            return self._cache.get(cache_key, [])

    async def get_vix(self) -> dict:
        if self._cache_valid("vix", 60.0):
            return self._cache["vix"]
        try:
            import yfinance as yf
            import numpy as np
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1y", interval="1d")
            if hist.empty:
                return {"level": 0, "regime": "unknown", "iv_percentile": 0}
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
                regime = "HIGH_VOL"
            elif current < 12:
                regime = "LOW_VOL"
            else:
                regime = "NORMAL"
            # Compute RV-IV spread
            try:
                spy = yf.Ticker("SPY")
                spy_hist = spy.history(period="30d", interval="1d")
                if len(spy_hist) >= 5:
                    returns = spy_hist["Close"].pct_change().dropna().values
                    rv = float(np.std(returns) * np.sqrt(252) * 100)
                else:
                    rv = current
            except Exception:
                rv = current
            data = {
                "level": round(current, 2),
                "regime": regime,
                "iv_percentile": round(iv_pctile, 1),
                "iv_rank": round(iv_rank, 1),
                "roc": round(roc, 2),
                "rv_iv_spread": round(rv - current, 2),
                "high_52w": round(vix_high, 2),
                "low_52w": round(vix_low, 2),
            }
            self._set_cache("vix", data)
            return data
        except Exception:
            logger.exception("Failed to get VIX")
            return self._cache.get("vix", {"level": 0, "regime": "unknown"})

    async def get_trade_history(self) -> list[dict]:
        """Load trade history from SQLite if available."""
        try:
            from src.data.trade_db import TradeDB
            db = TradeDB(self._settings.sqlite_db_path)
            db.connect()
            trades = db.get_trade_history(limit=100)
            stats = db.get_trade_stats()
            db.close()
            return {"trades": trades, "stats": stats}
        except Exception:
            return {"trades": [], "stats": {}}


def _json_serial(obj: Any) -> str:
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def create_app(settings: Settings = None) -> web.Application:
    """Create the aiohttp web application."""
    if settings is None:
        settings = get_settings()

    app = web.Application()
    data = DashboardData(settings)

    # Setup Jinja2 templates
    aiohttp_jinja2.setup(
        app,
        loader=jinja2.FileSystemLoader(str(BASE_DIR / "templates")),
    )

    # Store references
    app["settings"] = settings
    app["data"] = data

    # ── Routes ────────────────────────────────────────────────

    @aiohttp_jinja2.template("dashboard.html")
    async def index(request: web.Request) -> dict:
        return {
            "underlyings": settings.underlying_list,
            "trading_mode": settings.trading_mode.value,
            "web_port": settings.web_port,
        }

    async def api_account(request: web.Request) -> web.Response:
        result = await data.get_account()
        return web.json_response(result)

    async def api_prices(request: web.Request) -> web.Response:
        result = await data.get_prices()
        return web.json_response(result)

    async def api_positions(request: web.Request) -> web.Response:
        result = await data.get_positions()
        return web.json_response(result)

    async def api_orders(request: web.Request) -> web.Response:
        result = await data.get_orders()
        return web.json_response(result)

    async def api_chain(request: web.Request) -> web.Response:
        underlying = request.match_info.get("underlying", "SPY")
        result = await data.get_options_chain(underlying)
        return web.json_response(result)

    async def api_vix(request: web.Request) -> web.Response:
        result = await data.get_vix()
        return web.json_response(result)

    async def api_trades(request: web.Request) -> web.Response:
        result = await data.get_trade_history()
        return web.json_response(result, dumps=lambda x: json.dumps(x, default=_json_serial))

    async def api_config(request: web.Request) -> web.Response:
        return web.json_response({
            "underlyings": settings.underlying_list,
            "trading_mode": settings.trading_mode.value,
            "signal_confidence_threshold": settings.signal_confidence_threshold,
            "target_delta": settings.target_delta,
            "min_premium": settings.min_premium,
            "max_premium": settings.max_premium,
            "entry_start": settings.entry_start,
            "entry_cutoff": settings.entry_cutoff,
            "hard_close": settings.hard_close,
            "kelly_fraction": str(settings.kelly_fraction),
            "max_position_pct": str(settings.max_position_pct),
            "daily_drawdown_halt": str(settings.daily_drawdown_halt),
            "pt_profit_target_pct": settings.pt_profit_target_pct,
            "sl_stop_loss_pct": settings.sl_stop_loss_pct,
            "sl_trailing_pct": settings.sl_trailing_pct,
            "weights": {
                "technical": settings.weight_technical,
                "tick_momentum": settings.weight_tick_momentum,
                "gex": settings.weight_gex,
                "flow": settings.weight_flow,
                "vix": settings.weight_vix,
                "internals": settings.weight_internals,
                "sentiment": settings.weight_sentiment,
            },
            "greeks_limits": {
                "max_delta": settings.max_portfolio_delta,
                "max_gamma": settings.max_portfolio_gamma,
                "max_theta": settings.max_portfolio_theta,
                "max_vega": settings.max_portfolio_vega,
            },
        })

    # ── Register routes ───────────────────────────────────────
    app.router.add_get("/", index)
    app.router.add_get("/api/account", api_account)
    app.router.add_get("/api/prices", api_prices)
    app.router.add_get("/api/positions", api_positions)
    app.router.add_get("/api/orders", api_orders)
    app.router.add_get("/api/chain/{underlying}", api_chain)
    app.router.add_get("/api/vix", api_vix)
    app.router.add_get("/api/trades", api_trades)
    app.router.add_get("/api/config", api_config)

    # Static files
    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        app.router.add_static("/static", str(static_dir), name="static")

    # ── Startup / Cleanup ─────────────────────────────────────
    async def on_startup(app: web.Application) -> None:
        await data.connect()
        logger.info("Dashboard server started on port %d", settings.web_port)

    app.on_startup.append(on_startup)

    return app


def run_dashboard() -> None:
    """Run the dashboard as a standalone server."""
    import sys
    import os

    # Ensure we can find .env from project root
    project_root = Path(__file__).parent.parent.parent
    os.chdir(project_root)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = get_settings()
    app = create_app(settings)
    print(f"\n  Zero-DTE Options Scalper Dashboard")
    print(f"  Mode: {settings.trading_mode.value}")
    print(f"  URL:  http://localhost:{settings.web_port}\n")
    web.run_app(app, host="0.0.0.0", port=settings.web_port, print=None)


if __name__ == "__main__":
    run_dashboard()
