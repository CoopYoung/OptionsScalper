"""Public.com broker implementation for live trading.

Wraps the existing public_data.py functions for data, and adds
order execution via the Public.com SDK.

Returns data in the same dict shapes as alpaca.py (the Broker contract).
"""

import logging
from typing import Optional

from hybrid.broker.public_data import (
    _get_client,
    get_option_chain_with_greeks,
    get_public_portfolio,
)

logger = logging.getLogger(__name__)


class PublicBroker:
    """Public.com broker — for live trading with real money."""

    def get_account(self) -> dict:
        """Get account info matching Alpaca's format."""
        portfolio = get_public_portfolio()
        if "error" in portfolio:
            logger.error("Public portfolio error: %s", portfolio["error"])
            return {
                "equity": 0, "cash": 0, "buying_power": 0,
                "portfolio_value": 0, "day_trade_count": 0,
                "pattern_day_trader": False, "trading_blocked": False,
                "account_blocked": False, "status": "error",
            }

        bp = portfolio.get("buying_power", {})
        equity_breakdown = portfolio.get("equity_breakdown", [])
        total_equity = sum(e.get("value", 0) for e in equity_breakdown)

        return {
            "equity": total_equity,
            "cash": bp.get("cash_only", 0),
            "buying_power": bp.get("options_buying_power", bp.get("buying_power", 0)),
            "portfolio_value": total_equity,
            "day_trade_count": 0,  # Public doesn't expose this directly
            "pattern_day_trader": False,
            "trading_blocked": False,
            "account_blocked": False,
            "status": "ACTIVE",
        }

    def get_positions(self) -> list[dict]:
        """Get open positions matching Alpaca's format."""
        portfolio = get_public_portfolio()
        if "error" in portfolio:
            return []

        positions = []
        for p in portfolio.get("positions", []):
            pos_type = p.get("type", "")
            # Determine asset class
            if pos_type in ("OPTION", "option"):
                asset_class = "us_option"
            else:
                asset_class = "us_equity"

            entry_price = p.get("unit_cost", 0)
            current_price = p.get("last_price", 0)
            qty = p.get("quantity", 0)

            if entry_price and current_price and qty:
                unrealized_pl = (current_price - entry_price) * qty
                if asset_class == "us_option":
                    unrealized_pl *= 100  # Options are 100 shares/contract
                unrealized_plpc = (current_price - entry_price) / entry_price if entry_price else 0
            else:
                unrealized_pl = p.get("total_gain", 0)
                unrealized_plpc = (p.get("total_gain_pct", 0) or 0) / 100

            positions.append({
                "symbol": p["symbol"],
                "qty": qty,
                "side": "long",
                "avg_entry_price": entry_price,
                "current_price": current_price,
                "market_value": p.get("current_value", 0),
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
                "asset_class": asset_class,
            })

        return positions

    def get_orders(self, status: str = "open") -> list[dict]:
        """Get orders. Uses Public.com SDK."""
        client = _get_client()
        if not client:
            return []
        try:
            orders_resp = client.get_orders()
            orders = []
            for o in orders_resp.orders:
                order_status = str(getattr(o, "status", "")).lower()
                # Filter by status
                if status == "open" and order_status not in ("new", "pending", "partially_filled"):
                    continue
                if status == "closed" and order_status not in ("filled", "cancelled", "expired"):
                    continue

                orders.append({
                    "id": str(getattr(o, "order_id", "")),
                    "symbol": str(getattr(o.instrument, "symbol", "")) if o.instrument else "",
                    "side": str(getattr(o, "side", "")).lower(),
                    "qty": getattr(o, "quantity", 0),
                    "type": str(getattr(o, "type", "")).lower(),
                    "time_in_force": str(getattr(o, "time_in_force", "")).lower(),
                    "limit_price": getattr(o, "limit_price", None),
                    "status": order_status,
                    "filled_qty": getattr(o, "filled_quantity", 0),
                    "filled_avg_price": getattr(o, "average_fill_price", None),
                    "submitted_at": str(getattr(o, "created_at", "")),
                    "filled_at": str(getattr(o, "filled_at", "")),
                    "order_class": "",
                })
            return orders
        except Exception as e:
            logger.error("Public.com get_orders failed: %s", e)
            return []

    def get_stock_quotes(self, symbols: list[str]) -> dict[str, dict]:
        """Get equity quotes from Public.com."""
        client = _get_client()
        if not client:
            return {}
        try:
            from public_api_sdk import OrderInstrument, InstrumentType
            instruments = [
                OrderInstrument(symbol=s, type=InstrumentType.EQUITY) for s in symbols
            ]
            quotes = client.get_quotes(instruments)
            result = {}
            for q in quotes:
                sym = q.instrument.symbol
                bid = float(q.bid) if q.bid else 0
                ask = float(q.ask) if q.ask else 0
                result[sym] = {
                    "bid": bid,
                    "ask": ask,
                    "mid": round((bid + ask) / 2, 2) if bid and ask else 0,
                    "last": float(q.last) if q.last else 0,
                    "bid_size": 0,
                    "ask_size": 0,
                    "timestamp": "",
                }
            return result
        except Exception as e:
            logger.error("Public.com quotes failed: %s", e)
            return {}

    def get_stock_bars(self, symbol: str, timeframe: str = "5Min",
                       limit: int = 50) -> list[dict]:
        """Get bars — Public.com doesn't provide bars, fall back to yfinance."""
        try:
            import yfinance as yf
            tf_map = {"1Min": "1m", "5Min": "5m", "15Min": "15m", "1Hour": "1h", "1Day": "1d"}
            yf_interval = tf_map.get(timeframe, "5m")
            period = "1d" if yf_interval in ("1m", "5m") else "5d"
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=yf_interval)
            bars = []
            for ts, row in df.iterrows():
                bars.append({
                    "timestamp": ts.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                    "vwap": 0,  # yfinance doesn't provide VWAP
                })
            return bars[-limit:]
        except Exception as e:
            logger.error("yfinance bars fallback failed for %s: %s", symbol, e)
            return []

    def get_option_chain(self, underlying: str, expiration_date: str | None = None,
                         option_type: str | None = None) -> list[dict]:
        """Get options chain using Public.com chain-with-greeks."""
        if not expiration_date:
            return []
        result = get_option_chain_with_greeks(
            symbol=underlying,
            expiry=expiration_date,
            option_type=option_type,
        )
        if "error" in result:
            logger.error("Public chain error: %s", result["error"])
            return []

        # Flatten calls + puts into single list matching Alpaca format
        chain = []
        for opt_type in ("calls", "puts"):
            for c in result.get(opt_type, []):
                chain.append({
                    "symbol": c.get("symbol", ""),
                    "underlying": underlying,
                    "strike": c.get("strike", 0),
                    "expiration": expiration_date,
                    "option_type": "call" if opt_type == "calls" else "put",
                    "bid": float(c.get("bid", 0) or 0),
                    "ask": float(c.get("ask", 0) or 0),
                    "mid": c.get("mid", 0),
                    "spread": c.get("spread", 0),
                    "spread_pct": c.get("spread_pct", 999),
                    "last_trade": float(c.get("last", 0) or 0),
                    "volume": int(c.get("volume", 0) or 0),
                    "open_interest": int(c.get("open_interest", 0) or 0),
                    "iv": float(c.get("iv", 0) or 0),
                    "delta": float(c.get("delta", 0) or 0),
                    "gamma": float(c.get("gamma", 0) or 0),
                    "theta": float(c.get("theta", 0) or 0),
                    "vega": float(c.get("vega", 0) or 0),
                    "rho": float(c.get("rho", 0) or 0),
                })

        chain.sort(key=lambda x: (x["option_type"], x["strike"]))
        return chain

    def get_option_expirations(self, underlying: str) -> list[str]:
        """Get available expirations from Public.com."""
        client = _get_client()
        if not client:
            return []
        try:
            from public_api_sdk import OrderInstrument, InstrumentType
            instrument = OrderInstrument(symbol=underlying, type=InstrumentType.EQUITY)
            resp = client.get_option_expirations(instrument)
            return sorted([str(d) for d in resp.expirations]) if resp.expirations else []
        except Exception as e:
            logger.error("Public expirations failed for %s: %s", underlying, e)
            return []

    def get_option_quote(self, symbol: str) -> dict:
        """Get quote for a specific option contract."""
        from hybrid.broker.public_data import get_option_greeks
        result = get_option_greeks([symbol])
        if "error" in result or not result.get("greeks"):
            return {"symbol": symbol, "bid": 0, "ask": 0, "mid": 0}

        g = result["greeks"][0]
        return {
            "symbol": symbol,
            "bid": 0,  # Greeks endpoint doesn't return bid/ask
            "ask": 0,
            "mid": 0,
            "iv": g.get("iv", 0),
            "delta": g.get("delta", 0),
            "gamma": g.get("gamma", 0),
            "theta": g.get("theta", 0),
            "vega": g.get("vega", 0),
        }

    def place_order(self, symbol: str, qty: int, side: str, order_type: str,
                    time_in_force: str = "day",
                    limit_price: float | None = None) -> dict:
        """Place an order via Public.com SDK."""
        client = _get_client()
        if not client:
            raise RuntimeError("Public.com client not initialized")

        try:
            from public_api_sdk import (
                OrderInstrument, InstrumentType,
                OrderSide, OrderType, TimeInForce,
                PlaceOrderRequest,
            )

            instrument = OrderInstrument(symbol=symbol, type=InstrumentType.OPTION)
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            ot = OrderType.LIMIT if order_type == "limit" else OrderType.MARKET
            tif = TimeInForce.DAY if time_in_force == "day" else TimeInForce.GTC

            req = PlaceOrderRequest(
                instrument=instrument,
                side=order_side,
                type=ot,
                time_in_force=tif,
                quantity=qty,
                limit_price=limit_price,
            )

            resp = client.place_order(req)
            order_id = str(getattr(resp, "order_id", ""))
            status = str(getattr(resp, "status", "submitted"))

            logger.info("Public.com order placed: %s %d x %s @ %s → %s",
                        side, qty, symbol, limit_price, order_id)

            return {
                "order_id": order_id,
                "status": status,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "type": order_type,
                "limit_price": limit_price,
            }

        except Exception as e:
            logger.error("Public.com place_order failed: %s", e)
            raise

    def close_position(self, symbol: str, qty: int | None = None) -> dict:
        """Close a position by placing an opposite order."""
        # Get current position to determine qty
        positions = self.get_positions()
        pos = next((p for p in positions if p["symbol"] == symbol), None)

        if not pos:
            raise ValueError(f"No position found for {symbol}")

        close_qty = qty or pos["qty"]

        # Place a sell order to close
        return self.place_order(
            symbol=symbol,
            qty=close_qty,
            side="sell",
            order_type="market",
            time_in_force="day",
        )

    def cancel_order(self, order_id: str) -> dict:
        """Cancel an order via Public.com SDK."""
        client = _get_client()
        if not client:
            raise RuntimeError("Public.com client not initialized")
        try:
            client.cancel_order(order_id)
            logger.info("Public.com order cancelled: %s", order_id)
            return {"order_id": order_id, "status": "cancelled"}
        except Exception as e:
            logger.error("Public.com cancel_order failed: %s", e)
            raise
