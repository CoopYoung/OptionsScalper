"""Alpaca REST API wrapper for the hybrid trader.

Provides all the data Claude needs via tool calls:
- Account info, positions, orders
- Stock quotes, option chains, option greeks
- Order placement with preflight validation
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import requests

from hybrid.config import (
    ALPACA_API_KEY, ALPACA_BASE_URL, ALPACA_DATA_URL, ALPACA_SECRET_KEY,
)

logger = logging.getLogger(__name__)

HEADERS = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Accept": "application/json",
}


def _trading(path: str) -> str:
    return f"{ALPACA_BASE_URL}/v2{path}"


def _data(path: str) -> str:
    return f"{ALPACA_DATA_URL}/v1beta1{path}"


def _stock_data(path: str) -> str:
    return f"{ALPACA_DATA_URL}/v2{path}"


def _get(url: str, params: dict = None) -> dict | list:
    resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _post(url: str, data: dict = None) -> dict:
    resp = requests.post(url, headers=HEADERS, json=data, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _delete(url: str) -> dict | None:
    resp = requests.delete(url, headers=HEADERS, timeout=15)
    if resp.status_code == 204:
        return None
    resp.raise_for_status()
    return resp.json()


# ── Account & Portfolio ──────────────────────────────────────

def get_account() -> dict:
    """Get account info: cash, equity, buying power, day trade count."""
    raw = _get(_trading("/account"))
    return {
        "equity": float(raw.get("equity", 0)),
        "cash": float(raw.get("cash", 0)),
        "buying_power": float(raw.get("buying_power", 0)),
        "portfolio_value": float(raw.get("portfolio_value", 0)),
        "day_trade_count": int(raw.get("daytrade_count", 0)),
        "pattern_day_trader": raw.get("pattern_day_trader", False),
        "trading_blocked": raw.get("trading_blocked", False),
        "account_blocked": raw.get("account_blocked", False),
        "status": raw.get("status", ""),
    }


def get_positions() -> list[dict]:
    """Get all open positions with real P&L."""
    raw = _get(_trading("/positions"))
    positions = []
    for p in raw:
        positions.append({
            "symbol": p["symbol"],
            "qty": int(float(p.get("qty", 0))),
            "side": p.get("side", ""),
            "avg_entry_price": float(p.get("avg_entry_price", 0)),
            "current_price": float(p.get("current_price", 0)),
            "market_value": float(p.get("market_value", 0)),
            "unrealized_pl": float(p.get("unrealized_pl", 0)),
            "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
            "asset_class": p.get("asset_class", ""),
        })
    return positions


def get_orders(status: str = "open") -> list[dict]:
    """Get orders by status: open, closed, all."""
    raw = _get(_trading("/orders"), params={"status": status, "limit": 50})
    orders = []
    for o in raw:
        orders.append({
            "id": o["id"],
            "symbol": o["symbol"],
            "side": o["side"],
            "qty": o.get("qty"),
            "type": o["type"],
            "time_in_force": o["time_in_force"],
            "limit_price": o.get("limit_price"),
            "status": o["status"],
            "filled_qty": o.get("filled_qty"),
            "filled_avg_price": o.get("filled_avg_price"),
            "submitted_at": o.get("submitted_at"),
            "filled_at": o.get("filled_at"),
            "order_class": o.get("order_class", ""),
        })
    return orders


# ── Market Data: Stocks ──────────────────────────────────────

def get_stock_quotes(symbols: list[str]) -> dict[str, dict]:
    """Get real-time quotes for stocks/ETFs."""
    params = {"symbols": ",".join(symbols)}
    raw = _get(_stock_data("/stocks/quotes/latest"), params=params)
    quotes = {}
    for sym, q in raw.get("quotes", {}).items():
        quotes[sym] = {
            "bid": float(q.get("bp", 0)),
            "ask": float(q.get("ap", 0)),
            "mid": round((float(q.get("bp", 0)) + float(q.get("ap", 0))) / 2, 2),
            "last": float(q.get("ap", 0)),  # Approximate with ask
            "bid_size": int(q.get("bs", 0)),
            "ask_size": int(q.get("as", 0)),
            "timestamp": q.get("t", ""),
        }
    return quotes


def get_stock_bars(symbol: str, timeframe: str = "5Min",
                   limit: int = 50) -> list[dict]:
    """Get recent price bars for technical analysis."""
    params = {"timeframe": timeframe, "limit": limit}
    raw = _get(_stock_data(f"/stocks/{symbol}/bars"), params=params)
    bars = []
    for b in raw.get("bars", []):
        bars.append({
            "timestamp": b.get("t", ""),
            "open": float(b.get("o", 0)),
            "high": float(b.get("h", 0)),
            "low": float(b.get("l", 0)),
            "close": float(b.get("c", 0)),
            "volume": int(b.get("v", 0)),
            "vwap": float(b.get("vw", 0)),
        })
    return bars


# ── Market Data: Options ─────────────────────────────────────

def get_option_chain(underlying: str, expiration_date: str = None,
                     option_type: str = None) -> list[dict]:
    """Get options chain for an underlying.

    Args:
        underlying: e.g. "SPY"
        expiration_date: YYYY-MM-DD format, defaults to nearest
        option_type: "call" or "put" or None for both
    """
    params = {
        "underlying_symbols": underlying,
        "limit": 100,
        "feed": "indicative",
    }
    if expiration_date:
        params["expiration_date"] = expiration_date
    if option_type:
        params["type"] = option_type

    raw = _get(_data("/options/snapshots"), params=params)
    chain = []
    for sym, snap in raw.get("snapshots", {}).items():
        greeks = snap.get("greeks", {})
        quote = snap.get("latestQuote", {})
        trade = snap.get("latestTrade", {})

        bid = float(quote.get("bp", 0))
        ask = float(quote.get("ap", 0))
        mid = round((bid + ask) / 2, 2) if (bid + ask) > 0 else 0

        chain.append({
            "symbol": sym,
            "underlying": underlying,
            "strike": float(snap.get("strike_price", _parse_strike(sym))),
            "expiration": snap.get("expiration_date", _parse_expiry(sym)),
            "option_type": snap.get("type", _parse_type(sym)),
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": round(ask - bid, 2),
            "spread_pct": round((ask - bid) / mid * 100, 1) if mid > 0 else 999,
            "last_trade": float(trade.get("p", 0)),
            "volume": int(trade.get("s", 0)),
            "open_interest": int(snap.get("open_interest", 0)),
            "iv": round(float(snap.get("implied_volatility", 0)), 4),
            "delta": round(float(greeks.get("delta", 0)), 4),
            "gamma": round(float(greeks.get("gamma", 0)), 4),
            "theta": round(float(greeks.get("theta", 0)), 4),
            "vega": round(float(greeks.get("vega", 0)), 4),
            "rho": round(float(greeks.get("rho", 0)), 4),
        })

    # Sort by strike
    chain.sort(key=lambda x: (x["option_type"], x["strike"]))
    return chain


def get_option_expirations(underlying: str) -> list[str]:
    """Get available expiration dates for an underlying."""
    # Fetch a broad chain and extract unique expirations
    params = {
        "underlying_symbols": underlying,
        "limit": 200,
        "feed": "indicative",
    }
    raw = _get(_data("/options/snapshots"), params=params)
    expirations = set()
    for sym in raw.get("snapshots", {}):
        exp = _parse_expiry(sym)
        if exp:
            expirations.add(exp)
    return sorted(expirations)


def get_option_quote(symbol: str) -> dict:
    """Get quote for a specific option contract."""
    raw = _get(_data(f"/options/snapshots/{symbol}"))
    snap = raw.get("snapshot", raw)
    greeks = snap.get("greeks", {})
    quote = snap.get("latestQuote", {})

    bid = float(quote.get("bp", 0))
    ask = float(quote.get("ap", 0))

    return {
        "symbol": symbol,
        "bid": bid,
        "ask": ask,
        "mid": round((bid + ask) / 2, 2),
        "iv": round(float(snap.get("implied_volatility", 0)), 4),
        "delta": round(float(greeks.get("delta", 0)), 4),
        "gamma": round(float(greeks.get("gamma", 0)), 4),
        "theta": round(float(greeks.get("theta", 0)), 4),
        "vega": round(float(greeks.get("vega", 0)), 4),
    }


# ── Order Placement ──────────────────────────────────────────

def place_order(
    symbol: str,
    qty: int,
    side: str,           # "buy" or "sell"
    order_type: str,     # "market" or "limit"
    time_in_force: str = "day",
    limit_price: float = None,
) -> dict:
    """Place a single-leg order."""
    data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
    }
    if limit_price and order_type == "limit":
        data["limit_price"] = str(round(limit_price, 2))

    result = _post(_trading("/orders"), data=data)
    logger.info("Order placed: %s %s %d x %s @ %s → %s",
                side, symbol, qty, order_type, limit_price, result.get("id"))
    return {
        "order_id": result["id"],
        "status": result["status"],
        "symbol": result["symbol"],
        "side": result["side"],
        "qty": result.get("qty"),
        "type": result["type"],
        "limit_price": result.get("limit_price"),
    }


def place_bracket_order(
    symbol: str,
    qty: int,
    side: str,
    order_type: str,
    limit_price: float = None,
    take_profit_price: float = None,
    stop_loss_price: float = None,
    time_in_force: str = "day",
) -> dict:
    """Place an order with take-profit and stop-loss legs."""
    data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force,
        "order_class": "bracket",
    }
    if limit_price and order_type == "limit":
        data["limit_price"] = str(round(limit_price, 2))
    if take_profit_price:
        data["take_profit"] = {"limit_price": str(round(take_profit_price, 2))}
    if stop_loss_price:
        data["stop_loss"] = {"stop_price": str(round(stop_loss_price, 2))}

    result = _post(_trading("/orders"), data=data)
    return {
        "order_id": result["id"],
        "status": result["status"],
        "symbol": result["symbol"],
        "legs": result.get("legs", []),
    }


def cancel_order(order_id: str) -> dict:
    """Cancel an open order."""
    _delete(_trading(f"/orders/{order_id}"))
    logger.info("Order cancelled: %s", order_id)
    return {"order_id": order_id, "status": "cancelled"}


def close_position(symbol: str, qty: int = None) -> dict:
    """Close a position (full or partial)."""
    params = {}
    if qty:
        params["qty"] = str(qty)
    resp = requests.delete(
        _trading(f"/positions/{symbol}"),
        headers=HEADERS,
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    result = resp.json()
    logger.info("Position closed: %s qty=%s → %s", symbol, qty, result.get("id"))
    return {
        "order_id": result.get("id"),
        "status": result.get("status"),
        "symbol": symbol,
    }


# ── Helpers ──────────────────────────────────────────────────

def _parse_strike(symbol: str) -> float:
    """Parse strike from OCC option symbol like SPY250321C00580000."""
    try:
        strike_part = symbol[-8:]
        return int(strike_part) / 1000
    except (ValueError, IndexError):
        return 0.0


def _parse_expiry(symbol: str) -> str:
    """Parse expiration from OCC symbol like SPY250321C00580000."""
    try:
        # Find where the date starts (after underlying, which varies in length)
        for i in range(len(symbol)):
            if symbol[i].isdigit():
                date_str = symbol[i:i+6]
                return f"20{date_str[:2]}-{date_str[2:4]}-{date_str[4:6]}"
        return ""
    except (ValueError, IndexError):
        return ""


def _parse_type(symbol: str) -> str:
    """Parse call/put from OCC symbol."""
    for c in symbol:
        if c == 'C':
            return "call"
        if c == 'P':
            return "put"
    return ""
