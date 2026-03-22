"""Tool definitions for Claude's Anthropic API tool_use.

Each tool maps to an Alpaca API call. Claude calls these tools,
gets real data back, and makes decisions based on actual market state.
"""

import json
import logging
import traceback
from typing import Any

from hybrid.broker import alpaca

logger = logging.getLogger(__name__)

# ── Tool Schemas (sent to Claude API) ────────────────────────

TOOLS = [
    {
        "name": "get_account",
        "description": (
            "Get your brokerage account info: cash balance, equity, buying power, "
            "day trade count, and whether trading is blocked. Call this first every cycle."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_positions",
        "description": (
            "Get all open positions with current market value, unrealized P&L, "
            "and P&L percentage. Use this to decide whether to hold or close positions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_orders",
        "description": (
            "Get orders filtered by status. Use 'open' to see pending orders, "
            "'closed' to see recent fills."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["open", "closed", "all"],
                    "description": "Filter by order status",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_stock_quotes",
        "description": (
            "Get real-time bid/ask/mid quotes for one or more stock/ETF symbols. "
            "Use this to check current underlying prices before analyzing options."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of stock symbols, e.g. ['SPY', 'QQQ', 'IWM']",
                },
            },
            "required": ["symbols"],
        },
    },
    {
        "name": "get_stock_bars",
        "description": (
            "Get recent OHLCV price bars for technical analysis. "
            "Returns open, high, low, close, volume, and VWAP."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock symbol, e.g. 'SPY'",
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["1Min", "5Min", "15Min", "1Hour", "1Day"],
                    "description": "Bar timeframe (default: 5Min)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of bars to fetch (default: 50, max: 200)",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_option_chain",
        "description": (
            "Get the options chain for an underlying with strikes, bids, asks, "
            "Greeks (delta, gamma, theta, vega), IV, open interest, and volume. "
            "Use this to find specific contracts for spreads."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "underlying": {
                    "type": "string",
                    "description": "Underlying symbol, e.g. 'SPY'",
                },
                "expiration_date": {
                    "type": "string",
                    "description": "Expiration date in YYYY-MM-DD format",
                },
                "option_type": {
                    "type": "string",
                    "enum": ["call", "put"],
                    "description": "Filter by call or put (omit for both)",
                },
            },
            "required": ["underlying"],
        },
    },
    {
        "name": "get_option_expirations",
        "description": (
            "Get available expiration dates for an underlying. "
            "Use this to find 0-3 DTE expirations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "underlying": {
                    "type": "string",
                    "description": "Underlying symbol, e.g. 'SPY'",
                },
            },
            "required": ["underlying"],
        },
    },
    {
        "name": "get_option_quote",
        "description": (
            "Get current bid/ask/mid and Greeks for a specific option contract. "
            "Use this to check pricing on existing positions or before placing orders."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Full OCC option symbol, e.g. 'SPY250321C00580000'",
                },
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "place_order",
        "description": (
            "Place a single-leg order to buy or sell a stock or option. "
            "IMPORTANT: You must provide your reasoning BEFORE calling this tool. "
            "The order will be validated against risk rules before execution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol to trade (stock or option symbol)",
                },
                "qty": {
                    "type": "integer",
                    "description": "Number of shares or contracts",
                },
                "side": {
                    "type": "string",
                    "enum": ["buy", "sell"],
                    "description": "Buy or sell",
                },
                "order_type": {
                    "type": "string",
                    "enum": ["market", "limit"],
                    "description": "Order type",
                },
                "limit_price": {
                    "type": "number",
                    "description": "Limit price (required for limit orders)",
                },
                "time_in_force": {
                    "type": "string",
                    "enum": ["day", "gtc"],
                    "description": "Time in force (default: day)",
                },
            },
            "required": ["symbol", "qty", "side", "order_type"],
        },
    },
    {
        "name": "cancel_order",
        "description": "Cancel an open order by its order ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The order ID to cancel",
                },
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "close_position",
        "description": (
            "Close an open position entirely or partially. "
            "Use this to exit trades — it submits a market sell order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol of the position to close",
                },
                "qty": {
                    "type": "integer",
                    "description": "Number of contracts to close (omit for full close)",
                },
            },
            "required": ["symbol"],
        },
    },
]


# ── Tool Execution ───────────────────────────────────────────

def execute_tool(name: str, args: dict) -> str:
    """Execute a tool call and return JSON result.

    This is the only function that touches the broker API.
    Returns JSON string for Claude to process.
    """
    try:
        if name == "get_account":
            result = alpaca.get_account()
        elif name == "get_positions":
            result = alpaca.get_positions()
        elif name == "get_orders":
            result = alpaca.get_orders(status=args.get("status", "open"))
        elif name == "get_stock_quotes":
            result = alpaca.get_stock_quotes(args["symbols"])
        elif name == "get_stock_bars":
            result = alpaca.get_stock_bars(
                symbol=args["symbol"],
                timeframe=args.get("timeframe", "5Min"),
                limit=args.get("limit", 50),
            )
        elif name == "get_option_chain":
            result = alpaca.get_option_chain(
                underlying=args["underlying"],
                expiration_date=args.get("expiration_date"),
                option_type=args.get("option_type"),
            )
        elif name == "get_option_expirations":
            result = alpaca.get_option_expirations(args["underlying"])
        elif name == "get_option_quote":
            result = alpaca.get_option_quote(args["symbol"])
        elif name == "place_order":
            result = alpaca.place_order(
                symbol=args["symbol"],
                qty=args["qty"],
                side=args["side"],
                order_type=args["order_type"],
                limit_price=args.get("limit_price"),
                time_in_force=args.get("time_in_force", "day"),
            )
        elif name == "cancel_order":
            result = alpaca.cancel_order(args["order_id"])
        elif name == "close_position":
            result = alpaca.close_position(
                symbol=args["symbol"],
                qty=args.get("qty"),
            )
        else:
            result = {"error": f"Unknown tool: {name}"}

        return json.dumps(result, default=str)

    except Exception as e:
        logger.error("Tool %s failed: %s", name, e)
        return json.dumps({
            "error": str(e),
            "tool": name,
            "traceback": traceback.format_exc(),
        })
