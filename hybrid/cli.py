#!/usr/bin/env python3
"""CLI interface for broker operations.

Claude Code calls these via Bash tool. Each subcommand returns JSON
that Claude can parse and reason about.

Usage:
    python -m hybrid.cli account
    python -m hybrid.cli positions
    python -m hybrid.cli quotes SPY QQQ IWM
    python -m hybrid.cli bars SPY --timeframe 5Min --limit 50
    python -m hybrid.cli chain SPY --expiry 2026-03-25 --type call
    python -m hybrid.cli expirations SPY
    python -m hybrid.cli option-quote SPY250325C00580000
    python -m hybrid.cli order buy SPY250325C00580000 3 limit --price 2.50
    python -m hybrid.cli close SPY250325C00580000
    python -m hybrid.cli cancel ORDER_ID
    python -m hybrid.cli orders --status open
    python -m hybrid.cli daily-state
    python -m hybrid.cli validate buy SPY250325C00580000 3 --price 2.50
"""

import argparse
import json
import sys

from hybrid.broker import alpaca
from hybrid.risk.validator import (
    get_daily_state,
    is_market_hours,
    record_trade_pnl,
    should_force_close_all,
    validate_close,
    validate_new_order,
)


def _print_json(data):
    """Print JSON to stdout for Claude to read."""
    print(json.dumps(data, indent=2, default=str))


def cmd_account(args):
    _print_json(alpaca.get_account())


def cmd_positions(args):
    _print_json(alpaca.get_positions())


def cmd_quotes(args):
    _print_json(alpaca.get_stock_quotes(args.symbols))


def cmd_bars(args):
    _print_json(alpaca.get_stock_bars(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
    ))


def cmd_chain(args):
    _print_json(alpaca.get_option_chain(
        underlying=args.underlying,
        expiration_date=args.expiry,
        option_type=args.type,
    ))


def cmd_expirations(args):
    _print_json(alpaca.get_option_expirations(args.underlying))


def cmd_option_quote(args):
    _print_json(alpaca.get_option_quote(args.symbol))


def cmd_orders(args):
    _print_json(alpaca.get_orders(status=args.status))


def cmd_daily_state(args):
    state = get_daily_state()
    state["is_market_hours"] = is_market_hours()
    state["force_close"] = should_force_close_all()
    _print_json(state)


def cmd_validate(args):
    """Validate an order WITHOUT executing it."""
    account = alpaca.get_account()
    positions = alpaca.get_positions()
    result = validate_new_order(
        symbol=args.symbol,
        qty=args.qty,
        side=args.side,
        order_type="limit" if args.price else "market",
        limit_price=args.price,
        current_positions=positions,
        account=account,
    )
    _print_json(result)


def cmd_order(args):
    """Validate then execute an order."""
    # Always validate first
    account = alpaca.get_account()
    positions = alpaca.get_positions()

    order_type = args.order_type
    limit_price = args.price

    validation = validate_new_order(
        symbol=args.symbol,
        qty=args.qty,
        side=args.side,
        order_type=order_type,
        limit_price=limit_price,
        current_positions=positions,
        account=account,
    )

    if not validation["approved"]:
        _print_json({
            "error": "ORDER BLOCKED BY VALIDATOR",
            "reason": validation["reason"],
            "violations": validation.get("violations", []),
        })
        sys.exit(1)

    # Validation passed — execute
    result = alpaca.place_order(
        symbol=args.symbol,
        qty=args.qty,
        side=args.side,
        order_type=order_type,
        limit_price=limit_price,
        time_in_force=args.tif,
    )
    _print_json(result)


def cmd_close(args):
    """Validate then close a position."""
    positions = alpaca.get_positions()
    validation = validate_close(args.symbol, positions)

    if not validation["approved"]:
        _print_json({
            "error": "CLOSE BLOCKED",
            "reason": validation["reason"],
        })
        sys.exit(1)

    result = alpaca.close_position(args.symbol, qty=args.qty)
    _print_json(result)


def cmd_cancel(args):
    _print_json(alpaca.cancel_order(args.order_id))


def cmd_record_pnl(args):
    """Record a closed trade's P&L for daily tracking."""
    record_trade_pnl(args.pnl)
    _print_json(get_daily_state())


def main():
    parser = argparse.ArgumentParser(description="Hybrid Trader CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # account
    sub.add_parser("account", help="Get account info")

    # positions
    sub.add_parser("positions", help="Get open positions")

    # quotes
    p = sub.add_parser("quotes", help="Get stock quotes")
    p.add_argument("symbols", nargs="+", help="Stock symbols")

    # bars
    p = sub.add_parser("bars", help="Get price bars")
    p.add_argument("symbol", help="Stock symbol")
    p.add_argument("--timeframe", default="5Min", help="Bar timeframe")
    p.add_argument("--limit", type=int, default=50, help="Number of bars")

    # chain
    p = sub.add_parser("chain", help="Get options chain")
    p.add_argument("underlying", help="Underlying symbol")
    p.add_argument("--expiry", help="Expiration date YYYY-MM-DD")
    p.add_argument("--type", choices=["call", "put"], help="Option type filter")

    # expirations
    p = sub.add_parser("expirations", help="Get option expirations")
    p.add_argument("underlying", help="Underlying symbol")

    # option-quote
    p = sub.add_parser("option-quote", help="Get option quote")
    p.add_argument("symbol", help="OCC option symbol")

    # orders
    p = sub.add_parser("orders", help="Get orders")
    p.add_argument("--status", default="open", choices=["open", "closed", "all"])

    # daily-state
    sub.add_parser("daily-state", help="Get daily trading state")

    # validate (dry run)
    p = sub.add_parser("validate", help="Validate order without executing")
    p.add_argument("side", choices=["buy", "sell"])
    p.add_argument("symbol", help="Symbol to trade")
    p.add_argument("qty", type=int, help="Quantity")
    p.add_argument("--price", type=float, help="Limit price")

    # order (validate + execute)
    p = sub.add_parser("order", help="Place an order (validates first)")
    p.add_argument("side", choices=["buy", "sell"])
    p.add_argument("symbol", help="Symbol to trade")
    p.add_argument("qty", type=int, help="Quantity")
    p.add_argument("order_type", choices=["market", "limit"])
    p.add_argument("--price", type=float, help="Limit price")
    p.add_argument("--tif", default="day", choices=["day", "gtc"])

    # close
    p = sub.add_parser("close", help="Close a position")
    p.add_argument("symbol", help="Position symbol")
    p.add_argument("--qty", type=int, help="Partial close qty")

    # cancel
    p = sub.add_parser("cancel", help="Cancel an order")
    p.add_argument("order_id", help="Order ID")

    # record-pnl
    p = sub.add_parser("record-pnl", help="Record closed trade P&L")
    p.add_argument("pnl", type=float, help="P&L amount")

    args = parser.parse_args()

    # Dispatch
    cmd_map = {
        "account": cmd_account,
        "positions": cmd_positions,
        "quotes": cmd_quotes,
        "bars": cmd_bars,
        "chain": cmd_chain,
        "expirations": cmd_expirations,
        "option-quote": cmd_option_quote,
        "orders": cmd_orders,
        "daily-state": cmd_daily_state,
        "validate": cmd_validate,
        "order": cmd_order,
        "close": cmd_close,
        "cancel": cmd_cancel,
        "record-pnl": cmd_record_pnl,
    }

    try:
        cmd_map[args.command](args)
    except Exception as e:
        _print_json({"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
