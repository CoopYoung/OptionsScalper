#!/usr/bin/env python3
"""Mock CLI for backtesting — drop-in replacement for hybrid.cli.

Claude calls this exactly the same way as the real CLI:
    python3 -m hybrid.backtest.mock_cli account
    python3 -m hybrid.backtest.mock_cli quotes SPY QQQ
    python3 -m hybrid.backtest.mock_cli order buy SPY260321C00570000 2 limit --price 1.50

Instead of hitting live APIs, it reads from a pre-built snapshot file
and manages simulated account state.

Environment variables:
    BACKTEST_SNAPSHOT  — path to the current cycle's snapshot JSON
    BACKTEST_STATE     — path to the persistent state JSON
"""

import argparse
import json
import os
import sys
from pathlib import Path


SNAPSHOT_PATH = os.environ.get("BACKTEST_SNAPSHOT", "/tmp/bt_snapshot.json")
STATE_PATH = os.environ.get("BACKTEST_STATE", "/tmp/bt_state.json")
MAX_RISK_PER_TRADE = float(os.environ.get("BACKTEST_MAX_RISK", "500"))
MAX_DAILY_LOSS = float(os.environ.get("BACKTEST_MAX_DAILY_LOSS", "1000"))


def _load_snapshot() -> dict:
    with open(SNAPSHOT_PATH) as f:
        return json.load(f)


def _load_state() -> dict:
    if not Path(STATE_PATH).exists():
        return {}
    with open(STATE_PATH) as f:
        return json.load(f)


def _save_state(state: dict):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2, default=str)


def _print_json(data):
    print(json.dumps(data, indent=2, default=str))


# ── Command handlers ──────────────────────────────────────────

def cmd_account(args):
    snap = _load_snapshot()
    _print_json(snap.get("account", {}))


def cmd_positions(args):
    state = _load_state()
    _print_json(state.get("positions", []))


def cmd_quotes(args):
    snap = _load_snapshot()
    quotes = snap.get("quotes", {})
    # Filter to requested symbols
    if args.symbols:
        filtered = {s: quotes.get(s, {"error": f"No data for {s}"}) for s in args.symbols}
        _print_json(filtered)
    else:
        _print_json(quotes)


def cmd_bars(args):
    snap = _load_snapshot()
    all_bars = snap.get("bars", {})
    bars = all_bars.get(args.symbol, [])
    # Apply limit
    if args.limit and len(bars) > args.limit:
        bars = bars[-args.limit:]
    _print_json(bars)


def cmd_chain(args):
    snap = _load_snapshot()
    chain = snap.get("chain", [])
    # Filter by type if specified
    if args.type:
        chain = [c for c in chain if c.get("option_type") == args.type]
    _print_json(chain)


def cmd_expirations(args):
    snap = _load_snapshot()
    _print_json(snap.get("expirations", []))


def cmd_option_quote(args):
    snap = _load_snapshot()
    chain = snap.get("chain", [])
    for c in chain:
        if c["symbol"] == args.symbol:
            _print_json(c)
            return
    _print_json({"error": f"Option {args.symbol} not found in chain"})


def cmd_orders(args):
    state = _load_state()
    orders = state.get("orders", [])
    if args.status == "open":
        orders = [o for o in orders if o.get("status") in ("new", "pending")]
    elif args.status == "closed":
        orders = [o for o in orders if o.get("status") in ("filled", "cancelled")]
    _print_json(orders)


def cmd_daily_state(args):
    snap = _load_snapshot()
    ds = snap.get("daily-state", {})
    # Merge with actual state
    state = _load_state()
    ds["daily_pnl"] = state.get("daily_pnl", 0)
    ds["trades_today"] = state.get("daily_trades", 0)
    _print_json(ds)


def cmd_validate(args):
    """Validate an order against risk rules (simulated)."""
    state = _load_state()
    snap = _load_snapshot()

    # Check daily loss limit
    if state.get("daily_pnl", 0) <= -MAX_DAILY_LOSS:
        _print_json({"approved": False, "reason": f"Daily loss limit reached (-${MAX_DAILY_LOSS:.0f})"})
        return

    # Check position count
    positions = state.get("positions", [])
    if len(positions) >= 3:
        _print_json({"approved": False, "reason": "Max concurrent positions reached (3)"})
        return

    # Check risk per trade
    if args.price:
        risk = args.price * args.qty * 100
        if risk > MAX_RISK_PER_TRADE:
            _print_json({"approved": False, "reason": f"Risk ${risk:.0f} exceeds ${MAX_RISK_PER_TRADE:.0f} max"})
            return

    # Check market hours
    ds = snap.get("daily-state", {})
    if not ds.get("is_market_hours", False):
        _print_json({"approved": False, "reason": "Outside market hours"})
        return

    _print_json({"approved": True, "reason": "Validation passed"})


def cmd_order(args):
    """Simulate order execution."""
    state = _load_state()
    snap = _load_snapshot()

    # Run validation first
    positions = state.get("positions", [])
    if state.get("daily_pnl", 0) <= -MAX_DAILY_LOSS:
        _print_json({"error": "ORDER BLOCKED: Daily loss limit"})
        sys.exit(1)
    if len(positions) >= 3:
        _print_json({"error": "ORDER BLOCKED: Max positions"})
        sys.exit(1)

    # Simulate fill
    fill_price = args.price if args.price else 0
    if fill_price == 0:
        # Market order — find in chain
        chain = snap.get("chain", [])
        for c in chain:
            if c["symbol"] == args.symbol:
                fill_price = c["mid"]
                break

    if fill_price == 0:
        _print_json({"error": f"Cannot determine fill price for {args.symbol}"})
        sys.exit(1)

    # Add slippage (2% worse)
    if args.side == "buy":
        fill_price *= 1.02
    else:
        fill_price *= 0.98
    fill_price = round(fill_price, 2)

    time_str = snap.get("daily-state", {}).get("date", "") + " " + state.get("current_time_et", "")

    # Determine underlying from symbol (first 3 chars)
    underlying = args.symbol[:3]

    # Record position
    if "positions" not in state:
        state["positions"] = []
    if "orders" not in state:
        state["orders"] = []

    trade_id = state.get("daily_trades", 0) + 1

    state["positions"].append({
        "symbol": args.symbol,
        "underlying": underlying,
        "qty": args.qty,
        "side": "long",
        "avg_entry_price": fill_price,
        "current_price": fill_price,
        "market_value": fill_price * args.qty * 100,
        "unrealized_pl": 0.0,
        "unrealized_plpc": 0.0,
        "asset_class": "option",
        "entry_time": time_str,
    })

    state["orders"].append({
        "id": f"bt-{trade_id}",
        "symbol": args.symbol,
        "side": args.side,
        "qty": args.qty,
        "type": args.order_type,
        "time_in_force": args.tif,
        "limit_price": args.price,
        "status": "filled",
        "filled_qty": args.qty,
        "filled_avg_price": fill_price,
        "submitted_at": time_str,
        "filled_at": time_str,
    })

    cost = fill_price * args.qty * 100
    state["cash"] = state.get("cash", 69500) - cost
    state["daily_trades"] = trade_id

    _save_state(state)

    _print_json({
        "id": f"bt-{trade_id}",
        "symbol": args.symbol,
        "side": args.side,
        "qty": args.qty,
        "type": args.order_type,
        "status": "filled",
        "filled_avg_price": fill_price,
    })


def cmd_close(args):
    """Simulate closing a position."""
    state = _load_state()
    snap = _load_snapshot()

    positions = state.get("positions", [])
    pos = None
    for p in positions:
        if p["symbol"] == args.symbol:
            pos = p
            break

    if pos is None:
        _print_json({"error": f"No position found for {args.symbol}"})
        sys.exit(1)

    # Determine exit price from chain
    exit_price = pos["current_price"]
    chain = snap.get("chain", [])
    for c in chain:
        if c["symbol"] == args.symbol:
            exit_price = c["mid"] * 0.98  # Slippage
            break

    # Calculate P&L
    entry = pos["avg_entry_price"]
    qty = pos["qty"]
    pnl = round((exit_price - entry) * qty * 100, 2)

    state["cash"] = state.get("cash", 69500) + exit_price * qty * 100
    state["daily_pnl"] = state.get("daily_pnl", 0) + pnl
    state["positions"] = [p for p in positions if p["symbol"] != args.symbol]

    if pnl > 0:
        state["daily_wins"] = state.get("daily_wins", 0) + 1
    else:
        state["daily_losses"] = state.get("daily_losses", 0) + 1

    _save_state(state)

    _print_json({
        "symbol": args.symbol,
        "exit_price": round(exit_price, 2),
        "pnl": pnl,
        "status": "closed",
    })


def cmd_cancel(args):
    _print_json({"status": "cancelled", "order_id": args.order_id})


def cmd_record_pnl(args):
    state = _load_state()
    state["daily_pnl"] = state.get("daily_pnl", 0) + args.pnl
    _save_state(state)
    _print_json({"daily_pnl": state["daily_pnl"]})


# ── Market data commands (read from snapshot) ─────────────────

def cmd_vix(args):
    _print_json(_load_snapshot().get("vix", {}))

def cmd_fear_greed(args):
    _print_json(_load_snapshot().get("fear-greed", {}))

def cmd_sectors(args):
    _print_json(_load_snapshot().get("sectors", {}))

def cmd_sentiment(args):
    _print_json(_load_snapshot().get("sentiment", {}))

def cmd_news(args):
    _print_json(_load_snapshot().get("news", []))

def cmd_calendar(args):
    _print_json(_load_snapshot().get("calendar", []))

def cmd_earnings(args):
    _print_json(_load_snapshot().get("earnings", []))

def cmd_market_overview(args):
    _print_json(_load_snapshot().get("market-overview", {}))

def cmd_indices(args):
    _print_json(_load_snapshot().get("indices", {}))

def cmd_greeks(args):
    snap = _load_snapshot()
    chain = snap.get("chain", [])
    results = []
    for sym in args.symbols:
        for c in chain:
            if c["symbol"] == sym:
                results.append({
                    "symbol": sym,
                    "delta": c.get("delta"),
                    "gamma": c.get("gamma"),
                    "theta": c.get("theta"),
                    "vega": c.get("vega"),
                    "iv": c.get("iv"),
                })
                break
    _print_json({"greeks": results, "count": len(results)})

def cmd_chain_greeks(args):
    _print_json(_load_snapshot().get("chain-greeks", {}))

def cmd_public_portfolio(args):
    state = _load_state()
    _print_json({
        "account_id": "backtest",
        "buying_power": {"cash_only": state.get("cash", 69500)},
        "positions": state.get("positions", []),
    })


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backtest Mock CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("account")
    sub.add_parser("positions")

    p = sub.add_parser("quotes")
    p.add_argument("symbols", nargs="+")

    p = sub.add_parser("bars")
    p.add_argument("symbol")
    p.add_argument("--timeframe", default="5Min")
    p.add_argument("--limit", type=int, default=50)

    p = sub.add_parser("chain")
    p.add_argument("underlying")
    p.add_argument("--expiry")
    p.add_argument("--type", choices=["call", "put"])

    p = sub.add_parser("expirations")
    p.add_argument("underlying")

    p = sub.add_parser("option-quote")
    p.add_argument("symbol")

    p = sub.add_parser("orders")
    p.add_argument("--status", default="open", choices=["open", "closed", "all"])

    sub.add_parser("daily-state")

    p = sub.add_parser("validate")
    p.add_argument("side", choices=["buy", "sell"])
    p.add_argument("symbol")
    p.add_argument("qty", type=int)
    p.add_argument("--price", type=float)

    p = sub.add_parser("order")
    p.add_argument("side", choices=["buy", "sell"])
    p.add_argument("symbol")
    p.add_argument("qty", type=int)
    p.add_argument("order_type", choices=["market", "limit"])
    p.add_argument("--price", type=float)
    p.add_argument("--tif", default="day", choices=["day", "gtc"])

    p = sub.add_parser("close")
    p.add_argument("symbol")
    p.add_argument("--qty", type=int)

    p = sub.add_parser("cancel")
    p.add_argument("order_id")

    p = sub.add_parser("record-pnl")
    p.add_argument("pnl", type=float)

    sub.add_parser("vix")
    sub.add_parser("fear-greed")
    sub.add_parser("sectors")

    p = sub.add_parser("sentiment")
    p.add_argument("symbol", nargs="?", default="SPY")

    p = sub.add_parser("news")
    p.add_argument("--category", default="general")

    sub.add_parser("calendar")
    sub.add_parser("earnings")
    sub.add_parser("market-overview")

    p = sub.add_parser("indices")
    p.add_argument("symbols", nargs="*", default=["VIX", "SPX"])

    p = sub.add_parser("greeks")
    p.add_argument("symbols", nargs="+")

    p = sub.add_parser("chain-greeks")
    p.add_argument("underlying")
    p.add_argument("--expiry", required=True)
    p.add_argument("--type", choices=["call", "put"])
    p.add_argument("--range", type=int, default=10)

    sub.add_parser("public-portfolio")

    args = parser.parse_args()

    cmd_map = {
        "account": cmd_account, "positions": cmd_positions,
        "quotes": cmd_quotes, "bars": cmd_bars, "chain": cmd_chain,
        "expirations": cmd_expirations, "option-quote": cmd_option_quote,
        "orders": cmd_orders, "daily-state": cmd_daily_state,
        "validate": cmd_validate, "order": cmd_order,
        "close": cmd_close, "cancel": cmd_cancel, "record-pnl": cmd_record_pnl,
        "vix": cmd_vix, "fear-greed": cmd_fear_greed, "sectors": cmd_sectors,
        "sentiment": cmd_sentiment, "news": cmd_news, "calendar": cmd_calendar,
        "earnings": cmd_earnings, "market-overview": cmd_market_overview,
        "indices": cmd_indices, "greeks": cmd_greeks,
        "chain-greeks": cmd_chain_greeks, "public-portfolio": cmd_public_portfolio,
    }

    try:
        cmd_map[args.command](args)
    except Exception as e:
        _print_json({"error": str(e)})
        sys.exit(1)


if __name__ == "__main__":
    main()
