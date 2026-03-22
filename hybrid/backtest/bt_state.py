"""Simulated account state for backtesting.

Manages a virtual broker account that persists across Claude's
tool calls within a single cycle and across cycles within a day.

State lives in a JSON file so the mock CLI (invoked as a subprocess
by Claude) can read/write it.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = Path("/tmp/bt_state.json")


@dataclass
class SimPosition:
    """A simulated open position."""
    symbol: str
    underlying: str
    qty: int
    side: str  # "long"
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_plpc: float
    asset_class: str = "option"
    entry_time: str = ""


@dataclass
class SimOrder:
    """A simulated order."""
    id: str
    symbol: str
    side: str
    qty: int
    type: str
    time_in_force: str
    limit_price: Optional[float]
    status: str  # "filled", "cancelled"
    filled_qty: int = 0
    filled_avg_price: float = 0.0
    submitted_at: str = ""
    filled_at: str = ""


@dataclass
class BacktestState:
    """Full simulated account state."""
    # Simulated time
    current_date: str = ""
    current_time_et: str = ""

    # Account
    initial_capital: float = 69_500.0
    equity: float = 69_500.0
    cash: float = 69_500.0
    buying_power: float = 139_000.0

    # Positions & orders
    positions: list[dict] = field(default_factory=list)
    orders: list[dict] = field(default_factory=list)

    # Daily tracking
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_wins: int = 0
    daily_losses: int = 0

    # Cycle tracking
    cycle_count: int = 0
    decisions: list[dict] = field(default_factory=list)

    # Snapshot data path
    snapshot_path: str = ""


def load_state(path: Path = DEFAULT_STATE_PATH) -> BacktestState:
    """Load state from disk."""
    if not path.exists():
        return BacktestState()
    try:
        with open(path) as f:
            data = json.load(f)
        state = BacktestState(**{
            k: v for k, v in data.items()
            if k in BacktestState.__dataclass_fields__
        })
        return state
    except Exception as e:
        logger.error(f"Failed to load state: {e}")
        return BacktestState()


def save_state(state: BacktestState, path: Path = DEFAULT_STATE_PATH):
    """Save state to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(state), f, indent=2, default=str)


def init_day(state: BacktestState, day_date: str, capital: float = None):
    """Reset state for a new trading day."""
    if capital is not None:
        state.equity = capital
        state.cash = capital
        state.buying_power = capital * 2
    state.current_date = day_date
    state.positions = []
    state.orders = []
    state.daily_pnl = 0.0
    state.daily_trades = 0
    state.daily_wins = 0
    state.daily_losses = 0
    return state


def add_position(state: BacktestState, symbol: str, underlying: str,
                 qty: int, price: float, time_str: str):
    """Add a new position (simulate a fill)."""
    cost = price * qty * 100
    state.cash -= cost
    state.equity -= 0  # Equity stays same (cash → position)

    state.positions.append({
        "symbol": symbol,
        "underlying": underlying,
        "qty": qty,
        "side": "long",
        "avg_entry_price": price,
        "current_price": price,
        "market_value": cost,
        "unrealized_pl": 0.0,
        "unrealized_plpc": 0.0,
        "asset_class": "option",
        "entry_time": time_str,
    })

    state.orders.append({
        "id": f"bt-{state.daily_trades + 1}",
        "symbol": symbol,
        "side": "buy",
        "qty": qty,
        "type": "limit",
        "time_in_force": "day",
        "limit_price": price,
        "status": "filled",
        "filled_qty": qty,
        "filled_avg_price": price,
        "submitted_at": time_str,
        "filled_at": time_str,
    })

    state.daily_trades += 1
    return state


def close_position(state: BacktestState, symbol: str, exit_price: float,
                   time_str: str) -> float:
    """Close a position and return realized P&L."""
    pos = None
    for p in state.positions:
        if p["symbol"] == symbol:
            pos = p
            break

    if pos is None:
        return 0.0

    entry = pos["avg_entry_price"]
    qty = pos["qty"]
    pnl = (exit_price - entry) * qty * 100

    state.cash += exit_price * qty * 100
    state.daily_pnl += pnl
    state.equity = state.cash + sum(
        p["current_price"] * p["qty"] * 100
        for p in state.positions if p["symbol"] != symbol
    )
    state.buying_power = state.equity * 2

    if pnl > 0:
        state.daily_wins += 1
    elif pnl < 0:
        state.daily_losses += 1

    state.positions = [p for p in state.positions if p["symbol"] != symbol]

    state.orders.append({
        "id": f"bt-close-{state.daily_trades}",
        "symbol": symbol,
        "side": "sell",
        "qty": qty,
        "type": "market",
        "time_in_force": "day",
        "limit_price": None,
        "status": "filled",
        "filled_qty": qty,
        "filled_avg_price": exit_price,
        "submitted_at": time_str,
        "filled_at": time_str,
    })

    return pnl


def update_position_prices(state: BacktestState, price_map: dict[str, float]):
    """Update current prices for open positions."""
    for pos in state.positions:
        sym = pos.get("underlying", pos["symbol"][:3])
        if sym in price_map:
            # Rough reprice: use delta approximation
            # This is a simplification; the mock_cli handles detailed repricing
            pass

    state.equity = state.cash + sum(
        p["current_price"] * p["qty"] * 100 for p in state.positions
    )
    state.buying_power = state.equity * 2


def record_decision(state: BacktestState, decision: str, reasoning: str,
                    trade_details: dict = None):
    """Record Claude's decision for this cycle."""
    state.decisions.append({
        "cycle": state.cycle_count,
        "date": state.current_date,
        "time": state.current_time_et,
        "decision": decision,
        "reasoning": reasoning[:500],  # Truncate to save space
        "trade": trade_details,
    })
    state.cycle_count += 1
