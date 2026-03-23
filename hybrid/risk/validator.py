"""Trade validation layer — hard rules that Claude cannot bypass.

This sits between Claude's tool calls and the actual broker execution.
If a trade violates any rule, it's blocked before reaching the broker.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from hybrid.config import (
    ALLOWED_STRATEGIES,
    ENTRY_CUTOFF_ET,
    ENTRY_START_ET,
    HARD_CLOSE_ET,
    MAX_CONCURRENT_POSITIONS,
    MAX_CONTRACTS_PER_TRADE,
    MAX_DAILY_LOSS,
    MAX_RISK_PER_TRADE,
)

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# Daily state file — persists across cron cycles within a single day
_STATE_FILE = Path(__file__).parent.parent / "trade_logs" / ".daily_state.json"


def _load_daily_state() -> dict:
    """Load daily state (P&L, trade count) — resets each calendar day."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    if _STATE_FILE.exists():
        try:
            state = json.loads(_STATE_FILE.read_text())
            if state.get("date") == today:
                return state
        except (json.JSONDecodeError, KeyError):
            pass
    return {"date": today, "realized_pnl": 0.0, "trades_today": 0, "blocked_today": 0}


def _save_daily_state(state: dict) -> None:
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps(state))


def record_trade_pnl(pnl: float) -> None:
    """Record a closed trade's P&L for daily loss tracking."""
    state = _load_daily_state()
    state["realized_pnl"] += pnl
    state["trades_today"] += 1
    _save_daily_state(state)


def get_daily_state() -> dict:
    """Get current daily state for Claude's context."""
    return _load_daily_state()


class ValidationError(Exception):
    """Raised when a trade violates a hard rule."""
    pass


def validate_new_order(
    symbol: str,
    qty: int,
    side: str,
    order_type: str,
    limit_price: float | None,
    current_positions: list[dict],
    account: dict,
    confidence: int = 100,
) -> dict:
    """Validate a proposed order against all hard rules.

    Returns:
        dict with "approved": True/False and "reason" if blocked.

    Raises nothing — returns a result dict that the tool executor checks.
    """
    violations = []
    state = _load_daily_state()

    # ── Confidence threshold ──
    from hybrid.config import SIGNAL_CONFIDENCE_THRESHOLD
    if confidence < SIGNAL_CONFIDENCE_THRESHOLD:
        violations.append(
            f"Confidence {confidence} < threshold {SIGNAL_CONFIDENCE_THRESHOLD}"
        )

    # ── Time window check ──
    now_et = datetime.now(ET)
    current_time = now_et.strftime("%H:%M")

    if side == "buy":  # Only restrict entries, not exits
        if current_time < ENTRY_START_ET:
            violations.append(
                f"Too early for entries: {current_time} < {ENTRY_START_ET} ET"
            )
        if current_time > ENTRY_CUTOFF_ET:
            violations.append(
                f"Past entry cutoff: {current_time} > {ENTRY_CUTOFF_ET} ET"
            )

    # ── Daily loss limit ──
    if state["realized_pnl"] <= -MAX_DAILY_LOSS:
        violations.append(
            f"Daily loss limit reached: ${state['realized_pnl']:.2f} "
            f"(limit: -${MAX_DAILY_LOSS:.2f})"
        )

    # ── Contract size limit ──
    if qty > MAX_CONTRACTS_PER_TRADE:
        violations.append(
            f"Qty {qty} exceeds max {MAX_CONTRACTS_PER_TRADE} contracts per trade"
        )

    # ── Position count limit (for new entries) ──
    if side == "buy":
        option_positions = [
            p for p in current_positions
            if p.get("asset_class") == "us_option"
        ]
        if len(option_positions) >= MAX_CONCURRENT_POSITIONS:
            violations.append(
                f"Max concurrent positions reached: "
                f"{len(option_positions)}/{MAX_CONCURRENT_POSITIONS}"
            )

    # ── Buying power check ──
    if side == "buy" and limit_price:
        cost = limit_price * qty * 100  # Options are 100 shares per contract
        buying_power = account.get("buying_power", 0)
        if cost > buying_power:
            violations.append(
                f"Insufficient buying power: need ${cost:.2f}, "
                f"have ${buying_power:.2f}"
            )

    # ── Max risk per trade (for defined-risk, estimate from premium) ──
    if side == "buy" and limit_price:
        max_loss = limit_price * qty * 100
        if max_loss > MAX_RISK_PER_TRADE:
            violations.append(
                f"Max risk per trade exceeded: ${max_loss:.2f} > "
                f"${MAX_RISK_PER_TRADE:.2f}"
            )

    # ── PDT protection ──
    # Under $25K equity: max 3 day trades per rolling 5-day window
    # A "day trade" = opening AND closing the same position in one day
    # 0DTE options always count because they must close same day
    equity = account.get("equity", 0)
    if equity < 25_000:
        dt_count = account.get("day_trade_count", 0)
        if side == "buy" and dt_count >= 3:
            violations.append(
                f"PDT protection: {dt_count}/3 day trades used "
                f"(equity ${equity:,.0f} < $25K). "
                f"Cannot open new 0DTE positions — they count as day trades."
            )

    # ── Trading blocked check ──
    if account.get("trading_blocked"):
        violations.append("Trading is blocked on this account")

    if violations:
        state["blocked_today"] += 1
        _save_daily_state(state)
        reason = "; ".join(violations)
        logger.warning("Order BLOCKED: %s %d x %s — %s", side, qty, symbol, reason)
        return {"approved": False, "reason": reason, "violations": violations}

    logger.info("Order APPROVED: %s %d x %s", side, qty, symbol)
    return {"approved": True, "reason": "All checks passed"}


def validate_close(symbol: str, current_positions: list[dict]) -> dict:
    """Validate a position close — always allowed if position exists."""
    has_position = any(p["symbol"] == symbol for p in current_positions)
    if not has_position:
        return {
            "approved": False,
            "reason": f"No open position for {symbol}",
        }
    return {"approved": True, "reason": "Position exists, close allowed"}


def should_force_close_all() -> bool:
    """Check if we're past hard close time — must exit everything."""
    now_et = datetime.now(ET)
    current_time = now_et.strftime("%H:%M")
    return current_time >= HARD_CLOSE_ET


def is_market_hours() -> bool:
    """Check if we're within trading hours."""
    now_et = datetime.now(ET)
    weekday = now_et.weekday()
    if weekday >= 5:  # Saturday/Sunday
        return False
    current_time = now_et.strftime("%H:%M")
    return "09:30" <= current_time <= "16:00"
