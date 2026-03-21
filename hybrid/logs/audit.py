"""Structured audit logging — every cycle is recorded.

Writes JSON lines to trade_logs/audit.jsonl for full traceability.
Each line is a complete record of what Claude saw, decided, and did.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from hybrid.config import AUDIT_LOG

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def log_cycle(result: dict) -> None:
    """Append a cycle result to the audit log."""
    entry = {
        "timestamp_utc": datetime.utcnow().isoformat(),
        "timestamp_et": datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S"),
        "action": result.get("action", "UNKNOWN"),
        "trades": result.get("trades", []),
        "errors": result.get("errors", []),
        "token_usage": result.get("token_usage", {}),
        "daily_state": result.get("daily_state", {}),
        # Store reasoning but truncate if massive
        "reasoning_excerpt": _truncate(result.get("reasoning", ""), 2000),
    }

    try:
        AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT_LOG, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.error("Failed to write audit log: %s", e)


def get_recent_cycles(n: int = 10) -> list[dict]:
    """Read the last N cycle entries from the audit log."""
    if not AUDIT_LOG.exists():
        return []

    lines = AUDIT_LOG.read_text().strip().split("\n")
    recent = lines[-n:] if len(lines) > n else lines
    entries = []
    for line in recent:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def get_todays_trades() -> list[dict]:
    """Get all trades executed today."""
    today = datetime.now(ET).strftime("%Y-%m-%d")
    trades = []

    if not AUDIT_LOG.exists():
        return trades

    for line in AUDIT_LOG.read_text().strip().split("\n"):
        try:
            entry = json.loads(line)
            if entry.get("timestamp_et", "").startswith(today):
                for trade in entry.get("trades", []):
                    trade["cycle_time"] = entry["timestamp_et"]
                    trades.append(trade)
        except json.JSONDecodeError:
            continue

    return trades


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "... [truncated]"
