"""Telegram notifications for the hybrid trader.

Sends trade alerts, daily summaries, and error notifications.
Uses the same bot credentials as the original trading bot.
"""

import logging
from typing import Optional

import requests

from hybrid.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)

_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


def send_message(text: str, parse_mode: str = "HTML") -> bool:
    """Send a message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.debug("Telegram not configured, skipping notification")
        return False

    try:
        resp = requests.post(
            f"{_BASE_URL}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            return True
        logger.warning("Telegram send failed: %d %s", resp.status_code, resp.text)
        return False
    except Exception as e:
        logger.warning("Telegram send error: %s", e)
        return False


def notify_cycle_result(result: dict) -> None:
    """Send a notification for a completed analysis cycle."""
    action = result.get("action", "UNKNOWN")
    trades = result.get("trades", [])
    errors = result.get("errors", [])
    usage = result.get("token_usage", {})
    daily = result.get("daily_state", {})

    # Only notify on trades, errors, or force close — skip quiet cycles
    if action in ("NO_TRADE", "HOLD", "ANALYZED") and not errors:
        return

    lines = [f"<b>🤖 Trading Cycle — {action}</b>"]
    lines.append("")

    # Trades
    for trade in trades:
        emoji = "🟢" if trade["type"] == "entry" else "🔴"
        symbol = trade.get("symbol", "?")
        side = trade.get("side", "?")
        qty = trade.get("qty", "?")
        price = trade.get("limit_price", "market")
        lines.append(
            f"{emoji} {trade['type'].upper()}: {side} {qty}x {symbol} "
            f"@ ${price}"
        )

    # Daily state
    if daily:
        pnl = daily.get("realized_pnl", 0)
        pnl_emoji = "💰" if pnl >= 0 else "📉"
        lines.append("")
        lines.append(
            f"{pnl_emoji} Daily P&L: ${pnl:+.2f} | "
            f"Trades: {daily.get('trades_today', 0)} | "
            f"Blocked: {daily.get('blocked_today', 0)}"
        )

    # Token cost
    cost = usage.get("estimated_cost", 0)
    lines.append(f"⚡ Tokens: {usage.get('input_tokens', 0):,}+{usage.get('output_tokens', 0):,} (${cost:.4f})")

    # Errors
    if errors:
        lines.append("")
        lines.append("<b>⚠️ Errors:</b>")
        for err in errors[:3]:  # Limit to 3 errors
            lines.append(f"  • {err[:100]}")

    # Reasoning excerpt
    reasoning = result.get("reasoning", "")
    if reasoning and action not in ("NO_TRADE", "HOLD"):
        # Extract the CYCLE SUMMARY section
        if "CYCLE SUMMARY" in reasoning:
            summary = reasoning.split("CYCLE SUMMARY")[1][:500]
            lines.append("")
            lines.append(f"<i>{summary.strip()}</i>")

    send_message("\n".join(lines))


def notify_startup() -> None:
    """Send startup notification."""
    send_message(
        "🤖 <b>Hybrid Trader Started</b>\n"
        "Claude + Alpaca paper trading active.\n"
        "Monitoring SPY, QQQ, IWM for setups."
    )


def notify_shutdown(reason: str = "") -> None:
    """Send shutdown notification."""
    send_message(
        f"🛑 <b>Hybrid Trader Stopped</b>\n"
        f"Reason: {reason or 'Manual shutdown'}"
    )


def notify_daily_summary(daily_state: dict, positions: list, account: dict) -> None:
    """Send end-of-day summary."""
    pnl = daily_state.get("realized_pnl", 0)
    trades = daily_state.get("trades_today", 0)
    blocked = daily_state.get("blocked_today", 0)
    equity = account.get("equity", 0)

    lines = [
        "<b>📊 Daily Trading Summary</b>",
        "",
        f"💰 Realized P&L: ${pnl:+.2f}",
        f"📈 Portfolio Value: ${equity:,.2f}",
        f"🔄 Trades Executed: {trades}",
        f"🚫 Orders Blocked: {blocked}",
    ]

    if positions:
        lines.append("")
        lines.append(f"<b>Open Positions ({len(positions)}):</b>")
        for p in positions:
            pl = p.get("unrealized_pl", 0)
            emoji = "🟢" if pl >= 0 else "🔴"
            lines.append(
                f"  {emoji} {p['symbol']}: {p['qty']}x @ "
                f"${p['avg_entry_price']:.2f} → ${p['current_price']:.2f} "
                f"(${pl:+.2f})"
            )

    send_message("\n".join(lines))


def notify_error(error: str) -> None:
    """Send error notification."""
    send_message(f"🚨 <b>Error:</b> {error[:500]}")
