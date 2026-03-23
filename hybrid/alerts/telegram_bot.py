"""Telegram command bot — remote control for the trading bot.

Polls for commands from the configured Telegram chat. Runs in a
background thread alongside the main orchestrator loop.

Commands:
    /status   — Show bot state, P&L, positions
    /pause    — Stop entering new trades (still manages exits)
    /resume   — Resume trading
    /kill     — Force-close all positions and halt the bot
    /positions — Show open positions
    /digest   — Show the last digest sent to the LLM
    /help     — List available commands
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from hybrid.config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")

# Shared state — the orchestrator reads these
_state = {
    "paused": False,
    "kill_requested": False,
    "last_status_request": None,
}
_state_lock = threading.Lock()

# Store last digest for /digest command
_last_digest: str = ""


def is_paused() -> bool:
    """Check if trading is paused (orchestrator calls this before entries)."""
    with _state_lock:
        return _state["paused"]


def is_kill_requested() -> bool:
    """Check if a kill was requested (orchestrator calls this each cycle)."""
    with _state_lock:
        return _state["kill_requested"]


def set_last_digest(digest: str) -> None:
    """Store the last digest for the /digest command."""
    global _last_digest
    _last_digest = digest


def _send(text: str) -> None:
    """Send a message to the configured Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            },
            timeout=10,
        )
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)


def _get_updates(offset: int | None = None) -> list:
    """Poll for new messages."""
    if not TELEGRAM_BOT_TOKEN:
        return []
    try:
        params = {"timeout": 10, "allowed_updates": ["message"]}
        if offset is not None:
            params["offset"] = offset
        resp = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
            params=params,
            timeout=15,
        )
        if resp.status_code == 200:
            return resp.json().get("result", [])
    except Exception as e:
        logger.debug("Telegram poll error: %s", e)
    return []


def _handle_command(command: str, chat_id: str) -> None:
    """Process a command from Telegram."""
    # Only respond to the configured chat
    if str(chat_id) != str(TELEGRAM_CHAT_ID):
        return

    cmd = command.strip().lower().split()[0] if command.strip() else ""

    if cmd == "/status":
        _cmd_status()
    elif cmd == "/pause":
        _cmd_pause()
    elif cmd == "/resume":
        _cmd_resume()
    elif cmd == "/kill":
        _cmd_kill()
    elif cmd == "/positions":
        _cmd_positions()
    elif cmd == "/digest":
        _cmd_digest()
    elif cmd == "/help" or cmd == "/start":
        _cmd_help()
    else:
        _send(f"Unknown command: {cmd}\nSend /help for available commands.")


def _cmd_status() -> None:
    """Show bot status."""
    from hybrid.risk.validator import get_daily_state

    now = datetime.now(ET)
    daily = get_daily_state()
    paused = is_paused()
    kill = is_kill_requested()

    status_emoji = "🔴" if kill else ("⏸" if paused else "🟢")
    status_text = "KILLED" if kill else ("PAUSED" if paused else "RUNNING")

    lines = [
        f"<b>{status_emoji} Bot Status: {status_text}</b>",
        f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')} ET",
        "",
        f"Daily P&L: ${daily.get('realized_pnl', 0):+.2f}",
        f"Trades today: {daily.get('trades_today', 0)}",
        f"Blocked today: {daily.get('blocked_today', 0)}",
    ]

    # LLM info
    try:
        from hybrid import config
        model = config.OLLAMA_MODEL if config.LLM_PROVIDER == "ollama" else (
            config.CLAUDE_MODEL if config.LLM_PROVIDER == "anthropic" else config.OPENAI_MODEL
        )
        lines.append(f"Model: {config.LLM_PROVIDER} / {model}")
    except Exception:
        pass

    # Account info
    try:
        from hybrid.broker.broker_base import AlpacaBroker
        broker = AlpacaBroker()
        acct = broker.get_account()
        lines.append("")
        lines.append(f"Equity: ${acct.get('equity', 0):,.2f}")
        lines.append(f"Buying Power: ${acct.get('buying_power', 0):,.2f}")
    except Exception as e:
        lines.append(f"\n⚠️ Account fetch failed: {e}")

    _send("\n".join(lines))


def _cmd_pause() -> None:
    """Pause trading."""
    with _state_lock:
        _state["paused"] = True
    _send("⏸ <b>Trading PAUSED</b>\n"
          "Exits will still be managed. No new entries.\n"
          "Send /resume to continue trading.")
    logger.info("Trading PAUSED via Telegram")


def _cmd_resume() -> None:
    """Resume trading."""
    with _state_lock:
        _state["paused"] = False
    _send("▶️ <b>Trading RESUMED</b>\n"
          "Bot will enter new trades when signals appear.")
    logger.info("Trading RESUMED via Telegram")


def _cmd_kill() -> None:
    """Kill switch — close everything and halt."""
    with _state_lock:
        _state["kill_requested"] = True
        _state["paused"] = True

    _send("🔴 <b>KILL SWITCH ACTIVATED</b>\n"
          "Force-closing all positions and halting the bot.\n"
          "Restart the systemd service to resume.")
    logger.warning("KILL SWITCH activated via Telegram")

    # Attempt to close all positions immediately
    try:
        from hybrid.broker.broker_base import AlpacaBroker
        broker = AlpacaBroker()
        positions = broker.get_positions()
        option_positions = [p for p in positions if p.get("asset_class") == "us_option"]

        if option_positions:
            for pos in option_positions:
                try:
                    broker.close_position(pos["symbol"])
                    _send(f"  Closed {pos['symbol']}")
                except Exception as e:
                    _send(f"  ❌ Failed to close {pos['symbol']}: {e}")
            _send(f"🔴 Closed {len(option_positions)} positions")
        else:
            _send("No open positions to close.")
    except Exception as e:
        _send(f"❌ Kill failed: {e}")
        logger.error("Kill switch position close failed: %s", e)


def _cmd_positions() -> None:
    """Show open positions."""
    try:
        from hybrid.broker.broker_base import AlpacaBroker
        broker = AlpacaBroker()
        positions = broker.get_positions()
        option_positions = [p for p in positions if p.get("asset_class") == "us_option"]

        if not option_positions:
            _send("📋 No open positions")
            return

        lines = [f"<b>📋 Open Positions ({len(option_positions)})</b>"]
        for p in option_positions:
            pl = p.get("unrealized_pl", 0)
            pl_pct = p.get("unrealized_plpc", 0) * 100
            emoji = "🟢" if pl >= 0 else "🔴"
            lines.append(
                f"  {emoji} {p['symbol']}\n"
                f"    qty {p['qty']} | entry ${p['avg_entry_price']:.2f} | "
                f"now ${p['current_price']:.2f}\n"
                f"    P&L ${pl:+.2f} ({pl_pct:+.1f}%)"
            )
        _send("\n".join(lines))
    except Exception as e:
        _send(f"❌ Position fetch failed: {e}")


def _cmd_digest() -> None:
    """Show the last digest sent to the LLM."""
    if _last_digest:
        # Telegram has a 4096 char limit — truncate if needed
        text = _last_digest[:3900]
        if len(_last_digest) > 3900:
            text += f"\n\n... (truncated, {len(_last_digest)} chars total)"
        _send(f"<pre>{text}</pre>")
    else:
        _send("No digest generated yet. Wait for the next cycle.")


def _cmd_help() -> None:
    """Show help."""
    _send(
        "<b>🤖 OptionsScalper Commands</b>\n"
        "\n"
        "/status — Bot state, P&L, account info\n"
        "/positions — Open positions with P&L\n"
        "/pause — Stop new entries (exits still managed)\n"
        "/resume — Resume trading\n"
        "/kill — Force-close all positions and halt\n"
        "/digest — Show last market digest\n"
        "/help — This message"
    )


def start_polling() -> threading.Thread:
    """Start the Telegram command polling thread.

    Returns the thread so the caller can join/stop it.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.info("Telegram not configured — command bot disabled")
        return None

    def _poll_loop():
        logger.info("Telegram command bot started")
        offset = None

        while True:
            try:
                if is_kill_requested():
                    # Keep polling even after kill so we can report status
                    pass

                updates = _get_updates(offset)
                for update in updates:
                    offset = update["update_id"] + 1
                    message = update.get("message", {})
                    text = message.get("text", "")
                    chat_id = message.get("chat", {}).get("id", "")

                    if text.startswith("/"):
                        _handle_command(text, chat_id)

            except Exception as e:
                logger.warning("Telegram poll loop error: %s", e)
                time.sleep(5)

    thread = threading.Thread(target=_poll_loop, daemon=True, name="telegram-bot")
    thread.start()
    return thread
