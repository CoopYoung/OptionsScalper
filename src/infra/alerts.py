"""Telegram alerting for trade notifications and health monitoring."""

import logging
from decimal import Decimal
from typing import Optional

import aiohttp

from src.infra.config import Settings

logger = logging.getLogger(__name__)


class AlertManager:
    """Send alerts via Telegram bot API."""

    def __init__(self, settings: Settings) -> None:
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._enabled = bool(self._token and self._chat_id)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send(self, message: str) -> None:
        if not self._enabled:
            return
        try:
            session = await self._get_session()
            url = f"https://api.telegram.org/bot{self._token}/sendMessage"
            payload = {"chat_id": self._chat_id, "text": message, "parse_mode": "Markdown"}
            async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 400:
                    payload.pop("parse_mode")
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as retry:
                        if retry.status != 200:
                            logger.error("Telegram alert failed: %s", await retry.text())
                elif resp.status != 200:
                    logger.error("Telegram alert failed: %s", await resp.text())
        except Exception:
            logger.exception("Failed to send Telegram alert")

    async def trade_opened(self, **kwargs) -> None:
        underlying = kwargs.get("underlying", "?")
        option_type = kwargs.get("option_type", "?")
        strike = kwargs.get("strike", "?")
        contracts = kwargs.get("contracts", 0)
        premium = kwargs.get("premium", 0)
        confidence = kwargs.get("confidence", 0)
        reason = kwargs.get("reason", "")
        msg = (f"*Trade Opened*\n"
               f"Underlying: `{underlying}`\n"
               f"Type: {option_type} @ ${strike}\n"
               f"Contracts: {contracts}\n"
               f"Premium: ${premium}\n"
               f"Confidence: {confidence}")
        if reason:
            msg += f"\nReason: {reason}"
        await self.send(msg)

    async def trade_closed(self, **kwargs) -> None:
        underlying = kwargs.get("underlying", "?")
        pnl = kwargs.get("pnl", 0)
        reason = kwargs.get("reason", "")
        hold_time = kwargs.get("hold_time", "")
        entry_premium = kwargs.get("entry_premium", 0)
        exit_premium = kwargs.get("exit_premium", 0)
        sign = "+" if float(pnl) >= 0 else ""
        msg = (f"*Trade Closed*\n"
               f"Underlying: `{underlying}`\n"
               f"Entry: ${entry_premium} → Exit: ${exit_premium}\n"
               f"P&L: {sign}${pnl}\n"
               f"Hold: {hold_time}\n"
               f"Reason: {reason}")
        await self.send(msg)

    async def circuit_breaker_triggered(self, reason: str) -> None:
        await self.send(f"*CIRCUIT BREAKER*\nReason: {reason}")

    async def daily_summary(self, total_pnl: Decimal, trades: int,
                            win_rate: float, portfolio: Decimal) -> None:
        msg = (f"*Daily Summary*\n"
               f"P&L: ${total_pnl}\n"
               f"Trades: {trades}\n"
               f"Win rate: {win_rate:.1%}\n"
               f"Portfolio: ${portfolio}")
        await self.send(msg)

    async def quant_alert(self, alert_type: str, message: str) -> None:
        await self.send(f"*{alert_type}*\n{message}")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
