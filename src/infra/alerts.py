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
        score_breakdown = kwargs.get("score_breakdown", {})
        delta = kwargs.get("delta", 0)
        spot_price = kwargs.get("spot_price", 0)

        msg = (
            f"📈 *ENTRY* `{underlying}` {option_type.upper()} ${strike}\n"
            f"Contracts: {contracts} @ ${premium:.2f}\n"
            f"Confidence: {confidence}% | Delta: {delta:.2f}\n"
            f"Spot: ${spot_price:.2f}"
        )
        if score_breakdown:
            factors = []
            for factor, score in sorted(score_breakdown.items(), key=lambda x: abs(x[1]), reverse=True):
                if abs(score) >= 0.05:
                    arrow = "↑" if score > 0 else "↓"
                    factors.append(f"  {factor}: {arrow}{abs(score):.2f}")
            if factors:
                msg += "\n*Factors:*\n" + "\n".join(factors[:5])
        if reason:
            msg += f"\n_{reason}_"
        await self.send(msg)

    async def trade_closed(self, **kwargs) -> None:
        underlying = kwargs.get("underlying", "?")
        pnl = kwargs.get("pnl", 0)
        reason = kwargs.get("reason", "")
        hold_time = kwargs.get("hold_time", "")
        entry_premium = kwargs.get("entry_premium", 0)
        exit_premium = kwargs.get("exit_premium", 0)
        pnl_pct = kwargs.get("pnl_pct", 0)
        underlying_move = kwargs.get("underlying_move_pct", 0)
        day_pnl = kwargs.get("day_pnl", 0)

        pnl_float = float(pnl)
        emoji = "✅" if pnl_float >= 0 else "❌"
        sign = "+" if pnl_float >= 0 else ""
        day_sign = "+" if float(day_pnl) >= 0 else ""

        msg = (
            f"{emoji} *EXIT* `{underlying}`\n"
            f"${entry_premium:.2f} → ${exit_premium:.2f} ({sign}{pnl_pct:.1%})\n"
            f"P&L: {sign}${pnl_float:.2f} | Hold: {hold_time}\n"
            f"Reason: {reason}\n"
            f"Day P&L: {day_sign}${float(day_pnl):.2f}"
        )
        if underlying_move:
            msg += f" | Underlying: {'+' if underlying_move >= 0 else ''}{underlying_move:.2%}"
        await self.send(msg)

    async def signal_rejected(self, underlying: str, reason: str,
                              confidence: int = 0) -> None:
        """Notify when a signal is generated but blocked by risk/gate checks."""
        msg = (
            f"🚫 *Signal Rejected* `{underlying}`\n"
            f"Confidence: {confidence}%\n"
            f"Reason: {reason}"
        )
        await self.send(msg)

    async def circuit_breaker_triggered(self, reason: str,
                                        resume_time: str = "") -> None:
        msg = f"🛑 *CIRCUIT BREAKER TRIGGERED*\nReason: {reason}"
        if resume_time:
            msg += f"\nResumes: {resume_time}"
        await self.send(msg)

    async def vix_alert(self, vix: float, regime: str, action: str) -> None:
        """Alert on VIX regime changes that affect trading."""
        emoji = "🔴" if regime == "crisis" else "🟡" if regime == "high" else "🟢"
        msg = f"{emoji} *VIX Alert*\nVIX: {vix:.1f} | Regime: {regime}\nAction: {action}"
        await self.send(msg)

    async def daily_summary(self, total_pnl: Decimal, trades: int,
                            win_rate: float, portfolio: Decimal,
                            report: str = "") -> None:
        sign = "+" if total_pnl >= 0 else ""
        msg = (
            f"📊 *Daily Summary*\n"
            f"P&L: {sign}${total_pnl:.2f}\n"
            f"Trades: {trades} | Win Rate: {win_rate:.1%}\n"
            f"Portfolio: ${portfolio:,.2f}"
        )
        if report:
            msg += f"\n\n{report}"
        await self.send(msg)

    async def startup_status(self, checks: dict[str, bool]) -> None:
        """Send pre-market health check results on startup."""
        lines = ["🏁 *Bot Starting*"]
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            lines.append(f"  {status} {check}")
        all_ok = all(checks.values())
        lines.append(f"\n{'Ready to trade' if all_ok else 'ISSUES DETECTED — check logs'}")
        await self.send("\n".join(lines))

    async def quant_alert(self, alert_type: str, message: str) -> None:
        await self.send(f"📡 *{alert_type}*\n{message}")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
