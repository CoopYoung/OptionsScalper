"""Options risk manager: Kelly sizing, Greeks limits, PDT tracking.

Adapted from poly-trader with:
    - Greeks-based portfolio exposure limits
    - VIX-adjusted Kelly sizing
    - PDT (Pattern Day Trader) rule tracking
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from src.infra.config import Settings
from src.risk.circuit_breaker import CircuitBreaker
from src.strategy.base import OptionsContract, TradeDirection, TradeSignal

logger = logging.getLogger(__name__)


@dataclass
class PortfolioGreeks:
    """Aggregate portfolio Greek exposure."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0

    def add(self, contract: OptionsContract, qty: int) -> None:
        self.delta += contract.delta * qty * 100
        self.gamma += contract.gamma * qty * 100
        self.theta += contract.theta * qty * 100
        self.vega += contract.vega * qty * 100


@dataclass
class PDTTracker:
    """Pattern Day Trader rule tracking.

    PDT rule: 4+ round trips in 5 business days requires $25k equity.
    """
    day_trades: deque = field(default_factory=lambda: deque(maxlen=50))
    account_equity: Decimal = Decimal("0")

    def record_round_trip(self, date: date) -> None:
        self.day_trades.append(date)

    @property
    def rolling_5d_count(self) -> int:
        cutoff = date.today() - timedelta(days=7)  # ~5 business days
        return sum(1 for d in self.day_trades if d >= cutoff)

    @property
    def is_pdt_restricted(self) -> bool:
        """True if at PDT limit and under $25k."""
        return self.rolling_5d_count >= 3 and self.account_equity < Decimal("25000")

    @property
    def remaining_day_trades(self) -> int:
        if self.account_equity >= Decimal("25000"):
            return 999  # Unlimited
        return max(0, 3 - self.rolling_5d_count)

    def status(self) -> dict:
        return {
            "rolling_5d": self.rolling_5d_count,
            "remaining": self.remaining_day_trades,
            "restricted": self.is_pdt_restricted,
            "equity": str(self.account_equity),
        }


class OptionsRiskManager:
    """Risk management for 0DTE options trading."""

    def __init__(self, settings: Settings, circuit_breaker: CircuitBreaker) -> None:
        self._settings = settings
        self._cb = circuit_breaker
        self._pdt = PDTTracker()
        self._portfolio_greeks = PortfolioGreeks()
        self._portfolio_value = Decimal("10000")
        self._daily_pnl = Decimal("0")
        self._day_start_value = Decimal("10000")
        self._open_positions: dict[str, dict] = {}
        self._positions_per_underlying: dict[str, int] = {}
        self._trade_count_today: int = 0
        self._win_count: int = 0
        self._loss_count: int = 0
        self._total_pnl: Decimal = Decimal("0")

    def set_portfolio_value(self, value: Decimal) -> None:
        self._portfolio_value = value
        self._pdt.account_equity = value

    def set_day_start_value(self, value: Decimal) -> None:
        self._day_start_value = value

    # ── Pre-Trade Checks ──────────────────────────────────────

    def can_trade(self, signal: TradeSignal) -> tuple[bool, str]:
        """Full pre-trade risk check. Returns (allowed, reason)."""
        # Circuit breaker
        if self._cb.is_halted:
            return False, f"Circuit breaker: {self._cb.halt_reason}"

        # PDT check
        if self._pdt.is_pdt_restricted:
            return False, f"PDT restricted ({self._pdt.rolling_5d_count}/3 day trades, equity ${self._pdt.account_equity})"

        if self._pdt.remaining_day_trades <= 0:
            return False, "No day trades remaining (PDT rule)"

        # Daily drawdown check
        drawdown = self._compute_drawdown()
        if drawdown >= self._settings.daily_drawdown_halt:
            self._cb.check_drawdown(drawdown)
            return False, f"Daily drawdown {float(drawdown):.2%} exceeds limit"

        # Confidence threshold
        if signal.confidence < self._settings.signal_confidence_threshold:
            return False, f"Confidence {signal.confidence} < {self._settings.signal_confidence_threshold}"

        # Portfolio exposure
        if not self._check_portfolio_exposure(signal):
            return False, "Max portfolio exposure reached"

        # Greeks limits
        if signal.contract:
            greeks_ok, greeks_reason = self._check_greeks_room(signal.contract)
            if not greeks_ok:
                return False, greeks_reason

        # Per-underlying limit
        underlying = signal.underlying
        count = self._positions_per_underlying.get(underlying, 0)
        if count >= self._settings.max_positions_per_underlying:
            return False, f"Max positions for {underlying} ({count}/{self._settings.max_positions_per_underlying})"

        return True, "OK"

    def compute_position_size(
        self, signal: TradeSignal, vix_multiplier: float = 1.0,
    ) -> int:
        """Kelly-based position sizing with VIX adjustment.

        Uses option delta as proxy for win probability:
        - Delta 0.30 → ~30% chance of expiring ITM
        - For 0DTE scalps, we're not holding to expiry, so
          adjust win_prob based on confidence score.
        """
        if not signal.contract:
            return 0

        contract = signal.contract

        # Win probability from confidence (scaled 0-1)
        win_prob = signal.confidence / 100

        # Payout ratio: target profit / risk (premium)
        target_pct = self._settings.pt_profit_target_pct
        stop_pct = self._settings.sl_stop_loss_pct
        payout_ratio = target_pct / stop_pct if stop_pct > 0 else 1.0

        # Kelly criterion: f = (bp - q) / b
        q = 1 - win_prob
        b = payout_ratio
        kelly_f = (b * win_prob - q) / b if b > 0 else 0

        if kelly_f <= 0:
            return 0

        # Apply fraction and VIX adjustment
        adjusted_kelly = kelly_f * float(self._settings.kelly_fraction) * vix_multiplier

        # Circuit breaker multiplier
        adjusted_kelly *= float(self._cb.position_size_multiplier)

        # Max position as % of portfolio
        max_notional = float(self._portfolio_value) * float(self._settings.max_position_pct)
        premium_per_contract = float(contract.mid) * 100  # 100 shares per contract

        if premium_per_contract <= 0:
            return 0

        # Kelly-based contracts
        kelly_notional = float(self._portfolio_value) * adjusted_kelly
        kelly_contracts = int(kelly_notional / premium_per_contract)

        # Cap at max contracts and max notional
        max_by_notional = int(max_notional / premium_per_contract)
        contracts = min(
            kelly_contracts,
            max_by_notional,
            self._settings.max_contracts_per_trade,
        )

        # Ensure at least 1 if we got a signal
        contracts = max(1, contracts)

        # Final Greeks check: would adding these contracts breach limits?
        if not self._would_greeks_fit(contract, contracts):
            # Reduce until they fit
            while contracts > 0 and not self._would_greeks_fit(contract, contracts):
                contracts -= 1

        logger.info(
            "Sizing: kelly=%.4f adj=%.4f vix_mult=%.2f contracts=%d (premium=$%.2f/ea)",
            kelly_f, adjusted_kelly, vix_multiplier, contracts, premium_per_contract / 100,
        )
        return contracts

    # ── Greeks Management ─────────────────────────────────────

    def _check_greeks_room(self, contract: OptionsContract) -> tuple[bool, str]:
        """Check if we have room for 1 more contract within Greeks limits."""
        return self._would_greeks_fit_with_reason(contract, 1)

    def _would_greeks_fit(self, contract: OptionsContract, qty: int) -> bool:
        ok, _ = self._would_greeks_fit_with_reason(contract, qty)
        return ok

    def _would_greeks_fit_with_reason(self, contract: OptionsContract, qty: int) -> tuple[bool, str]:
        proposed_delta = self._portfolio_greeks.delta + contract.delta * qty * 100
        proposed_gamma = self._portfolio_greeks.gamma + contract.gamma * qty * 100
        proposed_theta = self._portfolio_greeks.theta + contract.theta * qty * 100
        proposed_vega = self._portfolio_greeks.vega + contract.vega * qty * 100

        if abs(proposed_delta) > self._settings.max_portfolio_delta:
            return False, f"Delta would be {proposed_delta:.1f} (limit ±{self._settings.max_portfolio_delta})"
        if abs(proposed_gamma) > self._settings.max_portfolio_gamma:
            return False, f"Gamma would be {proposed_gamma:.1f} (limit ±{self._settings.max_portfolio_gamma})"
        if proposed_theta < self._settings.max_portfolio_theta:
            return False, f"Theta would be {proposed_theta:.1f} (limit {self._settings.max_portfolio_theta})"
        if abs(proposed_vega) > self._settings.max_portfolio_vega:
            return False, f"Vega would be {proposed_vega:.1f} (limit ±{self._settings.max_portfolio_vega})"

        return True, "OK"

    def update_portfolio_greeks(self, positions: list[dict]) -> None:
        """Recalculate portfolio Greeks from all open positions."""
        self._portfolio_greeks = PortfolioGreeks()
        self._positions_per_underlying = {}

        for pos in positions:
            contract = pos.get("contract")
            qty = pos.get("qty", 0)
            underlying = pos.get("underlying", "")

            if contract and isinstance(contract, OptionsContract):
                self._portfolio_greeks.add(contract, qty)

            self._positions_per_underlying[underlying] = (
                self._positions_per_underlying.get(underlying, 0) + 1
            )

    # ── Portfolio Exposure ────────────────────────────────────

    def _check_portfolio_exposure(self, signal: TradeSignal) -> bool:
        """Check total portfolio exposure doesn't exceed limit."""
        if not signal.contract:
            return True

        current_exposure = sum(
            pos.get("notional", 0) for pos in self._open_positions.values()
        )
        new_notional = float(signal.contract.mid) * 100
        total = current_exposure + new_notional

        max_exposure = float(self._portfolio_value) * float(self._settings.max_portfolio_exposure)
        return total <= max_exposure

    def _compute_drawdown(self) -> Decimal:
        if self._day_start_value <= 0:
            return Decimal("0")
        return (self._day_start_value - self._portfolio_value) / self._day_start_value

    # ── Position Tracking ─────────────────────────────────────

    def record_open(self, symbol: str, underlying: str, qty: int, premium: Decimal, contract: OptionsContract = None) -> None:
        existing = self._open_positions.get(symbol)
        if existing:
            # Aggregating into existing position — update notional and qty
            self._open_positions[symbol] = {
                "underlying": underlying,
                "qty": qty,  # Already aggregated total from engine
                "entry_premium": premium,  # Already averaged from engine
                "notional": float(premium) * 100 * qty,
                "contract": contract or existing.get("contract"),
            }
            # Don't increment positions_per_underlying (same position)
        else:
            self._open_positions[symbol] = {
                "underlying": underlying,
                "qty": qty,
                "entry_premium": premium,
                "notional": float(premium) * 100 * qty,
                "contract": contract,
            }
            self._positions_per_underlying[underlying] = (
                self._positions_per_underlying.get(underlying, 0) + 1
            )
        if contract:
            self._portfolio_greeks.add(contract, qty)
        self._trade_count_today += 1

    def record_close(self, symbol: str, pnl: Decimal) -> None:
        pos = self._open_positions.pop(symbol, None)
        if pos:
            underlying = pos["underlying"]
            count = self._positions_per_underlying.get(underlying, 1)
            self._positions_per_underlying[underlying] = max(0, count - 1)

            # Update Greeks (subtract closed position)
            contract = pos.get("contract")
            qty = pos.get("qty", 0)
            if contract and isinstance(contract, OptionsContract):
                self._portfolio_greeks.delta -= contract.delta * qty * 100
                self._portfolio_greeks.gamma -= contract.gamma * qty * 100
                self._portfolio_greeks.theta -= contract.theta * qty * 100
                self._portfolio_greeks.vega -= contract.vega * qty * 100

        self._daily_pnl += pnl
        self._portfolio_value += pnl
        self._total_pnl += pnl

        if pnl > 0:
            self._win_count += 1
        elif pnl < 0:
            self._loss_count += 1
            self._cb.record_loss(pnl)

        # PDT tracking
        self._pdt.record_round_trip(date.today())

    def reset_daily(self) -> None:
        self._daily_pnl = Decimal("0")
        self._day_start_value = self._portfolio_value
        self._trade_count_today = 0

    # ── Exit Management ───────────────────────────────────────

    def should_exit(
        self, symbol: str, current_premium: Decimal, peak_premium: Decimal,
        entry_premium: Decimal, now: Optional[datetime] = None,
    ) -> tuple[bool, str]:
        """Check if a position should be exited."""
        now = now or datetime.now(timezone.utc)

        if entry_premium <= 0:
            return False, ""

        pnl_pct = float((current_premium - entry_premium) / entry_premium)

        # Profit target
        if pnl_pct >= self._settings.pt_profit_target_pct:
            return True, f"Profit target ({pnl_pct:.1%} >= {self._settings.pt_profit_target_pct:.1%})"

        # Stop loss
        if pnl_pct <= -self._settings.sl_stop_loss_pct:
            return True, f"Stop loss ({pnl_pct:.1%} <= -{self._settings.sl_stop_loss_pct:.1%})"

        # Trailing stop
        if peak_premium > entry_premium:
            trail_pct = float((peak_premium - current_premium) / peak_premium)
            if trail_pct >= self._settings.sl_trailing_pct:
                return True, f"Trailing stop ({trail_pct:.1%} from peak ${float(peak_premium):.2f})"

        # Time exit (hard close before 3:30 PM ET)
        from zoneinfo import ZoneInfo
        et = now.astimezone(ZoneInfo("America/New_York"))
        close_h, close_m = map(int, self._settings.hard_close.split(":"))
        if et.hour > close_h or (et.hour == close_h and et.minute >= close_m):
            return True, f"Time exit ({self._settings.hard_close} ET)"

        return False, ""

    # ── Status ────────────────────────────────────────────────

    def status(self) -> dict:
        total = self._win_count + self._loss_count
        return {
            "portfolio_value": str(self._portfolio_value),
            "daily_pnl": str(self._daily_pnl),
            "drawdown": f"{float(self._compute_drawdown()):.2%}",
            "open_positions": len(self._open_positions),
            "trades_today": self._trade_count_today,
            "win_rate": f"{self._win_count / total:.1%}" if total > 0 else "N/A",
            "total_pnl": str(self._total_pnl),
            "portfolio_greeks": {
                "delta": round(self._portfolio_greeks.delta, 1),
                "gamma": round(self._portfolio_greeks.gamma, 1),
                "theta": round(self._portfolio_greeks.theta, 1),
                "vega": round(self._portfolio_greeks.vega, 1),
            },
            "pdt": self._pdt.status(),
            "circuit_breaker": self._cb.status(),
        }
