"""Abstract strategy interface for 0DTE options trading."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class TradeDirection(str, Enum):
    BUY_CALL = "BUY_CALL"
    BUY_PUT = "BUY_PUT"
    HOLD = "HOLD"


@dataclass
class OptionsContract:
    """Represents a single options contract."""
    symbol: str              # e.g. "SPY250306C00550000"
    underlying: str          # e.g. "SPY"
    option_type: str         # "call" or "put"
    strike: Decimal
    expiration: str          # ISO date
    bid: Decimal = Decimal("0")
    ask: Decimal = Decimal("0")
    last: Decimal = Decimal("0")
    volume: int = 0
    open_interest: int = 0
    # Greeks
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    iv: float = 0.0          # Implied volatility

    @property
    def mid(self) -> Decimal:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def spread(self) -> Decimal:
        return self.ask - self.bid

    @property
    def spread_ratio(self) -> float:
        """Bid/ask ratio. 1.0 = no spread, 0.0 = infinitely wide."""
        if self.ask <= 0:
            return 0.0
        return float(self.bid / self.ask)


@dataclass
class TradeSignal:
    """Output of a strategy evaluation."""
    direction: TradeDirection
    confidence: int                          # 0-100
    underlying: str                          # "SPY", "QQQ", etc.
    contract: Optional[OptionsContract]      # Recommended contract to trade
    target_price: Decimal                    # Limit price for the order
    reason: str                              # Human-readable explanation
    score_breakdown: dict = field(default_factory=dict)  # Per-factor scores

    @property
    def should_trade(self) -> bool:
        return self.direction != TradeDirection.HOLD


class BaseStrategy(ABC):
    """Abstract base for all trading strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def evaluate(self, **kwargs) -> TradeSignal:
        ...

    def _make_hold(self, underlying: str, reason: str = "No signal") -> TradeSignal:
        return TradeSignal(
            direction=TradeDirection.HOLD,
            confidence=0,
            underlying=underlying,
            contract=None,
            target_price=Decimal("0"),
            reason=reason,
        )
