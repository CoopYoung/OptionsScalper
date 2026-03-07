"""Shared test fixtures for the options scalper test suite."""

import os
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

# Prevent pydantic-settings from trying to load .env files during tests
os.environ.setdefault("ALPACA_API_KEY", "test-key")
os.environ.setdefault("ALPACA_SECRET_KEY", "test-secret")

from src.infra.config import Settings
from src.risk.circuit_breaker import CircuitBreaker
from src.risk.manager import OptionsRiskManager, PDTTracker, PortfolioGreeks
from src.strategy.base import OptionsContract, TradeDirection, TradeSignal


@pytest.fixture
def settings() -> Settings:
    return Settings(
        alpaca_api_key="test-key",
        alpaca_secret_key="test-secret",
        alpaca_paper=True,
        signal_confidence_threshold=55,
        kelly_fraction=Decimal("0.20"),
        max_position_pct=Decimal("0.05"),
        max_portfolio_exposure=Decimal("0.30"),
        daily_drawdown_halt=Decimal("0.08"),
        max_portfolio_delta=50.0,
        max_portfolio_gamma=20.0,
        max_portfolio_theta=-100.0,
        max_portfolio_vega=30.0,
        max_positions_per_underlying=3,
        max_contracts_per_trade=10,
        pt_profit_target_pct=0.50,
        sl_stop_loss_pct=0.30,
        sl_trailing_pct=0.20,
        hard_close="15:15",
        entry_start="09:45",
        entry_cutoff="14:30",
        max_consecutive_losses_window=3,
        consecutive_loss_window_minutes=30,
        circuit_breaker_cooldown_hours=2,
        target_delta=0.30,
        min_premium=0.50,
        max_premium=5.00,
        min_spread_ratio=0.90,
        weight_technical=0.25,
        weight_tick_momentum=0.20,
        weight_gex=0.15,
        weight_flow=0.15,
        weight_vix=0.10,
        weight_internals=0.10,
        weight_sentiment=0.05,
    )


@pytest.fixture
def circuit_breaker(settings: Settings) -> CircuitBreaker:
    return CircuitBreaker(settings)


@pytest.fixture
def risk_manager(settings: Settings, circuit_breaker: CircuitBreaker) -> OptionsRiskManager:
    rm = OptionsRiskManager(settings, circuit_breaker)
    rm.set_portfolio_value(Decimal("100000"))
    rm.set_day_start_value(Decimal("100000"))
    return rm


def make_contract(
    symbol: str = "SPY250306C00550000",
    underlying: str = "SPY",
    option_type: str = "call",
    strike: Decimal = Decimal("550"),
    bid: Decimal = Decimal("2.00"),
    ask: Decimal = Decimal("2.20"),
    delta: float = 0.30,
    gamma: float = 0.02,
    theta: float = -0.05,
    vega: float = 0.08,
    iv: float = 0.25,
    volume: int = 500,
    open_interest: int = 1000,
) -> OptionsContract:
    return OptionsContract(
        symbol=symbol,
        underlying=underlying,
        option_type=option_type,
        strike=strike,
        expiration="2025-03-06",
        bid=bid,
        ask=ask,
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        iv=iv,
        volume=volume,
        open_interest=open_interest,
    )


def make_signal(
    direction: TradeDirection = TradeDirection.BUY_CALL,
    confidence: int = 70,
    underlying: str = "SPY",
    contract: OptionsContract | None = None,
    target_price: Decimal = Decimal("2.10"),
) -> TradeSignal:
    return TradeSignal(
        direction=direction,
        confidence=confidence,
        underlying=underlying,
        contract=contract or make_contract(),
        target_price=target_price,
        reason="Test signal",
    )
