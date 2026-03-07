"""Tests for strategy base types: OptionsContract, TradeSignal, TradeDirection."""

from decimal import Decimal

import pytest

from src.strategy.base import BaseStrategy, OptionsContract, TradeDirection, TradeSignal


class TestOptionsContract:
    def test_mid_price(self):
        c = OptionsContract(
            symbol="SPY250306C00550000", underlying="SPY",
            option_type="call", strike=Decimal("550"),
            expiration="2025-03-06",
            bid=Decimal("2.00"), ask=Decimal("2.20"),
        )
        assert c.mid == Decimal("2.10")

    def test_mid_fallback_to_last(self):
        c = OptionsContract(
            symbol="SPY250306C00550000", underlying="SPY",
            option_type="call", strike=Decimal("550"),
            expiration="2025-03-06",
            bid=Decimal("0"), ask=Decimal("0"), last=Decimal("1.50"),
        )
        assert c.mid == Decimal("1.50")

    def test_spread(self):
        c = OptionsContract(
            symbol="SPY250306C00550000", underlying="SPY",
            option_type="call", strike=Decimal("550"),
            expiration="2025-03-06",
            bid=Decimal("2.00"), ask=Decimal("2.20"),
        )
        assert c.spread == Decimal("0.20")

    def test_spread_ratio_tight(self):
        c = OptionsContract(
            symbol="SPY250306C00550000", underlying="SPY",
            option_type="call", strike=Decimal("550"),
            expiration="2025-03-06",
            bid=Decimal("2.00"), ask=Decimal("2.10"),
        )
        ratio = c.spread_ratio
        assert ratio > 0.90

    def test_spread_ratio_wide(self):
        c = OptionsContract(
            symbol="SPY250306C00550000", underlying="SPY",
            option_type="call", strike=Decimal("550"),
            expiration="2025-03-06",
            bid=Decimal("1.00"), ask=Decimal("3.00"),
        )
        assert c.spread_ratio < 0.50

    def test_spread_ratio_zero_ask(self):
        c = OptionsContract(
            symbol="SPY250306C00550000", underlying="SPY",
            option_type="call", strike=Decimal("550"),
            expiration="2025-03-06",
            bid=Decimal("0"), ask=Decimal("0"),
        )
        assert c.spread_ratio == 0.0


class TestTradeDirection:
    def test_values(self):
        assert TradeDirection.BUY_CALL == "BUY_CALL"
        assert TradeDirection.BUY_PUT == "BUY_PUT"
        assert TradeDirection.HOLD == "HOLD"


class TestTradeSignal:
    def test_should_trade_true(self):
        signal = TradeSignal(
            direction=TradeDirection.BUY_CALL,
            confidence=70,
            underlying="SPY",
            contract=None,
            target_price=Decimal("2.10"),
            reason="Test",
        )
        assert signal.should_trade is True

    def test_should_trade_false_for_hold(self):
        signal = TradeSignal(
            direction=TradeDirection.HOLD,
            confidence=0,
            underlying="SPY",
            contract=None,
            target_price=Decimal("0"),
            reason="No signal",
        )
        assert signal.should_trade is False

    def test_score_breakdown_default_empty(self):
        signal = TradeSignal(
            direction=TradeDirection.HOLD,
            confidence=0,
            underlying="SPY",
            contract=None,
            target_price=Decimal("0"),
            reason="No signal",
        )
        assert signal.score_breakdown == {}
