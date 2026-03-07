"""Tests for OptionsChainManager: strike selection, filtering, Greeks scoring."""

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data.options_chain import OptionsChainManager, StrikeCandidate
from src.strategy.base import OptionsContract

from tests.conftest import make_contract


@pytest.fixture
def chain_mgr(settings):
    client = MagicMock()
    return OptionsChainManager(settings, client)


def _build_chain(
    underlying: str = "SPY",
    n_calls: int = 5,
    n_puts: int = 5,
    base_strike: float = 545.0,
    strike_step: float = 1.0,
) -> list[OptionsContract]:
    """Build a synthetic options chain."""
    chain = []
    for i in range(n_calls):
        strike = Decimal(str(base_strike + i * strike_step))
        # Delta decreases as strike goes higher (OTM calls)
        delta = max(0.05, 0.50 - i * 0.10)
        chain.append(make_contract(
            symbol=f"SPY250306C{int(strike * 1000):08d}",
            underlying=underlying,
            option_type="call",
            strike=strike,
            bid=Decimal(str(max(0.30, 3.00 - i * 0.60))),
            ask=Decimal(str(max(0.35, 3.30 - i * 0.60))),
            delta=delta,
            gamma=0.02,
            theta=-0.05,
            vega=0.08,
            iv=0.25,
        ))
    for i in range(n_puts):
        strike = Decimal(str(base_strike - i * strike_step))
        delta = max(0.05, 0.50 - i * 0.10) * -1
        chain.append(make_contract(
            symbol=f"SPY250306P{int(strike * 1000):08d}",
            underlying=underlying,
            option_type="put",
            strike=strike,
            bid=Decimal(str(max(0.30, 3.00 - i * 0.60))),
            ask=Decimal(str(max(0.35, 3.30 - i * 0.60))),
            delta=delta,
            gamma=0.02,
            theta=-0.05,
            vega=0.08,
            iv=0.25,
        ))
    return chain


class TestSelectStrike:
    def test_selects_best_call(self, chain_mgr):
        chain = _build_chain()
        chain_mgr._chains["SPY"] = chain
        # Also put them in snapshots to simulate live data
        for c in chain:
            chain_mgr._snapshots[c.symbol] = c

        result = chain_mgr.select_strike("SPY", "call", Decimal("548"))
        assert result is not None
        assert isinstance(result, StrikeCandidate)
        assert result.contract.option_type == "call"
        assert result.total_score > 0

    def test_selects_best_put(self, chain_mgr):
        chain = _build_chain()
        chain_mgr._chains["SPY"] = chain
        for c in chain:
            chain_mgr._snapshots[c.symbol] = c

        result = chain_mgr.select_strike("SPY", "put", Decimal("548"))
        assert result is not None
        assert result.contract.option_type == "put"

    def test_no_chain_returns_none(self, chain_mgr):
        result = chain_mgr.select_strike("SPY", "call", Decimal("548"))
        assert result is None

    def test_filters_out_zero_bid(self, chain_mgr):
        chain = [make_contract(bid=Decimal("0"), ask=Decimal("1.00"))]
        chain_mgr._chains["SPY"] = chain
        chain_mgr._snapshots[chain[0].symbol] = chain[0]
        result = chain_mgr.select_strike("SPY", "call", Decimal("548"))
        assert result is None

    def test_filters_out_wide_spread(self, chain_mgr):
        # spread_ratio = 0.50/2.00 = 0.25 < 0.90
        chain = [make_contract(bid=Decimal("0.50"), ask=Decimal("2.00"))]
        chain_mgr._chains["SPY"] = chain
        chain_mgr._snapshots[chain[0].symbol] = chain[0]
        result = chain_mgr.select_strike("SPY", "call", Decimal("548"))
        assert result is None

    def test_filters_out_below_min_premium(self, chain_mgr):
        # mid = (0.10 + 0.12) / 2 = 0.11 < 0.50 min
        chain = [make_contract(bid=Decimal("0.10"), ask=Decimal("0.12"))]
        chain_mgr._chains["SPY"] = chain
        chain_mgr._snapshots[chain[0].symbol] = chain[0]
        result = chain_mgr.select_strike("SPY", "call", Decimal("548"))
        assert result is None

    def test_filters_out_above_max_premium(self, chain_mgr):
        # mid = (8.00 + 8.20) / 2 = 8.10 > 5.00 max
        chain = [make_contract(bid=Decimal("8.00"), ask=Decimal("8.20"))]
        chain_mgr._chains["SPY"] = chain
        chain_mgr._snapshots[chain[0].symbol] = chain[0]
        result = chain_mgr.select_strike("SPY", "call", Decimal("548"))
        assert result is None

    def test_prefers_near_target_delta(self, chain_mgr):
        # Two contracts: one near target delta (0.30), one far (0.10)
        near = make_contract(
            symbol="NEAR", delta=0.30,
            bid=Decimal("2.00"), ask=Decimal("2.10"),
        )
        far = make_contract(
            symbol="FAR", delta=0.10,
            bid=Decimal("2.00"), ask=Decimal("2.10"),
        )
        chain_mgr._chains["SPY"] = [near, far]
        chain_mgr._snapshots["NEAR"] = near
        chain_mgr._snapshots["FAR"] = far

        result = chain_mgr.select_strike("SPY", "call", Decimal("548"))
        assert result is not None
        assert result.contract.symbol == "NEAR"


class TestScoreGreeks:
    def test_high_gamma_scores_better(self, chain_mgr):
        low_gamma = make_contract(gamma=0.005, iv=0.25, theta=-0.03, vega=0.08)
        high_gamma = make_contract(gamma=0.04, iv=0.25, theta=-0.03, vega=0.08)
        score_low = chain_mgr._score_greeks(low_gamma, "call")
        score_high = chain_mgr._score_greeks(high_gamma, "call")
        assert score_high > score_low

    def test_extreme_theta_penalty(self, chain_mgr):
        mild_theta = make_contract(theta=-0.03, iv=0.25, gamma=0.02, vega=0.08)
        harsh_theta = make_contract(theta=-0.15, iv=0.25, gamma=0.02, vega=0.08)
        score_mild = chain_mgr._score_greeks(mild_theta, "call")
        score_harsh = chain_mgr._score_greeks(harsh_theta, "call")
        assert score_mild > score_harsh

    def test_moderate_iv_bonus(self, chain_mgr):
        good_iv = make_contract(iv=0.25, gamma=0.02, theta=-0.05, vega=0.08)
        high_iv = make_contract(iv=0.70, gamma=0.02, theta=-0.05, vega=0.08)
        score_good = chain_mgr._score_greeks(good_iv, "call")
        score_high = chain_mgr._score_greeks(high_iv, "call")
        assert score_good > score_high

    def test_score_bounded_0_to_1(self, chain_mgr):
        contract = make_contract()
        score = chain_mgr._score_greeks(contract, "call")
        assert 0.0 <= score <= 1.0


class TestChainHelpers:
    def test_get_chain_empty(self, chain_mgr):
        assert chain_mgr.get_chain("SPY") == []

    def test_get_chain_populated(self, chain_mgr):
        chain = _build_chain()
        chain_mgr._chains["SPY"] = chain
        assert len(chain_mgr.get_chain("SPY")) == 10

    def test_get_snapshot(self, chain_mgr):
        contract = make_contract()
        chain_mgr._snapshots[contract.symbol] = contract
        assert chain_mgr.get_snapshot(contract.symbol) is contract
        assert chain_mgr.get_snapshot("NONEXIST") is None

    def test_get_chain_summary(self, chain_mgr):
        chain = _build_chain(n_calls=3, n_puts=2)
        chain_mgr._chains["SPY"] = chain
        summary = chain_mgr.get_chain_summary("SPY")
        assert summary["total_contracts"] == 5
        assert summary["calls"] == 3
        assert summary["puts"] == 2
