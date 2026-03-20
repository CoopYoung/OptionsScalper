"""Tests for self-computed GEX: formula correctness, regime detection, scoring.

Validates:
    - Per-strike GEX formula: gamma × OI × spot² × 0.01 × 100
    - Call/put wall identification by gamma×OI (not raw OI)
    - Flip point linear interpolation
    - Data filtering (noise removal)
    - Intensity scoring (z-score based, not flat)
    - Proximity scoring (near support/resistance)
"""

from decimal import Decimal

import pytest

from src.quant.gex import GEXAnalyzer, GEXLevel, GEXRegime
from tests.conftest import make_contract


def _make_chain_contract(
    strike: float,
    option_type: str = "call",
    gamma: float = 0.01,
    open_interest: int = 1000,
    delta: float = 0.50,
    iv: float = 0.25,
    bid: float = 1.00,
    volume: int = 100,
) -> object:
    """Create an OptionsContract with specific GEX-relevant fields."""
    return make_contract(
        symbol=f"SPY260320{'C' if option_type == 'call' else 'P'}{int(strike*1000):08d}",
        underlying="SPY",
        option_type=option_type,
        strike=Decimal(str(strike)),
        bid=Decimal(str(bid)),
        ask=Decimal(str(bid + 0.20)),
        delta=delta if option_type == "call" else -delta,
        gamma=gamma,
        iv=iv,
        volume=volume,
        open_interest=open_interest,
    )


def _make_simple_chain(spot: float, strikes: list[float],
                       call_oi: list[int], put_oi: list[int],
                       call_gamma: list[float] = None,
                       put_gamma: list[float] = None) -> list:
    """Build a chain with matching call and put at each strike."""
    if call_gamma is None:
        call_gamma = [0.01] * len(strikes)
    if put_gamma is None:
        put_gamma = [0.01] * len(strikes)

    chain = []
    for i, strike in enumerate(strikes):
        chain.append(_make_chain_contract(
            strike=strike, option_type="call",
            gamma=call_gamma[i], open_interest=call_oi[i],
        ))
        chain.append(_make_chain_contract(
            strike=strike, option_type="put",
            gamma=put_gamma[i], open_interest=put_oi[i],
        ))
    return chain


@pytest.fixture
def gex(settings):
    return GEXAnalyzer(settings)


class TestGEXFormula:
    def test_gex_formula_basic(self, gex):
        """Verify per-strike GEX matches hand-computed values."""
        spot = 550.0
        # Single call: gamma=0.02, OI=5000
        # Expected: 0.02 * 5000 * 550^2 * 0.01 * 100 = 0.02 * 5000 * 302500 * 0.01 * 100
        #         = 0.02 * 5000 * 3025 = 302,500
        chain = [_make_chain_contract(
            strike=550, option_type="call", gamma=0.02, open_interest=5000,
        )]
        signals = gex._compute_from_chain("SPY", chain, spot)
        expected_gex = 0.02 * 5000 * (550 ** 2) * 0.01 * 100
        assert abs(signals.total_gex - expected_gex) < 1.0

    def test_puts_contribute_negative_gex(self, gex):
        """Put GEX should be negative (dealers short puts → sell on dips)."""
        spot = 550.0
        chain = [_make_chain_contract(
            strike=545, option_type="put", gamma=0.015, open_interest=3000,
        )]
        signals = gex._compute_from_chain("SPY", chain, spot)
        assert signals.total_gex < 0

    def test_net_gex_is_call_minus_put(self, gex):
        """Net GEX at a strike = call_gex - put_gex."""
        spot = 550.0
        chain = [
            _make_chain_contract(strike=550, option_type="call", gamma=0.02, open_interest=5000),
            _make_chain_contract(strike=550, option_type="put", gamma=0.02, open_interest=5000),
        ]
        signals = gex._compute_from_chain("SPY", chain, spot)
        # Equal gamma and OI → call and put GEX cancel out → net ≈ 0
        # Actually: call contributes +gex, put contributes -gex → they cancel
        assert abs(signals.total_gex) < 1.0


class TestGEXRegime:
    def test_positive_regime_call_dominant(self, gex):
        """More call gamma×OI → positive GEX → mean-reverting."""
        spot = 550.0
        chain = _make_simple_chain(
            spot=spot,
            strikes=[545, 550, 555],
            call_oi=[5000, 10000, 3000],
            put_oi=[1000, 1000, 1000],
        )
        signals = gex._compute_from_chain("SPY", chain, spot)
        assert signals.regime == GEXRegime.POSITIVE
        assert signals.total_gex > 0

    def test_negative_regime_put_dominant(self, gex):
        """More put gamma×OI → negative GEX → trending."""
        spot = 550.0
        chain = _make_simple_chain(
            spot=spot,
            strikes=[545, 550, 555],
            call_oi=[1000, 1000, 1000],
            put_oi=[5000, 10000, 3000],
        )
        signals = gex._compute_from_chain("SPY", chain, spot)
        assert signals.regime == GEXRegime.NEGATIVE
        assert signals.total_gex < 0


class TestWalls:
    def test_call_wall_by_gamma_oi(self, gex):
        """Call wall should be strike with max(call_gamma × call_OI), not raw OI."""
        spot = 550.0
        chain = _make_simple_chain(
            spot=spot,
            strikes=[545, 550, 555],
            call_oi=[5000, 3000, 1000],      # 545 has most raw OI
            put_oi=[100, 100, 100],
            call_gamma=[0.005, 0.020, 0.010],  # 550 has gamma×OI = 0.020*3000 = 60 > 0.005*5000 = 25
        )
        signals = gex._compute_from_chain("SPY", chain, spot)
        assert signals.call_wall == 550.0

    def test_put_wall_by_gamma_oi(self, gex):
        """Put wall should be strike with max(put_gamma × put_OI)."""
        spot = 550.0
        chain = _make_simple_chain(
            spot=spot,
            strikes=[545, 550, 555],
            call_oi=[100, 100, 100],
            put_oi=[2000, 1000, 8000],       # 555 has most raw OI
            put_gamma=[0.015, 0.005, 0.003],  # 545: 0.015*2000=30 > 555: 0.003*8000=24
        )
        signals = gex._compute_from_chain("SPY", chain, spot)
        assert signals.put_wall == 545.0


class TestFlipPoint:
    def test_flip_point_interpolation(self, gex):
        """Flip point should be linearly interpolated, not midpoint."""
        # Create levels with a sign change
        levels = [
            GEXLevel(strike=548.0, gex_value=300.0, is_call_wall=False, is_put_wall=False),
            GEXLevel(strike=552.0, gex_value=-100.0, is_call_wall=False, is_put_wall=False),
        ]
        flip = gex._find_flip_point(levels, 550.0)
        # g1=300, g2=-100, fraction = 300/(300+100) = 0.75
        # flip = 548 + 0.75 * (552 - 548) = 548 + 3.0 = 551.0
        assert abs(flip - 551.0) < 0.01

    def test_flip_point_no_sign_change(self, gex):
        """If no sign change, flip defaults to current price."""
        levels = [
            GEXLevel(strike=548.0, gex_value=100.0, is_call_wall=False, is_put_wall=False),
            GEXLevel(strike=552.0, gex_value=200.0, is_call_wall=False, is_put_wall=False),
        ]
        flip = gex._find_flip_point(levels, 550.0)
        assert flip == 550.0


class TestDataFiltering:
    def test_filters_low_oi(self, gex):
        """Contracts with OI <= 10 should be excluded."""
        chain = [_make_chain_contract(strike=550, open_interest=5)]
        filtered = gex._filter_chain(chain)
        assert len(filtered) == 0

    def test_filters_low_bid(self, gex):
        """Contracts with bid <= $0.05 should be excluded."""
        chain = [_make_chain_contract(strike=550, bid=0.03)]
        filtered = gex._filter_chain(chain)
        assert len(filtered) == 0

    def test_filters_extreme_iv(self, gex):
        """Contracts with IV > 200% or < 1% should be excluded."""
        chain = [
            _make_chain_contract(strike=550, iv=2.5),    # Too high
            _make_chain_contract(strike=551, iv=0.005),   # Too low
            _make_chain_contract(strike=552, iv=0.30),    # OK
        ]
        filtered = gex._filter_chain(chain)
        assert len(filtered) == 1
        assert float(filtered[0].strike) == 552.0

    def test_filters_deep_itm(self, gex):
        """Contracts with |delta| > 0.95 should be excluded."""
        chain = [_make_chain_contract(strike=550, delta=0.98)]
        filtered = gex._filter_chain(chain)
        assert len(filtered) == 0

    def test_empty_chain_returns_fallback(self, gex):
        """Empty chain should return NEUTRAL regime."""
        signals = gex._compute_from_chain("SPY", [], 550.0)
        assert signals.regime == GEXRegime.NEUTRAL


class TestGEXScoring:
    @pytest.mark.asyncio
    async def test_score_not_flat(self, gex):
        """Score should vary based on intensity, not just flat +0.2/-0.1."""
        spot = 550.0
        chain = _make_simple_chain(
            spot=spot,
            strikes=[545, 550, 555],
            call_oi=[10000, 20000, 5000],
            put_oi=[1000, 1000, 1000],
        )
        # Feed multiple updates to build history for z-score
        for _ in range(5):
            await gex.update("SPY", chain, spot)

        score_call = gex.get_score("SPY", "call")
        score_put = gex.get_score("SPY", "put")

        # Scores should differ between call and put
        assert score_call != score_put
        # At least one should be non-zero
        assert score_call != 0.0 or score_put != 0.0

    @pytest.mark.asyncio
    async def test_score_proximity_near_support(self, gex):
        """Price near put wall (support) → positive call score."""
        spot = 545.1  # Very close to put wall at 545
        chain = _make_simple_chain(
            spot=spot,
            strikes=[545, 550, 555],
            call_oi=[100, 100, 100],
            put_oi=[10000, 1000, 100],
        )
        # Build history first
        for _ in range(5):
            await gex.update("SPY", chain, spot)

        score = gex.get_score("SPY", "call")
        # Near put wall + call direction → should be positive (support bounce)
        assert score > 0

    @pytest.mark.asyncio
    async def test_score_proximity_near_resistance(self, gex):
        """Price near call wall (resistance) → negative call score."""
        spot = 554.9  # Very close to call wall at 555
        chain = _make_simple_chain(
            spot=spot,
            strikes=[545, 550, 555],
            call_oi=[100, 1000, 10000],
            put_oi=[100, 100, 100],
        )
        for _ in range(5):
            await gex.update("SPY", chain, spot)

        score = gex.get_score("SPY", "call")
        # Near call wall + call direction → should be negative (resistance)
        assert score < 0
