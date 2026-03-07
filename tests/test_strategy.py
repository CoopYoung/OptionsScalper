"""Tests for ZeroDTEStrategy: gate checks, factor scoring, ensemble evaluation."""

from datetime import datetime
from decimal import Decimal
from unittest.mock import MagicMock, PropertyMock
from zoneinfo import ZoneInfo

import pytest

from src.data.alpaca_stream import TickMomentum
from src.strategy.base import TradeDirection
from src.strategy.signals import (
    BollingerResult,
    MACDResult,
    RSIResult,
    SignalBundle,
    VolumeDeltaResult,
)
from src.strategy.zero_dte import ZeroDTEStrategy

from tests.conftest import make_contract


@pytest.fixture
def mock_deps(settings):
    """Create ZeroDTEStrategy with mocked quant dependencies."""
    chain_mgr = MagicMock()
    vix = MagicMock()
    gex = MagicMock()
    flow = MagicMock()
    sentiment = MagicMock()
    macro = MagicMock()
    internals = MagicMock()

    # Default: all gates pass
    macro.is_blackout.return_value = False
    vix.latest = None  # No VIX block

    # Default scores: all return neutral 0.0
    gex.get_score.return_value = 0.0
    flow.get_score.return_value = 0.0
    vix.get_score.return_value = 0.0
    internals.get_score.return_value = 0.0
    sentiment.get_score.return_value = 0.0

    # Chain manager returns a valid strike
    contract = make_contract()
    strike_candidate = MagicMock()
    strike_candidate.contract = contract
    chain_mgr.select_strike.return_value = strike_candidate

    strategy = ZeroDTEStrategy(
        settings=settings,
        chain_mgr=chain_mgr,
        vix=vix,
        gex=gex,
        flow=flow,
        sentiment=sentiment,
        macro=macro,
        internals=internals,
    )

    return {
        "strategy": strategy,
        "chain_mgr": chain_mgr,
        "vix": vix,
        "gex": gex,
        "flow": flow,
        "sentiment": sentiment,
        "macro": macro,
        "internals": internals,
    }


def _market_time(hour: int, minute: int = 0) -> datetime:
    """Create a datetime during market hours in ET."""
    return datetime(2025, 3, 6, hour, minute, tzinfo=ZoneInfo("America/New_York"))


# ── Gate Checks ────────────────────────────────────────────────────────


class TestGateChecks:
    def test_macro_blackout_blocks(self, mock_deps):
        strategy = mock_deps["strategy"]
        mock_deps["macro"].is_blackout.return_value = True
        mock_deps["macro"].latest = MagicMock(blackout_reason="FOMC")

        signal = strategy.evaluate("SPY", Decimal("550"), now=_market_time(11, 0))
        assert signal.direction == TradeDirection.HOLD
        assert "blackout" in signal.reason.lower()

    def test_before_entry_window_blocks(self, mock_deps):
        strategy = mock_deps["strategy"]
        # 9:30 AM is before 9:45 entry start
        signal = strategy.evaluate("SPY", Decimal("550"), now=_market_time(9, 30))
        assert signal.direction == TradeDirection.HOLD
        assert "Before entry window" in signal.reason

    def test_after_entry_cutoff_blocks(self, mock_deps):
        strategy = mock_deps["strategy"]
        # 3:00 PM is after 2:30 PM cutoff
        signal = strategy.evaluate("SPY", Decimal("550"), now=_market_time(15, 0))
        assert signal.direction == TradeDirection.HOLD
        assert "After entry cutoff" in signal.reason

    def test_within_entry_window_passes(self, mock_deps):
        strategy = mock_deps["strategy"]
        # 11:00 AM is within window
        signal = strategy.evaluate("SPY", Decimal("550"), now=_market_time(11, 0))
        # Gate passes, but signal may be HOLD if confidence is below threshold
        assert "Before entry window" not in signal.reason
        assert "After entry cutoff" not in signal.reason

    def test_vix_crisis_blocks(self, mock_deps):
        strategy = mock_deps["strategy"]
        vix_signals = MagicMock()
        vix_signals.should_trade = False
        vix_signals.block_reason = "VIX crisis (40.5)"
        mock_deps["vix"].latest = vix_signals

        signal = strategy.evaluate("SPY", Decimal("550"), now=_market_time(11, 0))
        assert signal.direction == TradeDirection.HOLD
        assert "VIX" in signal.reason


# ── Technical Scoring ──────────────────────────────────────────────────


class TestScoreTechnicals:
    def test_none_signals_returns_zero(self, mock_deps):
        strategy = mock_deps["strategy"]
        assert strategy._score_technicals(None) == 0.0

    def test_oversold_rsi_bullish(self, mock_deps):
        strategy = mock_deps["strategy"]
        bundle = SignalBundle(
            rsi=RSIResult(value=25, is_overbought=False, is_oversold=True),
        )
        score = strategy._score_technicals(bundle)
        assert score > 0  # Bullish

    def test_overbought_rsi_bearish(self, mock_deps):
        strategy = mock_deps["strategy"]
        bundle = SignalBundle(
            rsi=RSIResult(value=75, is_overbought=True, is_oversold=False),
        )
        score = strategy._score_technicals(bundle)
        assert score < 0  # Bearish

    def test_bullish_macd(self, mock_deps):
        strategy = mock_deps["strategy"]
        bundle = SignalBundle(
            macd=MACDResult(macd_line=1.0, signal_line=0.5, histogram=0.5, is_bullish=True),
        )
        score = strategy._score_technicals(bundle)
        assert score > 0

    def test_bearish_macd(self, mock_deps):
        strategy = mock_deps["strategy"]
        bundle = SignalBundle(
            macd=MACDResult(macd_line=-1.0, signal_line=-0.5, histogram=-0.5, is_bullish=False),
        )
        score = strategy._score_technicals(bundle)
        assert score < 0

    def test_bollinger_near_lower_band_bullish(self, mock_deps):
        strategy = mock_deps["strategy"]
        bundle = SignalBundle(
            bollinger=BollingerResult(
                upper=110, middle=100, lower=90, bandwidth=0.2,
                pct_b=0.05, is_squeeze=False,
            ),
        )
        score = strategy._score_technicals(bundle)
        assert score > 0

    def test_bollinger_squeeze_reduces_conviction(self, mock_deps):
        strategy = mock_deps["strategy"]
        no_squeeze = SignalBundle(
            rsi=RSIResult(value=25, is_overbought=False, is_oversold=True),
            bollinger=BollingerResult(
                upper=110, middle=100, lower=90, bandwidth=0.2,
                pct_b=0.05, is_squeeze=False,
            ),
        )
        with_squeeze = SignalBundle(
            rsi=RSIResult(value=25, is_overbought=False, is_oversold=True),
            bollinger=BollingerResult(
                upper=101, middle=100, lower=99, bandwidth=0.01,
                pct_b=0.05, is_squeeze=True,
            ),
        )
        score_no = strategy._score_technicals(no_squeeze)
        score_sq = strategy._score_technicals(with_squeeze)
        assert abs(score_sq) < abs(score_no)

    def test_buy_volume_delta_bullish(self, mock_deps):
        strategy = mock_deps["strategy"]
        bundle = SignalBundle(
            volume_delta=VolumeDeltaResult(delta=500, ratio=0.70),
        )
        score = strategy._score_technicals(bundle)
        assert score > 0

    def test_score_bounded(self, mock_deps):
        strategy = mock_deps["strategy"]
        # All bullish factors at once
        bundle = SignalBundle(
            rsi=RSIResult(value=20, is_overbought=False, is_oversold=True),
            macd=MACDResult(macd_line=2.0, signal_line=0.5, histogram=1.5, is_bullish=True),
            bollinger=BollingerResult(
                upper=110, middle=100, lower=90, bandwidth=0.2,
                pct_b=0.02, is_squeeze=False,
            ),
            volume_delta=VolumeDeltaResult(delta=1000, ratio=0.80),
        )
        score = strategy._score_technicals(bundle)
        assert -1.0 <= score <= 1.0


# ── Tick Momentum Scoring ──────────────────────────────────────────────


class TestScoreTickMomentum:
    def test_none_momentum_returns_zero(self, mock_deps):
        strategy = mock_deps["strategy"]
        assert strategy._score_tick_momentum(None) == 0.0

    def test_upward_momentum(self, mock_deps):
        strategy = mock_deps["strategy"]
        mom = TickMomentum()
        # Add rising prices
        import time
        base = time.time()
        for i in range(15):
            mom.add_tick(100 + i * 0.5, base + i)
        score = strategy._score_tick_momentum(mom)
        assert score > 0

    def test_downward_momentum(self, mock_deps):
        strategy = mock_deps["strategy"]
        mom = TickMomentum()
        import time
        base = time.time()
        for i in range(15):
            mom.add_tick(100 - i * 0.5, base + i)
        score = strategy._score_tick_momentum(mom)
        assert score < 0

    def test_flat_momentum_near_zero(self, mock_deps):
        strategy = mock_deps["strategy"]
        mom = TickMomentum()
        import time
        base = time.time()
        for i in range(15):
            mom.add_tick(100.0, base + i)
        score = strategy._score_tick_momentum(mom)
        assert score == pytest.approx(0.0, abs=0.1)

    def test_score_bounded(self, mock_deps):
        strategy = mock_deps["strategy"]
        mom = TickMomentum()
        import time
        base = time.time()
        for i in range(20):
            mom.add_tick(100 + i * 5, base + i)
        score = strategy._score_tick_momentum(mom)
        assert -1.0 <= score <= 1.0


# ── Full Ensemble Evaluation ───────────────────────────────────────────


class TestEnsembleEvaluation:
    def test_strong_bullish_signals_produce_buy_call(self, mock_deps):
        strategy = mock_deps["strategy"]
        # Set all quant scores to bullish (+0.8)
        mock_deps["gex"].get_score.return_value = 0.8
        mock_deps["flow"].get_score.return_value = 0.8
        mock_deps["vix"].get_score.return_value = 0.8
        mock_deps["internals"].get_score.return_value = 0.8
        mock_deps["sentiment"].get_score.return_value = 0.8

        bullish_bundle = SignalBundle(
            rsi=RSIResult(value=25, is_overbought=False, is_oversold=True),
            macd=MACDResult(macd_line=2.0, signal_line=0.5, histogram=1.5, is_bullish=True),
            bollinger=BollingerResult(
                upper=110, middle=100, lower=90, bandwidth=0.2,
                pct_b=0.05, is_squeeze=False,
            ),
            volume_delta=VolumeDeltaResult(delta=500, ratio=0.70),
        )

        mom = TickMomentum()
        import time
        base = time.time()
        for i in range(15):
            mom.add_tick(100 + i * 0.5, base + i)

        signal = strategy.evaluate(
            "SPY", Decimal("550"), signals=bullish_bundle,
            momentum=mom, now=_market_time(11, 0),
        )
        assert signal.direction == TradeDirection.BUY_CALL
        assert signal.confidence >= 55

    def test_below_threshold_produces_hold(self, mock_deps):
        strategy = mock_deps["strategy"]
        # All neutral scores → confidence below threshold
        signal = strategy.evaluate(
            "SPY", Decimal("550"), now=_market_time(11, 0),
        )
        assert signal.direction == TradeDirection.HOLD
        assert "Below threshold" in signal.reason

    def test_no_viable_strikes_produces_hold(self, mock_deps):
        strategy = mock_deps["strategy"]
        mock_deps["chain_mgr"].select_strike.return_value = None
        # Push strong bullish to cross threshold
        mock_deps["gex"].get_score.return_value = 0.9
        mock_deps["flow"].get_score.return_value = 0.9
        mock_deps["vix"].get_score.return_value = 0.9
        mock_deps["internals"].get_score.return_value = 0.9
        mock_deps["sentiment"].get_score.return_value = 0.9

        bullish_bundle = SignalBundle(
            rsi=RSIResult(value=25, is_overbought=False, is_oversold=True),
            macd=MACDResult(macd_line=2.0, signal_line=0.5, histogram=1.5, is_bullish=True),
        )
        mom = TickMomentum()
        import time
        base = time.time()
        for i in range(15):
            mom.add_tick(100 + i * 0.5, base + i)

        signal = strategy.evaluate(
            "SPY", Decimal("550"), signals=bullish_bundle,
            momentum=mom, now=_market_time(11, 0),
        )
        assert signal.direction == TradeDirection.HOLD
        assert "No viable" in signal.reason

    def test_signal_has_score_breakdown(self, mock_deps):
        strategy = mock_deps["strategy"]
        mock_deps["gex"].get_score.return_value = 0.8
        mock_deps["flow"].get_score.return_value = 0.8
        mock_deps["vix"].get_score.return_value = 0.8
        mock_deps["internals"].get_score.return_value = 0.8
        mock_deps["sentiment"].get_score.return_value = 0.8

        bullish_bundle = SignalBundle(
            rsi=RSIResult(value=20, is_overbought=False, is_oversold=True),
            macd=MACDResult(macd_line=2.0, signal_line=0.5, histogram=1.5, is_bullish=True),
        )
        mom = TickMomentum()
        import time
        base = time.time()
        for i in range(15):
            mom.add_tick(100 + i * 0.5, base + i)

        signal = strategy.evaluate(
            "SPY", Decimal("550"), signals=bullish_bundle,
            momentum=mom, now=_market_time(11, 0),
        )
        if signal.should_trade:
            assert "technical" in signal.score_breakdown
            assert "gex" in signal.score_breakdown
            assert "flow" in signal.score_breakdown

    def test_strategy_name(self, mock_deps):
        assert mock_deps["strategy"].name == "ZeroDTE-Ensemble-v1"
