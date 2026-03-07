"""Tests for technical indicators: RSI, MACD, Bollinger Bands, Volume Delta."""

from decimal import Decimal

import pytest

from src.strategy.signals import (
    compute_all_signals,
    compute_bollinger_bands,
    compute_macd,
    compute_rsi,
    compute_volume_delta,
)


# ── RSI ────────────────────────────────────────────────────────────────


class TestComputeRSI:
    def test_insufficient_data(self):
        closes = [Decimal(str(i)) for i in range(10)]
        assert compute_rsi(closes, period=14) is None

    def test_steady_uptrend_is_overbought(self):
        # 20 consecutive higher closes → RSI should be very high
        closes = [Decimal(str(100 + i)) for i in range(20)]
        result = compute_rsi(closes, period=14)
        assert result is not None
        assert result.value > 70
        assert result.is_overbought is True
        assert result.is_oversold is False

    def test_steady_downtrend_is_oversold(self):
        closes = [Decimal(str(200 - i)) for i in range(20)]
        result = compute_rsi(closes, period=14)
        assert result is not None
        assert result.value < 30
        assert result.is_oversold is True
        assert result.is_overbought is False

    def test_flat_prices_mid_rsi(self):
        # Alternating +1/-1 → roughly RSI ~50
        closes = [Decimal(str(100 + (i % 2))) for i in range(20)]
        result = compute_rsi(closes, period=14)
        assert result is not None
        assert 30 < result.value < 70
        assert result.is_overbought is False
        assert result.is_oversold is False

    def test_all_gains_rsi_100(self):
        # Every bar is a gain, no losses → RSI = 100
        closes = [Decimal(str(100 + i * 5)) for i in range(20)]
        result = compute_rsi(closes, period=14)
        assert result is not None
        assert result.value == 100.0

    def test_custom_thresholds(self):
        closes = [Decimal(str(100 + i)) for i in range(20)]
        result = compute_rsi(closes, period=14, overbought=90, oversold=10)
        assert result is not None
        # With thresholds at 90/10, a mild uptrend shouldn't hit overbought
        assert result.value > 70  # Still high RSI
        # Whether it's overbought depends on threshold

    def test_exact_period_plus_one(self):
        # Minimum data: period + 1 = 15 closes
        closes = [Decimal(str(100 + i * 0.5)) for i in range(15)]
        result = compute_rsi(closes, period=14)
        assert result is not None


# ── MACD ───────────────────────────────────────────────────────────────


class TestComputeMACD:
    def test_insufficient_data(self):
        closes = [Decimal(str(i)) for i in range(30)]
        assert compute_macd(closes, fast=12, slow=26, signal_period=9) is None

    def test_uptrend_is_bullish(self):
        # Clear uptrend: MACD should go bullish
        closes = [Decimal(str(100 + i * 0.5)) for i in range(50)]
        result = compute_macd(closes)
        assert result is not None
        assert result.is_bullish is True
        assert result.macd_line > result.signal_line

    def test_downtrend_is_bearish(self):
        closes = [Decimal(str(200 - i * 0.5)) for i in range(50)]
        result = compute_macd(closes)
        assert result is not None
        assert result.is_bullish is False

    def test_histogram_sign_matches_bullish(self):
        closes = [Decimal(str(100 + i * 0.3)) for i in range(50)]
        result = compute_macd(closes)
        assert result is not None
        if result.is_bullish:
            assert result.histogram > 0
        else:
            assert result.histogram <= 0

    def test_minimum_data(self):
        # slow + signal = 26 + 9 = 35
        closes = [Decimal(str(100 + i * 0.1)) for i in range(35)]
        result = compute_macd(closes)
        assert result is not None


# ── Bollinger Bands ────────────────────────────────────────────────────


class TestComputeBollingerBands:
    def test_insufficient_data(self):
        closes = [Decimal(str(i)) for i in range(15)]
        assert compute_bollinger_bands(closes, period=20) is None

    def test_basic_structure(self):
        closes = [Decimal(str(100 + (i % 5) - 2)) for i in range(30)]
        result = compute_bollinger_bands(closes)
        assert result is not None
        assert result.upper > result.middle > result.lower
        assert result.bandwidth > 0

    def test_price_above_upper_band_pct_b_above_1(self):
        # Flat then spike: last price well above upper band
        closes = [Decimal("100")] * 20 + [Decimal("200")]
        result = compute_bollinger_bands(closes, period=20)
        assert result is not None
        assert result.pct_b > 1.0  # Price above upper band

    def test_price_below_lower_band_pct_b_below_0(self):
        closes = [Decimal("100")] * 20 + [Decimal("50")]
        result = compute_bollinger_bands(closes, period=20)
        assert result is not None
        assert result.pct_b < 0.0  # Price below lower band

    def test_squeeze_detection(self):
        # Very tight prices → small bandwidth → squeeze
        closes = [Decimal(str(100 + i * 0.001)) for i in range(25)]
        result = compute_bollinger_bands(closes, period=20, squeeze_threshold=0.02)
        assert result is not None
        assert result.is_squeeze is True

    def test_no_squeeze_with_volatile_prices(self):
        closes = [Decimal(str(100 + (i % 2) * 10)) for i in range(25)]
        result = compute_bollinger_bands(closes, period=20, squeeze_threshold=0.02)
        assert result is not None
        assert result.is_squeeze is False


# ── Volume Delta ───────────────────────────────────────────────────────


class TestComputeVolumeDelta:
    def test_empty_candles(self):
        assert compute_volume_delta([]) is None

    def test_bullish_candles(self):
        # Close near high = mostly buy volume
        candles = [
            {"open": 100, "high": 105, "low": 99, "close": 104, "volume": 1000}
            for _ in range(5)
        ]
        result = compute_volume_delta(candles)
        assert result is not None
        assert result.delta > 0
        assert result.ratio > 0.5

    def test_bearish_candles(self):
        # Close near low = mostly sell volume
        candles = [
            {"open": 104, "high": 105, "low": 99, "close": 100, "volume": 1000}
            for _ in range(5)
        ]
        result = compute_volume_delta(candles)
        assert result is not None
        assert result.delta < 0
        assert result.ratio < 0.5

    def test_doji_candles_neutral(self):
        # Open = close at midpoint → ratio ~0.5
        candles = [
            {"open": 100, "high": 102, "low": 98, "close": 100, "volume": 1000}
            for _ in range(5)
        ]
        result = compute_volume_delta(candles)
        assert result is not None
        assert 0.45 <= result.ratio <= 0.55

    def test_zero_range_candle(self):
        candles = [{"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1000}]
        result = compute_volume_delta(candles)
        assert result is not None
        assert result.ratio == pytest.approx(0.5)

    def test_zero_volume(self):
        candles = [{"open": 100, "high": 105, "low": 95, "close": 102, "volume": 0}]
        result = compute_volume_delta(candles)
        assert result is not None
        assert result.delta == 0.0
        assert result.ratio == 0.5

    def test_only_last_20_candles_used(self):
        # 30 candles, but only last 20 should be used
        candles = (
            [{"open": 100, "high": 105, "low": 99, "close": 104, "volume": 1000}] * 10
            + [{"open": 104, "high": 105, "low": 99, "close": 100, "volume": 1000}] * 20
        )
        result = compute_volume_delta(candles)
        assert result is not None
        # Last 20 are bearish
        assert result.delta < 0


# ── compute_all_signals ────────────────────────────────────────────────


class TestComputeAllSignals:
    def test_returns_bundle_with_all_components(self):
        closes = [Decimal(str(100 + i * 0.3)) for i in range(50)]
        candles = [
            {"open": 100, "high": 105, "low": 99, "close": 103, "volume": 1000}
            for _ in range(25)
        ]
        bundle = compute_all_signals(closes, candles)
        assert bundle.rsi is not None
        assert bundle.macd is not None
        assert bundle.bollinger is not None
        assert bundle.volume_delta is not None

    def test_insufficient_data_returns_nones(self):
        closes = [Decimal("100")] * 5
        bundle = compute_all_signals(closes, [])
        assert bundle.rsi is None
        assert bundle.macd is None
        assert bundle.bollinger is None
        assert bundle.volume_delta is None
