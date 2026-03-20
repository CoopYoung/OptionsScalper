"""Tests for TickMomentum from alpaca_stream."""

import time

import pytest

from src.data.alpaca_stream import TickMomentum


class TestTickMomentumDirection:
    def test_insufficient_data(self):
        mom = TickMomentum()
        assert mom.direction == 0.0

    def test_two_ticks_insufficient(self):
        mom = TickMomentum()
        mom.add_tick(100.0, time.time())
        mom.add_tick(101.0, time.time())
        assert mom.direction == 0.0  # Need >= 3

    def test_upward_direction(self):
        mom = TickMomentum()
        base = time.time()
        for i in range(10):
            mom.add_tick(100 + i, base + i)
        assert mom.direction > 0

    def test_downward_direction(self):
        mom = TickMomentum()
        base = time.time()
        for i in range(10):
            mom.add_tick(100 - i, base + i)
        assert mom.direction < 0

    def test_flat_direction_zero(self):
        mom = TickMomentum()
        base = time.time()
        for i in range(10):
            mom.add_tick(100.0, base + i)
        assert mom.direction == 0.0

    def test_direction_bounded(self):
        mom = TickMomentum()
        base = time.time()
        for i in range(20):
            mom.add_tick(100 + i * 10, base + i)
        assert -1.0 <= mom.direction <= 1.0


class TestTickMomentumSpeed:
    def test_insufficient_data(self):
        mom = TickMomentum()
        assert mom.speed == 0.0

    def test_positive_speed(self):
        mom = TickMomentum()
        now = time.time()
        mom.add_tick(100.0, now)
        mom.add_tick(102.0, now + 1.0)
        assert mom.speed == pytest.approx(2.0, abs=0.01)

    def test_zero_time_delta(self):
        mom = TickMomentum()
        now = time.time()
        mom.add_tick(100.0, now)
        mom.add_tick(102.0, now)  # Same timestamp
        assert mom.speed == 0.0


class TestTickMomentumROC:
    def test_insufficient_data(self):
        mom = TickMomentum()
        for i in range(4):
            mom.add_tick(100.0, time.time() + i)
        assert mom.roc_pct == 0.0

    def test_positive_roc(self):
        mom = TickMomentum()
        base = time.time()
        prices = [100, 101, 102, 103, 105]
        for i, p in enumerate(prices):
            mom.add_tick(float(p), base + i)
        roc = mom.roc_pct
        assert roc > 0
        assert roc == pytest.approx(5.0, abs=0.01)

    def test_negative_roc(self):
        mom = TickMomentum()
        base = time.time()
        prices = [100, 99, 98, 97, 95]
        for i, p in enumerate(prices):
            mom.add_tick(float(p), base + i)
        assert mom.roc_pct < 0

    def test_zero_oldest_price(self):
        mom = TickMomentum()
        base = time.time()
        prices = [0, 1, 2, 3, 4]
        for i, p in enumerate(prices):
            mom.add_tick(float(p), base + i)
        assert mom.roc_pct == 0.0  # Division by zero protection


class TestTickMomentumLatestPrice:
    def test_empty(self):
        mom = TickMomentum()
        assert mom.latest_price is None

    def test_returns_last_price(self):
        mom = TickMomentum()
        mom.add_tick(100.0, time.time())
        mom.add_tick(105.0, time.time())
        assert mom.latest_price == 105.0


class TestTickMomentumMaxLen:
    def test_deque_maxlen(self):
        mom = TickMomentum()
        base = time.time()
        for i in range(100):
            mom.add_tick(float(i), base + i)
        assert len(mom.prices) == 60  # maxlen=60
