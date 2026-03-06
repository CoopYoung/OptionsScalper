"""Technical indicators: RSI, MACD, Bollinger Bands, Volume Delta.

Forked from poly-trader — these are 100% asset-agnostic.
Kalshi-specific OrderFlow computation removed (handled by quant layer instead).
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RSIResult:
    value: float
    is_overbought: bool
    is_oversold: bool


@dataclass
class MACDResult:
    macd_line: float
    signal_line: float
    histogram: float
    is_bullish: bool


@dataclass
class BollingerResult:
    upper: float
    middle: float
    lower: float
    bandwidth: float
    pct_b: float
    is_squeeze: bool


@dataclass
class VolumeDeltaResult:
    delta: float
    ratio: float


@dataclass
class SignalBundle:
    """All computed technical signals for a given point in time."""
    rsi: Optional[RSIResult] = None
    macd: Optional[MACDResult] = None
    bollinger: Optional[BollingerResult] = None
    volume_delta: Optional[VolumeDeltaResult] = None


def compute_rsi(
    closes: list[Decimal],
    period: int = 14,
    overbought: int = 70,
    oversold: int = 30,
) -> Optional[RSIResult]:
    if len(closes) < period + 1:
        return None

    prices = np.array([float(c) for c in closes])
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    alpha = 1.0 / period
    avg_gain = _ema(gains, alpha, period)
    avg_loss = _ema(losses, alpha, period)

    if avg_loss == 0:
        rsi_val = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_val = 100.0 - (100.0 / (1.0 + rs))

    return RSIResult(
        value=rsi_val,
        is_overbought=rsi_val >= overbought,
        is_oversold=rsi_val <= oversold,
    )


def _ema(data: np.ndarray, alpha: float, period: int) -> float:
    if len(data) < period:
        return float(np.mean(data))
    sma = float(np.mean(data[:period]))
    result = sma
    for val in data[period:]:
        result = alpha * float(val) + (1 - alpha) * result
    return result


def compute_macd(
    closes: list[Decimal],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Optional[MACDResult]:
    if len(closes) < slow + signal_period:
        return None

    prices = pd.Series([float(c) for c in closes])
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return MACDResult(
        macd_line=float(macd_line.iloc[-1]),
        signal_line=float(signal_line.iloc[-1]),
        histogram=float(histogram.iloc[-1]),
        is_bullish=float(macd_line.iloc[-1]) > float(signal_line.iloc[-1]),
    )


def compute_bollinger_bands(
    closes: list[Decimal],
    period: int = 20,
    num_std: float = 2.0,
    squeeze_threshold: float = 0.02,
) -> Optional[BollingerResult]:
    if len(closes) < period:
        return None

    prices = np.array([float(c) for c in closes])
    recent = prices[-period:]

    middle = float(np.mean(recent))
    std = float(np.std(recent, ddof=1))
    upper = middle + num_std * std
    lower = middle - num_std * std

    bandwidth = (upper - lower) / middle if middle != 0 else 0.0
    current = float(prices[-1])
    band_range = upper - lower
    pct_b = (current - lower) / band_range if band_range != 0 else 0.5

    return BollingerResult(
        upper=upper,
        middle=middle,
        lower=lower,
        bandwidth=bandwidth,
        pct_b=pct_b,
        is_squeeze=bandwidth < squeeze_threshold,
    )


def compute_volume_delta(candles: list[dict]) -> Optional[VolumeDeltaResult]:
    if not candles:
        return None

    total_buy = 0.0
    total_sell = 0.0

    for c in candles[-20:]:
        o = float(c.get("open", 0))
        h = float(c.get("high", 0))
        low = float(c.get("low", 0))
        cl = float(c.get("close", 0))
        v = float(c.get("volume", 0))

        candle_range = h - low
        buy_ratio = (cl - low) / candle_range if candle_range != 0 else 0.5
        total_buy += v * buy_ratio
        total_sell += v * (1 - buy_ratio)

    total = total_buy + total_sell
    if total == 0:
        return VolumeDeltaResult(delta=0.0, ratio=0.5)

    return VolumeDeltaResult(
        delta=total_buy - total_sell,
        ratio=total_buy / total,
    )


def compute_all_signals(
    closes: list[Decimal],
    candle_dicts: list[dict],
    rsi_period: int = 14,
    rsi_overbought: int = 70,
    rsi_oversold: int = 30,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    bb_period: int = 20,
    bb_std: float = 2.0,
) -> SignalBundle:
    return SignalBundle(
        rsi=compute_rsi(closes, rsi_period, rsi_overbought, rsi_oversold),
        macd=compute_macd(closes, macd_fast, macd_slow, macd_signal),
        bollinger=compute_bollinger_bands(closes, bb_period, bb_std),
        volume_delta=compute_volume_delta(candle_dicts),
    )
