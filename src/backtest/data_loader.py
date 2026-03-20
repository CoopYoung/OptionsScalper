"""Historical data loader for backtesting.

Fetches intraday bars from yfinance and constructs simulated options data
using Black-Scholes pricing for realistic 0DTE backtesting.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HistoricalBar:
    """A single intraday bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class SimulatedOption:
    """A simulated 0DTE option contract at a point in time."""
    symbol: str
    underlying: str
    option_type: str  # "call" or "put"
    strike: float
    expiration: str
    bid: float
    ask: float
    mid: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float


@dataclass
class BacktestDay:
    """All data needed for one backtest day."""
    date: date
    underlying: str
    bars: list[HistoricalBar]
    vix_close: float
    prev_close: float


class HistoricalDataLoader:
    """Loads and prepares historical data for backtesting."""

    def __init__(self, underlyings: list[str] = None) -> None:
        self._underlyings = underlyings or ["SPY", "QQQ", "IWM"]

    def load_days(
        self,
        underlying: str,
        start_date: date,
        end_date: date,
        interval: str = "1m",
    ) -> list[BacktestDay]:
        """Load intraday bars for a date range.

        Args:
            underlying: Ticker symbol.
            start_date: First date (inclusive).
            end_date: Last date (inclusive).
            interval: Bar interval ("1m", "2m", "5m").

        Returns:
            List of BacktestDay, one per trading day.
        """
        import yfinance as yf

        logger.info("Loading %s data: %s to %s (%s)", underlying, start_date, end_date, interval)

        # yfinance limits 1m data to 30 days at a time
        all_bars: list[pd.DataFrame] = []
        current = start_date
        while current <= end_date:
            chunk_end = min(current + timedelta(days=6), end_date + timedelta(days=1))
            try:
                ticker = yf.Ticker(underlying)
                df = ticker.history(
                    start=current.isoformat(),
                    end=chunk_end.isoformat(),
                    interval=interval,
                )
                if not df.empty:
                    all_bars.append(df)
            except Exception:
                logger.exception("Failed to load %s chunk %s-%s", underlying, current, chunk_end)
            current = chunk_end

        if not all_bars:
            logger.warning("No data loaded for %s", underlying)
            return []

        full_df = pd.concat(all_bars)
        full_df = full_df[~full_df.index.duplicated(keep="first")]
        full_df = full_df.sort_index()

        # Load VIX for the period
        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_df = vix_ticker.history(start=start_date.isoformat(),
                                         end=(end_date + timedelta(days=1)).isoformat(),
                                         interval="1d")
            vix_map = {d.date(): float(row["Close"]) for d, row in vix_df.iterrows()}
        except Exception:
            vix_map = {}

        # Load daily bars for prev_close
        try:
            daily = yf.Ticker(underlying).history(
                start=(start_date - timedelta(days=5)).isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                interval="1d",
            )
            daily_closes = {d.date(): float(row["Close"]) for d, row in daily.iterrows()}
        except Exception:
            daily_closes = {}

        # Group by trading day
        days: list[BacktestDay] = []
        grouped = full_df.groupby(full_df.index.date)

        for day_date, group in grouped:
            bars = []
            for ts, row in group.iterrows():
                bars.append(HistoricalBar(
                    timestamp=ts.to_pydatetime().replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                ))

            if not bars:
                continue

            # Find prev close
            sorted_dates = sorted(daily_closes.keys())
            prev_close = bars[0].open
            for d in sorted_dates:
                if d < day_date:
                    prev_close = daily_closes[d]

            days.append(BacktestDay(
                date=day_date,
                underlying=underlying,
                bars=bars,
                vix_close=vix_map.get(day_date, 20.0),
                prev_close=prev_close,
            ))

        logger.info("Loaded %d trading days for %s (%d total bars)",
                     len(days), underlying, sum(len(d.bars) for d in days))
        return days


class OptionPricer:
    """Black-Scholes option pricer for simulated 0DTE contracts."""

    @staticmethod
    def price_option(
        spot: float,
        strike: float,
        tte: float,  # Time to expiry in years
        iv: float,    # Implied volatility (annualized, e.g. 0.20 = 20%)
        r: float = 0.05,
        option_type: str = "call",
    ) -> dict:
        """Price an option using Black-Scholes and return price + Greeks."""
        from scipy.stats import norm

        if tte <= 0 or iv <= 0:
            intrinsic = max(0, spot - strike) if option_type == "call" else max(0, strike - spot)
            return {
                "price": intrinsic,
                "delta": (1.0 if spot > strike else 0.0) if option_type == "call" else (-1.0 if spot < strike else 0.0),
                "gamma": 0.0, "theta": 0.0, "vega": 0.0, "iv": iv,
            }

        sqrt_tte = np.sqrt(tte)
        d1 = (np.log(spot / strike) + (r + iv**2 / 2) * tte) / (iv * sqrt_tte)
        d2 = d1 - iv * sqrt_tte

        if option_type == "call":
            price = spot * norm.cdf(d1) - strike * np.exp(-r * tte) * norm.cdf(d2)
            delta = norm.cdf(d1)
        else:
            price = strike * np.exp(-r * tte) * norm.cdf(-d2) - spot * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1

        gamma = norm.pdf(d1) / (spot * iv * sqrt_tte)
        theta = (-(spot * norm.pdf(d1) * iv) / (2 * sqrt_tte)
                 - r * strike * np.exp(-r * tte) * (norm.cdf(d2) if option_type == "call" else norm.cdf(-d2)))
        theta = theta / 365  # Per-day, per-share
        # For 0DTE, theta can be extreme; cap to realistic broker-reported range
        theta = max(-0.50, min(0.0, theta))
        vega = spot * norm.pdf(d1) * sqrt_tte / 100  # Per 1% IV change

        return {
            "price": max(0.01, price),
            "delta": delta,
            "gamma": gamma,
            "theta": theta,
            "vega": vega,
            "iv": iv,
        }

    @staticmethod
    def generate_chain(
        spot: float,
        vix: float,
        minutes_to_close: float,
        underlying: str,
        expiration: str,
        strike_range: float = 0.03,
        strike_step: float = 1.0,
        spread_pct: float = 0.05,
    ) -> list[SimulatedOption]:
        """Generate a full simulated options chain for a point in time."""
        tte = max(minutes_to_close / (252 * 390), 1e-6)  # Trading minutes to years
        iv = vix / 100  # Convert VIX to decimal IV

        # Intraday IV term structure for 0DTE:
        # As expiration approaches, IV increases significantly (empirically observed).
        # This supports option prices and counteracts pure BS theta decay.
        # Scale: at 390 min (open), 1.3×; at 60 min, ~1.8×; at 15 min, ~2.5×
        iv_scale = 1.3 + 0.8 * max(0, 1 - minutes_to_close / 390) ** 1.5
        iv_0dte = iv * iv_scale

        low_strike = int(spot * (1 - strike_range))
        high_strike = int(spot * (1 + strike_range))
        strikes = np.arange(low_strike, high_strike + strike_step, strike_step)

        chain = []
        for strike in strikes:
            for otype in ["call", "put"]:
                greeks = OptionPricer.price_option(spot, strike, tte, iv_0dte, option_type=otype)
                price = greeks["price"]

                # Simulate bid/ask spread
                half_spread = price * spread_pct / 2
                bid = max(0.01, price - half_spread)
                ask = price + half_spread
                mid = (bid + ask) / 2

                # Build OCC-style symbol
                strike_str = f"{int(strike * 1000):08d}"
                type_char = "C" if otype == "call" else "P"
                symbol = f"{underlying}{expiration.replace('-', '')}{type_char}{strike_str}"

                chain.append(SimulatedOption(
                    symbol=symbol,
                    underlying=underlying,
                    option_type=otype,
                    strike=strike,
                    expiration=expiration,
                    bid=round(bid, 2),
                    ask=round(ask, 2),
                    mid=round(mid, 2),
                    delta=round(greeks["delta"], 4),
                    gamma=round(greeks["gamma"], 6),
                    theta=round(greeks["theta"], 4),
                    vega=round(greeks["vega"], 4),
                    iv=round(iv_0dte, 4),
                ))

        return chain
