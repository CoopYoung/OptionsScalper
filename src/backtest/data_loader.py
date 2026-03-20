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
    open_interest: int = 0
    volume: int = 0


@dataclass
class BacktestDay:
    """All data needed for one backtest day."""
    date: date
    underlying: str
    bars: list[HistoricalBar]
    vix_close: float
    prev_close: float


class HistoricalDataLoader:
    """Loads and prepares historical data for backtesting.

    For dates within yfinance's 60-day intraday window: uses real intraday bars.
    For dates beyond that: generates synthetic intraday bars from daily OHLCV
    using a Brownian bridge constrained to hit real daily OHLC values.
    """

    # yfinance intraday data horizon (days from today)
    _INTRADAY_HORIZON = 58  # conservative vs 60-day hard limit

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

        Automatically uses real intraday data where available (last 60 days)
        and synthetic intraday bars from daily OHLCV for older dates.

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

        # Determine cutoff: dates before this use synthetic bars
        today = date.today()
        intraday_cutoff = today - timedelta(days=self._INTRADAY_HORIZON)

        # Parse interval to minutes for synthetic bar generation
        interval_minutes = {"1m": 1, "2m": 2, "5m": 5}.get(interval, 2)

        # ── Phase 1: Load daily OHLCV + VIX for full range ────────
        try:
            daily_df = yf.Ticker(underlying).history(
                start=(start_date - timedelta(days=5)).isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                interval="1d",
            )
            daily_map = {}
            for ts, row in daily_df.iterrows():
                d = ts.date() if hasattr(ts, 'date') else ts
                daily_map[d] = {
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
        except Exception:
            logger.exception("Failed to load daily data for %s", underlying)
            daily_map = {}

        try:
            vix_ticker = yf.Ticker("^VIX")
            vix_df = vix_ticker.history(
                start=start_date.isoformat(),
                end=(end_date + timedelta(days=1)).isoformat(),
                interval="1d",
            )
            vix_map = {d.date(): float(row["Close"]) for d, row in vix_df.iterrows()}
        except Exception:
            vix_map = {}

        daily_closes = {d: v["close"] for d, v in daily_map.items()}

        # ── Phase 2: Load real intraday bars for recent dates ─────
        real_intraday_bars = {}
        real_start = max(start_date, intraday_cutoff)
        if real_start <= end_date:
            all_bars: list[pd.DataFrame] = []
            current = real_start
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
                    logger.debug("Failed to load %s intraday chunk %s-%s", underlying, current, chunk_end)
                current = chunk_end

            if all_bars:
                full_df = pd.concat(all_bars)
                full_df = full_df[~full_df.index.duplicated(keep="first")]
                full_df = full_df.sort_index()
                for day_date, group in full_df.groupby(full_df.index.date):
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
                    if bars:
                        real_intraday_bars[day_date] = bars

        # ── Phase 3: Build BacktestDay list ───────────────────────
        days: list[BacktestDay] = []
        synthetic_count = 0
        real_count = 0

        for day_date in sorted(daily_map.keys()):
            if day_date < start_date or day_date > end_date:
                continue

            # Find prev close
            prev_close = daily_map[day_date]["open"]
            for d in sorted(daily_closes.keys()):
                if d < day_date:
                    prev_close = daily_closes[d]

            if day_date in real_intraday_bars:
                bars = real_intraday_bars[day_date]
                real_count += 1
            else:
                # Generate synthetic intraday bars from daily OHLCV
                daily = daily_map[day_date]
                bars = self._generate_synthetic_bars(
                    day_date, underlying, daily, interval_minutes,
                )
                synthetic_count += 1

            if bars:
                days.append(BacktestDay(
                    date=day_date,
                    underlying=underlying,
                    bars=bars,
                    vix_close=vix_map.get(day_date, 20.0),
                    prev_close=prev_close,
                ))

        logger.info(
            "Loaded %d trading days for %s (%d real, %d synthetic, %d total bars)",
            len(days), underlying, real_count, synthetic_count,
            sum(len(d.bars) for d in days),
        )
        return days

    @staticmethod
    def _generate_synthetic_bars(
        day_date: date,
        underlying: str,
        daily: dict,
        interval_minutes: int,
    ) -> list["HistoricalBar"]:
        """Generate synthetic intraday bars from daily OHLCV.

        Uses a constrained Brownian bridge: the price path starts at Open,
        touches High and Low at realistic times, and ends at Close.
        Volume follows a U-shaped intraday pattern (high at open/close).

        This produces statistically realistic intraday paths that respect
        the actual daily range and direction, enabling long-range backtests
        beyond yfinance's 60-day intraday limit.
        """
        o, h, l, c = daily["open"], daily["high"], daily["low"], daily["close"]
        total_volume = daily["volume"]

        if o <= 0 or h <= 0 or l <= 0 or c <= 0:
            return []

        # Market hours: 9:30-16:00 ET = 390 minutes
        n_bars = 390 // interval_minutes
        if n_bars < 10:
            return []

        # Seed deterministically per day + underlying for reproducibility
        seed = hash((day_date.isoformat(), underlying)) & 0x7FFFFFFF
        rng = np.random.default_rng(seed)

        # ── Step 1: Generate normalized price path via Brownian bridge ──
        # Determine if bullish or bearish day
        bullish = c >= o
        daily_range = h - l
        if daily_range < 0.01:
            daily_range = 0.01

        # High and low touch points (when during the day)
        # On bullish days: tend to hit low first, then high
        # On bearish days: tend to hit high first, then low
        if bullish:
            low_bar = rng.integers(n_bars // 6, n_bars // 3)     # Low in first third
            high_bar = rng.integers(n_bars * 2 // 3, n_bars - 2)  # High in last third
        else:
            high_bar = rng.integers(n_bars // 6, n_bars // 3)
            low_bar = rng.integers(n_bars * 2 // 3, n_bars - 2)

        # Build price path using piecewise Brownian bridges
        # Segments: Open→first_extreme, first_extreme→second_extreme, second_extreme→Close
        prices = np.zeros(n_bars + 1)
        prices[0] = o
        prices[n_bars] = c

        if bullish:
            # Open → Low → High → Close
            pivot1_bar, pivot1_price = low_bar, l
            pivot2_bar, pivot2_price = high_bar, h
        else:
            # Open → High → Low → Close
            pivot1_bar, pivot1_price = high_bar, h
            pivot2_bar, pivot2_price = low_bar, l

        # Ensure pivots are ordered
        if pivot1_bar >= pivot2_bar:
            pivot1_bar, pivot2_bar = n_bars // 4, 3 * n_bars // 4
            if bullish:
                pivot1_price, pivot2_price = l, h
            else:
                pivot1_price, pivot2_price = h, l

        prices[pivot1_bar] = pivot1_price
        prices[pivot2_bar] = pivot2_price

        # Fill segments with Brownian bridges
        def _bridge(arr, i_start, i_end, rng_inst):
            """Fill arr[i_start+1..i_end-1] with a Brownian bridge."""
            n = i_end - i_start
            if n <= 1:
                return
            start_val = arr[i_start]
            end_val = arr[i_end]
            # Generate increments
            noise = rng_inst.normal(0, 1, n)
            # Brownian motion
            cumsum = np.cumsum(noise)
            # Scale noise to fit within daily range
            vol = daily_range / (n ** 0.5) * 0.3
            bridge = start_val + (end_val - start_val) * np.arange(1, n + 1) / n
            bridge += vol * (cumsum - np.arange(1, n + 1) / n * cumsum[-1])
            # Clamp to daily range
            bridge = np.clip(bridge, l * 0.999, h * 1.001)
            arr[i_start + 1:i_end] = bridge[:-1]

        _bridge(prices, 0, pivot1_bar, rng)
        _bridge(prices, pivot1_bar, pivot2_bar, rng)
        _bridge(prices, pivot2_bar, n_bars, rng)

        # Ensure exact endpoints
        prices[0] = o
        prices[n_bars] = c
        prices = np.clip(prices, l * 0.999, h * 1.001)

        # ── Step 2: Generate OHLCV bars from the path ──
        # U-shaped volume distribution (high at open/close, low midday)
        t = np.linspace(0, 1, n_bars)
        vol_shape = 1.5 + 2.0 * ((t - 0.5) ** 2)  # U-shape: min=1.5 at midday
        vol_shape /= vol_shape.sum()
        bar_volumes = (vol_shape * total_volume).astype(int)
        bar_volumes = np.maximum(bar_volumes, 100)

        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")

        bars = []
        base_dt = datetime(day_date.year, day_date.month, day_date.day,
                           9, 30, 0, tzinfo=ET)

        for i in range(n_bars):
            bar_open = prices[i]
            bar_close = prices[i + 1]
            # Add micro-noise for high/low within each bar
            bar_noise = rng.uniform(0, daily_range * 0.005, 2)
            bar_high = max(bar_open, bar_close) + bar_noise[0]
            bar_low = min(bar_open, bar_close) - bar_noise[1]
            bar_high = min(bar_high, h * 1.001)
            bar_low = max(bar_low, l * 0.999)

            ts = base_dt + timedelta(minutes=i * interval_minutes)

            bars.append(HistoricalBar(
                timestamp=ts,
                open=round(bar_open, 2),
                high=round(bar_high, 2),
                low=round(bar_low, 2),
                close=round(bar_close, 2),
                volume=int(bar_volumes[i]),
            ))

        return bars


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

        # Beta-adjust IV: VIX tracks SPY vol, so other underlyings need scaling.
        # Without this, QQQ options are priced too cheaply (its realized vol
        # exceeds VIX-implied vol), creating an artificial edge in backtests.
        beta_map = {"SPY": 1.0, "QQQ": 1.35, "IWM": 1.15}
        iv *= beta_map.get(underlying, 1.0)

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

                # Simulate realistic OI distribution:
                # - Peak OI near ATM, decaying with distance from spot
                # - Calls concentrated above spot, puts below
                # - Round number strikes get extra OI
                moneyness = abs(strike - spot) / spot
                atm_decay = np.exp(-moneyness * 30)  # Gaussian-like decay
                base_oi = int(5000 * atm_decay)

                # Directional bias: calls heavier above spot, puts below
                if otype == "call" and strike >= spot:
                    base_oi = int(base_oi * 1.3)
                elif otype == "put" and strike <= spot:
                    base_oi = int(base_oi * 1.3)

                # Round number bonus (e.g. 550, 560 get more OI)
                if strike % 10 == 0:
                    base_oi = int(base_oi * 2.0)
                elif strike % 5 == 0:
                    base_oi = int(base_oi * 1.4)

                # Add randomness (±30%)
                rng = np.random.default_rng(int(strike * 100 + (1 if otype == "call" else 2)))
                oi = max(0, int(base_oi * rng.uniform(0.7, 1.3)))

                # Volume: fraction of OI, higher near ATM
                vol_ratio = 0.1 + 0.3 * atm_decay  # 10-40% of OI
                volume = max(0, int(oi * vol_ratio * rng.uniform(0.5, 1.5)))

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
                    open_interest=oi,
                    volume=volume,
                ))

        return chain
