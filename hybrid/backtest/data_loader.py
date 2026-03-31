"""Historical data loader for hybrid backtester.

Fetches intraday bars from Alpaca (preferred) or yfinance (fallback),
VIX history from yfinance, and generates simulated option chains via
Black-Scholes pricing.
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Bar:
    """A single intraday OHLCV bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float = 0.0


@dataclass
class SimOption:
    """A simulated option contract with Greeks."""
    symbol: str
    strike: float
    option_type: str  # "call" or "put"
    expiry: str
    bid: float
    ask: float
    mid: float
    delta: float
    gamma: float
    theta: float
    vega: float
    iv: float
    volume: int = 0
    open_interest: int = 0


@dataclass
class DaySnapshot:
    """All data for one backtest day."""
    date: date
    underlying: str
    bars: list[Bar]
    vix: float
    prev_close: float
    fear_greed: int = 50  # Default neutral
    sector_breadth: str = "MIXED"


def load_alpaca_bars(
    symbol: str,
    start: date,
    end: date,
    timeframe: str = "5Min",
) -> list[Bar]:
    """Load intraday bars from Alpaca's historical data API."""
    import requests
    from hybrid.config import ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_DATA_URL

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    }

    bars = []
    current = start
    while current <= end:
        try:
            url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/bars"
            params = {
                "timeframe": timeframe,
                "start": current.isoformat() + "T09:30:00-04:00",
                "end": current.isoformat() + "T16:00:00-04:00",
                "limit": 10000,
                "feed": "iex",
                "adjustment": "raw",
            }
            resp = requests.get(url, params=params, headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                for b in data.get("bars", []) or []:
                    bars.append(Bar(
                        timestamp=datetime.fromisoformat(b["t"].replace("Z", "+00:00")),
                        open=float(b["o"]),
                        high=float(b["h"]),
                        low=float(b["l"]),
                        close=float(b["c"]),
                        volume=int(b["v"]),
                        vwap=float(b.get("vw", 0)),
                    ))
        except Exception as e:
            logger.warning(f"Alpaca bars failed for {symbol} on {current}: {e}")
        current += timedelta(days=1)

    return bars


def load_yfinance_bars(
    symbol: str,
    start: date,
    end: date,
    interval: str = "5m",
) -> list[Bar]:
    """Fallback: load bars from yfinance."""
    import yfinance as yf

    bars = []
    # yfinance limits 1m to 30 days, 5m to 60 days
    current = start
    while current <= end:
        chunk_end = min(current + timedelta(days=6), end + timedelta(days=1))
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=current.isoformat(),
                end=chunk_end.isoformat(),
                interval=interval,
            )
            for ts, row in df.iterrows():
                bars.append(Bar(
                    timestamp=ts.to_pydatetime() if ts.tzinfo else ts.to_pydatetime().replace(tzinfo=timezone.utc),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=int(row["Volume"]),
                ))
        except Exception as e:
            logger.warning(f"yfinance bars failed for {symbol} {current}-{chunk_end}: {e}")
        current = chunk_end

    return bars


def load_vix_history(start: date, end: date) -> dict[date, float]:
    """Load daily VIX closes from yfinance."""
    import yfinance as yf

    try:
        vix = yf.Ticker("^VIX")
        df = vix.history(
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            interval="1d",
        )
        return {d.date(): float(row["Close"]) for d, row in df.iterrows()}
    except Exception as e:
        logger.warning(f"VIX history failed: {e}")
        return {}


def load_days(
    symbol: str,
    start: date,
    end: date,
    timeframe: str = "5Min",
    use_alpaca: bool = True,
) -> list[DaySnapshot]:
    """Load and organize historical data into per-day snapshots."""
    # Load bars
    if use_alpaca:
        try:
            bars = load_alpaca_bars(symbol, start, end, timeframe)
        except Exception:
            logger.info("Alpaca bars failed, falling back to yfinance")
            interval = {"1Min": "1m", "5Min": "5m", "15Min": "15m"}.get(timeframe, "5m")
            bars = load_yfinance_bars(symbol, start, end, interval)
    else:
        interval = {"1Min": "1m", "5Min": "5m", "15Min": "15m"}.get(timeframe, "5m")
        bars = load_yfinance_bars(symbol, start, end, interval)

    if not bars:
        logger.error(f"No bars loaded for {symbol}")
        return []

    # Load VIX
    vix_map = load_vix_history(start - timedelta(days=5), end)

    # Group bars by date
    from collections import defaultdict
    by_date = defaultdict(list)
    for b in bars:
        by_date[b.timestamp.date()].append(b)

    # Build DaySnapshots
    days = []
    prev_close = bars[0].open
    sorted_dates = sorted(by_date.keys())

    for day_date in sorted_dates:
        day_bars = sorted(by_date[day_date], key=lambda b: b.timestamp)
        if not day_bars:
            continue

        days.append(DaySnapshot(
            date=day_date,
            underlying=symbol,
            bars=day_bars,
            vix=vix_map.get(day_date, 20.0),
            prev_close=prev_close,
        ))
        prev_close = day_bars[-1].close

    logger.info(f"Loaded {len(days)} days for {symbol} ({sum(len(d.bars) for d in days)} bars)")
    return days


# ── Black-Scholes Option Pricer ──────────────────────────────

def bs_price(
    spot: float,
    strike: float,
    tte_years: float,
    iv: float,
    r: float = 0.05,
    option_type: str = "call",
) -> dict:
    """Black-Scholes price + Greeks."""
    from scipy.stats import norm

    if tte_years <= 0 or iv <= 0:
        intrinsic = max(0, spot - strike) if option_type == "call" else max(0, strike - spot)
        delta = (1.0 if spot > strike else 0.0) if option_type == "call" else (-1.0 if spot < strike else 0.0)
        return {"price": intrinsic, "delta": delta, "gamma": 0, "theta": 0, "vega": 0, "iv": iv}

    sqrt_t = np.sqrt(tte_years)
    d1 = (np.log(spot / strike) + (r + iv**2 / 2) * tte_years) / (iv * sqrt_t)
    d2 = d1 - iv * sqrt_t

    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * np.exp(-r * tte_years) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = strike * np.exp(-r * tte_years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (spot * iv * sqrt_t)
    theta = (-(spot * norm.pdf(d1) * iv) / (2 * sqrt_t)
             - r * strike * np.exp(-r * tte_years) * (
                 norm.cdf(d2) if option_type == "call" else norm.cdf(-d2)
             ))
    theta = theta / 365
    vega = spot * norm.pdf(d1) * sqrt_t / 100

    return {
        "price": max(0.01, price),
        "delta": delta,
        "gamma": gamma,
        "theta": max(-0.50, min(0.0, theta)),
        "vega": vega,
        "iv": iv,
    }


def generate_chain(
    spot: float,
    vix: float,
    minutes_to_close: float,
    underlying: str,
    expiry: str,
    strike_range_pct: float = 0.03,
    strike_step: float = 1.0,
    spread_pct: float = 0.05,
) -> list[SimOption]:
    """Generate simulated options chain at a point in time."""
    tte = max(minutes_to_close / (252 * 390), 1e-6)
    iv = vix / 100

    # 0DTE IV term structure adjustment
    iv_scale = 1.3 + 0.8 * max(0, 1 - minutes_to_close / 390) ** 1.5
    iv_adj = iv * iv_scale

    low = int(spot * (1 - strike_range_pct))
    high = int(spot * (1 + strike_range_pct))
    strikes = np.arange(low, high + strike_step, strike_step)

    chain = []
    for strike in strikes:
        for otype in ["call", "put"]:
            g = bs_price(spot, strike, tte, iv_adj, option_type=otype)
            price = g["price"]
            half = price * spread_pct / 2
            bid = max(0.01, price - half)
            ask = price + half
            mid = (bid + ask) / 2

            strike_str = f"{int(strike * 1000):08d}"
            type_char = "C" if otype == "call" else "P"
            sym = f"{underlying}{expiry.replace('-', '')}{type_char}{strike_str}"

            chain.append(SimOption(
                symbol=sym, strike=strike, option_type=otype, expiry=expiry,
                bid=round(bid, 2), ask=round(ask, 2), mid=round(mid, 2),
                delta=round(g["delta"], 4), gamma=round(g["gamma"], 6),
                theta=round(g["theta"], 4), vega=round(g["vega"], 4),
                iv=round(iv_adj, 4),
            ))

    return chain
