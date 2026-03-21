"""Build complete CLI response snapshots from historical data.

Takes a day's bars + VIX + a time index and produces the JSON that
every CLI command would return at that point in time. This is what
the mock CLI serves to Claude during backtesting.
"""

import json
import logging
from datetime import datetime, date, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from hybrid.backtest.data_loader import (
    Bar, DaySnapshot, SimOption, generate_chain, load_days,
)
from hybrid.backtest.bt_state import BacktestState

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def build_snapshot(
    day: DaySnapshot,
    bar_index: int,
    state: BacktestState,
) -> dict:
    """Build a complete snapshot of all CLI command responses at a point in time.

    Args:
        day: The day's data (bars, VIX, etc.)
        bar_index: Which bar we're "at" (0 = first bar of day)
        state: Current simulated account state

    Returns:
        Dict keyed by CLI command name, values are the JSON responses.
    """
    if bar_index >= len(day.bars):
        bar_index = len(day.bars) - 1

    current_bar = day.bars[bar_index]
    et_time = current_bar.timestamp.astimezone(ET)

    # Historical bars up to this point (what Claude would see)
    historical_bars = day.bars[:bar_index + 1]

    snapshot = {}

    # ── account ──
    snapshot["account"] = {
        "equity": round(state.equity, 2),
        "cash": round(state.cash, 2),
        "buying_power": round(state.buying_power, 2),
        "portfolio_value": round(state.equity, 2),
        "day_trade_count": state.daily_trades,
        "pattern_day_trader": False,
        "trading_blocked": False,
        "account_blocked": False,
        "status": "ACTIVE",
    }

    # ── positions ──
    snapshot["positions"] = state.positions

    # ── orders ──
    snapshot["orders"] = [o for o in state.orders if o["status"] == "filled"]
    snapshot["orders_open"] = []

    # ── daily-state ──
    is_in_window = time(9, 45) <= et_time.time() <= time(15, 0)
    snapshot["daily-state"] = {
        "date": day.date.isoformat(),
        "daily_pnl": round(state.daily_pnl, 2),
        "trades_today": state.daily_trades,
        "wins": state.daily_wins,
        "losses": state.daily_losses,
        "is_market_hours": time(9, 30) <= et_time.time() <= time(16, 0),
        "force_close": et_time.time() >= time(15, 45),
    }

    # ── quotes ──
    quotes = {}
    for sym in ["SPY", "QQQ", "IWM"]:
        if sym == day.underlying:
            bar = current_bar
        else:
            # Approximate other underlyings from SPY ratio
            bar = current_bar  # Simplified: use same bar
        quotes[sym] = {
            "bid": round(bar.close * 0.9999, 2),
            "ask": round(bar.close * 1.0001, 2),
            "mid": round(bar.close, 2),
            "last": round(bar.close, 2),
            "bid_size": 500,
            "ask_size": 500,
            "timestamp": bar.timestamp.isoformat(),
        }
    snapshot["quotes"] = quotes

    # ── bars (last N 5-min bars) ──
    bars_data = []
    for b in historical_bars[-50:]:
        bars_data.append({
            "timestamp": b.timestamp.isoformat(),
            "open": round(b.open, 2),
            "high": round(b.high, 2),
            "low": round(b.low, 2),
            "close": round(b.close, 2),
            "volume": b.volume,
            "vwap": round(b.vwap, 2) if b.vwap else round(b.close, 2),
        })
    snapshot["bars"] = {day.underlying: bars_data}

    # ── expirations ──
    today = day.date.isoformat()
    tomorrow = (day.date + timedelta(days=1)).isoformat()
    day_after = (day.date + timedelta(days=2)).isoformat()
    snapshot["expirations"] = [today, tomorrow, day_after]

    # ── chain (0DTE options) ──
    minutes_left = _minutes_until(et_time, 16, 0)
    chain = generate_chain(
        spot=current_bar.close,
        vix=day.vix,
        minutes_to_close=minutes_left,
        underlying=day.underlying,
        expiry=today,
    )

    chain_data = []
    for opt in chain:
        chain_data.append({
            "symbol": opt.symbol,
            "strike": opt.strike,
            "expiration": opt.expiry,
            "option_type": opt.option_type,
            "bid": opt.bid,
            "ask": opt.ask,
            "mid": opt.mid,
            "delta": opt.delta,
            "gamma": opt.gamma,
            "theta": opt.theta,
            "vega": opt.vega,
            "iv": opt.iv,
            "open_interest": 500,
            "volume": 1000,
        })
    snapshot["chain"] = chain_data

    # ── chain-greeks (Public.com format) ──
    calls = [c for c in chain_data if c["option_type"] == "call"]
    puts = [c for c in chain_data if c["option_type"] == "put"]
    snapshot["chain-greeks"] = {
        "underlying": day.underlying,
        "expiry": today,
        "spot_price": current_bar.close,
        "calls": calls,
        "puts": puts,
        "contracts_returned": len(chain_data),
        "greeks_available": len(chain_data),
    }

    # ── vix ──
    vix = day.vix
    if vix < 15:
        regime, desc = "LOW_VOL", "Low volatility"
    elif vix < 20:
        regime, desc = "NORMAL", "Normal volatility"
    elif vix < 25:
        regime, desc = "ELEVATED", "Elevated volatility"
    elif vix < 35:
        regime, desc = "HIGH", "High volatility — reduce size, wider stops"
    else:
        regime, desc = "CRISIS", "Crisis volatility — consider standing aside"

    snapshot["vix"] = {
        "vix": round(vix, 2),
        "previous_close": round(vix * 0.97, 2),
        "change": round(vix * 0.03, 2),
        "change_pct": 3.0,
        "regime": regime,
        "description": desc,
    }

    # ── fear-greed (approximate from VIX) ──
    # High VIX correlates with fear; use a rough mapping
    if vix > 30:
        fg_score, fg_rating = 15, "extreme fear"
    elif vix > 25:
        fg_score, fg_rating = 25, "fear"
    elif vix > 20:
        fg_score, fg_rating = 40, "fear"
    elif vix > 15:
        fg_score, fg_rating = 50, "neutral"
    else:
        fg_score, fg_rating = 65, "greed"

    if fg_score <= 25:
        fg_signal, fg_advice = "CONTRARIAN_BULLISH", "Extreme fear — contrarian buying opportunity"
    elif fg_score <= 40:
        fg_signal, fg_advice = "LEAN_BULLISH", "Fear — market may be oversold"
    elif fg_score <= 60:
        fg_signal, fg_advice = "NEUTRAL", "Neutral — no strong signal"
    elif fg_score <= 75:
        fg_signal, fg_advice = "LEAN_BEARISH", "Greed — market may be overbought"
    else:
        fg_signal, fg_advice = "CONTRARIAN_BEARISH", "Extreme greed — pullback risk"

    snapshot["fear-greed"] = {
        "score": fg_score,
        "rating": fg_rating,
        "signal": fg_signal,
        "advice": fg_advice,
    }

    # ── sectors (simplified) ──
    snapshot["sectors"] = {
        "sectors": [],
        "up_count": 5,
        "down_count": 6,
        "breadth": "MIXED",
        "description": "Mixed market breadth",
    }

    # ── market-overview ──
    snapshot["market-overview"] = {
        "vix": snapshot["vix"],
        "fear_and_greed": snapshot["fear-greed"],
        "sectors": snapshot["sectors"],
    }

    # ── indices ──
    snapshot["indices"] = {
        "VIX": {"last": str(round(vix, 2)), "volume": 0, "regime": regime},
        "SPX": {"last": str(round(current_bar.close * 10, 2)), "volume": 0},
    }

    # ── news (empty for backtest) ──
    snapshot["news"] = []

    # ── calendar (empty for backtest) ──
    snapshot["calendar"] = []

    # ── earnings (empty for backtest) ──
    snapshot["earnings"] = []

    # ── sentiment ──
    snapshot["sentiment"] = {
        "symbol": day.underlying,
        "bullish_pct": 50.0,
        "bearish_pct": 50.0,
        "net_sentiment": 0.0,
        "signal": "NEUTRAL",
        "articles_this_week": 50,
        "buzz_ratio": 1.0,
        "buzz_description": "Normal coverage",
    }

    return snapshot


def _minutes_until(et_time: datetime, h: int, m: int) -> float:
    target = et_time.replace(hour=h, minute=m, second=0, microsecond=0)
    delta = (target - et_time).total_seconds() / 60
    return max(1, delta)


def save_snapshot(snapshot: dict, path: str):
    """Save snapshot to disk for mock CLI to read."""
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
