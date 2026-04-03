"""Data gathering, technical indicators, and prompt formatting.

Python does ALL the work here. The LLM just reads a pre-digested summary
and returns a JSON decision. No tool calling needed.
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from hybrid.broker.broker_base import Broker

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


# ══════════════════════════════════════════════════════════════
# Section A: Technical Indicators (pure math, no I/O)
# ══════════════════════════════════════════════════════════════

def compute_rsi(closes: list[float], period: int = 14) -> float | None:
    """Relative Strength Index. Returns 0-100 or None if insufficient data."""
    if len(closes) < period + 1:
        return None

    gains, losses = [], []
    for i in range(1, len(closes)):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0))
        losses.append(max(-delta, 0))

    # Use exponential moving average (Wilder's smoothing)
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 1)


def compute_macd(closes: list[float],
                 fast: int = 12, slow: int = 26, signal: int = 9) -> dict | None:
    """MACD with signal line and histogram. Returns None if insufficient data."""
    if len(closes) < slow + signal:
        return None

    def _ema(data: list[float], period: int) -> list[float]:
        multiplier = 2 / (period + 1)
        ema = [sum(data[:period]) / period]
        for price in data[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        return ema

    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)

    # Align: ema_fast starts at index (fast-1), ema_slow at (slow-1)
    offset = slow - fast
    macd_line = [f - s for f, s in zip(ema_fast[offset:], ema_slow)]

    if len(macd_line) < signal:
        return None

    signal_line = _ema(macd_line, signal)

    # Latest values
    macd_val = macd_line[-1]
    signal_val = signal_line[-1]
    histogram = macd_val - signal_val

    # Detect crossover
    if len(macd_line) >= 2 and len(signal_line) >= 2:
        prev_macd = macd_line[-2]
        prev_signal = signal_line[-2]
        if prev_macd <= prev_signal and macd_val > signal_val:
            crossover = "bullish_crossover"
        elif prev_macd >= prev_signal and macd_val < signal_val:
            crossover = "bearish_crossover"
        else:
            crossover = "none"
    else:
        crossover = "none"

    return {
        "macd": round(macd_val, 4),
        "signal": round(signal_val, 4),
        "histogram": round(histogram, 4),
        "crossover": crossover,
    }


def compute_bollinger(closes: list[float], period: int = 20,
                      num_std: float = 2.0) -> dict | None:
    """Bollinger Bands — where is price relative to bands."""
    if len(closes) < period:
        return None

    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)

    upper = middle + num_std * std
    lower = middle - num_std * std
    current = closes[-1]

    if upper == lower:
        position = "middle"
    else:
        pct = (current - lower) / (upper - lower)
        if pct > 1.0:
            position = "above_upper"
        elif pct > 0.5:
            position = "upper_half"
        elif pct > 0.0:
            position = "lower_half"
        else:
            position = "below_lower"

    return {
        "upper": round(upper, 2),
        "middle": round(middle, 2),
        "lower": round(lower, 2),
        "position": position,
        "pct_b": round((current - lower) / (upper - lower), 2) if upper != lower else 0.5,
    }


def compute_vwap(bars: list[dict]) -> float | None:
    """VWAP from intraday bars. Alpaca bars already include vwap field."""
    if not bars:
        return None
    # Prefer Alpaca's pre-computed VWAP
    if bars[-1].get("vwap", 0) > 0:
        return round(bars[-1]["vwap"], 2)
    # Manual fallback
    cum_vol = 0
    cum_tp_vol = 0.0
    for b in bars:
        tp = (b["high"] + b["low"] + b["close"]) / 3
        vol = b["volume"]
        cum_tp_vol += tp * vol
        cum_vol += vol
    return round(cum_tp_vol / cum_vol, 2) if cum_vol > 0 else None


def compute_volume_ratio(volumes: list[int], lookback: int = 20) -> float | None:
    """Current volume relative to N-bar average."""
    if len(volumes) < 2:
        return None
    avg_vol = sum(volumes[-lookback - 1:-1]) / min(lookback, len(volumes) - 1)
    if avg_vol == 0:
        return None
    return round(volumes[-1] / avg_vol, 2)


def compute_momentum(closes: list[float], period: int = 5) -> float | None:
    """Percentage change over N bars."""
    if len(closes) < period + 1:
        return None
    old = closes[-period - 1]
    if old == 0:
        return None
    return round((closes[-1] - old) / old * 100, 2)


# ── NEW: Put/Call Ratio from chain ─────────────────────────

def compute_put_call_ratio(chain: list[dict]) -> dict | None:
    """Compute P/C ratio from the full options chain.

    Returns volume-based and OI-based ratios with interpretation.
    P/C > 1.2 = bearish (contrarian bullish), < 0.7 = bullish (contrarian bearish).
    """
    if not chain:
        return None

    call_vol, put_vol = 0, 0
    call_oi, put_oi = 0, 0

    for c in chain:
        otype = c.get("option_type", "")
        vol = int(c.get("volume", 0) or 0)
        oi = int(c.get("open_interest", 0) or 0)
        if otype == "call":
            call_vol += vol
            call_oi += oi
        elif otype == "put":
            put_vol += vol
            put_oi += oi

    vol_ratio = round(put_vol / call_vol, 2) if call_vol > 0 else None
    oi_ratio = round(put_oi / call_oi, 2) if call_oi > 0 else None

    # Interpret
    if vol_ratio is not None:
        if vol_ratio > 1.2:
            sentiment = "BEARISH_FLOW (contrarian bullish)"
        elif vol_ratio < 0.7:
            sentiment = "BULLISH_FLOW (contrarian bearish)"
        else:
            sentiment = "NEUTRAL_FLOW"
    else:
        sentiment = "unknown"

    return {
        "volume_pcr": vol_ratio,
        "oi_pcr": oi_ratio,
        "call_volume": call_vol,
        "put_volume": put_vol,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "sentiment": sentiment,
    }


# ── NEW: Options flow imbalance (net delta flow) ──────────

def compute_options_flow(chain: list[dict]) -> dict | None:
    """Compute net delta exposure from options flow.

    Sum of (delta × volume × mid × 100) across all contracts.
    Large negative net flow = smart money buying puts.
    """
    if not chain:
        return None

    net_delta_flow = 0.0
    total_premium_vol = 0.0
    unusual_activity: list[dict] = []

    for c in chain:
        delta = float(c.get("delta", 0) or 0)
        vol = int(c.get("volume", 0) or 0)
        oi = int(c.get("open_interest", 0) or 0)
        mid = float(c.get("mid", 0) or 0)

        if vol <= 0 or mid <= 0:
            continue

        premium_flow = delta * vol * mid * 100
        net_delta_flow += premium_flow
        total_premium_vol += vol * mid * 100

        # Detect unusual activity: volume > 3x open interest
        if oi > 0 and vol > oi * 3:
            unusual_activity.append({
                "symbol": c.get("symbol", ""),
                "type": c.get("option_type", ""),
                "strike": c.get("strike", 0),
                "vol": vol,
                "oi": oi,
                "ratio": round(vol / oi, 1),
            })

    if total_premium_vol == 0:
        return None

    # Normalize to -1 to +1 scale
    normalized = net_delta_flow / total_premium_vol if total_premium_vol else 0

    if normalized > 0.15:
        bias = "BULLISH_FLOW"
    elif normalized < -0.15:
        bias = "BEARISH_FLOW"
    else:
        bias = "NEUTRAL"

    return {
        "net_delta_flow": round(net_delta_flow, 0),
        "normalized_flow": round(normalized, 3),
        "bias": bias,
        "unusual_activity": sorted(unusual_activity,
                                    key=lambda x: x["ratio"], reverse=True)[:3],
    }


# ── NEW: IV Percentile / IV Rank ──────────────────────────

def compute_iv_percentile(current_iv: float, vix_history_5d: list[float]) -> dict | None:
    """Compute IV rank relative to recent VIX history.

    Uses VIX as a proxy for broad IV. If current ATM IV is in the 80th
    percentile of recent VIX range → options are expensive (favor selling).
    """
    if not vix_history_5d or current_iv <= 0:
        return None

    all_values = vix_history_5d + [current_iv * 100]  # Convert IV to VIX scale
    sorted_vals = sorted(all_values)
    rank = sorted_vals.index(current_iv * 100) / len(sorted_vals) * 100 if len(sorted_vals) > 1 else 50

    # IV rank: (current - min) / (max - min)
    min_iv = min(all_values)
    max_iv = max(all_values)
    iv_rank = ((current_iv * 100) - min_iv) / (max_iv - min_iv) * 100 if max_iv > min_iv else 50

    if iv_rank > 70:
        regime = "HIGH_IV (favor selling premium / tighter stops)"
    elif iv_rank < 30:
        regime = "LOW_IV (favor buying premium / wider targets)"
    else:
        regime = "NORMAL_IV"

    return {
        "iv_rank": round(iv_rank, 0),
        "iv_percentile": round(rank, 0),
        "regime": regime,
    }


# ── NEW: Intraday price action narrative ──────────────────

def compute_intraday_narrative(bars: list[dict]) -> dict | None:
    """Build a summary of today's intraday price action.

    Tells the LLM the story: "Opened at $655, sold off to $648 at 10:15,
    now bouncing at $651."
    """
    if not bars or len(bars) < 3:
        return None

    opens = [b["open"] for b in bars]
    highs = [b["high"] for b in bars]
    lows = [b["low"] for b in bars]
    closes = [b["close"] for b in bars]

    day_open = opens[0]
    day_high = max(highs)
    day_low = min(lows)
    current = closes[-1]

    # Find when high/low occurred (bar index)
    high_idx = highs.index(day_high)
    low_idx = lows.index(day_low)

    # Determine pattern
    if high_idx < low_idx:
        pattern = "SELL_OFF"  # Hit high first, then sold off
        narrative = (f"Opened ${day_open:.2f}, rallied to ${day_high:.2f} "
                     f"then sold off to ${day_low:.2f}, now ${current:.2f}")
    elif low_idx < high_idx:
        pattern = "RECOVERY"  # Hit low first, then bounced
        narrative = (f"Opened ${day_open:.2f}, dipped to ${day_low:.2f} "
                     f"then recovered to ${day_high:.2f}, now ${current:.2f}")
    else:
        pattern = "RANGE"
        narrative = (f"Opened ${day_open:.2f}, ranging between "
                     f"${day_low:.2f}-${day_high:.2f}, now ${current:.2f}")

    # Gap analysis (open vs previous close — approximated from first bar)
    gap_pct = 0.0
    gap_type = "FLAT"

    # Calculate day's range position
    day_range = day_high - day_low
    if day_range > 0:
        range_position = (current - day_low) / day_range
        range_label = "top" if range_position > 0.75 else (
            "upper" if range_position > 0.5 else (
                "lower" if range_position > 0.25 else "bottom"))
    else:
        range_position = 0.5
        range_label = "middle"

    return {
        "day_open": round(day_open, 2),
        "day_high": round(day_high, 2),
        "day_low": round(day_low, 2),
        "current": round(current, 2),
        "day_range": round(day_range, 2),
        "range_position": range_label,
        "pattern": pattern,
        "narrative": narrative,
    }


# ── NEW: Overnight gap ────────────────────────────────────

def compute_overnight_gap(broker: Broker, symbol: str,
                          current_open: float) -> dict | None:
    """Compute overnight gap by comparing today's open to yesterday's close."""
    if current_open <= 0:
        return None
    try:
        daily_bars = broker.get_stock_bars(symbol, "1Day", 2)
        if daily_bars and len(daily_bars) >= 2:
            prev_close = daily_bars[-2]["close"]
            gap = current_open - prev_close
            gap_pct = (gap / prev_close) * 100 if prev_close else 0

            if gap_pct > 0.3:
                gap_type = "GAP_UP"
            elif gap_pct < -0.3:
                gap_type = "GAP_DOWN"
            else:
                gap_type = "FLAT_OPEN"

            return {
                "prev_close": round(prev_close, 2),
                "today_open": round(current_open, 2),
                "gap": round(gap, 2),
                "gap_pct": round(gap_pct, 2),
                "type": gap_type,
            }
    except Exception as e:
        logger.debug("Overnight gap calc failed for %s: %s", symbol, e)
    return None


# ── NEW: Time-of-day regime ───────────────────────────────

def get_time_regime(now_et: datetime | None = None) -> dict:
    """Classify the current time into a trading regime.

    0DTE theta decay and volatility vary significantly by time of day.
    """
    if now_et is None:
        now_et = datetime.now(ET)

    hour = now_et.hour
    minute = now_et.minute
    time_mins = hour * 60 + minute

    if time_mins < 10 * 60 + 30:  # Before 10:30
        regime = "OPENING_VOLATILITY"
        description = "High volatility, wide spreads, fast moves. Be selective."
        theta_note = "Theta decay slow — full day of extrinsic value remains."
    elif time_mins < 12 * 60:  # 10:30 - 12:00
        regime = "MID_MORNING_TREND"
        description = "Trend establishment. Best window for directional entries."
        theta_note = "Moderate theta — good risk/reward for 0DTE."
    elif time_mins < 13 * 60 + 30:  # 12:00 - 1:30
        regime = "LUNCH_LULL"
        description = "Low volume, range-bound. Avoid new entries."
        theta_note = "Theta accelerating — existing positions losing value faster."
    elif time_mins < 14 * 60 + 30:  # 1:30 - 2:30
        regime = "AFTERNOON_SESSION"
        description = "Volume returns. Watch for trend continuation or reversal."
        theta_note = "Significant theta decay — need strong conviction for entries."
    else:  # 2:30+
        regime = "POWER_HOUR"
        description = "Final hour — maximum theta crush. Only take high-conviction trades."
        theta_note = "Extreme theta decay — options losing value rapidly."

    return {
        "regime": regime,
        "description": description,
        "theta_note": theta_note,
    }


# ── NEW: Recent trade history ─────────────────────────────

def get_recent_trades() -> list[dict]:
    """Load last 5 trades from the audit log for pattern awareness."""
    from hybrid import config
    audit_file = config.AUDIT_LOG

    if not audit_file.exists():
        return []

    trades: list[dict] = []
    try:
        lines = audit_file.read_text().strip().split("\n")
        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                trade = entry.get("trade")
                decision = entry.get("decision", {})
                exits = entry.get("exits", [])

                # Record trades
                if trade and not trade.get("dry_run"):
                    trades.append({
                        "symbol": trade.get("symbol", ""),
                        "side": trade.get("side", ""),
                        "price": trade.get("limit_price", 0),
                        "confidence": decision.get("confidence", 0),
                        "reasoning": decision.get("reasoning", "")[:60],
                        "timestamp": entry.get("timestamp", ""),
                    })

                # Record exits
                for ex in exits:
                    trades.append({
                        "symbol": ex.get("symbol", ""),
                        "action": "EXIT",
                        "reason": ex.get("reason", ""),
                        "pnl": ex.get("pnl", 0),
                        "timestamp": entry.get("timestamp", ""),
                    })

                if len(trades) >= 5:
                    break
            except json.JSONDecodeError:
                continue
    except Exception as e:
        logger.debug("Failed to read trade history: %s", e)

    return trades[:5]


# ══════════════════════════════════════════════════════════════
# Section B: Data Gathering (I/O, graceful degradation)
# ══════════════════════════════════════════════════════════════

def gather_market_context() -> dict:
    """Gather all market context. Each source wrapped in try/except."""
    from hybrid.broker import market_data

    ctx: dict = {}

    # VIX
    try:
        ctx["vix"] = market_data.get_vix()
    except Exception as e:
        logger.warning("VIX fetch failed: %s", e)
        ctx["vix"] = {"error": str(e)}

    # Fear & Greed
    try:
        ctx["fear_greed"] = market_data.get_fear_greed()
    except Exception as e:
        logger.warning("F&G fetch failed: %s", e)
        ctx["fear_greed"] = {"error": str(e)}

    # Sectors
    try:
        ctx["sectors"] = market_data.get_sector_performance()
    except Exception as e:
        logger.warning("Sectors fetch failed: %s", e)
        ctx["sectors"] = {"error": str(e)}

    # Economic calendar
    try:
        ctx["calendar"] = market_data.get_economic_calendar()
    except Exception as e:
        logger.warning("Calendar fetch failed: %s", e)
        ctx["calendar"] = {"error": str(e)}

    # Earnings
    try:
        ctx["earnings"] = market_data.get_earnings_calendar()
    except Exception as e:
        logger.warning("Earnings fetch failed: %s", e)
        ctx["earnings"] = {"error": str(e)}

    # News headlines
    try:
        ctx["news"] = market_data.get_market_news("general")
    except Exception as e:
        logger.warning("News fetch failed: %s", e)
        ctx["news"] = {"error": str(e)}

    # Time regime (zero cost — computed locally)
    ctx["time_regime"] = get_time_regime()

    # Recent trade history (zero cost — reads local file)
    ctx["recent_trades"] = get_recent_trades()

    return ctx


def gather_underlying_analysis(broker: Broker, symbol: str,
                                expiry: str | None,
                                vix_history_5d: list[float] | None = None) -> dict:
    """Gather price data + compute technicals for one underlying."""
    analysis: dict = {"symbol": symbol}

    # Quote
    try:
        quotes = broker.get_stock_quotes([symbol])
        analysis["quote"] = quotes.get(symbol, {})
    except Exception as e:
        logger.warning("Quote failed for %s: %s", symbol, e)
        analysis["quote"] = {"error": str(e)}

    # Bars + technicals
    bars = []
    try:
        bars = broker.get_stock_bars(symbol, "5Min", 50)
        if bars:
            closes = [b["close"] for b in bars]
            highs = [b["high"] for b in bars]
            lows = [b["low"] for b in bars]
            volumes = [b["volume"] for b in bars]

            analysis["price"] = closes[-1] if closes else 0
            analysis["rsi"] = compute_rsi(closes)
            analysis["macd"] = compute_macd(closes)
            analysis["bollinger"] = compute_bollinger(closes)
            analysis["vwap"] = compute_vwap(bars)
            analysis["volume_ratio"] = compute_volume_ratio(volumes)
            analysis["momentum_5"] = compute_momentum(closes, 5)
            analysis["momentum_20"] = compute_momentum(closes, 20)

            # Support / resistance from recent swings
            if len(highs) >= 10:
                analysis["resistance"] = round(max(highs[-20:]), 2)
                analysis["support"] = round(min(lows[-20:]), 2)

            # NEW: Intraday narrative
            analysis["intraday"] = compute_intraday_narrative(bars)
        else:
            analysis["bars_error"] = "No bars returned"
    except Exception as e:
        logger.warning("Bars/technicals failed for %s: %s", symbol, e)
        analysis["bars_error"] = str(e)

    # NEW: Overnight gap (uses 1Day bars — separate API call)
    day_open = bars[0]["open"] if bars else 0
    if day_open > 0:
        analysis["overnight_gap"] = compute_overnight_gap(broker, symbol, day_open)

    # Options chain (0DTE)
    chain = []
    if expiry:
        try:
            chain = broker.get_option_chain(symbol, expiry)
            spot = analysis.get("price", 0) or analysis.get("quote", {}).get("mid", 0)
            analysis["calls"] = _filter_options(chain, spot, "call")
            analysis["puts"] = _filter_options(chain, spot, "put")

            # NEW: Put/Call ratio from full chain
            analysis["put_call_ratio"] = compute_put_call_ratio(chain)

            # NEW: Options flow imbalance
            analysis["options_flow"] = compute_options_flow(chain)

            # NEW: IV percentile (use ATM options' IV)
            atm_ivs = [c.get("iv", 0) for c in chain
                       if abs(c.get("delta", 0)) > 0.35
                       and abs(c.get("delta", 0)) < 0.65
                       and c.get("iv", 0) > 0]
            if atm_ivs:
                avg_atm_iv = sum(atm_ivs) / len(atm_ivs)
                analysis["iv_percentile"] = compute_iv_percentile(
                    avg_atm_iv, [float(v) for v in
                                  (vix_history_5d or [])]
                )
                analysis["atm_iv"] = round(avg_atm_iv, 4)

        except Exception as e:
            logger.warning("Chain failed for %s: %s", symbol, e)
            analysis["chain_error"] = str(e)

    return analysis


def _filter_options(chain: list[dict], spot: float, opt_type: str,
                    max_spread_pct: float = 15.0,
                    min_delta: float = 0.15,
                    max_delta: float = 0.55,
                    max_count: int = 4) -> list[dict]:
    """Filter chain to tradeable near-money contracts."""
    from hybrid.config import MAX_OPTIONS_PER_TYPE

    max_count = MAX_OPTIONS_PER_TYPE
    filtered = []
    for c in chain:
        if c.get("option_type", "") != opt_type:
            continue
        # Spread check
        spread_pct = c.get("spread_pct", 999)
        if spread_pct > max_spread_pct:
            continue
        # Delta check
        delta = abs(c.get("delta", 0))
        if delta < min_delta or delta > max_delta:
            continue
        # Near money check (within 3% of spot)
        if spot > 0:
            strike = c.get("strike", 0)
            if abs(strike - spot) / spot > 0.03:
                continue
        # Must have a bid
        if c.get("bid", 0) <= 0:
            continue
        filtered.append(c)

    # Sort by volume descending, take top N
    filtered.sort(key=lambda x: x.get("volume", 0), reverse=True)
    return filtered[:max_count]


def _find_todays_expiry(broker: Broker, symbol: str) -> str | None:
    """Check if today is a valid 0DTE expiry for this underlying.

    Since 2022, SPY has DAILY expirations (Mon-Fri). QQQ also has daily expirations.
    IWM has Mon/Wed/Fri + some others. Rather than hardcoding schedules,
    we always try to fetch the chain — it returns empty if no expiry exists.
    """
    today = datetime.now(ET)
    today_str = today.strftime("%Y-%m-%d")
    weekday = today.weekday()  # 0=Mon, 4=Fri

    if weekday >= 5:  # Weekend
        return None

    # Note: SPY and QQQ have daily 0DTE since CBOE expanded in 2022.
    # IWM may not have every day. Always try the API — it's the source of truth.

    # Try to fetch a small chain for today's expiry
    try:
        chain = broker.get_option_chain(symbol, today_str)
        if chain and len(chain) > 0:
            return today_str
    except Exception as e:
        logger.debug("Chain check for %s %s: %s", symbol, today_str, e)

    return None


# ══════════════════════════════════════════════════════════════
# Section C: Prompt Formatting
# ══════════════════════════════════════════════════════════════

def build_digest(
    account: dict,
    positions: list[dict],
    daily_state: dict,
    market_context: dict,
    analyses: dict[str, dict],
    now_et: datetime | None = None,
) -> str:
    """Format everything into the single digest prompt for the LLM."""
    from hybrid import config

    if now_et is None:
        now_et = datetime.now(ET)

    lines: list[str] = []

    # ── Header ──
    lines.append("=== 0DTE OPTIONS SCALPER — MARKET SNAPSHOT ===")
    lines.append(f"Time: {now_et.strftime('%Y-%m-%d %H:%M')} ET")
    lines.append("")

    # ── Time Regime (NEW #6) ──
    time_regime = market_context.get("time_regime", {})
    if time_regime:
        lines.append(f"TIME REGIME: {time_regime.get('regime', '?')} — "
                     f"{time_regime.get('description', '')}")
        lines.append(f"  Theta: {time_regime.get('theta_note', '')}")
        lines.append("")

    # ── Account ──
    equity = account.get("equity", 0)
    cash = account.get("cash", 0)
    bp = account.get("buying_power", 0)
    dt_count = account.get("day_trade_count", 0)
    lines.append(f"ACCOUNT: Equity ${equity:,.0f} | Cash ${cash:,.0f} | "
                 f"Buying Power ${bp:,.0f} | Day Trades: {dt_count}/3")

    # ── Positions ──
    option_positions = [p for p in positions if p.get("asset_class") == "us_option"]
    if option_positions:
        lines.append(f"POSITIONS ({len(option_positions)}):")
        for p in option_positions:
            pl = p.get("unrealized_pl", 0)
            pl_pct = p.get("unrealized_plpc", 0) * 100
            lines.append(f"  {p['symbol']} | qty {p['qty']} | "
                         f"entry ${p['avg_entry_price']:.2f} | "
                         f"now ${p['current_price']:.2f} | "
                         f"P&L ${pl:.2f} ({pl_pct:+.1f}%)")
    else:
        lines.append("POSITIONS: None")

    # ── Daily P&L ──
    dpnl = daily_state.get("realized_pnl", 0)
    trades = daily_state.get("trades_today", 0)
    lines.append(f"DAILY P&L: ${dpnl:.2f} | Trades: {trades}")

    # ── Recent Trade History (NEW #5) ──
    recent_trades = market_context.get("recent_trades", [])
    if recent_trades:
        lines.append("")
        lines.append("RECENT TRADES:")
        for t in recent_trades:
            if t.get("action") == "EXIT":
                emoji = "W" if t.get("pnl", 0) > 0 else "L"
                lines.append(f"  [{emoji}] {t.get('symbol', '?')} {t.get('reason', '')} "
                             f"P&L ${t.get('pnl', 0):+.2f}")
            else:
                lines.append(f"  [E] {t.get('symbol', '?')} @ ${t.get('price', 0):.2f} "
                             f"(conf {t.get('confidence', 0)}) "
                             f"{t.get('reasoning', '')}")

    lines.append("")

    # ── Market Context ──
    lines.append("--- MARKET CONTEXT ---")

    # VIX
    vix = market_context.get("vix", {})
    if "error" not in vix:
        vix_val = vix.get("vix", "?")
        lines.append(f"VIX: {vix_val} ({vix.get('regime', '?')}) | "
                     f"Change: {vix.get('change_pct', 0):+.1f}%")
        # VIX 5-day history for IV context
        hist = vix.get("history_5d", [])
        if hist:
            hist_vals = [h.get("close", 0) for h in hist if h.get("close")]
            if hist_vals:
                lines.append(f"  VIX 5d range: {min(hist_vals):.1f} - {max(hist_vals):.1f} "
                             f"(avg {sum(hist_vals)/len(hist_vals):.1f})")
    else:
        lines.append("VIX: unavailable")

    # Fear & Greed
    fg = market_context.get("fear_greed", {})
    if "error" not in fg:
        lines.append(f"Fear & Greed: {fg.get('score', '?')} ({fg.get('rating', '?')}) "
                     f"→ {fg.get('signal', '?')}")
    else:
        lines.append("Fear & Greed: unavailable")

    # Sectors
    sectors = market_context.get("sectors", {})
    if "error" not in sectors:
        lines.append(f"Sectors: {sectors.get('up_count', '?')} up / "
                     f"{sectors.get('down_count', '?')} down | "
                     f"{sectors.get('breadth', '?')}")

    # Calendar (high-impact events only)
    cal = market_context.get("calendar", {})
    if isinstance(cal, list):
        high_impact = [e for e in cal if e.get("impact") == "HIGH"]
        if high_impact:
            lines.append(f"HIGH-IMPACT EVENTS: {', '.join(e['event'] for e in high_impact[:3])}")
            lines.append("  ⚠ Consider avoiding entries around these events")
        else:
            lines.append("Calendar: No high-impact events today")

    # Earnings
    earn = market_context.get("earnings", {})
    if isinstance(earn, list) and earn:
        syms = [e["symbol"] for e in earn[:5]]
        lines.append(f"Earnings today: {', '.join(syms)}")

    # News headlines (NEW #8)
    news = market_context.get("news", {})
    if isinstance(news, list) and news:
        lines.append("Headlines:")
        for n in news[:3]:
            lines.append(f"  • {n.get('headline', '?')[:80]}")

    lines.append("")

    # ── Cross-Underlying Comparison (NEW #9) ──
    if len(analyses) > 1:
        lines.append("--- CROSS-UNDERLYING COMPARISON ---")
        comparison_parts = []
        for sym, a in analyses.items():
            price = a.get("price", 0)
            mom = a.get("momentum_5")
            if price > 0 and mom is not None:
                comparison_parts.append(f"{sym}: ${price:.2f} ({mom:+.2f}%)")
        if comparison_parts:
            lines.append("  " + " | ".join(comparison_parts))

        # Check correlation
        momentums = {s: a.get("momentum_5", 0) for s, a in analyses.items()
                     if a.get("momentum_5") is not None}
        if len(momentums) >= 2:
            all_positive = all(m > 0 for m in momentums.values())
            all_negative = all(m < 0 for m in momentums.values())
            if all_positive:
                lines.append("  → All underlyings moving UP together (correlated rally)")
            elif all_negative:
                lines.append("  → All underlyings moving DOWN together (correlated selloff)")
            else:
                lines.append("  → Divergence detected (sector rotation — trade selectively)")
        lines.append("")

    # ── Per-Underlying Analysis ──
    for symbol, analysis in analyses.items():
        lines.append(f"--- {symbol} ANALYSIS ---")
        price = analysis.get("price", 0)
        vwap_val = analysis.get("vwap")

        # Price + VWAP
        price_str = f"Price: ${price:.2f}"
        if vwap_val:
            rel = "above" if price > vwap_val else "below"
            price_str += f" | VWAP: ${vwap_val:.2f} ({rel})"
        lines.append(price_str)

        # NEW #2: Overnight gap
        gap = analysis.get("overnight_gap")
        if gap:
            lines.append(f"  Overnight: {gap['type']} | Prev close ${gap['prev_close']:.2f} "
                         f"→ Open ${gap['today_open']:.2f} ({gap['gap_pct']:+.2f}%)")

        # NEW #4: Intraday narrative
        intraday = analysis.get("intraday")
        if intraday:
            lines.append(f"  Intraday: {intraday['narrative']}")
            lines.append(f"  Day range: ${intraday['day_low']:.2f}-${intraday['day_high']:.2f} "
                         f"(${intraday['day_range']:.2f}) | Position: {intraday['range_position']}")

        # Technicals
        techs = []
        rsi = analysis.get("rsi")
        if rsi is not None:
            label = "oversold" if rsi < 30 else ("overbought" if rsi > 70 else "neutral")
            techs.append(f"RSI(14): {rsi} ({label})")

        macd = analysis.get("macd")
        if macd:
            techs.append(f"MACD: {macd['crossover']}")

        bb = analysis.get("bollinger")
        if bb:
            techs.append(f"BB: {bb['position']} (%B={bb['pct_b']})")

        vol_ratio = analysis.get("volume_ratio")
        if vol_ratio:
            label = "high" if vol_ratio > 1.5 else ("low" if vol_ratio < 0.5 else "normal")
            techs.append(f"Vol: {vol_ratio}x ({label})")

        mom5 = analysis.get("momentum_5")
        if mom5 is not None:
            techs.append(f"Mom(5): {mom5:+.2f}%")

        if techs:
            lines.append("  " + " | ".join(techs))

        # Support/Resistance
        sup = analysis.get("support")
        res = analysis.get("resistance")
        if sup and res:
            lines.append(f"  Support: ${sup:.2f} | Resistance: ${res:.2f}")

        # NEW #3: IV percentile
        iv_pct = analysis.get("iv_percentile")
        atm_iv = analysis.get("atm_iv")
        if iv_pct and atm_iv:
            lines.append(f"  ATM IV: {atm_iv*100:.1f}% | IV Rank: {iv_pct['iv_rank']:.0f}% "
                         f"→ {iv_pct['regime']}")

        # NEW #1: Put/Call ratio
        pcr = analysis.get("put_call_ratio")
        if pcr and pcr.get("volume_pcr") is not None:
            lines.append(f"  Put/Call: {pcr['volume_pcr']:.2f} (vol) "
                         f"| {pcr.get('oi_pcr', '?')} (OI) → {pcr['sentiment']}")

        # NEW #7: Options flow
        flow = analysis.get("options_flow")
        if flow:
            lines.append(f"  Flow: Net Δ${flow['net_delta_flow']:+,.0f} | "
                         f"Normalized: {flow['normalized_flow']:+.3f} → {flow['bias']}")
            unusual = flow.get("unusual_activity", [])
            if unusual:
                for u in unusual[:2]:
                    lines.append(f"    ⚡ Unusual: {u['type'].upper()} ${u['strike']:.0f} "
                                 f"vol {u['vol']:,} vs OI {u['oi']:,} ({u['ratio']}x)")

        # Options chains
        for opt_type, label in [("calls", "CALLS"), ("puts", "PUTS")]:
            opts = analysis.get(opt_type, [])
            if opts:
                lines.append(f"  0DTE {label}:")
                for o in opts:
                    delta = o.get("delta", 0)
                    iv = o.get("iv", 0)
                    lines.append(
                        f"    ${o['strike']:.0f} | mid ${o.get('mid', 0):.2f} | "
                        f"Δ{delta:+.2f} | IV {iv*100:.0f}% | "
                        f"spread {o.get('spread_pct', 0):.0f}% | "
                        f"vol {o.get('volume', 0):,}"
                    )

        if not analysis.get("calls") and not analysis.get("puts"):
            if analysis.get("chain_error"):
                lines.append(f"  Chain: {analysis['chain_error']}")
            else:
                lines.append("  No tradeable 0DTE options found")

        # NEW #10: Historical price level context
        if sup and res and price > 0:
            dist_to_sup = ((price - sup) / price) * 100
            dist_to_res = ((res - price) / price) * 100
            if dist_to_sup < 0.5:
                lines.append(f"  📍 Price near SUPPORT (${sup:.2f}, {dist_to_sup:.1f}% away) "
                             f"— potential bounce zone")
            elif dist_to_res < 0.5:
                lines.append(f"  📍 Price near RESISTANCE (${res:.2f}, {dist_to_res:.1f}% away) "
                             f"— potential rejection zone")

            # Round number magnetism
            round_level = round(price / 5) * 5  # Nearest $5 level
            dist_to_round = abs(price - round_level) / price * 100
            if dist_to_round < 0.3:
                lines.append(f"  📍 Near round number ${round_level:.0f} "
                             f"(options pinning magnet)")

        lines.append("")

    # ── Rules ──
    lines.append("=== TRADING RULES ===")
    lines.append(f"- Only trade if your confidence > {config.SIGNAL_CONFIDENCE_THRESHOLD}")
    lines.append(f"- Max risk per trade: ${config.MAX_RISK_PER_TRADE:.0f} "
                 f"(premium x qty x 100)")
    lines.append(f"- Max {config.MAX_CONCURRENT_POSITIONS} concurrent positions")
    lines.append(f"- Stop loss: {config.STOP_LOSS_PCT:.0f}% | "
                 f"Profit target: {config.PROFIT_TARGET_PCT:.0f}%")
    lines.append("- LIMIT orders only (no market orders)")
    lines.append("- If VIX > 35: NO TRADE (crisis regime)")
    lines.append("- If high-impact macro event in next 30 min: NO TRADE")
    lines.append("- Prefer contracts with tight spreads (<10%), "
                 "good volume (>500), delta 0.20-0.45")
    lines.append("- High IV rank (>70%) favors tighter stops; "
                 "Low IV rank (<30%) favors wider targets")
    lines.append("- Put/Call > 1.2 is contrarian bullish; < 0.7 is contrarian bearish")
    lines.append("- Unusual options activity (vol >> OI) may signal smart money")
    lines.append("")

    # ── JSON schema ──
    lines.append("=== RESPOND WITH ONLY THIS JSON ===")
    lines.append("""{
  "decision": "TRADE" or "NO_TRADE",
  "underlying": "SPY" or "QQQ" or "IWM",
  "direction": "CALL" or "PUT",
  "strike": 650.0,
  "contract_symbol": "SPY260321P00650000",
  "qty": 1,
  "limit_price": 2.15,
  "confidence": 72,
  "reasoning": "Brief explanation of why this trade or why no trade"
}""")
    lines.append("")
    lines.append("If NO_TRADE, only 'decision', 'confidence', and 'reasoning' are required.")

    return "\n".join(lines)
