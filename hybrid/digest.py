"""Data gathering, technical indicators, and prompt formatting.

Python does ALL the work here. The LLM just reads a pre-digested summary
and returns a JSON decision. No tool calling needed.
"""

import logging
import math
from datetime import datetime
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

    return ctx


def gather_underlying_analysis(broker: Broker, symbol: str,
                                expiry: str | None) -> dict:
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
        else:
            analysis["bars_error"] = "No bars returned"
    except Exception as e:
        logger.warning("Bars/technicals failed for %s: %s", symbol, e)
        analysis["bars_error"] = str(e)

    # Options chain (0DTE)
    if expiry:
        try:
            chain = broker.get_option_chain(symbol, expiry)
            spot = analysis.get("price", 0) or analysis.get("quote", {}).get("mid", 0)
            analysis["calls"] = _filter_options(chain, spot, "call")
            analysis["puts"] = _filter_options(chain, spot, "put")
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

    SPY has Mon/Wed/Fri expirations, QQQ has Mon/Wed/Fri, IWM has monthly + some weeklies.
    Rather than fetching all expirations (heavy API call), we use day-of-week heuristics
    and try to fetch the chain. If the chain returns contracts, today is valid.
    """
    today = datetime.now(ET)
    today_str = today.strftime("%Y-%m-%d")
    weekday = today.weekday()  # 0=Mon, 4=Fri

    # SPY and QQQ have Mon/Wed/Fri 0DTE. IWM has Fri + some others.
    # Quick heuristic: try today for all — the chain call will just return empty if invalid.
    if weekday >= 5:  # Weekend
        return None

    # For SPY/QQQ, check M/W/F
    if symbol in ("SPY", "QQQ") and weekday not in (0, 2, 4):
        # Tue/Thu — these tickers usually don't have 0DTE
        # But check anyway in case of special expirations
        pass

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
    lines.append("")

    # ── Market Context ──
    lines.append("--- MARKET CONTEXT ---")

    # VIX
    vix = market_context.get("vix", {})
    if "error" not in vix:
        lines.append(f"VIX: {vix.get('vix', '?')} ({vix.get('regime', '?')}) | "
                     f"Change: {vix.get('change_pct', 0):+.1f}%")
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

    lines.append("")

    # ── Per-Underlying Analysis ──
    for symbol, analysis in analyses.items():
        lines.append(f"--- {symbol} ANALYSIS ---")
        price = analysis.get("price", 0)
        vwap_val = analysis.get("vwap")

        price_str = f"Price: ${price:.2f}"
        if vwap_val:
            rel = "above" if price > vwap_val else "below"
            price_str += f" | VWAP: ${vwap_val:.2f} ({rel})"
        lines.append(price_str)

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
