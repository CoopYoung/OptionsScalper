"""LLM replay backtester — test how the model reacts to historical days.

Replays historical market data through the same digest→LLM→decision pipeline
used in production, but with simulated option chains (Black-Scholes).

Usage:
    python -m hybrid.backtest.replay --date 2026-03-26 --underlying SPY
    python -m hybrid.backtest.replay --date 2026-03-25 --underlying SPY --step 4
    python -m hybrid.backtest.replay --date 2026-03-26 --underlying SPY,QQQ,IWM
"""

import argparse
import logging
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from hybrid.backtest.data_loader import (
    Bar,
    DaySnapshot,
    generate_chain,
    load_days,
)
from hybrid.digest import (
    build_digest,
    compute_bollinger,
    compute_intraday_narrative,
    compute_macd,
    compute_momentum,
    compute_rsi,
    compute_vwap,
    compute_volume_ratio,
    compute_put_call_ratio,
    compute_options_flow,
    _filter_options,
)
from hybrid.llm import get_llm_client

logger = logging.getLogger(__name__)
ET = ZoneInfo("America/New_York")


def _bars_to_dicts(bars: list[Bar]) -> list[dict]:
    """Convert Bar dataclasses to dicts matching broker format."""
    return [
        {
            "timestamp": b.timestamp.isoformat(),
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
            "vwap": b.vwap,
        }
        for b in bars
    ]


def _chain_to_dicts(chain) -> list[dict]:
    """Convert SimOption list to dicts matching broker chain format."""
    result = []
    for c in chain:
        mid = c.mid
        bid = c.bid
        ask = c.ask
        spread = round(ask - bid, 2)
        spread_pct = round(spread / mid * 100, 1) if mid > 0 else 999
        result.append({
            "symbol": c.symbol,
            "underlying": c.symbol[:3],
            "strike": c.strike,
            "expiration": c.expiry,
            "option_type": c.option_type,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
            "spread_pct": spread_pct,
            "last_trade": mid,
            "volume": c.volume,
            "open_interest": c.open_interest,
            "iv": c.iv,
            "delta": c.delta,
            "gamma": c.gamma,
            "theta": c.theta,
            "vega": c.vega,
            "rho": 0.0,
        })
    return result


def _build_analysis(bars: list[Bar], chain_dicts: list[dict],
                    symbol: str, spot: float) -> dict:
    """Build the per-underlying analysis dict from historical data."""
    analysis = {"symbol": symbol, "price": spot}

    bar_dicts = _bars_to_dicts(bars)
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]
    volumes = [b.volume for b in bars]

    if len(closes) >= 5:
        analysis["rsi"] = compute_rsi(closes)
        analysis["macd"] = compute_macd(closes)
        analysis["bollinger"] = compute_bollinger(closes)
        analysis["vwap"] = compute_vwap(bar_dicts)
        analysis["volume_ratio"] = compute_volume_ratio(volumes)
        analysis["momentum_5"] = compute_momentum(closes, 5)
        analysis["momentum_20"] = compute_momentum(closes, 20) if len(closes) >= 20 else None

    if len(highs) >= 10:
        analysis["resistance"] = round(max(highs[-20:]), 2)
        analysis["support"] = round(min(lows[-20:]), 2)

    if bar_dicts:
        analysis["intraday"] = compute_intraday_narrative(bar_dicts)

    # Overnight gap
    if len(bars) >= 2:
        analysis["overnight_gap"] = {
            "type": "GAP_UP" if bars[0].open > bars[0].close else "GAP_DOWN",
            "prev_close": bars[0].open,  # Approximate
            "today_open": bars[0].open,
            "gap_pct": 0.0,
        }

    # Quote (for cross-underlying comparison)
    analysis["quote"] = {"mid": spot, "bid": spot - 0.01, "ask": spot + 0.01}

    # Filter options for the digest
    analysis["calls"] = _filter_options(chain_dicts, spot, "call")
    analysis["puts"] = _filter_options(chain_dicts, spot, "put")
    analysis["put_call_ratio"] = compute_put_call_ratio(chain_dicts)
    analysis["options_flow"] = compute_options_flow(chain_dicts)

    return analysis


def replay_day(
    day: DaySnapshot,
    step: int = 6,
    provider: str | None = None,
    all_days: dict[str, DaySnapshot] | None = None,
) -> list[dict]:
    """Replay one day through the LLM decision pipeline.

    Args:
        day: Historical day data
        step: Number of 5-min bars between cycles (6 = 30min, 2 = 10min)
        provider: LLM provider override
        all_days: Dict of symbol→DaySnapshot for multi-underlying
    """
    from hybrid import config

    llm = get_llm_client(provider)
    results = []
    underlyings = all_days or {day.underlying: day}

    # Market close at 16:00 ET = 390 minutes from 9:30
    market_open_utc = datetime(
        day.date.year, day.date.month, day.date.day,
        13, 30, tzinfo=ET.key and None,
    )

    total_bars = len(day.bars)
    print(f"\n{'='*70}")
    print(f"REPLAY: {day.date} | {', '.join(underlyings.keys())} | "
          f"{total_bars} bars | step={step} ({step*5}min cycles)")
    print(f"{'='*70}")

    held_underlyings: set[str] = set()
    open_positions: list[dict] = []
    realized_pnl = 0.0
    trades_taken = 0

    for i in range(14, total_bars, step):  # Start after enough bars for RSI
        bar = day.bars[i]
        bar_time = bar.timestamp
        # Convert to ET
        if bar_time.tzinfo is None:
            from zoneinfo import ZoneInfo
            bar_time = bar_time.replace(tzinfo=ZoneInfo("UTC"))
        et_time = bar_time.astimezone(ET)
        time_str = et_time.strftime("%H:%M")

        # Skip if outside entry window
        if time_str < config.ENTRY_START_ET or time_str > "15:30":
            continue

        minutes_to_close = max(1, (16 * 60) - (et_time.hour * 60 + et_time.minute))

        # Build analyses for all underlyings
        analyses = {}
        for sym, sym_day in underlyings.items():
            if i >= len(sym_day.bars):
                continue
            spot = sym_day.bars[i].close
            expiry_str = sym_day.date.isoformat()

            # Generate simulated chain at this point in time
            chain = generate_chain(
                spot=spot,
                vix=sym_day.vix,
                minutes_to_close=minutes_to_close,
                underlying=sym,
                expiry=expiry_str,
            )
            chain_dicts = _chain_to_dicts(chain)
            bars_so_far = sym_day.bars[:i + 1]
            analyses[sym] = _build_analysis(bars_so_far, chain_dicts, sym, spot)

        if not analyses:
            continue

        # Build market context (simplified for backtest)
        market_context = {
            "vix": {
                "vix": day.vix,
                "regime": "HIGH" if day.vix > 25 else ("LOW" if day.vix < 15 else "NORMAL"),
                "change_pct": 0.0,
                "history_5d": [],
            },
            "fear_greed": {
                "score": day.fear_greed,
                "rating": "extreme fear" if day.fear_greed <= 25 else (
                    "fear" if day.fear_greed <= 40 else "neutral"),
                "signal": "EXTREME_FEAR" if day.fear_greed <= 25 else (
                    "FEAR" if day.fear_greed <= 40 else "NEUTRAL"),
            },
            "sectors": {"up_count": 5, "down_count": 6, "breadth": day.sector_breadth},
            "time_regime": _get_time_regime(et_time),
            "recent_trades": [],
        }

        # Build digest
        account = {"equity": 69500, "cash": 69500, "buying_power": 278000}
        daily_state = {
            "realized_pnl": realized_pnl,
            "trades_today": trades_taken,
        }

        digest = build_digest(
            account=account,
            positions=open_positions,
            daily_state=daily_state,
            market_context=market_context,
            analyses=analyses,
            now_et=et_time,
            held_underlyings=held_underlyings,
        )

        # Call LLM
        t0 = time.time()
        decision = llm.decide(digest)
        elapsed = time.time() - t0

        d = decision.get("decision", "?")
        direction = decision.get("direction", "")
        underlying = decision.get("underlying", "")
        strike = decision.get("strike", "")
        conf = decision.get("confidence", 0)
        reasoning = decision.get("reasoning", "")[:80]

        # Show the spot prices
        spots = {sym: underlyings[sym].bars[i].close
                 for sym in underlyings if i < len(underlyings[sym].bars)}
        spot_str = " | ".join(f"{s}=${p:.2f}" for s, p in spots.items())

        if d == "TRADE":
            # Show what the technicals said
            a = analyses.get(underlying, {})
            rsi = a.get("rsi", "?")
            mom = a.get("momentum_5")
            mom_str = f"{mom:+.2f}%" if mom is not None else "?"

            print(f"  {time_str} | {spot_str}")
            print(f"         → TRADE {underlying} {direction} ${strike} "
                  f"conf={conf} | RSI={rsi} Mom={mom_str}")
            print(f"         {reasoning}")
        else:
            print(f"  {time_str} | {spot_str} → NO_TRADE conf={conf} | {reasoning[:60]}")

        results.append({
            "time": time_str,
            "spots": spots,
            "decision": decision,
            "elapsed_s": round(elapsed, 1),
        })

    # Summary
    print(f"\n--- SUMMARY ---")
    trades = [r for r in results if r["decision"].get("decision") == "TRADE"]
    no_trades = len(results) - len(trades)
    calls = sum(1 for t in trades if t["decision"].get("direction", "").upper() == "CALL")
    puts = sum(1 for t in trades if t["decision"].get("direction", "").upper() == "PUT")
    print(f"Cycles: {len(results)} | Trades: {len(trades)} (CALL:{calls} PUT:{puts}) | "
          f"No-Trade: {no_trades}")

    # Check what the right answer was
    first_bar = day.bars[0]
    last_bar = day.bars[-1]
    day_change = (last_bar.close - first_bar.open) / first_bar.open * 100
    correct_dir = "PUT" if day_change < -0.3 else ("CALL" if day_change > 0.3 else "EITHER")
    print(f"Market: ${first_bar.open:.2f} → ${last_bar.close:.2f} ({day_change:+.2f}%) "
          f"→ Correct bias: {correct_dir}")

    if trades:
        directions = set(t["decision"].get("direction", "").upper() for t in trades)
        if correct_dir in directions:
            print(f"✓ Model DID pick {correct_dir}")
        elif correct_dir != "EITHER":
            print(f"✗ Model picked {directions} but market favored {correct_dir}")

    return results


def _get_time_regime(now: datetime) -> dict:
    """Simple time regime for backtest."""
    hour = now.hour
    minute = now.minute
    mins_since_open = (hour - 9) * 60 + (minute - 30)

    if mins_since_open <= 30:
        return {"regime": "OPENING", "description": "First 30 min — volatile",
                "theta_note": "Low theta impact early"}
    elif mins_since_open <= 120:
        return {"regime": "MORNING", "description": "Morning session — trend-following",
                "theta_note": "Moderate theta"}
    elif mins_since_open <= 270:
        return {"regime": "MIDDAY", "description": "Midday chop — be selective",
                "theta_note": "Theta accelerating"}
    elif mins_since_open <= 330:
        return {"regime": "AFTERNOON", "description": "Afternoon — trends resume",
                "theta_note": "Significant theta decay"}
    else:
        return {"regime": "POWER_HOUR", "description": "Final hour — max theta",
                "theta_note": "Extreme theta decay"}


def main():
    parser = argparse.ArgumentParser(description="LLM Replay Backtester")
    parser.add_argument("--date", required=True,
                        help="Date to replay (YYYY-MM-DD)")
    parser.add_argument("--underlying", default="SPY",
                        help="Comma-separated underlyings (default: SPY)")
    parser.add_argument("--step", type=int, default=6,
                        help="Bars between cycles (6=30min, 2=10min, default: 6)")
    parser.add_argument("--provider", default=None,
                        help="LLM provider (default: from config)")
    parser.add_argument("--days", type=int, default=1,
                        help="Number of trading days to replay")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    replay_date = date.fromisoformat(args.date)
    underlyings = [s.strip() for s in args.underlying.split(",")]
    end_date = replay_date + timedelta(days=max(0, args.days - 1))

    # Load historical data for all underlyings
    print(f"Loading historical data for {underlyings} "
          f"from {replay_date} to {end_date}...")

    all_underlying_days: dict[str, list[DaySnapshot]] = {}
    for sym in underlyings:
        days = load_days(sym, replay_date, end_date, use_alpaca=True)
        if days:
            all_underlying_days[sym] = days
            print(f"  {sym}: {len(days)} days, "
                  f"{sum(len(d.bars) for d in days)} bars")
        else:
            print(f"  {sym}: no data found")

    if not all_underlying_days:
        print("No data loaded — exiting")
        return

    # Replay each day
    first_sym = list(all_underlying_days.keys())[0]
    for day_idx, primary_day in enumerate(all_underlying_days[first_sym]):
        # Gather all underlyings for this date
        day_underlyings = {}
        for sym, sym_days in all_underlying_days.items():
            if day_idx < len(sym_days):
                day_underlyings[sym] = sym_days[day_idx]

        replay_day(
            day=primary_day,
            step=args.step,
            provider=args.provider,
            all_days=day_underlyings,
        )


if __name__ == "__main__":
    main()
