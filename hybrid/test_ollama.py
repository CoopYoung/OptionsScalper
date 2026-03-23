#!/usr/bin/env python3
"""Test Ollama model with simulated market data — works anytime, market open or closed.

Usage:
    python -m hybrid.test_ollama                    # Bullish scenario
    python -m hybrid.test_ollama --scenario bearish  # Bearish scenario
    python -m hybrid.test_ollama --scenario choppy   # Range-bound / no trade
    python -m hybrid.test_ollama --scenario all      # Run all three back-to-back
    python -m hybrid.test_ollama --verbose           # Show full digest + raw output
"""

import argparse
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from hybrid.digest import build_digest, get_time_regime
from hybrid.llm import get_llm_client

ET = ZoneInfo("America/New_York")

# ── Simulated Scenarios ──────────────────────────────────────

SCENARIOS = {
    "bullish": {
        "description": "Strong bullish setup — RSI oversold bounce, MACD crossover, high P/C ratio",
        "account": {"equity": 69500, "cash": 69500, "buying_power": 139000, "day_trade_count": 1},
        "positions": [],
        "daily_state": {"realized_pnl": 0.0, "trades_today": 0},
        "market_context": {
            "vix": {"vix": 18.5, "regime": "NORMAL", "change_pct": -3.2},
            "fear_greed": {"score": 25, "rating": "extreme fear", "signal": "CONTRARIAN_BULLISH"},
            "sectors": {"up_count": 8, "down_count": 3, "breadth": "BROAD_RALLY"},
            "calendar": [],
            "earnings": [],
            "news": [
                {"headline": "Fed signals potential rate cut in June meeting minutes"},
                {"headline": "Tech earnings beat expectations across the board"},
                {"headline": "Consumer confidence rebounds to 6-month high"},
            ],
            "time_regime": get_time_regime(datetime(2026, 3, 23, 10, 45, tzinfo=ET)),
            "recent_trades": [],
        },
        "analyses": {
            "SPY": {
                "symbol": "SPY",
                "price": 562.30,
                "rsi": 28.5,
                "macd": {"macd": 0.15, "signal": -0.08, "histogram": 0.23, "crossover": "bullish_crossover"},
                "bollinger": {"upper": 575.00, "middle": 567.00, "lower": 559.00, "position": "lower_half", "pct_b": 0.21},
                "vwap": 560.80,
                "volume_ratio": 1.65,
                "momentum_5": 0.35,
                "momentum_20": -1.20,
                "support": 558.00,
                "resistance": 572.00,
                "intraday": {
                    "day_open": 559.00, "day_high": 563.50, "day_low": 557.80,
                    "current": 562.30, "day_range": 5.70, "range_position": "upper",
                    "pattern": "RECOVERY",
                    "narrative": "Opened $559.00, dipped to $557.80 then recovered to $563.50, now $562.30",
                },
                "overnight_gap": {
                    "prev_close": 561.50, "today_open": 559.00,
                    "gap": -2.50, "gap_pct": -0.45, "type": "GAP_DOWN",
                },
                "put_call_ratio": {
                    "volume_pcr": 1.45, "oi_pcr": 1.18,
                    "call_volume": 38000, "put_volume": 55100,
                    "sentiment": "BEARISH_FLOW (contrarian bullish)",
                },
                "options_flow": {
                    "net_delta_flow": -85000, "normalized_flow": -0.18,
                    "bias": "BEARISH_FLOW",
                    "unusual_activity": [],
                },
                "atm_iv": 0.22,
                "iv_percentile": {"iv_rank": 35, "iv_percentile": 38, "regime": "NORMAL_IV"},
                "calls": [
                    {"strike": 562, "mid": 1.85, "delta": 0.45, "iv": 0.22, "spread_pct": 4, "volume": 12500},
                    {"strike": 564, "mid": 1.15, "delta": 0.35, "iv": 0.21, "spread_pct": 6, "volume": 8200},
                    {"strike": 565, "mid": 0.85, "delta": 0.28, "iv": 0.21, "spread_pct": 8, "volume": 5600},
                ],
                "puts": [
                    {"strike": 560, "mid": 1.50, "delta": -0.40, "iv": 0.23, "spread_pct": 5, "volume": 9800},
                    {"strike": 558, "mid": 0.95, "delta": -0.30, "iv": 0.22, "spread_pct": 7, "volume": 6100},
                ],
            },
        },
        "time": datetime(2026, 3, 23, 10, 45, tzinfo=ET),
    },

    "bearish": {
        "description": "Bearish breakdown — RSI overbought, MACD bearish, VIX rising",
        "account": {"equity": 69500, "cash": 69500, "buying_power": 139000, "day_trade_count": 1},
        "positions": [],
        "daily_state": {"realized_pnl": 0.0, "trades_today": 0},
        "market_context": {
            "vix": {"vix": 28.5, "regime": "HIGH", "change_pct": 12.5},
            "fear_greed": {"score": 72, "rating": "greed", "signal": "CONTRARIAN_BEARISH"},
            "sectors": {"up_count": 2, "down_count": 9, "breadth": "BROAD_DECLINE"},
            "calendar": [{"event": "FOMC Minutes", "impact": "HIGH", "time": "14:00"}],
            "earnings": [],
            "news": [
                {"headline": "Inflation data comes in hotter than expected at 3.8%"},
                {"headline": "Treasury yields spike to 4.7% on hawkish Fed comments"},
                {"headline": "Tech stocks lead market decline on growth concerns"},
            ],
            "time_regime": get_time_regime(datetime(2026, 3, 23, 11, 15, tzinfo=ET)),
            "recent_trades": [],
        },
        "analyses": {
            "SPY": {
                "symbol": "SPY",
                "price": 555.20,
                "rsi": 72.5,
                "macd": {"macd": -0.55, "signal": -0.22, "histogram": -0.33, "crossover": "bearish_crossover"},
                "bollinger": {"upper": 570.00, "middle": 562.00, "lower": 554.00, "position": "below_lower", "pct_b": 0.08},
                "vwap": 558.50,
                "volume_ratio": 2.10,
                "momentum_5": -0.85,
                "momentum_20": -2.40,
                "support": 552.00,
                "resistance": 560.00,
                "intraday": {
                    "day_open": 560.00, "day_high": 561.20, "day_low": 554.80,
                    "current": 555.20, "day_range": 6.40, "range_position": "bottom",
                    "pattern": "SELL_OFF",
                    "narrative": "Opened $560.00, rallied to $561.20 then sold off to $554.80, now $555.20",
                },
                "overnight_gap": {
                    "prev_close": 563.00, "today_open": 560.00,
                    "gap": -3.00, "gap_pct": -0.53, "type": "GAP_DOWN",
                },
                "put_call_ratio": {
                    "volume_pcr": 0.65, "oi_pcr": 0.72,
                    "call_volume": 62000, "put_volume": 40300,
                    "sentiment": "BULLISH_FLOW (contrarian bearish)",
                },
                "options_flow": {
                    "net_delta_flow": 145000, "normalized_flow": 0.25,
                    "bias": "BULLISH_FLOW",
                    "unusual_activity": [
                        {"symbol": "SPY260323C00565000", "type": "call", "strike": 565, "vol": 22000, "oi": 4500, "ratio": 4.9},
                    ],
                },
                "atm_iv": 0.32,
                "iv_percentile": {"iv_rank": 82, "iv_percentile": 85, "regime": "HIGH_IV (favor selling premium / tighter stops)"},
                "calls": [
                    {"strike": 556, "mid": 1.90, "delta": 0.42, "iv": 0.31, "spread_pct": 6, "volume": 7200},
                    {"strike": 558, "mid": 1.20, "delta": 0.33, "iv": 0.30, "spread_pct": 8, "volume": 4500},
                ],
                "puts": [
                    {"strike": 554, "mid": 2.40, "delta": -0.48, "iv": 0.33, "spread_pct": 4, "volume": 15600},
                    {"strike": 552, "mid": 1.65, "delta": -0.38, "iv": 0.32, "spread_pct": 5, "volume": 11200},
                    {"strike": 550, "mid": 1.05, "delta": -0.28, "iv": 0.31, "spread_pct": 7, "volume": 8900},
                ],
            },
        },
        "time": datetime(2026, 3, 23, 11, 15, tzinfo=ET),
    },

    "choppy": {
        "description": "Range-bound chop — no clear direction, lunch lull, FOMC in 30 min",
        "account": {"equity": 69500, "cash": 69200, "buying_power": 138400, "day_trade_count": 2},
        "positions": [],
        "daily_state": {"realized_pnl": -45.00, "trades_today": 1},
        "market_context": {
            "vix": {"vix": 22.0, "regime": "NORMAL", "change_pct": 0.5},
            "fear_greed": {"score": 48, "rating": "neutral", "signal": "NEUTRAL"},
            "sectors": {"up_count": 5, "down_count": 6, "breadth": "MIXED"},
            "calendar": [{"event": "FOMC Rate Decision", "impact": "HIGH", "time": "14:00"}],
            "earnings": [{"symbol": "NVDA"}, {"symbol": "COST"}],
            "news": [
                {"headline": "Markets tread water ahead of Fed rate decision"},
                {"headline": "NVDA earnings after close — options imply 8% move"},
            ],
            "time_regime": get_time_regime(datetime(2026, 3, 23, 12, 30, tzinfo=ET)),
            "recent_trades": [
                {"symbol": "SPY260323C00560000", "action": "EXIT", "reason": "STOP_LOSS: -28%", "pnl": -45.00, "timestamp": "2026-03-23T11:15:00"},
            ],
        },
        "analyses": {
            "SPY": {
                "symbol": "SPY",
                "price": 559.80,
                "rsi": 48.2,
                "macd": {"macd": 0.02, "signal": 0.01, "histogram": 0.01, "crossover": "none"},
                "bollinger": {"upper": 565.00, "middle": 560.00, "lower": 555.00, "position": "lower_half", "pct_b": 0.48},
                "vwap": 559.90,
                "volume_ratio": 0.45,
                "momentum_5": 0.05,
                "momentum_20": -0.30,
                "support": 557.00,
                "resistance": 563.00,
                "intraday": {
                    "day_open": 560.50, "day_high": 562.00, "day_low": 558.20,
                    "current": 559.80, "day_range": 3.80, "range_position": "lower",
                    "pattern": "RANGE",
                    "narrative": "Opened $560.50, ranging between $558.20-$562.00, now $559.80",
                },
                "overnight_gap": {
                    "prev_close": 560.00, "today_open": 560.50,
                    "gap": 0.50, "gap_pct": 0.09, "type": "FLAT_OPEN",
                },
                "put_call_ratio": {
                    "volume_pcr": 1.05, "oi_pcr": 0.98,
                    "call_volume": 42000, "put_volume": 44100,
                    "sentiment": "NEUTRAL_FLOW",
                },
                "options_flow": {
                    "net_delta_flow": -12000, "normalized_flow": -0.03,
                    "bias": "NEUTRAL",
                    "unusual_activity": [],
                },
                "atm_iv": 0.26,
                "iv_percentile": {"iv_rank": 55, "iv_percentile": 52, "regime": "NORMAL_IV"},
                "calls": [
                    {"strike": 560, "mid": 1.35, "delta": 0.42, "iv": 0.26, "spread_pct": 8, "volume": 5200},
                    {"strike": 562, "mid": 0.75, "delta": 0.30, "iv": 0.25, "spread_pct": 12, "volume": 2800},
                ],
                "puts": [
                    {"strike": 558, "mid": 1.20, "delta": -0.40, "iv": 0.27, "spread_pct": 9, "volume": 4600},
                    {"strike": 556, "mid": 0.70, "delta": -0.28, "iv": 0.26, "spread_pct": 13, "volume": 2100},
                ],
            },
        },
        "time": datetime(2026, 3, 23, 12, 30, tzinfo=ET),
    },
}


def run_scenario(name: str, verbose: bool = False) -> dict:
    """Run a single test scenario against the LLM."""
    scenario = SCENARIOS[name]
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {name.upper()}")
    print(f"  {scenario['description']}")
    print(f"{'='*60}")

    # Build digest from simulated data
    digest = build_digest(
        account=scenario["account"],
        positions=scenario["positions"],
        daily_state=scenario["daily_state"],
        market_context=scenario["market_context"],
        analyses=scenario["analyses"],
        now_et=scenario["time"],
    )

    if verbose:
        print("\n--- DIGEST ---")
        print(digest)
        print(f"--- {len(digest)} chars, ~{len(digest.split()) * 3 // 4} tokens ---\n")

    # Call LLM
    print(f"\n  Calling Ollama (0xroyce/plutus)...")
    llm = get_llm_client("ollama")

    start = time.time()
    result = llm.decide(digest)
    elapsed = time.time() - start

    # Display results
    decision = result.get("decision", "?")
    confidence = result.get("confidence", 0)
    reasoning = result.get("reasoning", "No reasoning provided")

    emoji = "🟢" if decision == "TRADE" else "⏸"
    print(f"\n  {emoji} Decision: {decision}")
    print(f"  Confidence: {confidence}")
    print(f"  Reasoning: {reasoning}")

    if decision == "TRADE":
        print(f"  Underlying: {result.get('underlying', '?')}")
        print(f"  Direction: {result.get('direction', '?')}")
        print(f"  Strike: ${result.get('strike', '?')}")
        print(f"  Qty: {result.get('qty', '?')}")
        print(f"  Limit: ${result.get('limit_price', '?')}")

    print(f"\n  Response time: {elapsed:.1f}s")
    print(f"  Parse error: {result.get('parse_error', False)}")

    if verbose and result.get("_raw_output"):
        print(f"\n--- RAW OUTPUT ---")
        print(result["_raw_output"])
        print("---")

    # Score the response
    print(f"\n  --- QUALITY CHECK ---")
    issues = []

    if result.get("parse_error"):
        issues.append("❌ Failed to parse JSON from output")

    if name == "bullish" and decision == "NO_TRADE":
        issues.append("⚠ Expected TRADE on a bullish setup")
    elif name == "bullish" and decision == "TRADE":
        direction = result.get("direction", "").upper()
        if direction == "PUT":
            issues.append("⚠ Chose PUT on a bullish setup — questionable")
        elif direction == "CALL":
            print("  ✅ Correctly chose CALL on bullish setup")

    if name == "bearish" and decision == "TRADE":
        direction = result.get("direction", "").upper()
        if direction == "CALL":
            issues.append("⚠ Chose CALL on a bearish setup — questionable")
        elif direction == "PUT":
            print("  ✅ Correctly chose PUT on bearish setup")

    if name == "choppy":
        if decision == "NO_TRADE":
            print("  ✅ Correctly avoided trade in choppy/FOMC conditions")
        else:
            issues.append("⚠ Traded in choppy conditions with FOMC 90 min away — risky")

    if confidence > 0 and confidence < 20:
        issues.append("⚠ Very low confidence — model may not understand the prompt")

    if not issues:
        print("  ✅ Response looks reasonable")
    else:
        for issue in issues:
            print(f"  {issue}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Test Ollama with simulated market scenarios")
    parser.add_argument("--scenario", choices=["bullish", "bearish", "choppy", "all"],
                        default="bullish", help="Which scenario to test")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full digest and raw LLM output")
    args = parser.parse_args()

    print("🧪 OptionsScalper — Ollama Model Test")
    print(f"Testing against: {args.scenario} scenario(s)")

    if args.scenario == "all":
        results = {}
        for name in ["bullish", "bearish", "choppy"]:
            results[name] = run_scenario(name, args.verbose)
            print()

        # Summary
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)
        for name, r in results.items():
            d = r.get("decision", "?")
            c = r.get("confidence", 0)
            t = r.get("_elapsed_s", "?")
            pe = "⚠ PARSE ERROR" if r.get("parse_error") else ""
            print(f"  {name:8s}: {d:8s} (conf {c:3d}) | {t}s {pe}")
    else:
        run_scenario(args.scenario, args.verbose)


if __name__ == "__main__":
    main()
