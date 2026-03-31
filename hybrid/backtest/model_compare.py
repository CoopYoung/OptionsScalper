"""Compare multiple Ollama models on the same historical day.

Usage:
    python -m hybrid.backtest.model_compare --date 2026-03-26
"""

import argparse
import logging
import os
import time
from datetime import date, timedelta

from hybrid.backtest.data_loader import load_days
from hybrid.backtest.replay import replay_day


def main():
    parser = argparse.ArgumentParser(description="Model Comparison Backtester")
    parser.add_argument("--date", required=True, help="Date to test (YYYY-MM-DD)")
    parser.add_argument("--underlying", default="SPY")
    parser.add_argument("--step", type=int, default=12, help="Bars between cycles (default: 12 = 60min)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated models (default: all installed)")

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    replay_date = date.fromisoformat(args.date)
    underlying = args.underlying

    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in resp.json()["models"]]

    # Load data once
    print(f"Loading data for {underlying} on {replay_date}...")
    days = load_days(underlying, replay_date, replay_date, use_alpaca=True)
    if not days:
        print("No data found")
        return

    day = days[0]
    day_change = (day.bars[-1].close - day.bars[0].open) / day.bars[0].open * 100
    correct = "PUT" if day_change < -0.3 else ("CALL" if day_change > 0.3 else "EITHER")

    print(f"Market: ${day.bars[0].open:.2f} → ${day.bars[-1].close:.2f} ({day_change:+.2f}%)")
    print(f"Correct bias: {correct}")
    print(f"Testing {len(models)} models with step={args.step} ({args.step * 5}min cycles)")
    print()

    # Run each model
    summaries = []
    for model in models:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model}")
        print(f"{'#'*70}")

        # Override the model
        os.environ["OLLAMA_MODEL"] = model

        # Force reimport to pick up new model
        import importlib
        import hybrid.config
        importlib.reload(hybrid.config)

        t0 = time.time()
        results = replay_day(day, step=args.step, all_days={underlying: day})
        elapsed = time.time() - t0

        trades = [r for r in results if r["decision"].get("decision") == "TRADE"]
        calls = sum(1 for t in trades if t["decision"].get("direction", "").upper() == "CALL")
        puts = sum(1 for t in trades if t["decision"].get("direction", "").upper() == "PUT")
        no_trades = len(results) - len(trades)
        parse_errors = sum(1 for r in results if r["decision"].get("parse_error"))

        avg_time = sum(r["elapsed_s"] for r in results) / len(results) if results else 0

        directions = set(t["decision"].get("direction", "").upper() for t in trades)
        got_it = correct in directions or correct == "EITHER"

        summaries.append({
            "model": model,
            "calls": calls,
            "puts": puts,
            "no_trades": no_trades,
            "parse_errors": parse_errors,
            "correct": got_it,
            "avg_response_s": round(avg_time, 1),
            "total_s": round(elapsed, 0),
        })

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON — {underlying} {replay_date} ({day_change:+.2f}%) → Correct: {correct}")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'CALL':>4} {'PUT':>4} {'SKIP':>4} {'ERR':>4} {'Correct?':>8} {'Avg(s)':>7} {'Total':>6}")
    print("-" * 80)
    for s in summaries:
        mark = "  ✓" if s["correct"] else "  ✗"
        print(f"{s['model']:<30} {s['calls']:>4} {s['puts']:>4} {s['no_trades']:>4} "
              f"{s['parse_errors']:>4} {mark:>8} {s['avg_response_s']:>6.1f}s {s['total_s']:>5.0f}s")


if __name__ == "__main__":
    main()
